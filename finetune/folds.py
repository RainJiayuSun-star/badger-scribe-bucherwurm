"""Leave-one-document-out folds for evaluating on unseen writers.

Every test page comes from a document that is not in train.csv, so a model has
to be scored on documents it never trained on. Each fold holds out documents
from one category; the model trains on every other page (all categories) and
is scored on the held-out documents.

* German letters: one fold per document (4 folds).
* Survey notes: one fold per document (9 folds).
* Account books: 18 small documents in 6 groups, balanced by page count.

Fold files list page IDs only, no label text.

    python -m finetune.folds          # writes runs/folds/<fold>.json and prints a summary
    python -m finetune.folds --data-root ~/badger-scribe-data-1118 --account-docs-from badger-scribe-data/train.csv

--account-docs-from groups the account books listed in that older train.csv exactly as that
release did, so held-out pages match earlier runs. Newer books are always trained on, and a
book that has since moved to test.csv drops out of its group.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import ROOT, default_data_root

LEAVE_ONE_OUT = ("kade_letters", "survey_notes")
GROUPED = {"dominy_accounts": 6}
# Folds whose held-out set is too small or too unusual to rank methods on alone.
NOISY_MIN_PAGES = 15


def group_documents(page_counts: dict[str, int], groups: int) -> list[list[str]]:
    """Split documents into groups with similar page totals, deterministically.

    Largest documents first, each to the group with the fewest pages so far
    (ties go to the lower-numbered group).
    """
    buckets: list[list[str]] = [[] for _ in range(groups)]
    totals = [0] * groups
    for doc in sorted(page_counts, key=lambda d: (-page_counts[d], d)):
        target = min(range(groups), key=lambda i: (totals[i], i))
        buckets[target].append(doc)
        totals[target] += page_counts[doc]
    return [sorted(bucket) for bucket in buckets if bucket]


def build_folds(rows: list[dict[str, str]], group_counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """group_counts, if given, are the page counts (from an older release) that the account groups are built from."""
    pages_by_doc: dict[str, list[str]] = defaultdict(list)
    category_of: dict[str, str] = {}
    for row in rows:
        pages_by_doc[row["doc_id"]].append(row["page_id"])
        category_of[row["doc_id"]] = row["category"]
    docs_by_category: dict[str, list[str]] = defaultdict(list)
    for doc in sorted(pages_by_doc):
        docs_by_category[category_of[doc]].append(doc)

    held_out: list[tuple[str, str, list[str]]] = []
    for category in LEAVE_ONE_OUT:
        held_out += [(f"{doc}", category, [doc]) for doc in docs_by_category.get(category, [])]
    for category, groups in GROUPED.items():
        counts = group_counts if group_counts is not None else {doc: len(pages_by_doc[doc]) for doc in docs_by_category.get(category, [])}
        for index, docs in enumerate(group_documents(counts, groups), start=1):
            docs = [doc for doc in docs if doc in pages_by_doc]  # a book moved to test since that release
            if docs:
                held_out.append((f"{category.split('_')[0]}_group{index}", category, docs))

    folds = []
    for name, category, docs in held_out:
        eval_docs = set(docs)
        eval_ids = [row["page_id"] for row in rows if row["doc_id"] in eval_docs]
        train_ids = [row["page_id"] for row in rows if row["doc_id"] not in eval_docs]
        human = sum(row["label_source"] == "human" for row in rows if row["doc_id"] in eval_docs)
        folds.append({
            "name": name,
            "category": category,
            "eval_docs": sorted(eval_docs),
            "eval_ids": eval_ids,
            "train_ids": train_ids,
            "eval_human_pages": human,
            "noisy": len(eval_ids) < NOISY_MIN_PAGES,
        })
    return folds


def check_folds(folds: list[dict[str, Any]], rows: list[dict[str, str]], always_train: set[str] = frozenset()) -> None:
    """Fail loudly on leakage or gaps. Documents in always_train must never be held out."""
    doc_of = {row["page_id"]: row["doc_id"] for row in rows}
    covered: Counter[str] = Counter()
    for fold in folds:
        eval_docs = {doc_of[p] for p in fold["eval_ids"]}
        train_docs = {doc_of[p] for p in fold["train_ids"]}
        if eval_docs & train_docs:
            raise ValueError(f"{fold['name']}: documents on both sides: {sorted(eval_docs & train_docs)}")
        if len(fold["eval_ids"]) + len(fold["train_ids"]) != len(rows):
            raise ValueError(f"{fold['name']}: pages missing from the fold")
        covered.update(fold["eval_ids"])
    leaked = [row["page_id"] for row in rows if row["doc_id"] in always_train and covered[row["page_id"]]]
    if leaked:
        raise ValueError(f"{len(leaked)} pages of always-trained documents are held out, e.g. {leaked[:3]}")
    unscored = [row["page_id"] for row in rows if row["doc_id"] not in always_train and covered[row["page_id"]] != 1]
    if unscored:
        raise ValueError(f"{len(unscored)} pages are not held out exactly once, e.g. {unscored[:3]}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "folds")
    parser.add_argument("--account-docs-from", type=Path, help="older train.csv whose account books are the only ones grouped")
    args = parser.parse_args()
    rows = read_csv(args.data_root / "train.csv")
    group_counts = None
    if args.account_docs_from:
        group_counts = Counter(row["doc_id"] for row in read_csv(args.account_docs_from) if row["category"] in GROUPED)
    always_train = {row["doc_id"] for row in rows if row["category"] in GROUPED and group_counts is not None
                    and row["doc_id"] not in group_counts}
    folds = build_folds(rows, group_counts)
    check_folds(folds, rows, always_train)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        (args.out_dir / f"{fold['name']}.json").write_text(json.dumps(fold, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(folds)} folds to {args.out_dir}")
    if always_train:
        pages = sum(row["doc_id"] in always_train for row in rows)
        print(f"  {len(always_train)} newer account books ({pages} pages) are trained on in every fold")
    for fold in folds:
        flag = "  (noisy: small held-out set)" if fold["noisy"] else ""
        print(f"  {fold['name']:<18} {fold['category']:<16} eval {len(fold['eval_ids']):>3} pages "
              f"({fold['eval_human_pages']} human), train {len(fold['train_ids'])}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
