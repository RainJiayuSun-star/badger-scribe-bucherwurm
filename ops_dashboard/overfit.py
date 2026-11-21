"""Overfitting-test verdicts for the Training tab: the same rule as finetune/overfit.py, kept here so the
dashboard runs without the training package."""

NOISE = 0.005  # repeating identical settings moved CER by about this much (EXPERIMENT_LOG.md)


def verdict(held, train):
    """held, train: {epoch: CER}. A short verdict for one run."""
    if len(held) < 2:
        return 'too few epochs'
    best, last = min(held, key=held.get), max(held)
    if held[last] - held[best] <= NOISE:
        return f'no overfitting seen (best epoch {best})'
    if best in train and last in train and train[last] < train[best]:
        return f'overfitting after epoch {best}: held-out +{held[last] - held[best]:.4f}, training -{train[best] - train[last]:.4f}'
    return f'unstable: held-out +{held[last] - held[best]:.4f} after epoch {best}, training CER not falling'


def family(run_id, fold):
    """The test a fold run belongs to: its ID without the trailing fold name."""
    return run_id[:-len(fold) - 1] if fold and run_id.endswith('-' + fold) else run_id


def mean_curve(curves):
    """Mean CER per epoch over the epochs every curve has."""
    shared = sorted(set.intersection(*[set(c) for c in curves])) if curves else []
    return [[epoch, sum(c[epoch] for c in curves) / len(curves)] for epoch in shared]


def summarize(rows):
    """Group runs that logged training-page CER into tests, with per-run verdicts and the epoch count to use."""
    groups = {}
    for row in rows:
        train = {e: c for e, c in row.get('train', [])}
        if not train:
            continue
        held = {e: c for e, c in row['val']}
        last = max(held) if held else None
        groups.setdefault(family(row['id'], row.get('fold')), []).append({
            'id': row['id'], 'fold': row.get('fold'), 'model': row['model'], 'held': held, 'train': train,
            'best_epoch': min(held, key=held.get) if held else None, 'last_epoch': last,
            'last_held': held.get(last), 'last_train': train.get(last), 'verdict': verdict(held, train)})
    tests = []
    for name, runs in sorted(groups.items()):
        held_mean = mean_curve([r['held'] for r in runs if r['held']])
        pick = None
        if held_mean:
            best = min(c for _, c in held_mean)
            pick = min(e for e, c in held_mean if c - best <= NOISE)  # fewest epochs as good as we can measure
        tests.append({'name': name, 'model': runs[0]['model'], 'runs': [{k: v for k, v in r.items() if k not in ('held', 'train')} for r in runs],
                      'held_mean': held_mean, 'train_mean': mean_curve([r['train'] for r in runs]), 'epochs_to_use': pick})
    return tests
