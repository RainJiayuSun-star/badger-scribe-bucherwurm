"""Build the MLM26 draft DOCX and its two aggregate-metric figures."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "MLM26_DRAFT.md"
OUT = HERE / "BadgerScribe_Bucherwurm.docx"
FIGURES = HERE / "figures"
LINK = re.compile(r"\[([^]]+)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[([^]]+)\]\(([^)]+)\)")


def figures() -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    runs = ["01", "02", "03", "04", "05", "06", "07"]
    scores = [0.143, 0.150, 0.138, 0.129, 0.126, 0.124, 0.124]
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.plot(runs, scores, marker="o", linewidth=2.2, color="#22577A")
    for x, y in zip(runs, scores):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Submission")
    ax.set_ylabel("Public CER  lower is better")
    ax.set_ylim(0.118, 0.155)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "leaderboard_progression.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    categories = ["German letters", "Survey notes", "Account books"]
    zero = [0.215, 0.292, 0.459]
    tuned = [0.0872, 0.077, 0.2452]
    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    width = 0.34
    ax.bar([i - width / 2 for i in x], zero, width, label="Zero shot", color="#9DB4C0")
    ax.bar([i + width / 2 for i in x], tuned, width, label="Selected fine tuned", color="#22577A")
    for i, (a, b) in enumerate(zip(zero, tuned)):
        ax.text(i - width / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x), categories)
    ax.set_ylabel("Held out CER  lower is better")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "category_cer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    rel = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    node = OxmlElement("w:hyperlink")
    node.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "22577A")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    node.append(run)
    paragraph._p.append(node)


def add_inline(paragraph, text: str) -> None:
    def add_text(value: str) -> None:
        parts = value.split("`")
        for index, part in enumerate(parts):
            run = paragraph.add_run(part)
            if index % 2:
                run.font.name = "Aptos Mono"
                run.font.size = Pt(9.5)

    pos = 0
    for match in LINK.finditer(text):
        add_text(text[pos : match.start()])
        hyperlink(paragraph, match.group(1), match.group(2))
        pos = match.end()
    add_text(text[pos:])


def build() -> None:
    figures()
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.52)
    section.bottom_margin = Inches(0.48)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.2)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.0
    for name, size, before, after in (("Title", 20, 0, 5), ("Heading 1", 13, 7, 2), ("Heading 2", 11.5, 5, 2)):
        style = styles[name]
        style.font.name = "Aptos Display" if name != "Normal" else "Aptos"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name != "Title" or True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    if "Figure Caption" not in styles:
        cap = styles.add_style("Figure Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Figure Caption"]
    cap.font.name = "Aptos"
    cap.font.size = Pt(8.5)
    cap.font.italic = True
    cap.font.color.rgb = RGBColor(70, 70, 70)
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(5)
    cap.paragraph_format.keep_with_next = False

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        image = IMAGE.fullmatch(line)
        if image:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            paragraph.add_run().add_picture(str(HERE / image.group(2)), width=Inches(6.65))
            caption = doc.add_paragraph(image.group(1), style="Figure Caption")
            caption.alignment = WD_ALIGN_PARAGRAPH.LEFT
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], style="Title")
            continue
        if line.startswith("## "):
            doc.add_paragraph(line[3:], style="Heading 1")
            continue
        if line.startswith("### "):
            doc.add_paragraph(line[4:], style="Heading 2")
            continue
        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.left_indent = Inches(0.22)
            paragraph.paragraph_format.first_line_indent = Inches(-0.13)
            paragraph.paragraph_format.space_after = Pt(2)
            add_inline(paragraph, line[2:])
            continue
        paragraph = doc.add_paragraph()
        add_inline(paragraph, line)

    for paragraph in doc.paragraphs:
        if paragraph.style.name == "Title":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if paragraph.text.startswith("Repository:"):
            paragraph.paragraph_format.space_after = Pt(7)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(70, 70, 70)

    doc.core_properties.title = "Badger Scribe Team Bücherwurm Research Draft"
    doc.core_properties.author = "Team Bücherwurm"
    doc.core_properties.subject = "MLM26 draft writeup"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
