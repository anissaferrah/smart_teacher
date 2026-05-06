"""Markdown → PDF converter (pure-Python, reportlab).

Limited dialect supporting :
  - # / ## / ### / #### headings
  - paragraphs, blank-line separators
  - bullet lists (- or *)
  - inline code `like this` and ``` code fences ```
  - tables with | separators
  - **bold** / *italic*
  - horizontal rule ---
  - links [text](url) → keep text only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem,
    Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def _make_styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=22, spaceAfter=10, spaceBefore=14, textColor=colors.HexColor("#1a365d"), alignment=TA_LEFT, fontName="Helvetica-Bold"),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=17, spaceAfter=8, spaceBefore=14, textColor=colors.HexColor("#2c5282"), fontName="Helvetica-Bold"),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=13, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#2b6cb0"), fontName="Helvetica-Bold"),
        "h4": ParagraphStyle("h4", parent=base["Heading4"], fontSize=11, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#2d3748"), fontName="Helvetica-Bold"),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=4, textColor=colors.HexColor("#1a202c")),
        "code": ParagraphStyle("code", parent=base["BodyText"], fontSize=8, leading=10, fontName="Courier", backColor=colors.HexColor("#f7fafc"), borderColor=colors.HexColor("#e2e8f0"), borderWidth=0.5, borderPadding=4, spaceAfter=6, spaceBefore=2),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=12.5, leftIndent=14, bulletIndent=4, spaceAfter=2),
        "quote": ParagraphStyle("quote", parent=base["BodyText"], fontSize=9.5, leading=13, leftIndent=18, textColor=colors.HexColor("#4a5568"), italic=True, spaceAfter=6),
    }


_BOLD_RE   = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_CODE_RE   = re.compile(r"`([^`\n]+?)`")
_LINK_RE   = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Protect code spans first — replace with placeholders so italic/bold
    # regexes don't see the * / ** inside them (e.g. "qa_cache:*" must stay
    # literal, not become "qa_cache:<i>...</i>").
    code_chunks: list[str] = []
    def _stash(m):
        code_chunks.append(f'<font face="Courier" size="9" color="#c05621">{m.group(1)}</font>')
        return f"\x00CODE{len(code_chunks)-1}\x00"
    text = _CODE_RE.sub(_stash, text)

    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _LINK_RE.sub(r"\1", text)

    # Restore code spans
    for i, chunk in enumerate(code_chunks):
        text = text.replace(f"\x00CODE{i}\x00", chunk)
    return text


def _parse_lines(lines):
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()
        if stripped.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].rstrip().startswith("```"):
                buf.append(lines[j])
                j += 1
            yield ("code", "\n".join(buf))
            i = j + 1
            continue
        if re.fullmatch(r"\s*---+\s*", stripped):
            yield ("hr", None)
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if m:
            yield (f"h{len(m.group(1))}", m.group(2).strip())
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|\s*[:\- ]+", lines[i + 1]):
            j = i
            rows = []
            while j < n and lines[j].rstrip().startswith("|"):
                rows.append(lines[j].rstrip())
                j += 1
            yield ("table", rows)
            i = j
            continue
        if re.match(r"^\s*[-*]\s+", stripped):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i].rstrip()):
                buf.append(re.sub(r"^\s*[-*]\s+", "", lines[i].rstrip()))
                i += 1
            yield ("ul", buf)
            continue
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].rstrip().startswith(">"):
                buf.append(lines[i].rstrip().lstrip(">").lstrip())
                i += 1
            yield ("quote", " ".join(buf))
            continue
        if not stripped:
            yield ("blank", None)
            i += 1
            continue
        buf = [stripped]
        i += 1
        while i < n:
            s = lines[i].rstrip()
            if (not s or s.startswith("#") or s.startswith("```")
                or re.match(r"^\s*[-*]\s+", s) or s.startswith("|") or s.startswith(">")
                or re.fullmatch(r"\s*---+\s*", s)):
                break
            buf.append(s)
            i += 1
        yield ("p", " ".join(buf))


def _render_table(rows, styles):
    parsed = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    if len(parsed) >= 2 and all(re.fullmatch(r"[:\- ]+", c) for c in parsed[1]):
        header, body = parsed[0], parsed[2:]
    else:
        header, body = parsed[0], parsed[1:]
    data = [[Paragraph(_inline(c), styles["body"]) for c in row] for row in [header] + body]
    if not data:
        return None
    n_cols = len(data[0])
    avail = 17 * cm
    if n_cols == 2:
        widths = [avail * 0.32, avail * 0.68]
    elif n_cols == 3:
        widths = [avail * 0.28, avail * 0.20, avail * 0.52]
    else:
        widths = [avail / n_cols] * n_cols
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",(0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e0")),
    ]))
    return t


def md_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    styles = _make_styles()
    flows = []
    for kind, payload in _parse_lines(lines):
        if kind in ("h1", "h2", "h3", "h4"):
            flows.append(Paragraph(_inline(payload), styles[kind]))
        elif kind == "p":
            flows.append(Paragraph(_inline(payload), styles["body"]))
        elif kind == "code":
            flows.append(Preformatted(payload, styles["code"]))
        elif kind == "ul":
            items = [ListItem(Paragraph(_inline(p), styles["bullet"]), leftIndent=10) for p in payload]
            flows.append(ListFlowable(items, bulletType="bullet", leftIndent=14, bulletFontSize=8))
            flows.append(Spacer(1, 4))
        elif kind == "table":
            t = _render_table(payload, styles)
            if t is not None:
                flows.append(t)
                flows.append(Spacer(1, 6))
        elif kind == "quote":
            flows.append(Paragraph(_inline(payload), styles["quote"]))
        elif kind == "hr":
            flows.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e0"), spaceBefore=6, spaceAfter=6))
        elif kind == "blank":
            flows.append(Spacer(1, 4))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Smart Teacher — Architecture & Scores Reference",
        author="Smart Teacher project",
    )

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(2 * cm, 1.2 * cm, "Smart Teacher — Architecture & Scores Reference")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(flows, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"OK : {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "SMART_TEACHER_ARCHITECTURE.md"
    pdf = md.with_suffix(".pdf")
    md_to_pdf(md, pdf)
