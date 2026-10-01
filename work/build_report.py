from __future__ import annotations

import json
from pathlib import Path
from project_fonts import korean_font

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
FIG = ROOT / "figures"
OUT = ROOT / "output" / "pdf" / "DS-MINI-Design-울산_3반-이효준.pdf"
FONT = korean_font()

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#167D87")
MINT = colors.HexColor("#DCEFEA")
ORANGE = colors.HexColor("#F28E2B")
RED = colors.HexColor("#C95454")
INK = colors.HexColor("#24313C")
MUTED = colors.HexColor("#63717D")
LINE = colors.HexColor("#D9E1E7")
PALE = colors.HexColor("#F4F7F9")


pdfmetrics.registerFont(TTFont("KR", FONT))


class NumberedDocTemplate(BaseDocTemplate):
    pass


def page_chrome(canvas, doc):
    canvas.saveState()
    canvas.setTitle("DS Mini Project DAY 1 - Battery Cycle Life Analysis")
    canvas.setAuthor("이효준")
    canvas.setSubject("초기 100사이클 기반 배터리 수명 EDA 및 모델 설계")
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 11 * mm, width, 11 * mm, fill=1, stroke=0)
    canvas.setFont("KR", 7.5)
    canvas.setFillColor(colors.white)
    canvas.drawString(18 * mm, height - 7.3 * mm, "DS MINI PROJECT · DAY 1 DESIGN REPORT")
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "이효준 · 울산 3반")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="KRBody", fontName="KR", fontSize=9.1, leading=14, textColor=INK, wordWrap="CJK", spaceAfter=5))
styles.add(ParagraphStyle(name="KRSmall", fontName="KR", fontSize=7.7, leading=11.2, textColor=MUTED, wordWrap="CJK"))
styles.add(ParagraphStyle(name="KRTitle", fontName="KR", fontSize=23, leading=31, textColor=NAVY, wordWrap="CJK", spaceAfter=12))
styles.add(ParagraphStyle(name="KRH1", fontName="KR", fontSize=17, leading=23, textColor=NAVY, wordWrap="CJK", spaceAfter=9))
styles.add(ParagraphStyle(name="KRH2", fontName="KR", fontSize=11.5, leading=16, textColor=TEAL, wordWrap="CJK", spaceBefore=5, spaceAfter=5))
styles.add(ParagraphStyle(name="KRCall", fontName="KR", fontSize=10.2, leading=15, textColor=NAVY, wordWrap="CJK", leftIndent=5, rightIndent=5, spaceAfter=5))
styles.add(ParagraphStyle(name="KRTable", fontName="KR", fontSize=7.6, leading=10, textColor=INK, wordWrap="CJK"))
styles.add(ParagraphStyle(name="KRTableHeader", fontName="KR", fontSize=7.6, leading=10, textColor=colors.white, wordWrap="CJK"))
styles.add(ParagraphStyle(name="KRCover", fontName="KR", fontSize=8.8, leading=13, textColor=colors.HexColor("#D8E6EE"), wordWrap="CJK"))


def p(text: str, style: str = "KRBody") -> Paragraph:
    return Paragraph(text, styles[style])


def section_no(n: str, title: str, subtitle: str | None = None):
    parts = [p(f"<font color='#167D87'>{n}</font>  {title}", "KRH1")]
    if subtitle:
        parts.append(p(subtitle, "KRSmall"))
        parts.append(Spacer(1, 3 * mm))
    return parts


def callout(title: str, body: str, color=MINT):
    table = Table([[p(f"<b>{title}</b><br/>{body}", "KRCall")]], colWidths=[171 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), .6, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 6 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    return table


def data_table(rows, widths, header=True):
    cooked = []
    for row_idx, row in enumerate(rows):
        style_name = "KRTableHeader" if header and row_idx == 0 else "KRTable"
        cooked.append([cell if hasattr(cell, "wrap") else p(str(cell), style_name) for cell in row])
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), .4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    table.setStyle(TableStyle(commands))
    return table


def fig(name: str, width=171 * mm, height=65 * mm):
    image = Image(str(FIG / name), width=width, height=height)
    image.hAlign = "CENTER"
    return image


def fmt_p(value: float) -> str:
    return "< 0.001" if value < .001 else f"= {value:.3f}"


if __name__ == "__main__":
    from day1_design import build
    build()
