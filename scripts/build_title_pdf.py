from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = ROOT / "build"

TITLE_RU = "Система кластеризации научных статей"
TITLE_EN = "Scientific article clustering system"
STUDENT = "Белкин Андрей Александрович"
SUPERVISOR = "старший преподаватель А. А. Паринов"
SIGNATURE_PATH = ROOT / "figures" / "student_signature.png"

FONT_NAME = "TimesNewRoman"
FONT_PATHS = [
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    Path("/Library/Fonts/Times New Roman.ttf"),
]


def find_template() -> Path:
    matches = sorted(DOCS.glob("Титуль*МОЦП.docx"))
    if not matches:
        raise FileNotFoundError("Не найден шаблон титульного листа в docs/")
    return matches[0]


def register_font() -> str:
    for path in FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            return FONT_NAME
    return "Times-Roman"


def export_template(template: Path, output: Path) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    script = [
        f'set inputPath to POSIX file "{template}"',
        f'set outputPath to POSIX file "{output}"',
        'tell application "Pages"',
        "open inputPath",
        "delay 2",
        "set theDoc to front document",
        "export theDoc to outputPath as PDF",
        "close theDoc saving no",
        "end tell",
    ]
    args: list[str] = ["osascript"]
    for line in script:
        args.extend(["-e", line])
    subprocess.run(args, check=True)


def fit_size(text: str, font_name: str, start: float, max_width: float) -> float:
    size = start
    while size > 8 and pdfmetrics.stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    return size


def draw_text(
    layer: canvas.Canvas,
    font_name: str,
    text: str,
    x: float,
    y: float,
    size: float,
    *,
    max_width: float | None = None,
) -> None:
    if max_width is not None:
        size = fit_size(text, font_name, size, max_width)
    layer.setFillColorRGB(0, 0, 0)
    layer.setFont(font_name, size)
    layer.drawString(x, y, text)


def redraw_line(layer: canvas.Canvas, x0: float, x1: float, y0: float) -> None:
    layer.setFillColorRGB(1, 1, 1)
    layer.rect(x0 - 1.5, y0 - 1.0, x1 - x0 + 3.0, 15.0, stroke=0, fill=1)
    layer.setStrokeColorRGB(0, 0, 0)
    layer.setLineWidth(0.6)
    layer.line(x0, y0 + 1.2, x1, y0 + 1.2)


def cover_area(layer: canvas.Canvas, x: float, y: float, width: float, height: float) -> None:
    layer.setFillColorRGB(1, 1, 1)
    layer.rect(x, y, width, height, stroke=0, fill=1)


def make_overlay(base_pdf: Path, output: Path) -> None:
    font_name = register_font()
    reader = PdfReader(str(base_pdf))
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    packet = BytesIO()
    layer = canvas.Canvas(packet, pagesize=(width, height))

    redraw_line(layer, 160.1, 524.1, 535.9)
    redraw_line(layer, 85.0, 501.1, 504.4)
    redraw_line(layer, 168.7, 532.7, 467.0)
    redraw_line(layer, 85.0, 501.1, 435.5)
    cover_area(layer, 331.4, 295.6, 180.0, 15.0)
    cover_area(layer, 370, 278, 100, 17)
    redraw_line(layer, 332.9, 508.4, 274.5)
    redraw_line(layer, 332.9, 508.4, 184.6)

    draw_text(layer, font_name, TITLE_RU, 168, 539.3, 14)
    draw_text(layer, font_name, TITLE_EN, 177, 470.4, 14)
    draw_text(layer, font_name, STUDENT, 335, 300.2, 12.0)
    draw_text(layer, font_name, "(Ф.И.О., подпись)", 376, 258.8, 11)
    draw_text(layer, font_name, SUPERVISOR, 335, 205.0, 11.0)
    if SIGNATURE_PATH.exists():
        signature = ImageReader(str(SIGNATURE_PATH))
        layer.drawImage(signature, 368, 274, width=100, height=22, mask="auto", preserveAspectRatio=True)

    layer.setFillColorRGB(1, 1, 1)
    layer.rect(width - 55, 38, 45, 35, stroke=0, fill=1)
    layer.save()

    packet.seek(0)
    overlay_page = PdfReader(packet).pages[0]
    page.merge_page(overlay_page)

    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as fout:
        writer.write(fout)


def main() -> int:
    template = find_template()
    base_pdf = BUILD / "title_page_template.pdf"
    output = BUILD / "title_page.pdf"
    export_template(template, base_pdf)
    make_overlay(base_pdf, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
