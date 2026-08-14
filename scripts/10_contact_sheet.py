"""
Hata kirpintilarindan numarali kontakt sayfasi uretir (gorsel inceleme icin).

Kirpintilari tek tek acmak yerine 4x6'lik izgaralara dizer; her hucre kendi
indeksiyle etiketlenir ve yan tarafa indeks -> dosya/metadata eslemesi CSV
olarak yazilir. Boylece izgaraya bakip "3 numara dalga kopugu" denebiliyor.

Ornek:
  python scripts/10_contact_sheet.py --tag baseline_a_pilot --bucket fp_background
  python scripts/10_contact_sheet.py --tag baseline_a_pilot --bucket fn --per-sheet 24
"""
import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 190          # hucre kenari (px)
PAD = 4
LABEL_H = 26
BG = (252, 252, 251)
INK = (11, 11, 11)
MUTED = (137, 135, 129)

NAME_RE = re.compile(r"^(\d+)_(?:conf([\d.]+)_)?size(\d+)px_img(\d+)\.png$")


def parse(path: Path) -> dict:
    m = NAME_RE.match(path.name)
    parts = path.parts
    return {
        "file": str(path),
        "category": parts[-3],
        "size_band_dir": parts[-2],
        "conf": m.group(2) if m and m.group(2) else "",
        "edge_px": m.group(3) if m else "",
        "image_id": m.group(4) if m else "",
    }


def build_sheet(items: list[dict], cols: int, rows: int, start_idx: int) -> Image.Image:
    w = cols * (CELL + PAD) + PAD
    h = rows * (CELL + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(sheet)

    for k, it in enumerate(items):
        r, c = divmod(k, cols)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + LABEL_H + PAD)
        with Image.open(it["file"]) as im:
            im = im.convert("RGB")
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            ox = x + (CELL - im.width) // 2
            oy = y + (CELL - im.height) // 2
            sheet.paste(im, (ox, oy))
        draw.rectangle([x, y, x + CELL, y + CELL], outline=MUTED, width=1)
        idx = start_idx + k
        cap = f"#{idx}"
        if it["conf"]:
            cap += f" c{it['conf']}"
        cap += f" {it['edge_px']}px"
        draw.text((x + 2, y + CELL + 5), cap, fill=INK)
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--bucket", required=True,
                   help="fp_background | fp_class_confusion | fp_localization | fp_duplicate | fn")
    p.add_argument("--ea-dir", type=Path, default=Path("outputs/error_analysis"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    root = args.ea_dir / args.tag / "crops" / args.bucket
    if not root.exists():
        raise SystemExit(f"{root} yok")
    out_dir = args.out_dir or (args.ea_dir / args.tag / "contact_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --bucket alt yol da olabilir (orn. "fn/swimmer"); dosya adinda '/' olamaz
    slug = args.bucket.replace("/", "_")
    files = sorted(root.rglob("*.png"))
    if args.limit:
        files = files[: args.limit]
    items = [parse(f) for f in files]

    per = args.cols * args.rows
    index_rows = []
    for s in range(0, len(items), per):
        chunk = items[s : s + per]
        sheet = build_sheet(chunk, args.cols, args.rows, s)
        name = f"{slug}_sheet{s // per:02d}.png"
        sheet.save(out_dir / name)
        for k, it in enumerate(chunk):
            index_rows.append({"idx": s + k, "sheet": name, **it})

    idx_csv = out_dir / f"{slug}_index.csv"
    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        wr.writeheader()
        wr.writerows(index_rows)

    print(f"{len(items)} kirpinti -> {(len(items) + per - 1) // per} sayfa")
    print(f"  {out_dir}/  ({idx_csv.name})")


if __name__ == "__main__":
    main()
