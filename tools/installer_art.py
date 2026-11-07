"""Render the Windows setup's artwork from the ATLAS brand palette (no stock or AI imagery).

    python tools/installer_art.py          -> installer/art/*.png, installer/art/setup.ico

Shapes follow public_overlay/brand/mark.svg (a white "A" with a teal dot on a dark rounded square) and the brand's
concentric "sound" rings; drawn at 4x and downsampled. The wordmark uses Inter (SIL OFL), rendered into the images,
so Setup itself ships no font; regenerating the art needs Inter SemiBold installed (C:/Windows/Fonts/Inter-SemiBold.ttf).
Each image comes at 100%, 150% and 200% scale (Inno, and the setup script for its own images, pick the best fit).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "installer" / "art"
INK, TEAL, CYAN = (243, 245, 247), (94, 211, 198), (34, 211, 238)
TOP, BOT = (21, 27, 36), (10, 14, 19)          # brand gradient (#151B24 -> #0A0E13)
APP_BG = (11, 15, 23)                         # the app's window background #0B0F17
FONT_SEMI = "C:/Windows/Fonts/Inter-SemiBold.ttf"
SS = 4
SCALES = (100, 150, 200)
# Inno's background-image area (aspect 497:360) at WizardSizePercent=130: its documented sizes for the default 120,
# times 130/120. Inno stretches the best-fitting file to the window.
BACK = {100: (646, 468), 150: (1077, 780), 200: (1378, 999)}
SMALL = {100: 58, 150: 97, 200: 124}           # Inno's small-image area (inner-page corner), square
LOCKUP_W, LOCKUP_H = 260, 72


def gradient(w: int, h: int, top=TOP, bot=BOT) -> Image.Image:
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bot)))
    return im


def draw_mark(size: int, with_tile: bool = True) -> Image.Image:
    """The ATLAS mark (mark.svg geometry, 512 units) at `size` px, RGBA."""
    s = size * SS / 512
    im = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    if with_tile:
        tile = gradient(size * SS, size * SS).convert("RGBA")
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], radius=int(116 * s), fill=255)
        im.paste(tile, (0, 0), mask)
    d = ImageDraw.Draw(im)
    w = int(52 * s)
    pts = [(146 * s, 380 * s), (256 * s, 120 * s), (366 * s, 380 * s)]
    d.line(pts, fill=INK, width=w, joint="curve")
    for x, y in (pts[0], pts[2]):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=INK)
    r = 30 * s
    d.ellipse([256 * s - r, 296 * s - r, 256 * s + r, 296 * s + r], fill=TEAL)
    return im.resize((size, size), Image.Resampling.LANCZOS)


def back(scale: int) -> Image.Image:
    """Window background: the brand gradient with faint concentric rings at the lower right."""
    w, h = BACK[scale]
    W, H = w * SS, h * SS
    im = gradient(W, H, TOP, APP_BG).convert("RGBA")
    # a soft cyan glow behind the rings
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = int(W * 0.92), int(H * 0.86)
    gr = int(H * 0.55)
    gd.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=CYAN + (22,))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=gr * 0.45))
    im = Image.alpha_composite(im, glow)
    rings = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rings)
    lw = max(1, int(1.2 * SS * scale / 100))
    for i, rr in enumerate(range(int(H * 0.16), int(H * 1.35), int(H * 0.115))):
        a = max(10, 46 - i * 4)
        rd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=TEAL + (a,), width=lw)
    im = Image.alpha_composite(im, rings)
    return im.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)


def lockup(scale: int) -> Image.Image:
    """Mark + letter-spaced ATLAS wordmark on a transparent background."""
    w, h = LOCKUP_W * scale // 100, LOCKUP_H * scale // 100
    W, H = w * SS, h * SS
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    m = draw_mark(H // SS, with_tile=False).resize((H, H), Image.Resampling.LANCZOS)
    im.alpha_composite(m, (0, 0))
    font = ImageFont.truetype(FONT_SEMI, int(H * 0.46))
    d = ImageDraw.Draw(im)
    x, spacing = int(H * 1.08), int(H * 0.11)
    _, top, _, bottom = d.textbbox((0, 0), "A", font=font)
    y = (H - (bottom - top)) // 2 - top
    for ch in "ATLAS":
        d.text((x, y), ch, font=font, fill=INK)
        x += d.textlength(ch, font=font) + spacing
    return im.resize((w, h), Image.Resampling.LANCZOS)


STEP = 20                                      # step-list icon, px at 100%
DIM = (58, 69, 84)                             # a later step: a dim ring


def step_icon(kind: str, scale: int, frame: int = 0) -> Image.Image:
    """Step-list icons: done (cyan disc with a dark check), now (a cyan dot that pulses over 4 frames), todo (dim ring)."""
    n = STEP * scale // 100
    N = n * SS
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = N / 2
    if kind == "done":
        r = N * 0.46
        d.ellipse([c - r, c - r, c + r, c + r], fill=CYAN)
        w = max(2, int(N * 0.11))
        pts = [(N * 0.29, N * 0.52), (N * 0.44, N * 0.66), (N * 0.72, N * 0.36)]
        d.line(pts, fill=APP_BG, width=w, joint="curve")
        for x, y in (pts[0], pts[2]):
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=APP_BG)
    elif kind == "now":
        halo = (0.34, 0.40, 0.46, 0.40)[frame]          # the halo breathes: grow, peak, shrink
        alpha = (70, 50, 30, 50)[frame]
        r = N * halo
        d.ellipse([c - r, c - r, c + r, c + r], fill=CYAN + (alpha,))
        r = N * 0.22
        d.ellipse([c - r, c - r, c + r, c + r], fill=CYAN)
    else:
        r, w = N * 0.40, max(2, int(N * 0.09))
        d.ellipse([c - r, c - r, c + r, c + r], outline=DIM, width=w)
    return im.resize((n, n), Image.Resampling.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in SCALES:
        back(sc).save(OUT / f"back_{sc}.png", optimize=True)
        draw_mark(SMALL[sc]).save(OUT / f"mark_{sc}.png", optimize=True)
        lockup(sc).save(OUT / f"lockup_{sc}.png", optimize=True)
        step_icon("done", sc).save(OUT / f"step_done_{sc}.png", optimize=True)
        step_icon("todo", sc).save(OUT / f"step_todo_{sc}.png", optimize=True)
        for f in range(4):
            step_icon("now", sc, f).save(OUT / f"step_now{f + 1}_{sc}.png", optimize=True)
    big = draw_mark(256)
    big.save(OUT / "setup.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:16s} {p.stat().st_size // 1024:5d} KB")


if __name__ == "__main__":
    main()
