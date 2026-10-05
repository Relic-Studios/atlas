"""Render the ATLAS brand set to PNG with PIL (no browser, no AI).

Shapes mirror mark.svg; drawn at 4x and downsampled.
Text uses Inter (SIL OFL). Usage: python render_png.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "png"
OUT.mkdir(exist_ok=True)
INK, TEAL = (243, 245, 247), (94, 211, 198)
TOP, BOT = (21, 27, 36), (10, 14, 19)
FONT = "C:/Windows/Fonts/Inter-SemiBold.ttf"
FONT_R = "C:/Windows/Fonts/Inter-Regular.ttf"
SS = 4


def gradient(w, h, top=TOP, bot=BOT):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bot)))
    return im


def stroke(d, pts, width, fill=INK):
    """Polyline with round caps and joins."""
    d.line(pts, fill=fill, width=width, joint="curve")
    r = width / 2
    for x, y in (pts[0], pts[-1]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def dot(d, x, y, r, fill=TEAL):
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def mark_layer(size, k, d, ox=0, oy=0):
    """The ATLAS 'A' on a 512 grid scaled by k (already supersampled)."""
    P = lambda x, y: (ox + x * k, oy + y * k)
    stroke(d, [P(146, 380), P(256, 120), P(366, 380)], int(52 * k))
    dot(d, *P(256, 296), 30 * k)


def icon(name, size, kind="mark", rounded=True):
    S = size * SS
    bg = gradient(S, S)
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    if rounded:
        md.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 116 / 512), fill=255)
    else:
        md.ellipse([0, 0, S - 1, S - 1], fill=255)
    d = ImageDraw.Draw(bg)
    k = S / 512
    if kind == "mark":
        mark_layer(S, k, d)
    else:  # bot: T plus teal dot, thin ring
        d.ellipse([24 * k, 24 * k, S - 24 * k, S - 24 * k], outline=(54, 110, 106), width=int(6 * k))
        P = lambda x, y: (x * k, y * k)
        stroke(d, [P(150, 156), P(362, 156)], int(52 * k))
        stroke(d, [P(256, 156), P(256, 352)], int(52 * k))
        dot(d, *P(352, 352), 30 * k)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    im.paste(bg, (0, 0), mask)
    im.resize((size, size), Image.LANCZOS).save(OUT / f"{name}.png")


def banner(name, w, h, title="ATLAS", sub="Local voice agents for group conversation", mark_h=0.42,
           show_sub=True):
    S_w, S_h = w * SS, h * SS
    im = gradient(S_w, S_h, (18, 23, 31), (9, 12, 16))
    d = ImageDraw.Draw(im)
    # faint concentric arcs on the right: the 'listening' motif, no imagery
    cx, cy = S_w * 0.86, S_h * 0.5
    for i, r in enumerate(range(int(S_h * 0.25), int(S_w * 0.55), int(S_h * 0.13))):
        a = max(10, 46 - i * 5)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(a, a + 18, a + 16), width=SS * 2)
    mh = S_h * mark_h
    k = mh / 512
    mx = S_w * 0.08
    my = (S_h - mh) / 2
    mark_layer(mh, k, d, mx, my)
    f = ImageFont.truetype(FONT, int(mh * 0.62))
    tx = mx + mh * 0.95
    tb = d.textbbox((0, 0), title, font=f)
    ty = S_h / 2 - (tb[3] - tb[1]) / 2 - tb[1] - (mh * 0.14 if show_sub else 0)
    d.text((tx, ty), title, font=f, fill=INK)
    if show_sub:
        fs = ImageFont.truetype(FONT_R, int(mh * 0.17))
        d.text((tx + SS * 2, ty + (tb[3]) + mh * 0.10), sub, font=fs, fill=(150, 160, 170))
    im.resize((w, h), Image.LANCZOS).save(OUT / f"{name}.png")


if __name__ == "__main__":
    icon("atlas_icon_512", 512)
    icon("atlas_icon_1024", 1024)
    icon("server_icon_1024", 1024, rounded=False)
    banner("github_social_1280x640", 1280, 640, mark_h=0.30)
    banner("discord_invite_1920x1080", 1920, 1080, mark_h=0.22)
    banner("x_header_1500x500", 1500, 500, mark_h=0.32)
    print(sorted(p.name for p in OUT.iterdir()))
