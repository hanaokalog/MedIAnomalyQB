import cairosvg, os

W, H = 1200, 1060
out = []
def add(s): out.append(s)

C = dict(
    enc=("#dce8f7", "#2b5c8f"), dec=("#dff0dc", "#2f7a3a"), qb=("#fde3c9", "#c0561a"),
    conv=("#f3f3f3", "#6f6f6f"), io=("#ffffff", "#333333"), panel=("#fbfbfb", "#9a9a9a"),
    op=("#ffffff", "#555555"))
FS, FS2 = 18, 16          # main / secondary font size

def rect(x, y, w, h, kind, rx=7, dash=False, sw=1.6):
    f, s = C[kind]
    d = ' stroke-dasharray="6,4"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')

def text(x, y, lines, size=FS2, weight="normal", anchor="middle", color="#1a1a1a", lh=None, italic=False):
    if isinstance(lines, str): lines = [lines]
    lh = lh or size * 1.22
    y0 = y - (len(lines) - 1) * lh / 2
    st = ' font-style="italic"' if italic else ''
    for i, l in enumerate(lines):
        w_, sz = weight, size
        if l.startswith("**"): l = l[2:]; w_ = "bold"
        add(f'<text x="{x}" y="{y0 + i*lh + sz*0.35:.1f}" font-size="{sz}" font-weight="{w_}" '
            f'text-anchor="{anchor}" fill="{color}"{st}>{l}</text>')

def box(x, y, w, h, kind, lines, size=FS2, lh=None, weight="normal"):
    rect(x, y, w, h, kind)
    text(x + w/2, y + h/2, lines, size=size, lh=lh, weight=weight)

def arrow(pts, color="#444", sw=1.8):
    p = " ".join(f"{a},{b}" for a, b in pts)
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arr)"/>')

def plus(cx, cy, r=13):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="#444" stroke-width="1.8"/>')
    add(f'<line x1="{cx-r+5}" y1="{cy}" x2="{cx+r-5}" y2="{cy}" stroke="#444" stroke-width="1.8"/>')
    add(f'<line x1="{cx}" y1="{cy-r+5}" x2="{cx}" y2="{cy+r-5}" stroke="#444" stroke-width="1.8"/>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
    'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---------------- (a) overall architecture ----------------
text(20, 18, "**(a) Overall architecture", size=FS, anchor="start")
res = [128, 64, 32, 16, 8]
ch = [16, 32, 64, 128, 256]
kq = [1, 2, 4, 8]
Y = [140, 255, 370, 485, 600]          # row centres
BH = 70
EX, EW = 20, 230
C1X, C1W = 285, 62                     # 1x1 conv (pre)
QX, QW = 380, 360                      # QB
C2X = 773                              # 1x1 conv (post)
DX, DW = 870, 310                      # decoder

# input / output
box(EX, 38, EW, 52, "io", ["input x", "(+ blob noise, train)"], size=FS2)
arrow([(EX + EW/2, 90), (EX + EW/2, Y[0] - BH/2)])
box(DX, 38, DW, 52, "io", ["reconstruction x̂", "score: L2 + VGG perceptual"], size=FS2)

for i in range(5):
    y = Y[i]
    box(EX, y - BH/2, EW, BH, "enc", [f"**Conv block · {ch[i]} ch", f"{res[i]} × {res[i]}"], size=FS2)
    if i < 4:
        arrow([(EX + EW/2, y + BH/2), (EX + EW/2, Y[i+1] - BH/2)])
        text(EX + EW/2 + 10, (y + BH/2 + Y[i+1] - BH/2) / 2, "ResidualDown (c)", size=15,
             anchor="start", color="#2b5c8f")

for i in range(4):
    y = Y[i]; n = kq[i] * res[i] ** 2
    arrow([(EX + EW, y), (C1X, y)])
    box(C1X, y - 22, C1W, 44, "conv", ["1×1"], size=FS2)
    arrow([(C1X + C1W, y), (QX, y)])
    box(QX, y - 28, QW, 56, "qb", [f"**QB (b)  {kq[i]}×{res[i]}×{res[i]} = {n:,}"], size=FS2)
    arrow([(QX + QW, y), (C2X, y)])
    box(C2X, y - 22, C1W, 44, "conv", ["1×1"], size=FS2)
    arrow([(C2X + C1W, y), (DX, y)])
    attn = ["no attention", "+ channel attention", "+ spatial attention", "+ spatial attention"][i]
    box(DX, y - 38, DW, 76, "dec", [f"**Up block · {ch[i]} ch", "CBAM gate + conv block", attn], size=15, lh=19)
    if i < 3:
        arrow([(DX + DW/2, Y[i+1] - 38), (DX + DW/2, y + 38)])
        text(DX + DW/2 + 10, (y + 38 + Y[i+1] - 38) / 2, "ResidualUp (d)", size=15,
             anchor="start", color="#2f7a3a")
arrow([(DX + DW/2, Y[0] - 38), (DX + DW/2, 90)])

# bottleneck row (v31: attention top mixer, no FC)
by, bh = 668, 90
arrow([(EX + EW/2, Y[4] + BH/2), (EX + EW/2, by)])
box(EX, by, 330, bh, "conv", ["**Top mixer (encoder side)", "(SpatialAttn + FFN) × 2", "1×1 conv 256→512 · GN · ParamAtan"], size=15, lh=21)
arrow([(EX + 330, by + bh/2), (QX, by + bh/2)])
box(QX, by + 12, QW, bh - 24, "qb", ["**QB (b)  512×8×8 = 32,768"], size=FS2)
arrow([(QX + QW, by + bh/2), (DX, by + bh/2)])
box(DX, by, DW, bh, "conv", ["**Top mixer (decoder side)", "1×1 conv 512→256", "(SpatialAttn + FFN) × 2"], size=15, lh=21)
arrow([(DX + DW/2, by), (DX + DW/2, Y[3] + 38)])
text(DX + DW/2 + 10, (Y[3] + 38 + by) / 2, "ResidualUp (d)", size=15, anchor="start", color="#2f7a3a")
text((EX + EW + DX) / 2, Y[4] - 12, "decoder receives only QB outputs:", size=15, color="#444")
text((EX + EW + DX) / 2, Y[4] + 12, "16,384 + 8,192 + 4,096 + 2,048 + 32,768 = 63,488 channels", size=15, color="#444")

# ---------------- insets ----------------
PY, PH, PW = 785, 265, 375
PX = [20, 412, 805]
titles = ["(b) QB layer", "(c) ResidualDown", "(d) ResidualUp"]
subs = ["", "C×H×W → C×H/2×W/2", "C×H×W → C′×2H×2W"]
for j in range(3):
    rect(PX[j], PY, PW, PH, "panel", rx=8, sw=1.2)
    text(PX[j] + 12, PY + 22, "**" + titles[j], size=FS, anchor="start")
    text(PX[j] + PW - 12, PY + 22, subs[j], size=14, anchor="end", color="#444")

# (b) QB
x0, yb = PX[0], PY + 95
text(x0 + 26, yb, "h", size=FS, italic=True)
arrow([(x0 + 40, yb), (x0 + 72, yb)])
box(x0 + 72, yb - 22, 58, 44, "conv", ["σ(·)"], size=FS)
arrow([(x0 + 130, yb), (x0 + 196, yb)])
text(x0 + 163, yb - 16, "z", size=FS2, italic=True)
plus(x0 + 210, yb)
text(x0 + 210, yb - 50, "Lap(0, 1/ε)", size=15)
arrow([(x0 + 210, yb - 38), (x0 + 210, yb - 14)])
arrow([(x0 + 224, yb), (x0 + 300, yb)])
text(x0 + 322, yb, "z̃", size=FS, italic=True)
text(x0 + 16, PY + 185, ["train:  z̃ = σ(h) + n,  n ~ Lap(0, 1/ε)",
                         "test:   z̃ = σ(h)   or   1[σ(h) > 0.5]",
                         "ε-local DP per channel (Δ = 1);",
                         "Heaviside at test: ≤ 1 bit / channel"], size=15, anchor="start", lh=22)

# (c) ResidualDown and (d) ResidualUp share a layout
def residual_panel(x0, main, short, outlab):
    ym, ys, yc = PY + 90, PY + 168, PY + 129
    text(x0 + 24, yc, "x", size=FS, italic=True)
    arrow([(x0 + 36, yc), (x0 + 48, yc), (x0 + 48, ym), (x0 + 62, ym)])
    arrow([(x0 + 48, yc), (x0 + 48, ys), (x0 + 62, ys)])
    bw = 118
    for k_, (row, labs) in enumerate([(ym, main), (ys, short)]):
        box(x0 + 62, row - 26, bw, 52, "conv", labs[0], size=14, lh=17)
        arrow([(x0 + 62 + bw, row), (x0 + 196, row)])
        box(x0 + 196, row - 26, bw, 52, "conv", labs[1], size=14, lh=17)
    px_ = x0 + 346
    arrow([(x0 + 314, ym), (px_, ym), (px_, yc - 13)])
    arrow([(x0 + 314, ys), (px_, ys), (px_, yc + 13)])
    plus(px_, yc)
    text(x0 + 190, ym - 38, "learned path", size=14, color="#555", italic=True)
    text(x0 + 190, ys + 40, "parameter-free shortcut", size=14, color="#555", italic=True)
    text(x0 + PW / 2, PY + PH - 20, outlab, size=15, color="#333")

residual_panel(PX[1],
               (["Conv 3×3", "C → C/4"], ["Pixel-", "Unshuffle(2)"]),
               (["Pixel-", "Unshuffle(2)"], ["mean over", "4-ch groups"]),
               "sum → C × H/2 × W/2")
residual_panel(PX[2],
               (["Conv 3×3", "C → 4C′"], ["Pixel-", "Shuffle(2)"]),
               (["duplicate ch", "× 4C′/C"], ["Pixel-", "Shuffle(2)"]),
               "sum → C′ × 2H × 2W")

add('</svg>')
svg = "\n".join(out)
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "qbae_network")
open(base + ".svg", "w").write(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=base + ".pdf")
cairosvg.svg2png(bytestring=svg.encode(), write_to=base + ".png", output_width=W * 2)
print("ok")
