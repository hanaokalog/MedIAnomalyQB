import cairosvg, os

W, H = 1600, 2650 + 165
out = []
def add(s): out.append(s)

C = dict(
    enc=("#dce8f7", "#2b5c8f"), dec=("#dff0dc", "#2f7a3a"), qb=("#fde3c9", "#c0561a"),
    conv=("#f3f3f3", "#6f6f6f"), io=("#ffffff", "#333333"), panel=("#fbfbfb", "#9a9a9a"),
    att=("#ece1f6", "#6b3fa0"), norm=("#fff7d6", "#a08a2a"), act=("#ffffff", "#888888"))
FS, FS2, FS3 = 19, 16, 14

def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def rect(x, y, w, h, kind, rx=7, dash=False, sw=1.6):
    f, s = C[kind]
    d = ' stroke-dasharray="6,4"' if dash else ''
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')

def text(x, y, lines, size=FS2, weight="normal", anchor="middle", color="#1a1a1a", lh=None, italic=False):
    if isinstance(lines, str): lines = [lines]
    lh = lh or size * 1.25
    y0 = y - (len(lines) - 1) * lh / 2
    st = ' font-style="italic"' if italic else ''
    for i, l in enumerate(lines):
        w_ = weight
        if l.startswith("**"): l = l[2:]; w_ = "bold"
        add(f'<text x="{x:.1f}" y="{y0 + i*lh + size*0.35:.1f}" font-size="{size}" font-weight="{w_}" '
            f'text-anchor="{anchor}" fill="{color}"{st}>{esc(l)}</text>')

def box(x, y, w, h, kind, lines, size=FS2, lh=None, dash=False):
    rect(x, y, w, h, kind, dash=dash)
    text(x + w/2, y + h/2, lines, size=size, lh=lh)

def arrow(pts, color="#444", sw=1.8, dash=False):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    d = ' stroke-dasharray="6,4"' if dash else ''
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#arr)"/>')

def line(pts, color="#444", sw=1.8):
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"/>')

def opcircle(cx, cy, sym="+", r=13):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="white" stroke="#444" stroke-width="1.8"/>')
    if sym == "+":
        add(f'<line x1="{cx-r+5:.1f}" y1="{cy:.1f}" x2="{cx+r-5:.1f}" y2="{cy:.1f}" stroke="#444" stroke-width="1.8"/>')
        add(f'<line x1="{cx:.1f}" y1="{cy-r+5:.1f}" x2="{cx:.1f}" y2="{cy+r-5:.1f}" stroke="#444" stroke-width="1.8"/>')
    else:  # ×
        k = (r - 5) * 0.75
        add(f'<line x1="{cx-k:.1f}" y1="{cy-k:.1f}" x2="{cx+k:.1f}" y2="{cy+k:.1f}" stroke="#444" stroke-width="1.8"/>')
        add(f'<line x1="{cx-k:.1f}" y1="{cy+k:.1f}" x2="{cx+k:.1f}" y2="{cy-k:.1f}" stroke="#444" stroke-width="1.8"/>')

def panel(x, y, w, h, title, sub=""):
    rect(x, y, w, h, "panel", rx=8, sw=1.2)
    text(x + 14, y + 24, "**" + title, size=FS, anchor="start")
    if sub: text(x + w - 14, y + 24, sub, size=FS3, anchor="end", color="#444")

def vflow(cx, y, w, items, gap=20, size=FS3, lh=None):
    """items: (lines, kind, h). Draws boxes top->bottom with arrows. Returns list of (top, bottom)."""
    pos = []
    for j, (lines, kind, h) in enumerate(items):
        box(cx - w/2, y, w, h, kind, lines, size=size, lh=lh)
        pos.append((y, y + h))
        if j < len(items) - 1:
            arrow([(cx, y + h), (cx, y + h + gap)])
        y += h + gap
    return pos

def residual(cx, y_in, y_plus, side_x, label=None):
    """bypass from (cx, y_in) via side_x down to a ⊕ at (cx, y_plus)."""
    line([(cx, y_in), (side_x, y_in), (side_x, y_plus)])
    arrow([(side_x, y_plus), (cx + 13 if side_x > cx else cx - 13, y_plus)])
    opcircle(cx, y_plus)
    if label: text(side_x + (8 if side_x > cx else -8), (y_in + y_plus) / 2, label, size=13,
                   anchor="start" if side_x > cx else "end", color="#555", italic=True)

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
    'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# =====================================================================================
# (a) overall architecture
# =====================================================================================
text(20, 20, "**(a) Overall architecture  (wf = 4, depth = 7, 128 × 128 input, 60.7 M parameters)", size=FS, anchor="start")
res = [128, 64, 32, 16, 8, 4, 2]
ch = [16, 32, 64, 128, 256, 512, 1024]
kq = [1, 2, 4, 8, 16, 32]
gn = {16: 4, 32: 8, 64: 8, 128: 16, 256: 16, 512: 32, 1024: 32}
Y = [178, 300, 422, 544, 666, 788, 910]
BH = 84
EX, EW = 20, 260
C1X, C1W = 318, 96
QX, QW = 452, 390
C2X = 880
DX, DW = 1014, 566
cin = ["C_in", "16", "32", "64", "128", "256", "512"]

box(EX, 44, EW, 66, "io", ["input x  (C_in × 128²)", "x − 1  (+ blob noise, train)"], size=FS3)
arrow([(EX + EW/2, 110), (EX + EW/2, Y[0] - BH/2)])
box(DX, 44, DW, 66, "io", ["**output head:  x̂ = head(·) + 1",
                           "LReLU · Conv3×3 16→16 · LReLU · Conv3×3 16→C_in"], size=FS3)

for i in range(7):
    y = Y[i]
    box(EX, y - BH/2, EW, BH, "enc", [f"**ConvBlock (e) {cin[i]}→{ch[i]}",
                                       f"{res[i]} × {res[i]} · GN({gn[ch[i]]})",
                                       "1×1-conv shortcut"], size=FS3, lh=19)
    if i < 6:
        arrow([(EX + EW/2, y + BH/2), (EX + EW/2, Y[i+1] - BH/2)])
        text(EX + EW/2 + 10, (y + BH/2 + Y[i+1] - BH/2) / 2, f"ResidualDown (c) {ch[i]}→{ch[i]}", size=FS3,
             anchor="start", color="#2b5c8f")

up_attn = ["attention: none", "attention: ChannelAttn + FFN (h)"] + ["attention: SpatialAttn + FFN (h)"] * 4
for i in range(6):
    y = Y[i]; n = kq[i] * res[i] ** 2
    arrow([(EX + EW, y), (C1X, y)])
    box(C1X, y - 26, C1W, 52, "conv", ["1×1 conv", f"{ch[i]}→{kq[i]}"], size=FS3, lh=18)
    arrow([(C1X + C1W, y), (QX, y)])
    box(QX, y - 30, QW, 60, "qb", [f"**QB (b)  {kq[i]}×{res[i]}×{res[i]} = {n:,}", "flatten → QB → reshape"], size=FS3, lh=19)
    arrow([(QX + QW, y), (C2X, y)])
    box(C2X, y - 26, C1W, 52, "conv", ["1×1 conv", f"{kq[i]}→{ch[i]}"], size=FS3, lh=18)
    arrow([(C2X + C1W, y), (DX, y)])
    text((C2X + C1W + DX) / 2, y - 14, "ŝ", size=FS2, italic=True)
    box(DX, y - 50, DW, 100, "dec", [f"**Up block (f) · {ch[i]} ch · {res[i]} × {res[i]}",
                                      f"ResidualUp (d) {2*ch[i]}→{ch[i]} · CBAM gate (g), F_int = {ch[i]//2}",
                                      f"concat → ConvBlock (e) {2*ch[i]}→{ch[i]}, GN({gn[ch[i]]})",
                                      up_attn[i]], size=FS3, lh=21)
    if i < 5:
        arrow([(DX + DW/2, Y[i+1] - 50), (DX + DW/2, y + 50)])
arrow([(DX + DW/2, Y[0] - 50), (DX + DW/2, 110)])

# bottleneck row: dense FC around the 2x2 bottom QB
by, bh = 1000, 110
arrow([(EX + EW/2, Y[6] + BH/2), (EX + EW/2, by)])
box(EX, by, 390, bh, "conv", ["**FC top, encoder side (i)", "1×1 conv 1024→128 · GN · LReLU · flatten",
                              "FC 512→512 · GN(1) (+ 1×1-conv shortcut)", "ParamAtan"], size=FS3, lh=22)
arrow([(EX + 390, by + bh/2), (QX + 30, by + bh/2)])
box(QX + 30, by + 22, QW - 60, bh - 44, "qb", ["**QB (b)  128×2×2 = 512"], size=FS2)
arrow([(QX + QW - 30, by + bh/2), (DX, by + bh/2)])
box(DX, by, DW, bh, "conv", ["**FC top, decoder side (i)", "FC 512→512 · reshape 128×2×2 · LReLU · GN",
                             "1×1 conv 128→1024  (+ 1×1-conv shortcut)"], size=FS3, lh=22)
arrow([(DX + DW/2, by), (DX + DW/2, Y[5] + 50)])
text(DX + DW/2 + 10, (Y[5] + 50 + by) / 2, "1024 × 2 × 2", size=FS3, anchor="start", color="#2f7a3a")
text((EX + EW + DX) / 2 + 40, Y[6] - 12, "decoder receives only QB outputs:", size=FS3, color="#444")
text((EX + EW + DX) / 2 + 40, Y[6] + 12, "16,384 + 8,192 + 4,096 + 2,048 + 1,024 + 512 + 512 = 32,768 = 2¹⁵", size=FS3, color="#444")

# =====================================================================================
# row 1: (b) QB, (c) ResidualDown, (d) ResidualUp
# =====================================================================================
PY, PH, PW = 990 + 165, 290, 513
PX = [20, 543, 1067]
panel(PX[0], PY, PW, PH, "(b) QB layer", "N channels, per sample")
panel(PX[1], PY, PW, PH, "(c) ResidualDown", "C×H×W → C×H/2×W/2")
panel(PX[2], PY, PW, PH, "(d) ResidualUp", "C×H×W → C′×2H×2W")

x0, yb = PX[0] + 50, PY + 100
text(x0 + 26, yb, "h", size=FS, italic=True)
arrow([(x0 + 40, yb), (x0 + 72, yb)])
box(x0 + 72, yb - 22, 62, 44, "conv", ["σ(·)"], size=FS)
arrow([(x0 + 134, yb), (x0 + 200, yb)])
text(x0 + 167, yb - 16, "z", size=FS2, italic=True)
opcircle(x0 + 214, yb)
text(x0 + 214, yb - 52, "n ~ Lap(0, 1/ε)", size=FS3)
arrow([(x0 + 214, yb - 40), (x0 + 214, yb - 14)])
arrow([(x0 + 228, yb), (x0 + 300, yb)])
box(x0 + 300, yb - 22, 72, 44, "conv", ["1[·>½]"], size=FS3)
text(x0 + 336, yb + 38, "test option", size=13, color="#555", italic=True)
arrow([(x0 + 372, yb), (x0 + 404, yb)])
text(x0 + 418, yb, "z̃", size=FS, italic=True)
text(PX[0] + 18, PY + 215, ["train:  z̃ = σ(h) + n   (noise on)",
                            "test:   z̃ = σ(h)  |  1[σ(h) > ½]  |  σ(h) + n (LDP)",
                            "ε-local DP per channel (sensitivity Δ = 1);",
                            "Heaviside at test: ≤ 1 bit / channel;  ε = 0: bypass"], size=FS3, anchor="start", lh=22)

def residual_panel(px, main, short, outlab):
    x0 = px + (PW - 375) / 2
    ym, ys, yc = PY + 95, PY + 175, PY + 135
    text(x0 + 24, yc, "x", size=FS, italic=True)
    arrow([(x0 + 36, yc), (x0 + 48, yc), (x0 + 48, ym), (x0 + 62, ym)])
    arrow([(x0 + 48, yc), (x0 + 48, ys), (x0 + 62, ys)])
    bw = 118
    for row, labs in [(ym, main), (ys, short)]:
        box(x0 + 62, row - 27, bw, 54, "conv", labs[0], size=FS3, lh=18)
        arrow([(x0 + 62 + bw, row), (x0 + 196, row)])
        box(x0 + 196, row - 27, bw, 54, "conv", labs[1], size=FS3, lh=18)
    pxx = x0 + 346
    arrow([(x0 + 314, ym), (pxx, ym), (pxx, yc - 13)])
    arrow([(x0 + 314, ys), (pxx, ys), (pxx, yc + 13)])
    opcircle(pxx, yc)
    text(x0 + 190, ym - 42, "learned path", size=FS3, color="#555", italic=True)
    text(x0 + 190, ys + 44, "parameter-free shortcut", size=FS3, color="#555", italic=True)
    text(px + PW / 2, PY + PH - 24, outlab, size=FS3, color="#333")

residual_panel(PX[1], (["Conv 3×3", "C → C/4"], ["Pixel-", "Unshuffle(2)"]),
               (["Pixel-", "Unshuffle(2)"], ["mean over", "4-ch groups"]),
               "sum → C × H/2 × W/2   (no norm / activation)")
residual_panel(PX[2], (["Conv 3×3", "C → 4C′"], ["Pixel-", "Shuffle(2)"]),
               (["repeat ch", "× 4C′/C"], ["Pixel-", "Shuffle(2)"]),
               "sum → C′ × 2H × 2W   (here C = 2C′: repeat × 2)")

# =====================================================================================
# row 2: (e) ConvBlock, (f) Up block, (g) CBAM cross-attention gate
# =====================================================================================
RY, RH = 1300 + 165, 660
ex, ew = 20, 400
fx, fw = 440, 500
gx, gw = 960, 620
panel(ex, RY, ew, RH, "(e) ConvBlock", "C_in → C")
panel(fx, RY, fw, RH, "(f) Up block", "2C×H×W, ŝ → C×2H×2W")
panel(gx, RY, gw, RH, "(g) CBAM cross-attention gate", "F_int = C/2")

# (e)
cx = ex + 170
text(cx, RY + 62, "x  (C_in × H × W)", size=FS3, italic=True)
arrow([(cx, RY + 74), (cx, RY + 98)])
items = [(["WS-Conv 3×3, C_in → C", "(reflection pad 1)"], "conv", 50),
         (["Swish"], "act", 34),
         ([f"GroupNorm(g(C), C)"], "norm", 34),
         (["WS-Conv 3×3, C → C", "(reflection pad 1)"], "conv", 50),
         (["Swish"], "act", 34),
         (["GroupNorm(g(C), C)"], "norm", 34)]
pos = vflow(cx, RY + 98, 230, items, gap=16)
yp = pos[-1][1] + 30
arrow([(cx, pos[-1][1]), (cx, yp - 13)])
line([(cx, RY + 86), (ex + 340, RY + 86), (ex + 340, yp - 60)])
box(ex + 296, (RY + 86 + yp) / 2 - 34, 88, 68, "conv", ["1×1 conv", "if C_in ≠ C", "else id"], size=13, lh=17)
arrow([(ex + 340, (RY + 86 + yp) / 2 + 34), (ex + 340, yp), (cx + 13, yp)])
opcircle(cx, yp)
arrow([(cx, yp + 13), (cx, yp + 40)])
text(cx, yp + 52, "C × H × W", size=FS3, italic=True)
text(ex + 16, RY + RH - 60, ["WS = weight-standardised conv;  g(C): 16→4,",
                             "32→8, 64→8, 128→16, 256→16, 512/1024→32;",
                             "shortcut on by default",
                             "(--no-using_identity_connection removes it)"], size=13, anchor="start", lh=19)

# (f)
lx, rx = fx + 130, fx + 370
text(lx, RY + 60, ["x from deeper stage", "2C × H × W"], size=FS3, lh=18)
text(rx, RY + 60, ["ŝ from QB path", "C × 2H × 2W"], size=FS3, lh=18)
arrow([(lx, RY + 80), (lx, RY + 104)])
box(lx - 100, RY + 104, 200, 50, "dec", ["ResidualUp (d)", "2C → C"], size=FS3, lh=18)
gy = RY + 196
arrow([(lx, RY + 154), (lx, gy)])
text(lx + 10, RY + 176, "u", size=FS2, anchor="start", italic=True)
arrow([(rx, RY + 80), (rx, gy)])
box(fx + 30, gy, fw - 60, 50, "att", ["CBAM cross-attention gate (g)", "→ a ∈ [0,1]^(1×2H×2W)"], size=FS3, lh=18)
oy = gy + 90
arrow([(lx, gy + 50), (lx, oy - 18)])
arrow([(rx, gy + 50), (rx, oy - 18)])
text(lx, oy, "u ⊙ (1 − a)", size=FS2)
text(rx, oy, "ŝ ⊙ a", size=FS2)
cy_ = oy + 44
arrow([(lx, oy + 16), (lx, cy_), (fx + fw/2 - 60, cy_)])
arrow([(rx, oy + 16), (rx, cy_), (fx + fw/2 + 60, cy_)])
box(fx + fw/2 - 60, cy_ - 20, 120, 40, "conv", ["concat → 2C"], size=FS3)
items = [(["ConvBlock (e) 2C → C", "with 1×1-conv shortcut"], "dec", 52),
         (["stage attention (h)", "128²: none · 64²: ChannelAttn + FFN", "32² … 4²: SpatialAttn + FFN"], "att", 70)]
pos = vflow(fx + fw/2, cy_ + 46, 380, items, gap=22)
arrow([(fx + fw/2, cy_ + 20), (fx + fw/2, cy_ + 46)])
arrow([(fx + fw/2, pos[-1][1]), (fx + fw/2, pos[-1][1] + 30)])
text(fx + fw/2, pos[-1][1] + 44, "C × 2H × 2W", size=FS3, italic=True)
text(fx + 16, RY + RH - 34, ["skips are only gated, never bypass the QB;",
                             "attention blocks use gradient checkpointing"], size=13, anchor="start", lh=19)

# (g)
gcx = gx + gw/2
text(gx + 150, RY + 58, "g = u  (C × 2H × 2W)", size=FS3, italic=True)
text(gx + gw - 150, RY + 58, "x = ŝ  (C × 2H × 2W)", size=FS3, italic=True)
arrow([(gx + 150, RY + 70), (gx + 150, RY + 90)])
arrow([(gx + gw - 150, RY + 70), (gx + gw - 150, RY + 90)])
box(gx + 50, RY + 90, 200, 46, "conv", ["1×1 conv C→F_int", f"GroupNorm"], size=FS3, lh=18)
box(gx + gw - 250, RY + 90, 200, 46, "conv", ["1×1 conv C→F_int", "GroupNorm"], size=FS3, lh=18)
py_ = RY + 168
arrow([(gx + 150, RY + 136), (gx + 150, py_), (gcx - 13, py_)])
arrow([(gx + gw - 150, RY + 136), (gx + gw - 150, py_), (gcx + 13, py_)])
opcircle(gcx, py_)
items = [(["ReLU  →  m  (F_int × 2H × 2W)"], "act", 36),
         (["**channel attention", "avg-pool, max-pool over H, W",
           "shared MLP  F_int → h → F_int  (ReLU)", "sum → sigmoid → m ⊙ w_c"], "att", 92),
         (["**spatial attention", "[mean_c, max_c] → Conv 7×7, 2 → 1",
           "sigmoid → m ⊙ w_s"], "att", 74),
         (["Conv 1×1 F_int → 1 · GroupNorm(1, 1) · sigmoid", "→ a  (1 × 2H × 2W)"], "conv", 52)]
pos = vflow(gcx, py_ + 36, 420, items, gap=22, lh=19)
arrow([(gcx, py_ + 13), (gcx, py_ + 36)])
yo = pos[-1][1] + 42
arrow([(gcx, pos[-1][1]), (gcx, yo - 14)])
text(gcx, yo, "outputs:  ŝ ⊙ a  and  u ⊙ (1 − a)", size=FS2)
text(gx + 16, RY + RH - 34, ["MLP hidden width h = max(4, F_int/16): 4, 4, 4, 4, 8, 16",
                             "(F_int = 8, 16, 32, 64, 128, 256 for the 128² … 4² stages)"], size=13, anchor="start", lh=19)

# =====================================================================================
# row 3: (h) attention / FFN blocks, (i) top mixer, (k) head is in (a)
# =====================================================================================
AY, AH = 1980 + 165, 510
hx, hw = 20, 960
ix, iw = 1000, 580
panel(hx, AY, hw, AH, "(h) Attention and FFN blocks", "all pre-norm residual, output proj. zero-initialised")
panel(ix, AY, iw, AH, "(i) FC top around the bottom QB", "2 × 2, 1024 ch")

cols = [hx + 165, hx + 480, hx + 795]
heads = ["SpatialAttn(C), 8 heads", "ChannelAttn(C), 4 heads", "FFN(C)"]
flows = [
    [(["GroupNorm(C/4, C)"], "norm", 34),
     (["1×1 conv C → 3C", "q, k, v"], "conv", 46),
     (["softmax attention over", "H·W tokens (per head)"], "att", 50),
     (["1×1 conv C → C", "(zero-init)"], "conv", 46)],
    [(["GroupNorm(C/4, C)"], "norm", 34),
     (["1×1 conv C → 3C", "depthwise 3×3 → q, k, v"], "conv", 46),
     (["L2-norm q, k;  softmax(τ q kᵀ) v", "(C/h × C/h affinity)"], "att", 50),
     (["1×1 conv C → C", "(zero-init)"], "conv", 46)],
    [(["GroupNorm(C/4, C)"], "norm", 34),
     (["1×1 conv C → 4C", "GLU → 2C"], "conv", 46),
     (["depthwise 3×3 (2C)", "SiLU"], "conv", 46),
     (["1×1 conv 2C → C", "(zero-init)"], "conv", 46)],
]
for cxh, hd, fl in zip(cols, heads, flows):
    text(cxh, AY + 66, "**" + hd, size=FS3)
    text(cxh, AY + 96, "x", size=FS2, italic=True)
    arrow([(cxh, AY + 106), (cxh, AY + 130)])
    pos = vflow(cxh, AY + 130, 250, fl, gap=20)
    yp = pos[-1][1] + 34
    arrow([(cxh, pos[-1][1]), (cxh, yp - 13)])
    residual(cxh, AY + 118, yp, cxh + 140)
    arrow([(cxh, yp + 13), (cxh, yp + 40)])
text(hx + 16, AY + AH - 46, ["Used as (Attn + FFN) pairs:  decoder 4² … 32² stages → SpatialAttn + FFN;  64² → ChannelAttn + FFN;  128² → none.",
                             "SpatialAttn uses scaled-dot-product attention; ChannelAttn (Restormer-style) has a learnable temperature τ per head."],
     size=13, anchor="start", lh=20)

icx = ix + iw/2
text(icx, AY + 58, "encoder output  1024 × 2 × 2", size=FS3, italic=True)
arrow([(icx, AY + 70), (icx, AY + 88)])
items = [(["1×1 conv 1024 → 128 · GroupNorm(16) · LReLU", "flatten → 512"], "conv", 46),
         (["FC 512 → 512 · GroupNorm(1, 512)", "+ 1×1 conv 1024 → 128 (flattened shortcut)"], "conv", 46),
         (["ParamAtan:  α·atan(β h + γ) + δ   (β₀ = 0.01)"], "conv", 34),
         (["QB (b), N = 512"], "qb", 34),
         (["FC 512 → 512 · reshape 128 × 2 × 2", "LReLU · GroupNorm(16)"], "conv", 46),
         (["1×1 conv 128 → 1024", "+ 1×1 conv 128 → 1024 shortcut from QB output"], "conv", 46)]
pos = vflow(icx, AY + 88, 500, items, gap=14)
arrow([(icx, pos[-1][1]), (icx, pos[-1][1] + 22)])
text(icx, pos[-1][1] + 36, "→ first Up block (4²)", size=FS3, italic=True)
text(ix + 16, AY + AH - 38, ["FC layers have position-specific weights over the 2 × 2 grid (≈ 0.5 M params);",
                             "ParamAtan keeps σ(·) in the QB away from saturation at init"], size=13, anchor="start", lh=19)

# =====================================================================================
# (j) training / scoring
# =====================================================================================
JY, JH = 2510 + 165, 130
panel(20, JY, 1560, JH, "(j) Training objective and anomaly score")
text(36, JY + 78, [
    "train:  AdamW, lr 1e-3, cosine decay to 1e-5 after 5 warm-up epochs, gradient clipping 1.0;  input x + blob noise;  all QB layers add Laplace noise;  "
    "L = mean (x − x̂)²  +  λ_p · L_perc(x, x̂)  [ + optional KL sparsity on σ(h) ]",
    "L_perc: relative L1 between VGG19 relu4_2 features (ImageNet weights), random shift ≤ 8 px in training;  "
    "variance head unused (--not_use_log_var)",
    "test:  QB noise off (identity), Heaviside or LDP mode;  anomaly map = (x − x̂)² + λ_p · upsampled VGG map;  "
    "image score = perceptual term (AUC_perceptual)"],
    size=13.5, anchor="start", lh=24)

add('</svg>')
svg = "\n".join(out)
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "qbae_network_detailed")
open(base + ".svg", "w").write(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=base + ".pdf")
cairosvg.svg2png(bytestring=svg.encode(), write_to=base + ".png", output_width=W * 2)
print("ok")
