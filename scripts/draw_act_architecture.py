"""Render the planned ACT architecture; never modifies model code.

Run: MPLCONFIGDIR=/tmp/mini-wam-mpl python3 scripts/draw_act_architecture.py
Source of project contracts: SPEC.md sections 3-4, dated 2026-09-14.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "architecture"
W, H = 1800, 1580
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "none"})
fig = plt.figure(figsize=(18, 15.8), dpi=150, facecolor="white")
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")

COLORS = {
    "input": "#e7f1d7", "merge": "#d6e8c5", "encoder": "#f6d2a3",
    "model": "#c5deeb", "loss": "#e4d9ee", "gray": "#e8e8e8",
}
texts = []
boxes = []


def label(x, y, text, size=19, weight="normal", color="#222", ha="center"):
    item = ax.text(x, y, text, fontsize=size * 0.72, weight=weight,
                   color=color, ha=ha, va="center", linespacing=1.4, zorder=4)
    texts.append(item)
    return item


def panel(x, y, w, h, fill="#fcfcfc", dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=15",
        facecolor=fill, edgecolor="#d1b48c" if dashed else "#c2c2c2",
        linewidth=0.95, linestyle=(0, (5, 4)) if dashed else "solid", zorder=0))


def box(x, y, w, h, lines, color="model", size=20, gap=27):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
        facecolor=COLORS[color], edgecolor="#4b4b4b", linewidth=1.15, zorder=2))
    start = y + h / 2 - (len(lines) - 1) * gap / 2
    for i, line in enumerate(lines):
        t = label(x + w / 2, start + i * gap, line, size=size)
        boxes.append((t, (x, y, w, h)))


def arrow(points, dashed=False, color="#333"):
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.25, color=color,
                                linestyle=(0, (4, 3)) if dashed else "solid", zorder=1))


label(900, 43, "Mini-WAM | ACT Architecture", size=35, weight="bold")
label(900, 80, "ACT (Action Chunking with Transformers) · bottom-to-top data flow · tensor shapes per batch",
      size=18, color="#666")
panel(30, 110, 1060, 1255)
panel(1115, 110, 655, 1255)
label(560, 150, "(a) Action Prediction Policy", size=28, weight="bold")
label(560, 185, "Planned project architecture · used in training and inference", size=18, color="#666")
label(1442, 150, "(b) Training-only Latent Inference", size=27, weight="bold")
label(1442, 185, "CVAE (Conditional Variational Autoencoder) posterior branch", size=16, color="#666")

# Shared policy, read from the bottom upward.
box(60, 1260, 355, 80, ["Current Image  o(t)", "[B, 3, 96, 96]"], "input")
box(440, 1260, 260, 80, ["Current Position  p(t)", "[B, 2]"], "input")
box(730, 1260, 330, 80, ["Inference: z = 0", "[B, d_z]"], "gray")

box(60, 1110, 355, 115,
    ["ResNet-18 (Residual Network)", "Spatial features → project + flatten", "Visual tokens: [B, N, d]"], "encoder", 19)
box(440, 1110, 260, 115,
    ["Position Projection", "2 → d", "Position token: [B, 1, d]"], "encoder", 19)
box(730, 1110, 330, 115,
    ["Latent Projection", "d_z → d", "Latent token: [B, 1, d]"], "encoder", 19)
for x in (237.5, 570, 895):
    arrow([(x, 1260), (x, 1225)], dashed=(x == 895))

box(330, 975, 500, 95,
    ["Concatenate Visual + Position + Latent Tokens", "[B, N + 2, d]", "+ spatial / token position encodings"], "merge", 19, 26)
arrow([(237.5, 1110), (237.5, 1090), (420, 1090), (420, 1070)])
arrow([(570, 1110), (570, 1070)])
arrow([(895, 1110), (895, 1090), (740, 1090), (740, 1070)])

box(330, 815, 500, 110,
    ["Transformer Encoder", "Self-attention + feed-forward layers", "Observation memory: [B, N + 2, d]"], size=21)
arrow([(580, 975), (580, 925)])

box(330, 610, 500, 150,
    ["Transformer Decoder", "Query self-attention", "Cross-attention to observation memory", "+ feed-forward layers → [B, 16, d]"], size=21, gap=28)
arrow([(580, 815), (580, 760)])
label(695, 790, "memory", size=16, color="#666")
box(65, 635, 225, 105,
    ["16 Action Queries", "Learned slot embeddings", "[16, d]"], "gray", 18, 26)
arrow([(290, 687), (330, 687)])

box(330, 480, 500, 85,
    ["Shared Per-step Linear Action Head", "d → 2"], size=21)
arrow([(580, 610), (580, 565)])
box(330, 350, 500, 85,
    ["Predicted Action Chunk  â(t), …, â(t+15)", "[B, 16, 2] · normalized absolute target coordinates"], "input", 19)
arrow([(580, 480), (580, 435)])

box(330, 220, 500, 85,
    ["Masked L1 Action Reconstruction", "L_action = mean over valid action coordinates"], "loss", 20)
arrow([(580, 350), (580, 305)])
label(781, 330, "+ targets / mask", size=15, color="#6b526e")

# The expert action sequence is available only in the posterior / loss path.
panel(1160, 355, 565, 990, fill="#fff9ef", dashed=True)
box(1205, 220, 480, 85,
    ["Total Training Objective", "L_total = L_action + β × L_KL"], "loss", 22)
arrow([(830, 262), (1205, 262)])
label(1442, 382, "Training-only posterior supervision", size=21, weight="bold")
box(1205, 415, 480, 95,
    ["KL (Kullback–Leibler) Regularization", "L_KL = mean_B KL(q(z | p, A) ∥ N(0, I))", "β: configured separately"], "loss", 19, 26)
arrow([(1445, 415), (1445, 305)])

box(1210, 630, 470, 95,
    ["Reparameterized Latent Sample", "z = μ + exp(0.5 × log σ²) ⊙ ε", "ε ∼ N(0, I)     z: [B, d_z]"], "merge", 21, 26)
arrow([(1210, 677), (1137, 677), (1137, 1170), (1060, 1170)])
label(1221, 592, "sample z to policy", size=16, color="#666", ha="left")

box(1210, 790, 470, 95,
    ["Posterior Distribution  q(z | p, A)", "Summary output → two linear heads", "μ, log σ²: each [B, d_z]"], "model", 20, 27)
arrow([(1445, 790), (1445, 725)])
arrow([(1680, 835), (1706, 835), (1706, 462), (1685, 462)])

box(1210, 940, 470, 110,
    ["Variational Transformer Encoder", "Self-attention + feed-forward layers", "Read summary token: [B, d]"], "model", 21)
arrow([(1445, 940), (1445, 885)])
box(1210, 1095, 470, 115,
    ["Summary + Position + Action Tokens", "[B, 1 + 1 + 16, d] = [B, 18, d]", "+ fixed sinusoidal time encoding", "Padding actions masked in attention"], "encoder", 19, 25)
arrow([(1445, 1095), (1445, 1050)])
box(1210, 1250, 470, 90,
    ["Current Position [B, 2] + Expert Actions [B, 16, 2]", "A = a(t), …, a(t+15)", "action_valid_mask: [B, 16] · True = valid"], "input", 18, 25)
arrow([(1445, 1250), (1445, 1210)])

# Clarifying annotations in the open space keep dimensions deliberately symbolic.
label(190, 863, "Spatial grid retained\nN = feature height × width\nNo global pooling", size=17, color="#666")
label(944, 884, "Training: sampled z\nInference: zero z", size=17, color="#666")

label(900, 1400,
      "Inference: current image + current position + z = 0 → predict 16 actions → denormalize and clip → execute first 4 → observe again.",
      size=19, weight="bold")
label(900, 1434,
      "All 16 actions are predicted together. The expert-action encoder is removed at inference; temporal ensembling is disabled in the initial setup.",
      size=18)
label(900, 1472,
      "B = batch size · N = visual token count · d = model width · d_z = latent width · β = regularization weight",
      size=17, color="#666")
label(900, 1502,
      "Planned specification: widths, layer counts and β are not fixed yet. Mini-WAM: Mini World Action Model (project name).",
      size=17, color="#666")
label(900, 1533,
      "Source: SPEC.md §§3–4 (2026-09-14) + official ACT reference structure · act.py is currently empty; implementation and training are pending.",
      size=16, color="#666")

OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
# Check actual rendered text bounds, including all multiline labels.
for t, (x, y, w, h) in boxes:
    bounds = t.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bounds.x0 >= x + 5 and bounds.x1 <= x + w - 5, t.get_text()
for t in texts:
    bounds = t.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bounds.x0 >= 0 and bounds.x1 <= W, t.get_text()

for extension in ("svg", "png"):
    target = OUT / f"act_architecture.{extension}"
    fig.savefig(target, dpi=180, facecolor="white")
    print(target)
plt.close(fig)
