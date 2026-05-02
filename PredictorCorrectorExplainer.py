import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
plt.subplots_adjust(wspace=0.28)

# ============================================================
# Shared setup: a smooth "boundary" curve (ellipse-like)
# ============================================================
t_full = np.linspace(0, 2 * np.pi, 300)
cx, cy = 0.0, 0.0
rx, ry = 2.5, 1.8
boundary_x = cx + rx * np.cos(t_full)
boundary_y = cy + ry * np.sin(t_full)

# Colors
SAFE_COLOR = "#d4edda"
DEAD_COLOR = "#f8d7da"
BOUNDARY_COLOR = "#1a1a2e"
ACCENT_1 = "#e63946"
ACCENT_2 = "#457b9d"
ACCENT_3 = "#2a9d8f"
GRADIENT_COLOR = "#e76f51"
TANGENT_COLOR = "#264653"
PREDICT_COLOR = "#f4a261"
CORRECT_COLOR = "#2a9d8f"
SEED_COLOR = "#e63946"
TRACED_COLOR = "#1d3557"

for ax in axes:
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15)

# ============================================================
# PANEL 1: The Problem — uniform grid wastes evaluations
# ============================================================
ax = axes[0]
ax.set_title(
    "1. Coarse Grid + Gradient Scan\nFinds Seed Points",
    fontsize=11,
    fontweight="bold",
    pad=10,
)

# Fill regions
theta = np.linspace(0, 2 * np.pi, 200)
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.6, zorder=0)
ax.fill_between(
    [-4.2, 4.2], [-3.5, -3.5], [3.5, 3.5], color=SAFE_COLOR, alpha=0.4, zorder=-1
)
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.6, zorder=0)

# Draw true boundary
ax.plot(
    boundary_x,
    boundary_y,
    color=BOUNDARY_COLOR,
    linewidth=2,
    zorder=5,
    label="True boundary",
)

# Uniform grid
gx = np.linspace(-3.8, 3.8, 14)
gy = np.linspace(-3.0, 3.0, 11)
count_total = 0
count_wasted = 0
for x in gx:
    for y in gy:
        inside = (x / rx) ** 2 + (y / ry) ** 2 < 1
        near_boundary = abs((x / rx) ** 2 + (y / ry) ** 2 - 1) < 0.25
        ax.plot(x, y, "o", color="#adb5bd", markersize=3, zorder=4, alpha=0.5)
        count_total += 1

ax.text(
    0,
    -3.2,
    "Grey circles show the seed points",
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#555",
)
ax.text(2.0, 2.6, "Safe", fontsize=15, color="#28a745", fontweight="bold")
ax.text(0, 0, "Dead-end", fontsize=15, color=ACCENT_1, fontweight="bold", ha="center")

# ============================================================
# PANEL 2: Find one boundary point (Newton corrector)
# ============================================================
ax = axes[1]
ax.set_title(
    "2. Newton Corrector\nProjects Onto $g(a) = 0$",
    fontsize=11,
    fontweight="bold",
    pad=10,
)

# Regions + boundary
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)
ax.fill_between(
    [-4.2, 4.2], [-3.5, -3.5], [3.5, 3.5], color=SAFE_COLOR, alpha=0.2, zorder=-1
)
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)
ax.plot(boundary_x, boundary_y, color=BOUNDARY_COLOR, linewidth=2, alpha=0.4, zorder=3)

# Starting point (in safe region)
a0 = np.array([3.2, 1.0])
ax.plot(*a0, "o", color=ACCENT_2, markersize=10, zorder=10)
ax.annotate(
    "$a_0$: start\n(safe side)",
    [a0[0], a0[1] + 0.2],
    textcoords="offset points",
    xytext=(-35, 10),
    fontsize=9,
    color=ACCENT_2,
    fontweight="bold",
)

# Newton iterations toward boundary
points = [a0]
for i in range(3):
    p = points[-1]
    g_val = (p[0] / rx) ** 2 + (p[1] / ry) ** 2 - 1
    grad = np.array([2 * p[0] / rx**2, 2 * p[1] / ry**2])
    grad_norm_sq = np.dot(grad, grad)
    p_new = p - (g_val / grad_norm_sq) * grad
    points.append(p_new)

# Draw Newton steps
for i in range(len(points) - 1):
    p1, p2 = points[i], points[i + 1]
    ax.annotate(
        "", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color=ACCENT_1, lw=2)
    )
    if i > 0:
        ax.plot(*p1, "o", color=ACCENT_1, markersize=6, zorder=10)

# Draw gradient arrow at a0
grad_at_a0 = np.array([2 * a0[0] / rx**2, 2 * a0[1] / ry**2])
grad_at_a0_norm = grad_at_a0 / np.linalg.norm(grad_at_a0) * 0.9
ax.annotate(
    "",
    xy=a0 + grad_at_a0_norm,
    xytext=a0,
    arrowprops=dict(arrowstyle="->", color=GRADIENT_COLOR, lw=2.5, linestyle="-"),
)
ax.text(
    a0[0] + grad_at_a0_norm[0] - 0.3,
    a0[1] + grad_at_a0_norm[1] - 0.6,
    "$\\nabla g$",
    fontsize=11,
    color=GRADIENT_COLOR,
    fontweight="bold",
)

# Legend for arrow meanings
arrow_handles = [
    Line2D([0], [0], color=GRADIENT_COLOR, lw=2.5, label="Gradient (normal)"),
    Line2D([0], [0], color=ACCENT_1, lw=2, label="Newton update"),
]
ax.legend(handles=arrow_handles, loc="upper left", fontsize=8, framealpha=0.9)

# Final point on boundary
ax.plot(
    *points[-1],
    "*",
    color=CORRECT_COLOR,
    markersize=18,
    zorder=11,
    markeredgecolor="white",
    markeredgewidth=0.5,
)
ax.annotate(
    "$a^*$: on boundary\n($g = 0$)",
    points[-1],
    textcoords="offset points",
    xytext=(-55, -25),
    fontsize=9,
    color=CORRECT_COLOR,
    fontweight="bold",
)

ax.text(
    0,
    -3.2,
    "Newton steps: project onto $g(a)=0$\nusing $\\nabla_a g$ from backprop",
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#555",
)

# ============================================================
# PANEL 3: Predictor-Corrector step
# ============================================================
ax = axes[2]
ax.set_title(
    "3. Predictor-Corrector Step\nPredict → Correct",
    fontsize=11,
    fontweight="bold",
    pad=10,
)

# Regions + boundary
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)
ax.fill_between(
    [-4.2, 4.2], [-3.5, -3.5], [3.5, 3.5], color=SAFE_COLOR, alpha=0.2, zorder=-1
)
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)
ax.plot(boundary_x, boundary_y, color=BOUNDARY_COLOR, linewidth=2, alpha=0.4, zorder=3)

# Current boundary point
t_star = 0.4
a_star = np.array([rx * np.cos(t_star), ry * np.sin(t_star)])
ax.plot(
    *a_star,
    "*",
    color=TRACED_COLOR,
    markersize=16,
    zorder=10,
    markeredgecolor="white",
    markeredgewidth=0.5,
)
ax.annotate(
    "$a$",
    a_star,
    textcoords="axes fraction",
    xytext=(0.7, 0.45),
    fontsize=7.8,
    color=TRACED_COLOR,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=TRACED_COLOR, lw=1),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

# Gradient (normal to boundary)
grad = np.array([2 * a_star[0] / rx**2, 2 * a_star[1] / ry**2])
grad_unit = grad / np.linalg.norm(grad)
ax.annotate(
    "",
    xy=a_star + grad_unit * 1.0,
    xytext=a_star,
    arrowprops=dict(arrowstyle="->", color=GRADIENT_COLOR, lw=2.5),
)
ax.annotate(
    "$\\nabla g$ (normal)",
    xy=a_star + grad_unit * 1.0,
    xytext=(0.805, 0.58),
    textcoords="axes fraction",
    fontsize=7.0,
    color=GRADIENT_COLOR,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

# Tangent (perpendicular to gradient)
tangent = np.array([-grad_unit[1], grad_unit[0]])
step_size = 1.2

# Predicted point (along tangent)
a_tilde = a_star + step_size * tangent
ax.annotate(
    "",
    xy=a_tilde,
    xytext=a_star,
    arrowprops=dict(arrowstyle="->", color=TANGENT_COLOR, lw=2.5, linestyle="--"),
)
ax.plot(
    *a_tilde,
    "D",
    color=PREDICT_COLOR,
    markersize=10,
    zorder=10,
    markeredgecolor="white",
    markeredgewidth=1,
)
ax.annotate(
    "$\\tilde{a}$: predicted\n(slightly off)",
    a_tilde,
    textcoords="axes fraction",
    xytext=(0.75, 0.88),
    fontsize=7.0,
    color=PREDICT_COLOR,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=PREDICT_COLOR, lw=1),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

# Tangent label
mid = a_star + 0.5 * step_size * tangent
ax.annotate(
    "tangent $t$",
    xy=mid,
    xytext=(0.8, 0.78),
    textcoords="axes fraction",
    fontsize=8,
    color=TANGENT_COLOR,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=TANGENT_COLOR, lw=1),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

# Correction arrow (Newton back to boundary)
t_closest = np.arctan2(a_tilde[1] / ry, a_tilde[0] / rx)
a_corrected = np.array([rx * np.cos(t_closest), ry * np.sin(t_closest)])

ax.annotate(
    "",
    xy=a_corrected,
    xytext=a_tilde,
    arrowprops=dict(arrowstyle="->", color=CORRECT_COLOR, lw=2.5),
)
ax.plot(
    *a_corrected,
    "*",
    color=CORRECT_COLOR,
    markersize=16,
    zorder=11,
    markeredgecolor="white",
    markeredgewidth=0.5,
)
ax.annotate(
    "$a^*$: corrected\n(back on boundary)",
    a_corrected,
    textcoords="axes fraction",
    xytext=(0.38, 0.86),
    fontsize=7.0,
    color=CORRECT_COLOR,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=CORRECT_COLOR, lw=1),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

# # Label the two phases
# ax.annotate(
#     "PREDICT",
#     xy=(a_star[0] - 1.4, a_star[1] - 1.6),
#     fontsize=7.5,
#     color=TANGENT_COLOR,
#     fontweight="bold",
#     bbox=dict(
#         boxstyle="round,pad=0.3", facecolor="white", edgecolor=TANGENT_COLOR, alpha=0.8
#     ),
# )
# ax.annotate(
#     "CORRECT",
#     xy=(a_tilde[0] + 0.8, a_tilde[1] + 0.1),
#     fontsize=7.5,
#     color=CORRECT_COLOR,
#     fontweight="bold",
#     bbox=dict(
#         boxstyle="round,pad=0.3", facecolor="white", edgecolor=CORRECT_COLOR, alpha=0.8
#     ),
# )

ax.text(
    0,
    -3.2,
    "Step along tangent (predict),\nthen Newton back to boundary (correct)",
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#555",
)

# ============================================================
# PANEL 4: Full traced boundary
# ============================================================
ax = axes[3]
ax.set_title(
    "4. Full Boundary Traced\nSmooth Curve from ~300 Evals",
    fontsize=11,
    fontweight="bold",
    pad=10,
)

# Regions
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)
ax.fill_between(
    [-4.2, 4.2], [-3.5, -3.5], [3.5, 3.5], color=SAFE_COLOR, alpha=0.2, zorder=-1
)
ax.fill(rx * np.cos(theta), ry * np.sin(theta), color=DEAD_COLOR, alpha=0.3, zorder=0)

# Ground truth boundary (dashed)
ax.plot(
    boundary_x,
    boundary_y,
    "--",
    color="#adb5bd",
    linewidth=2,
    zorder=3,
    label="Ground truth",
)

# Traced boundary (solid, with points)
n_trace = 40
t_trace = np.linspace(0, 2 * np.pi, n_trace, endpoint=False)
trace_x = rx * np.cos(t_trace) + np.random.normal(0, 0.01, n_trace)
trace_y = ry * np.sin(t_trace) + np.random.normal(0, 0.01, n_trace)

# Close the loop
trace_x = np.append(trace_x, trace_x[0])
trace_y = np.append(trace_y, trace_y[0])

ax.plot(
    trace_x,
    trace_y,
    "-",
    color=TRACED_COLOR,
    linewidth=2.5,
    zorder=5,
    label="Traced boundary",
)
ax.plot(trace_x[:-1], trace_y[:-1], "o", color=TRACED_COLOR, markersize=3.5, zorder=6)

# Show a few direction arrows along the trace
for i in range(0, n_trace, 5):
    dx = trace_x[i + 1] - trace_x[i]
    dy = trace_y[i + 1] - trace_y[i]
    norm = np.sqrt(dx**2 + dy**2)
    ax.annotate(
        "",
        xy=(trace_x[i] + dx * 0.6, trace_y[i] + dy * 0.6),
        xytext=(trace_x[i], trace_y[i]),
        arrowprops=dict(arrowstyle="->", color=ACCENT_2, lw=1.5),
    )

# Show the seed point
ax.plot(
    trace_x[0],
    trace_y[0],
    "*",
    color=SEED_COLOR,
    markersize=15,
    zorder=10,
    markeredgecolor="white",
    markeredgewidth=0.5,
)
ax.annotate(
    "seed",
    (trace_x[0], trace_y[0]-0.3),
    textcoords="offset points",
    xytext=(10, 8),
    fontsize=9,
    color=SEED_COLOR,
    fontweight="bold",
)

# Coarse grid in background
gx = np.linspace(-3.8, 3.8, 8)
gy = np.linspace(-3.0, 3.0, 6)
for x in gx:
    for y in gy:
        ax.plot(x, y, "s", color="#dee2e6", markersize=3, zorder=1)

ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

# Evaluation count comparison
ax.text(
    0,
    -3.2,
    "Cost: ~300 evals (vs 10,000 for grid)\nOutput: ordered curve, not scattered points",
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#555",
)

ax.text(2.0, 2.6, "Safe", fontsize=15, color="#28a745", fontweight="bold")
ax.text(
    0, 0, "Dead-end", fontsize=15, color=ACCENT_1, fontweight="bold", ha="center"
)


fig_path = "./predictor_corrector_visual.png"
os.remove(fig_path)
plt.savefig(
    fig_path,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("Done")
