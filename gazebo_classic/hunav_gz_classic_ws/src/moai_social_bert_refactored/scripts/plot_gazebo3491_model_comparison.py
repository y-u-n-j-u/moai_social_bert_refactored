#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_JSON = (
    PROJECT_ROOT
    / "figures"
    / "gazebo_3491_guided_map20m"
    / "guided_mgp_visualization_metrics.json"
)

FINETUNE_JSON = (
    PROJECT_ROOT
    / "figures"
    / "gazebo_3491_guided_ethucy_pt"
    / "guided_mgp_visualization_metrics.json"
)

OUTPUT_DIR = PROJECT_ROOT / "figures" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "gazebo3491_scratch_vs_ethucy_finetune.png"
OUTPUT_PDF = OUTPUT_DIR / "gazebo3491_scratch_vs_ethucy_finetune.pdf"


# ============================================================
# 2. 디자인 설정
# ============================================================

BASELINE_COLOR = "#2563EB"
FINETUNE_COLOR = "#F97316"
TEXT_COLOR = "#172033"
GRID_COLOR = "#CBD5E1"
BACKGROUND_COLOR = "#F8FAFC"
GOOD_COLOR = "#DCFCE7"
BAD_COLOR = "#FEE2E2"
NEUTRAL_COLOR = "#F1F5F9"


def configure_font():
    """설치된 Noto Sans CJK 폰트를 찾아 한글 깨짐을 방지한다."""

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]

    for path in candidates:
        font_path = Path(path)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def load_metrics(path):
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("status") != "PASS":
        raise RuntimeError(f"검증을 통과하지 못한 결과입니다: {path}")

    return payload["test_inference"]


def add_bar_labels(ax, bars, suffix="", decimals=3):
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)


# ============================================================
# 3. 데이터 읽기
# ============================================================

configure_font()

baseline = load_metrics(BASELINE_JSON)
finetune = load_metrics(FINETUNE_JSON)

model_names = [
    "Gazebo-only",
    "ETH/UCY → Gazebo",
]

error_metric_names = [
    "ADE",
    "FDE",
    "GDE",
]

baseline_errors = np.array([
    baseline["all_sample_ade"],
    baseline["all_sample_fde"],
    baseline["all_sample_gde"],
])

finetune_errors = np.array([
    finetune["all_sample_ade"],
    finetune["all_sample_fde"],
    finetune["all_sample_gde"],
])

safety_metric_names = [
    "Safe candidates",
    "Selected goal valid",
    "Trajectory map-safe",
    "Execution valid",
]

baseline_safety = np.array([
    baseline["safe_candidate_rate"],
    baseline["selected_goal_valid_rate"],
    baseline["trajectory_map_safe_rate"],
    baseline["execution_valid_rate"],
]) * 100.0

finetune_safety = np.array([
    finetune["safe_candidate_rate"],
    finetune["selected_goal_valid_rate"],
    finetune["trajectory_map_safe_rate"],
    finetune["execution_valid_rate"],
]) * 100.0


# ============================================================
# 4. Figure 레이아웃
# ============================================================

fig = plt.figure(
    figsize=(15, 10),
    facecolor=BACKGROUND_COLOR,
)

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[2.0, 1.35],
    hspace=0.38,
    wspace=0.25,
)

ax_error = fig.add_subplot(grid[0, 0])
ax_safety = fig.add_subplot(grid[0, 1])
ax_table = fig.add_subplot(grid[1, :])

fig.suptitle(
    "Gazebo-only vs ETH/UCY 사전학습 후 Fine-tuning",
    fontsize=23,
    fontweight="bold",
    color=TEXT_COLOR,
    y=0.98,
)

fig.text(
    0.5,
    0.942,
    "동일한 Gazebo test split 524개 · Best-validation checkpoint · Seed 20",
    ha="center",
    fontsize=11,
    color="#64748B",
)


# ============================================================
# 5. ADE/FDE/GDE 오차 그래프
# ============================================================

x_error = np.arange(len(error_metric_names))
bar_width = 0.34

baseline_error_bars = ax_error.bar(
    x_error - bar_width / 2,
    baseline_errors,
    width=bar_width,
    color=BASELINE_COLOR,
    label=model_names[0],
    edgecolor="white",
    linewidth=0.8,
)

finetune_error_bars = ax_error.bar(
    x_error + bar_width / 2,
    finetune_errors,
    width=bar_width,
    color=FINETUNE_COLOR,
    label=model_names[1],
    edgecolor="white",
    linewidth=0.8,
)

add_bar_labels(ax_error, baseline_error_bars, suffix=" m", decimals=3)
add_bar_labels(ax_error, finetune_error_bars, suffix=" m", decimals=3)

ax_error.set_title(
    "Trajectory·Goal 오차 ↓",
    fontsize=15,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=14,
)

ax_error.set_ylabel("Error (m)", fontsize=11)
ax_error.set_xticks(x_error)
ax_error.set_xticklabels(error_metric_names, fontsize=11)
ax_error.set_ylim(
    0,
    max(baseline_errors.max(), finetune_errors.max()) * 1.25,
)

style_axis(ax_error)


# ============================================================
# 6. 안전성 그래프
# ============================================================

x_safety = np.arange(len(safety_metric_names))

baseline_safety_bars = ax_safety.bar(
    x_safety - bar_width / 2,
    baseline_safety,
    width=bar_width,
    color=BASELINE_COLOR,
    label=model_names[0],
    edgecolor="white",
    linewidth=0.8,
)

finetune_safety_bars = ax_safety.bar(
    x_safety + bar_width / 2,
    finetune_safety,
    width=bar_width,
    color=FINETUNE_COLOR,
    label=model_names[1],
    edgecolor="white",
    linewidth=0.8,
)

add_bar_labels(
    ax_safety,
    baseline_safety_bars,
    suffix="%",
    decimals=1,
)

add_bar_labels(
    ax_safety,
    finetune_safety_bars,
    suffix="%",
    decimals=1,
)

ax_safety.set_title(
    "Goal·Trajectory 안전율 ↑",
    fontsize=15,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=14,
)

ax_safety.set_ylabel("Rate (%)", fontsize=11)
ax_safety.set_xticks(x_safety)
ax_safety.set_xticklabels(
    [
        "Safe\ncandidates",
        "Selected goal\nvalid",
        "Trajectory\nmap-safe",
        "Execution\nvalid",
    ],
    fontsize=9.5,
)

# 비율 그래프의 축을 0부터 시작해 차이를 과장하지 않는다.
ax_safety.set_ylim(0, 108)

style_axis(ax_safety)


# 두 그래프 공통 범례
handles, labels = ax_error.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.915),
    ncol=2,
    frameon=False,
    fontsize=11,
)


# ============================================================
# 7. 결과 표
# ============================================================

ax_table.axis("off")

rows = [
    ("ADE ↓", baseline_errors[0], finetune_errors[0], "error"),
    ("FDE ↓", baseline_errors[1], finetune_errors[1], "error"),
    ("GDE ↓", baseline_errors[2], finetune_errors[2], "error"),
    (
        "Safe candidates ↑",
        baseline_safety[0],
        finetune_safety[0],
        "rate",
    ),
    (
        "Selected goal valid ↑",
        baseline_safety[1],
        finetune_safety[1],
        "rate",
    ),
    (
        "Trajectory map-safe ↑",
        baseline_safety[2],
        finetune_safety[2],
        "rate",
    ),
    (
        "Execution valid ↑",
        baseline_safety[3],
        finetune_safety[3],
        "rate",
    ),
]

table_data = []
delta_is_good = []

for metric_name, baseline_value, finetune_value, metric_type in rows:
    if metric_type == "error":
        delta_percent = (
            (finetune_value - baseline_value)
            / baseline_value
            * 100.0
        )

        delta_text = f"{delta_percent:+.1f}%"
        baseline_text = f"{baseline_value:.3f} m"
        finetune_text = f"{finetune_value:.3f} m"
        is_good = finetune_value < baseline_value

    else:
        delta_pp = finetune_value - baseline_value

        delta_text = f"{delta_pp:+.2f}%p"
        baseline_text = f"{baseline_value:.2f}%"
        finetune_text = f"{finetune_value:.2f}%"
        is_good = finetune_value > baseline_value

    table_data.append([
        metric_name,
        baseline_text,
        finetune_text,
        delta_text,
    ])

    delta_is_good.append(is_good)

table = ax_table.table(
    cellText=table_data,
    colLabels=[
        "평가 지표",
        "Gazebo-only",
        "ETH/UCY → Gazebo",
        "Fine-tuning 변화",
    ],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.27, 0.22, 0.28, 0.23],
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.65)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_linewidth(2)

    if row == 0:
        cell.set_facecolor(TEXT_COLOR)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("white")
        cell.get_text().set_color(TEXT_COLOR)

        if col == 0:
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor(NEUTRAL_COLOR)

        if col == 3:
            if delta_is_good[row - 1]:
                cell.set_facecolor(GOOD_COLOR)
                cell.get_text().set_color("#166534")
            else:
                cell.set_facecolor(BAD_COLOR)
                cell.get_text().set_color("#991B1B")

            cell.get_text().set_fontweight("bold")


# ============================================================
# 8. 결론과 주석
# ============================================================

fig.text(
    0.5,
    0.075,
    "결론: ETH/UCY 사전학습은 ADE·FDE와 TGP 실행 안전율을 개선했지만, "
    "GDE와 MGP 후보 안전율은 소폭 감소함",
    ha="center",
    fontsize=11.5,
    fontweight="bold",
    color=TEXT_COLOR,
)


fig.text(
    0.5,
    0.045,
    "※ 단일 seed 결과이므로 최종 결론을 위해 여러 seed 반복 실험이 필요함",
    ha="center",
    fontsize=9.5,
    color="#64748B",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=240,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.close(fig)

print(f"PNG 저장 완료: {OUTPUT_PNG}")
print(f"PDF 저장 완료: {OUTPUT_PDF}")
