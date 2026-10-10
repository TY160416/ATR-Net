import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

try:
    matplotlib.use("tkagg")
except ImportError:
    pass
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

file_mapping = {
    "BiLSTM": "HSI-BiLSTM_Predictions.csv",
    "Transformer": "HSI-Transformer_Predictions.csv",
    "CNN-LSTM": "HSI-CNN_LSTM_Predictions.csv",
    "PatchTST": "HSI-PatchTST_Predictions.csv",
    "TimesNet": "HSI-TimesNet_Predictions.csv",
    "Mamba": "HSI-Mamba_Predictions.csv",
    "iTransformer": "HSI-iTransformer_Predictions.csv",
    "BiLSTM-TCN": "HSI-BiLSTMTCN_Predictions.csv",
    "TCN-Transformer": "HSI-TCN_Transformer_Predictions.csv",
    "ModernTCN": "HSI-modernTCN_Predictions.csv",
    "ATR-Net": "HSI-ATR_Net_Predictions.csv"
}

color_true = 'lightsteelblue'

colors_main = [
    'gold',           # BiLSTM
    'magenta',        # Transformer
    'darkorchid',     # CNN-LSTM
    'sandybrown',     # PatchTST
    'brown',          # TimesNet
    'coral',          # Mamba
    'mediumseagreen', # iTransformer
    'blue',           # BiLSTM-TCN
    'cyan',           # TCN-Transformer
    'teal',           # ModernTCN
    'k'               # ATR-Net
]

model_names = list(file_mapping.keys())
line_styles = ['--', '--', '--', '--', '-.', '-.', '--', '--', '--', ':', '-']

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1], projection='polar')

y_true = None
predictions = {}

current_dir = os.path.dirname(os.path.abspath(__file__))

for model_name, file_name in file_mapping.items():
    full_path = os.path.join(current_dir, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"找不到文件: '{full_path}'。")

    df = pd.read_csv(full_path, header=0)

    col_1_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
    col_2_data = pd.to_numeric(df.iloc[:, 2], errors='coerce').dropna().values

    if y_true is None:
        y_true = col_1_data

    predictions[model_name] = col_2_data

min_len = len(y_true)
for p in predictions.values():
    min_len = min(min_len, len(p))

y_true = y_true[:min_len]

rmse_vals, mae_vals, mape_vals, r2_vals = [], [], [], []
error_data_list = []

ax1.plot(y_true, color=color_true, linewidth=3.5, label='True Price', zorder=1)

for i, model_name in enumerate(model_names):
    y_p = predictions[model_name][:min_len]
    error_data_list.append(y_p - y_true)

    rmse_vals.append(np.sqrt(mean_squared_error(y_true, y_p)))
    mae_vals.append(mean_absolute_error(y_true, y_p))
    mape_vals.append(np.mean(np.abs((y_true - y_p) / (y_true + 1e-8))) * 100)
    r2_vals.append(r2_score(y_true, y_p))

    is_proposed = (model_name == 'ATR-Net')
    line_w = 2.5 if is_proposed else 1.2
    z_ord = 10 if is_proposed else 5

    ax1.plot(y_p, color=colors_main[i], linestyle=line_styles[i], linewidth=line_w, label=model_name, alpha=0.9,
             zorder=z_ord)

ax1.set_xlabel('Time Step', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)

x_offset = 0.25
for i, model_name in enumerate(model_names):
    model_data = error_data_list[i]
    is_proposed = (model_name == 'ATR-Net')

    alpha_value = 0.6 if is_proposed else 0.4

    violin = ax2.violinplot(model_data, positions=[i], showextrema=False, showmedians=False, widths=0.45)
    for pc in violin['bodies']:
        pc.set_facecolor(colors_main[i])
        pc.set_edgecolor(colors_main[i])
        pc.set_alpha(alpha_value)
        pc.set_linewidth(1.5 if is_proposed else 1.0)
        vertices = pc.get_paths()[0].vertices
        vertices[:, 0] = np.clip(vertices[:, 0], i, i + x_offset)
        pc.set_paths([vertices])

    box_linewidth = 1.5 if is_proposed else 1.2
    bp = ax2.boxplot(model_data, positions=[i], widths=0.08, patch_artist=True,
                     boxprops={'facecolor': 'white', 'edgecolor': colors_main[i], 'linewidth': box_linewidth},
                     whiskerprops={'color': colors_main[i], 'linewidth': box_linewidth},
                     capprops={'color': colors_main[i], 'linewidth': box_linewidth},
                     medianprops={'color': colors_main[i], 'linewidth': 1.5},
                     showfliers=False)

    x_jitter = np.random.normal(i - x_offset, 0.04, size=len(model_data))
    ax2.scatter(x_jitter, model_data, color=colors_main[i], alpha=0.7 if is_proposed else 0.5,
                s=10 if is_proposed else 7, edgecolors='none')

ax2.set_ylabel('Prediction Error', fontsize=14, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels([])
ax2.tick_params(axis='x', length=0)

y_min, y_max = ax2.get_ylim()
y_range = y_max - y_min
ax2.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis='both', which='major', width=2.0, length=6)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')

ax3.set_rorigin(-0.25)
ax3.spines['polar'].set_linewidth(2.0)

metrics = ['RMSE', 'MAE', 'MAPE', 'R²']
N_metrics, N_models = len(metrics), len(model_names)

theta_metrics = np.linspace(0.0, 2 * np.pi, N_metrics, endpoint=False) + (np.pi / 4)

sector_width = (2 * np.pi / N_metrics) * 0.85
bar_width = sector_width / N_models
offsets = np.linspace(-(N_models - 1) / 2 * bar_width, (N_models - 1) / 2 * bar_width, N_models)

norm_data = [
    np.array(rmse_vals) / np.max(rmse_vals),
    np.array(mae_vals) / np.max(mae_vals),
    np.array(mape_vals) / np.max(mape_vals),
    np.array(r2_vals) / np.max(r2_vals)
]
raw_data = [rmse_vals, mae_vals, mape_vals, r2_vals]

formats = ["{:.4f}", "{:.4f}", "{:.4f}%", "{:.4f}"]

def draw_gradient_polar_bar(ax, angle, total_height, width, color_end, segments=30):
    cmap = LinearSegmentedColormap.from_list('grad', ['#FFFFFF', color_end])
    segment_height = total_height / segments
    for j in range(segments):
        bottom = j * segment_height
        color = cmap(j / (segments - 1))
        ax.bar(angle, segment_height, width=width, bottom=bottom, color=color, edgecolor='none', alpha=0.9, zorder=3)
    ax.bar(angle, total_height, width=width, bottom=0, color='none', edgecolor='black', linewidth=0.8, zorder=4)

boundary_angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
max_radius = 2.35

for k in range(N_metrics):
    if k % 2 == 1:
        center_quadrant = boundary_angles[k] + np.pi / 4
        ax3.bar(center_quadrant, max_radius, width=np.pi / 2, bottom=0,
                color='#F3F5F8', alpha=0.55, edgecolor='none', zorder=1)

for b_ang in boundary_angles:
    ax3.plot([b_ang, b_ang], [0, max_radius], color='#7F8C8D', linestyle='-.',
             linewidth=1.4, alpha=0.8, zorder=2)

for m_idx in range(N_metrics):
    center_theta = theta_metrics[m_idx]
    for c_idx in range(N_models):
        current_angle = center_theta + offsets[c_idx]
        bar_height = norm_data[m_idx][c_idx]

        draw_gradient_polar_bar(ax3, current_angle, bar_height, bar_width, colors_main[c_idx])

        val_text = formats[m_idx].format(raw_data[m_idx][c_idx])

        angle_deg = np.degrees(current_angle) % 360

        if 90 < angle_deg < 270:
            rot_text = angle_deg - 180
            ha_val = 'right'
        else:
            rot_text = angle_deg
            ha_val = 'left'

        text_r = max(bar_height + 0.08, 1.08)
        ax3.text(current_angle, text_r, val_text,
                 ha=ha_val, va='center',
                 fontsize=9.5, fontweight='bold', color='black',
                 rotation=rot_text, rotation_mode='anchor', zorder=5)

ax3.set_xticks(theta_metrics)
ax3.set_xticklabels(metrics, fontsize=15, fontweight='bold')
ax3.set_yticklabels([])
ax3.grid(True, linestyle='--', alpha=0.35, zorder=1)

ax3.set_ylim(0, max_radius)
ax3.tick_params(axis='x', pad=28)

bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none")
ax1.text(0.01, 0.96, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='left', zorder=20,
         bbox=bbox_props)
ax2.text(0.02, 0.96, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='left', zorder=20,
         bbox=bbox_props)
ax3.text(-0.05, 1.05, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='left', zorder=20,
         bbox=bbox_props)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6,
           frameon=False, prop={'weight': 'bold', 'size': 14})

plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.92])
plt.savefig('HSI_Predictions.png', bbox_inches='tight', dpi=600)
plt.show()