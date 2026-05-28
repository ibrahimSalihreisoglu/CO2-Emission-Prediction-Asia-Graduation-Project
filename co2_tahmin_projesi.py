# =============================================================================
# PROJE  : ÇEVRESEL SÜRDÜRÜLEBİLİRLİKTE MAKİNE ÖĞRENMESİ YÖNTEMLERİ
# KONU   : 16 Asya Ülkesi için Kişi Başı CO₂ Emisyonu Tahmini (1990–2020)
# OKUL   : Dokuz Eylül Üniversitesi, İstatistik Bölümü
# =============================================================================

# Colab'da çalıştırırken aşağıdaki satırın başındaki # işaretini kaldırın:
# !pip install catboost xgboost lightgbm shap streamlit google-generativeai openpyxl

import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import kruskal
from scipy import stats

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import google.generativeai as genai

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# =============================================================================
# GÖRSEL TEMA
# =============================================================================
PALETTE_MAIN   = "#1B2A4A"
PALETTE_ACCENT = "#E8553E"
PALETTE_SOFT   = "#5B8DB8"
PALETTE_GREEN  = "#3DAA72"
PALETTE_WARN   = "#F4A623"
PALETTE_PURPLE = "#7C5CBF"
BG_COLOR       = "#F8F9FB"
GRID_COLOR     = "#DDE3EC"

plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : "#FFFFFF",
    "axes.edgecolor"   : "#C8D0DC",
    "axes.linewidth"   : 0.8,
    "axes.grid"        : True,
    "grid.color"       : GRID_COLOR,
    "grid.linewidth"   : 0.6,
    "grid.alpha"       : 0.7,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.titlepad"    : 12,
    "axes.labelsize"   : 10,
    "axes.labelcolor"  : "#2C3E50",
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "xtick.color"      : "#555555",
    "ytick.color"      : "#555555",
    "legend.framealpha": 0.92,
    "legend.edgecolor" : "#CCCCCC",
    "legend.fontsize"  : 9,
    "figure.dpi"       : 130,
    "savefig.dpi"      : 200,
    "savefig.bbox"     : "tight",
})

def watermark(fig, text="CO2 Tahmin Projesi"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color="#BBBBBB", style="italic")


# =============================================================================
# 1. VERİ YÜKLEME VE ON ISLEME
# =============================================================================
df = pd.read_excel("Final.xlsx")

df.rename(columns={
    'Country'       : 'Ulke',
    'Year'          : 'Yil',
    'energy_pc'     : 'Enerji_Tuketimi_KWh',
    'gdp_pc_usd'    : 'GSYIH_KisiBasi_USD',
    'urban_pct'     : 'Kentsel_Nufus_Orani',
    'renewable_pct' : 'Yenilenebilir_Enerji_Orani',
    'forest_pct'    : 'Orman_Orani',
    'agri_land_pct' : 'Tarim_Arazisi_Orani',
    'trade_gdp_pct' : 'Ticaret_GSYIH_Orani',
    'pop_total'     : 'Toplam_Nufus',
    'pm25'          : 'PM25_Hava_Kirliligi',
    'co2_pc'        : 'CO2_KisiBasi_Ton',
}, inplace=True)

sayisal = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Veri seti: {df.shape[0]} gozlem, {df['Ulke'].nunique()} ulke, "
      f"{df['Yil'].min()}-{df['Yil'].max()} donemi")

# Eksik deger kontrolu
eksik = df.isnull().sum()
if eksik.sum() > 0:
    for col in sayisal:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  '{col}' sutunundaki eksik degerler medyan ile dolduruldu.")
else:
    print("Eksik deger yok.")

print("\nTanimLayici istatistikler:")
print(df[sayisal].describe().T.round(2))

print("\nCarpiklik degerleri:")
carpiklik = df[sayisal].skew().sort_values(ascending=False)
print(carpiklik.round(3))


# =============================================================================
# 2. EDA GORSELLEŞTIRMELERI
# =============================================================================
ulke_listesi  = sorted(df['Ulke'].unique())
pal_ulke      = sns.color_palette("tab20", n_colors=len(ulke_listesi))
ulke_renk_map = {u: c for u, c in zip(ulke_listesi, pal_ulke)}

# --- A. Korelasyon matrisi ---
cmap_corr = LinearSegmentedColormap.from_list(
    "corr_cmap", ["#C0392B", "#FFFFFF", "#1A5276"], N=256)
corr = df[sayisal].corr()

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(BG_COLOR)
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
            annot_kws={"size": 8, "weight": "bold"},
            cmap=cmap_corr, center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor="#DDEEFF",
            cbar_kws={"shrink": 0.75, "label": "Korelasyon Katsayisi (r)"},
            square=True)
co2_idx = list(corr.columns).index("CO2_KisiBasi_Ton")
ax.add_patch(plt.Rectangle((co2_idx, 0), 1, len(corr),
             fill=False, edgecolor=PALETTE_ACCENT, lw=2.5, clip_on=False))
ax.add_patch(plt.Rectangle((0, co2_idx), len(corr), 1,
             fill=False, edgecolor=PALETTE_ACCENT, lw=2.5, clip_on=False))
ax.set_title("Degiskenler Arasi Korelasyon Matrisi",
             fontsize=14, fontweight="bold", color=PALETTE_MAIN, pad=14)
ax.text(0.5, -0.08, "Not: Turuncu cerceve hedef degisken CO2'yi vurgular.",
        transform=ax.transAxes, ha="center", fontsize=8.5,
        color="#6B7A8D", style="italic")
ax.tick_params(axis='x', rotation=45, labelsize=8.5)
ax.tick_params(axis='y', rotation=0,  labelsize=8.5)
watermark(fig)
plt.tight_layout()
plt.show()

# --- B. Yillik CO2 trendi ---
yillik_ort = df.groupby('Yil')['CO2_KisiBasi_Ton'].mean()
yillik_std = df.groupby('Yil')['CO2_KisiBasi_Ton'].std()

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.axvspan(yillik_ort.index.min(), 2015, color="#EAF4FF", alpha=0.6)
ax.axvspan(2015, yillik_ort.index.max(), color="#FFF4EA", alpha=0.6)
ax.fill_between(yillik_ort.index,
                yillik_ort - yillik_std,
                yillik_ort + yillik_std,
                color=PALETTE_SOFT, alpha=0.15, label="+-1 Std. Sapma")
ax.plot(yillik_ort.index, yillik_ort.values, "o-",
        color=PALETTE_SOFT, linewidth=2.5, markersize=6,
        markerfacecolor="white", markeredgewidth=2, zorder=4)
idx_max = yillik_ort.idxmax()
idx_min = yillik_ort.idxmin()
ax.scatter([idx_max], [yillik_ort[idx_max]],
           color=PALETTE_ACCENT, zorder=5, s=80, label=f"Zirve ({idx_max})")
ax.scatter([idx_min], [yillik_ort[idx_min]],
           color=PALETTE_GREEN, zorder=5, s=80, label=f"Dip ({idx_min})")
ax.annotate(f"  {yillik_ort[idx_max]:.2f} t",
            (idx_max, yillik_ort[idx_max]),
            fontsize=8.5, color=PALETTE_ACCENT, fontweight="bold")
ax.annotate(f"  {yillik_ort[idx_min]:.2f} t",
            (idx_min, yillik_ort[idx_min]),
            fontsize=8.5, color=PALETTE_GREEN, fontweight="bold")
ax.axvline(x=2015, color=PALETTE_ACCENT, linestyle="--",
           linewidth=1.8, label="Train/Test Siniri (2015)")
ax.text(2015.3, ax.get_ylim()[0] * 1.02, "<- Egitim   Test ->",
        fontsize=8.5, color=PALETTE_ACCENT, va="bottom")
ax.set_title("16 Ulke Ortalamasi - Yillik CO2 Trendi (1990-2020)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_ylabel("Kisi Basi CO2 (ton)")
ax.set_xlabel("Yil")
ax.legend(loc="upper right", framealpha=0.9)
watermark(fig)
plt.tight_layout()
plt.show()

# --- C. Ulke bazli CO2 kutu grafigi ---
ulke_sirasi = (df.groupby('Ulke')['CO2_KisiBasi_Ton']
               .median().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG_COLOR)
sns.boxplot(data=df, x="Ulke", y="CO2_KisiBasi_Ton",
            order=ulke_sirasi,
            palette=sns.color_palette("RdYlGn_r", len(ulke_sirasi)),
            width=0.55, linewidth=1.1,
            flierprops=dict(marker="o", markersize=4,
                            markerfacecolor="#AAAAAA", alpha=0.5), ax=ax)
medyanlar = df.groupby("Ulke")["CO2_KisiBasi_Ton"].median()[ulke_sirasi]
for i, (ulke, med) in enumerate(medyanlar.items()):
    ax.text(i, med + 0.15, f"{med:.1f}", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#2C3E50")
global_med = df["CO2_KisiBasi_Ton"].median()
ax.axhline(global_med, color=PALETTE_SOFT, linestyle="--",
           linewidth=1.5, label=f"Kuresel Medyan: {global_med:.2f} t")
ax.set_title("Ulke Bazli CO2 Emisyon Dagilimi (1990-2020)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_ylabel("Kisi Basi CO2 (ton)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=40, labelsize=8.5)
ax.legend(loc="upper right", framealpha=0.9)
watermark(fig)
plt.tight_layout()
plt.show()

# --- D. Ulke bazli CO2 zaman serisi ---
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
for ulke in ulke_listesi:
    alt = df[df['Ulke'] == ulke].sort_values('Yil')
    ax.plot(alt['Yil'], alt['CO2_KisiBasi_Ton'],
            linewidth=1.8, alpha=0.85, color=ulke_renk_map[ulke])
    son = alt.iloc[-1]
    ax.text(son['Yil'] + 0.3, son['CO2_KisiBasi_Ton'], ulke,
            fontsize=6.5, va='center',
            color=ulke_renk_map[ulke], fontweight='bold')
ax.axvline(x=2015, color=PALETTE_ACCENT, linestyle="--",
           linewidth=1.5, label="Train/Test Siniri (2015)")
ax.set_title("Ulkelerin CO2 Emisyon Yolculuklari (1990-2020)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_ylabel("Kisi Basi CO2 (ton)")
ax.set_xlabel("Yil")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
          fontsize=7.5, ncol=1, framealpha=0.9,
          title="Ulkeler", title_fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

# --- E. Cevresel Kuznets Egrisi (EKC) ---
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG_COLOR)
for ulke in ulke_listesi:
    alt = df[df['Ulke'] == ulke]
    ax.scatter(alt['GSYIH_KisiBasi_USD'], alt['CO2_KisiBasi_Ton'],
               color=ulke_renk_map[ulke], alpha=0.55, s=18, zorder=3)
gdp_vals = df['GSYIH_KisiBasi_USD'].values
co2_vals = df['CO2_KisiBasi_Ton'].values
koef = np.polyfit(gdp_vals, co2_vals, 2)
poly = np.poly1d(koef)
x_fit = np.linspace(gdp_vals.min(), gdp_vals.max(), 300)
ax.plot(x_fit, poly(x_fit), color=PALETTE_ACCENT, linewidth=2.5,
        linestyle="--", zorder=5, label="Kuadratik Fit (EKC)")
if koef[0] < 0:
    gdp_zirve = -koef[1] / (2 * koef[0])
    co2_zirve = poly(gdp_zirve)
    if gdp_vals.min() < gdp_zirve < gdp_vals.max():
        ax.axvline(gdp_zirve, color=PALETTE_WARN, linestyle=":",
                   linewidth=1.5, label=f"EKC Zirve ~ ${gdp_zirve:,.0f}")
        ax.scatter([gdp_zirve], [co2_zirve], color=PALETTE_WARN,
                   s=120, zorder=6, edgecolors="white", linewidth=1.5)
        print(f"EKC zirve noktasi: ${gdp_zirve:,.0f} kisi basi GSYiH")
ulke_handles = [mpatches.Patch(color=ulke_renk_map[u], label=u)
                for u in ulke_listesi]
leg1 = ax.legend(handles=ulke_handles, loc='upper left', fontsize=7,
                 ncol=2, framealpha=0.9, title="Ulkeler", title_fontsize=8)
ax.add_artist(leg1)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title("Cevresel Kuznets Egrisi (EKC): GSYiH ve CO2 iliskisi",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("Kisi Basi GSYiH (USD)")
ax.set_ylabel("Kisi Basi CO2 (ton)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.98, 0.05,
        "Ters-U sekli EKC hipotezini destekler:\nGelir arttikca CO2 once yukselir, sonra duzer.",
        transform=ax.transAxes, ha='right', fontsize=8.5,
        color="#6B7A8D", style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=PALETTE_SOFT, alpha=0.85))
watermark(fig)
plt.tight_layout()
plt.show()

# --- F. Yenilenebilir enerji ve orman ortusu vs CO2 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Yenilenebilir Enerji ve Orman Ortusunun CO2 Uzerindeki Etkisi",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
for ax, (xcol, xlab, renk) in zip(axes, [
    ('Yenilenebilir_Enerji_Orani', 'Yenilenebilir Enerji Orani (%)', PALETTE_GREEN),
    ('Orman_Orani',                'Orman Ortusu Orani (%)',          '#5D8A4E'),
]):
    for ulke in ulke_listesi:
        alt = df[df['Ulke'] == ulke]
        ax.scatter(alt[xcol], alt['CO2_KisiBasi_Ton'],
                   color=ulke_renk_map[ulke], alpha=0.55, s=20, zorder=3)
    x_ = df[xcol].values
    y_ = df['CO2_KisiBasi_Ton'].values
    m, b, r, p, _ = stats.linregress(x_, y_)
    x_line = np.linspace(x_.min(), x_.max(), 200)
    p_str = '<0.001' if p < 0.001 else f'={p:.3f}'
    ax.plot(x_line, m * x_line + b, color=renk, linewidth=2.2,
            linestyle="--", label=f"r = {r:.2f}  (p{p_str})")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Kisi Basi CO2 (ton)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

# --- G. CO2 yillik degisim hizi heatmap ---
pivot_yillik  = df.pivot_table(index='Ulke', columns='Yil',
                                values='CO2_KisiBasi_Ton')
pivot_degisim = pivot_yillik.pct_change(axis=1) * 100
pivot_degisim = pivot_degisim.dropna(axis=1, how='all')
cmap_div = LinearSegmentedColormap.from_list(
    "div", ["#1A5276", "#FFFFFF", "#C0392B"], N=256)

fig, ax = plt.subplots(figsize=(18, 7))
fig.patch.set_facecolor(BG_COLOR)
sns.heatmap(pivot_degisim, ax=ax, cmap=cmap_div, center=0,
            annot=True, fmt=".1f", annot_kws={"size": 6.5},
            linewidths=0.3, linecolor="#EEEEEE",
            cbar_kws={"label": "Yillik CO2 Degisimi (%)", "shrink": 0.7})
ax.set_title("Ulke x Yil - CO2 Yillik Degisim Hizi (%)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("Yil")
ax.set_ylabel("Ulke")
ax.tick_params(axis='x', rotation=45, labelsize=7.5)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
ax.text(0.5, -0.1, "Mavi: CO2 azaliyor  |  Kirmizi: CO2 artiyor",
        transform=ax.transAxes, ha='center', fontsize=8.5,
        color="#6B7A8D", style='italic')
watermark(fig)
plt.tight_layout()
plt.show()

# --- H. PM2.5 ve kentlesme vs CO2 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Kentlesme ve Hava Kirliliginin CO2 ile iliskisi",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
co2_norm = ((df['CO2_KisiBasi_Ton'] - df['CO2_KisiBasi_Ton'].min()) /
            (df['CO2_KisiBasi_Ton'].max() - df['CO2_KisiBasi_Ton'].min()))
for ax, (xcol, xlab) in zip(axes, [
    ('Kentsel_Nufus_Orani', 'Kentlesme Orani (%)'),
    ('PM25_Hava_Kirliligi', 'PM2.5 Hava Kirliligi (ug/m3)'),
]):
    sc = ax.scatter(df[xcol], df['CO2_KisiBasi_Ton'],
                    c=co2_norm, cmap='RdYlGn_r', s=22,
                    alpha=0.65, edgecolors='none', zorder=3)
    x_ = df[xcol].values
    y_ = df['CO2_KisiBasi_Ton'].values
    m, b, r, p, _ = stats.linregress(x_, y_)
    x_line = np.linspace(x_.min(), x_.max(), 200)
    ax.plot(x_line, m * x_line + b, color=PALETTE_MAIN,
            linewidth=2, linestyle="--", label=f"r = {r:.2f}")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Kisi Basi CO2 (ton)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
cb = fig.colorbar(sc, ax=axes[1], shrink=0.8,
                  label="CO2 Duzeyi (normallesmis)")
cb.ax.tick_params(labelsize=8)
watermark(fig)
plt.tight_layout()
plt.show()


# =============================================================================
# 3. OYLAMA TABANLI OZELLIK SECIMI (VBFS)
# =============================================================================
HEDEF = 'CO2_KisiBasi_Ton'
# Enerji tuketimi cikarildi: CO2 ile r=0.95 -> multicollinearity riski
CIKAR = [HEDEF, 'Yil', 'Ulke', 'Enerji_Tuketimi_KWh']

X = df.drop(columns=CIKAR).select_dtypes(include=[np.number])
y = df[HEDEF].reset_index(drop=True)

# Kronolojik bolme: panel veride veri sizintisini onler
train_mask = df['Yil'] <= 2015
test_mask  = df['Yil'] >  2015

X_train_raw  = X[train_mask.values].reset_index(drop=True)
X_test_raw   = X[test_mask.values].reset_index(drop=True)
y_train      = y[train_mask.values].reset_index(drop=True)
y_test       = y[test_mask.values].reset_index(drop=True)
groups_train = df.loc[train_mask, 'Ulke'].reset_index(drop=True)

# RobustScaler: medyan/IQR tabanli, aykiri degerlere dayanikli
scaler     = RobustScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw),
                           columns=X_train_raw.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test_raw),
                           columns=X_test_raw.columns)

print(f"Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")

# Uc bagimsiz kriter, cogunluk oylamasi (>= 2 oy -> secildi)
K_ADET      = 6
selector_f  = SelectKBest(score_func=f_regression,       k=K_ADET).fit(X_train_sc, y_train)
f_cols      = X_train_sc.columns[selector_f.get_support()].tolist()
selector_mi = SelectKBest(score_func=mutual_info_regression, k=K_ADET).fit(X_train_sc, y_train)
mi_cols     = X_train_sc.columns[selector_mi.get_support()].tolist()
rf_sel      = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_sc, y_train)
rf_cols     = X_train_sc.columns[np.argsort(rf_sel.feature_importances_)[-K_ADET:]].tolist()

oy_sayilari         = pd.Series(f_cols + mi_cols + rf_cols).value_counts()
secilen_degiskenler = oy_sayilari[oy_sayilari >= 2].index.tolist()

print("\nVBFS oylama sonuclari:")
print(oy_sayilari)
print(f"\nSecilen {len(secilen_degiskenler)} degisken: {secilen_degiskenler}")

X_train_sc = X_train_sc[secilen_degiskenler]
X_test_sc  = X_test_sc[secilen_degiskenler]


# =============================================================================
# 4. MODEL EGITIMI - GroupKFold + GridSearchCV
# =============================================================================
# Regularizasyon stratejisi: max_depth dusuruldu, min_samples/min_data arttirildi,
# gradient boosting modellerine L1+L2 cezalari eklendi.
param_grids = {
    "Dogrusal Regresyon": {
        "model" : LinearRegression(),
        "params": {"fit_intercept": [True, False]}
    },
    "Random Forest": {
        "model" : RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators"     : [100, 200],
            "max_depth"        : [3, 4, 5],
            "min_samples_leaf" : [20, 30, 40],
            "min_samples_split": [30, 40, 50],
        }
    },
    "XGBoost": {
        "model" : XGBRegressor(random_state=42,
                               objective='reg:squarederror', verbosity=0),
        "params": {
            "n_estimators"    : [100, 200],
            "learning_rate"   : [0.01, 0.05],
            "max_depth"       : [2, 3],
            "subsample"       : [0.6, 0.7],
            "colsample_bytree": [0.6, 0.7],
            "min_child_weight": [10, 15, 20],
            "reg_lambda"      : [5, 10, 20],
            "reg_alpha"       : [0.1, 1.0],
        }
    },
    "LightGBM": {
        "model" : LGBMRegressor(random_state=42, verbose=-1),
        "params": {
            "n_estimators"     : [100, 200],
            "learning_rate"    : [0.01, 0.05],
            "num_leaves"       : [8, 12, 15],
            "min_child_samples": [30, 50, 70],
            "subsample"        : [0.6, 0.7],
            "reg_lambda"       : [5, 10, 20],
            "reg_alpha"        : [0.1, 1.0],
        }
    },
    "CatBoost": {
        "model" : CatBoostRegressor(random_state=42, verbose=0),
        "params": {
            "iterations"      : [100, 200],
            "learning_rate"   : [0.01, 0.05],
            "depth"           : [3, 4],
            "l2_leaf_reg"     : [10, 20, 30],
            "min_data_in_leaf": [20, 30, 40],
        }
    },
    "SVR": {
        "model" : SVR(),
        "params": {"kernel": ['rbf', 'linear'], "C": [0.1, 1, 10]}
    },
    "Yapay Sinir Aglari": {
        "model" : MLPRegressor(max_iter=1000, random_state=42,
                               early_stopping=True),
        "params": {
            "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
            "alpha"             : [0.1, 1.0, 10.0],
        }
    },
}

RENKLER = {
    "Dogrusal Regresyon": "#95A5A6",
    "LightGBM"          : "#2980B9",
    "Random Forest"     : "#27AE60",
    "CatBoost"          : "#F1C40F",
    "Yapay Sinir Aglari": "#E74C3C",
    "XGBoost"           : "#F39C12",
    "SVR"               : "#8E44AD",
}

en_iyi_modeller = {}
sonuclar        = []
gkf             = GroupKFold(n_splits=5)

for isim, config in param_grids.items():
    print(f"{isim} egitiliyor...")
    grid = GridSearchCV(
        config["model"], config["params"],
        cv=gkf.split(X_train_sc, y_train, groups=groups_train),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    grid.fit(X_train_sc, y_train)
    en_iyi_modeller[isim] = grid.best_estimator_

    y_pred_train = grid.best_estimator_.predict(X_train_sc)
    y_pred_test  = grid.best_estimator_.predict(X_test_sc)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)
    gap      = r2_train - r2_test
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae      = mean_absolute_error(y_test, y_pred_test)
    mape     = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    durum = ("Asiri Ogrenme" if gap > 0.10
             else "Dikkat"   if gap > 0.05
             else "Iyi")

    sonuclar.append({
        "Model"    : isim,
        "Train R2" : round(r2_train, 4),
        "Test R2"  : round(r2_test,  4),
        "Gap"      : round(gap,      4),
        "RMSE"     : round(rmse,     4),
        "MAE"      : round(mae,      4),
        "MAPE (%)" : round(mape,     2),
        "Durum"    : durum,
    })
    print(f"  Train R2={r2_train:.4f} | Test R2={r2_test:.4f} | "
          f"Gap={gap:.4f} | RMSE={rmse:.4f} | {durum}")

sonuc_df = pd.DataFrame(sonuclar).sort_values("RMSE").reset_index(drop=True)
print("\nModel karsilastirma tablosu:")
print(sonuc_df.to_string(index=False))

# Model secimi: Gap < 0.05 (overfitting-guvenli) ve en dusuk RMSE
guvenli = sonuc_df[sonuc_df['Gap'] < 0.05]
if len(guvenli) > 0:
    en_iyi_model_ismi = guvenli.sort_values("RMSE").iloc[0]["Model"]
else:
    en_iyi_model_ismi = sonuc_df.iloc[0]["Model"]
    print("Hicbir model Gap<0.05 esigini gecemedi, en dusuk RMSE secildi.")

print(f"\nKazanan model: {en_iyi_model_ismi}")
en_iyi_model = en_iyi_modeller[en_iyi_model_ismi]


# =============================================================================
# 5. PERFORMANS GORSELLEŞTIRMELERI
# =============================================================================
model_sirali = sonuc_df["Model"].tolist()

# --- Dort metrik panel ---
metrikler_panel = [
    ("Train R2", "Train R2\n(Egitim Basarisi)", True),
    ("Test R2",  "Test R2\n(Genelleme)",         True),
    ("RMSE",     "RMSE\n(Dusuk = Iyi)",          False),
    ("MAE",      "MAE\n(Dusuk = Iyi)",           False),
]
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Tum Algoritmalarin Performans Karsilastirmasi",
             fontsize=15, fontweight="bold", color=PALETTE_MAIN, y=1.01)
for ax, (metrik, baslik, higher_better) in zip(axes, metrikler_panel):
    vals = sonuc_df[metrik].values
    renk = [RENKLER.get(m, "#34495E") for m in sonuc_df["Model"]]
    bars = ax.bar(model_sirali, vals, color=renk, alpha=0.88,
                  edgecolor="white", linewidth=0.6, width=0.6)
    best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
    bars[best_idx].set_edgecolor(PALETTE_ACCENT)
    bars[best_idx].set_linewidth(2.5)
    ymax = max(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#2C3E50")
    ax.set_title(baslik, fontsize=11, fontweight="bold", color=PALETTE_MAIN)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.set_ylim(min(0, min(vals) * 1.05), max(vals) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    best_label = "En Iyi (yukari)" if higher_better else "En Iyi (asagi)"
    ax.text(best_idx, vals[best_idx] + ymax * 0.13, best_label,
            ha="center", fontsize=7.5, color=PALETTE_ACCENT, fontweight="bold")
watermark(fig)
plt.tight_layout()
plt.show()

# --- Train vs Test R2 overfitting haritasi ---
s = sonuc_df.sort_values("Test R2", ascending=True).reset_index(drop=True)
y_pos = np.arange(len(s))

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)
bar_train = ax.barh(y_pos - 0.20, s["Train R2"], 0.38,
                    color="#AFA9EC", alpha=0.9,
                    label="Train R2", edgecolor="white")
bar_test  = ax.barh(y_pos + 0.20, s["Test R2"],  0.38,
                    color="#534AB7", alpha=0.95,
                    label="Test R2", edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(s["Model"], fontsize=9.5)
ax.set_xlabel("R2 Skoru")
ax.set_title("Train vs Test R2 - Asiri Ogrenme Kontrolu",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
for i, (_, row) in enumerate(s.iterrows()):
    gap = row["Gap"]
    if   gap > 0.10: renk_gap, sembol = "#E24B4A", "x"
    elif gap > 0.05: renk_gap, sembol = PALETTE_WARN, "~"
    else:            renk_gap, sembol = PALETTE_GREEN, "ok"
    sag = max(row["Train R2"], row["Test R2"])
    ax.text(sag + 0.005, i, f"{sembol} Gap: {gap:.3f}",
            va="center", fontsize=9, color=renk_gap, fontweight="bold")
patch_iyi  = mpatches.Patch(color=PALETTE_GREEN, label="ok  Gap < 0.05")
patch_dikk = mpatches.Patch(color=PALETTE_WARN,  label="~   Gap 0.05-0.10")
patch_over = mpatches.Patch(color="#E24B4A",       label="x   Gap > 0.10")
ax.legend(handles=[bar_train, bar_test, patch_iyi, patch_dikk, patch_over],
          loc="lower right", framealpha=0.9, fontsize=8.5)
ax.set_xlim(0, 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(1.0, color="#CCCCCC", linewidth=0.8, linestyle=":")
watermark(fig)
plt.tight_layout()
plt.show()


# =============================================================================
# 6. SHAP YORUMLANABILIRLIK ANALIZI
# =============================================================================
if en_iyi_model_ismi in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
    explainer   = shap.TreeExplainer(en_iyi_model)
    shap_values = explainer.shap_values(X_test_sc)
else:
    background  = shap.kmeans(X_train_sc, 50)
    explainer   = shap.KernelExplainer(en_iyi_model.predict, background)
    shap_values = explainer.shap_values(X_test_sc)

# --- SHAP bar plot (degisken onem siralaması) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
shap.summary_plot(shap_values, X_test_sc,
                  feature_names=X_test_sc.columns,
                  plot_type="bar", show=False, color=PALETTE_SOFT)
ax = plt.gca()
ax.set_title(f"{en_iyi_model_ismi} - Degisken Onem Siralaması (SHAP)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("Ortalama |SHAP Degeri|")
ax.spines[["top", "right"]].set_visible(False)
bars_shap = ax.patches
if bars_shap:
    max_bar = max(bars_shap, key=lambda b: b.get_width())
    max_bar.set_edgecolor(PALETTE_ACCENT)
    max_bar.set_linewidth(2.2)
watermark(fig)
plt.tight_layout()
plt.show()

# --- SHAP beeswarm (etki yonu analizi) ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
shap.summary_plot(shap_values, X_test_sc,
                  feature_names=X_test_sc.columns,
                  show=False, alpha=0.75)
ax = plt.gca()
ax.set_title(f"{en_iyi_model_ismi} - Ozelliklerin CO2 Uzerindeki Etki Yonu",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("SHAP Degeri  (pozitif = CO2 arttirir)")
ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.03,
         "Solda mavi: dusuk ozellik degeri   |   Sagda kirmizi: yuksek ozellik degeri",
         ha="center", fontsize=8.5, color="#6B7A8D", style="italic")
watermark(fig)
plt.tight_layout()
plt.show()

# --- Ulke bazli SHAP stacked bar ---
shap_df       = pd.DataFrame(shap_values, columns=X_test_sc.columns)
shap_df['Ulke'] = df.loc[test_mask, 'Ulke'].values
ulke_shap_ort   = shap_df.groupby('Ulke').apply(
    lambda x: x.abs().mean(numeric_only=True))

kritik_degiskenler = secilen_degiskenler
var_colors = [PALETTE_ACCENT, PALETTE_SOFT, PALETTE_GREEN,
              PALETTE_WARN, PALETTE_PURPLE, "#E67E22"][:len(kritik_degiskenler)]

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG_COLOR)
plot_data = ulke_shap_ort[kritik_degiskenler]
bottom    = np.zeros(len(plot_data))
for var, color in zip(kritik_degiskenler, var_colors):
    vals = plot_data[var].values
    bars = ax.bar(plot_data.index, vals, bottom=bottom, label=var,
                  color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, val, bot in zip(bars, vals, bottom):
        if val > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                    f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    bottom += vals
totals = plot_data.sum(axis=1)
for i, (ulke, total) in enumerate(totals.items()):
    ax.text(i, total + 0.01, f"{total:.2f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#2C3E50")
ax.set_title("Ulkelere Gore Temel Gostergelerin CO2 Uzerindeki Etki Buyuklugu (SHAP)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_ylabel("Ortalama Mutlak SHAP Degeri")
ax.set_xlabel("Ulkeler")
ax.tick_params(axis="x", rotation=40, labelsize=8.5)
ax.legend(title="Degiskenler", framealpha=0.9, fontsize=9,
          title_fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

print("\nUlke bazli ortalama mutlak SHAP degerleri:")
print(ulke_shap_ort[kritik_degiskenler].round(3).to_string())


# =============================================================================
# 7. ISTATISTIKSEL TESTLER - Kruskal-Wallis
# =============================================================================
print("\nKruskal-Wallis testi (non-parametrik, normallik varsayimi gerekmez):")
for degisken in kritik_degiskenler:
    grup_verileri = [shap_df[shap_df['Ulke'] == u][degisken].values
                     for u in shap_df['Ulke'].unique()]
    stat_kw, p_val_kw = kruskal(*grup_verileri)
    sonuc = "H0 reddedildi (anlamli fark var)" if p_val_kw < 0.05 else "H0 reddedilemedi"
    print(f"  {degisken:<40} H={stat_kw:.4f}  p={p_val_kw:.6f}  -> {sonuc}")


# =============================================================================
# 8. TAHMIN SONRASI GOERSELLER
# =============================================================================
y_pred_test_all = en_iyi_model.predict(X_test_sc)
residuals       = y_test.values - y_pred_test_all

# --- Artik analizi ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle(f"{en_iyi_model_ismi} - Artik (Residual) Analizi",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
axes[0].scatter(y_pred_test_all, residuals, color=PALETTE_SOFT,
                alpha=0.65, s=30, edgecolors='white', linewidth=0.3)
axes[0].axhline(0, color=PALETTE_ACCENT, linestyle='--', linewidth=1.5)
std_r = residuals.std()
axes[0].axhspan(-std_r, std_r, color=PALETTE_SOFT, alpha=0.08)
axes[0].set_xlabel("Tahmin Edilen CO2 (ton/kisi)")
axes[0].set_ylabel("Artik (Gercek - Tahmin)")
axes[0].set_title("Artik vs Tahmin\n(Homojen dagilim = iyi)")
axes[0].spines[["top", "right"]].set_visible(False)

mu, sigma = residuals.mean(), residuals.std()
axes[1].hist(residuals, bins=20, color=PALETTE_SOFT,
             alpha=0.8, edgecolor='white', linewidth=0.5)
x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
axes[1].plot(x_norm,
             stats.norm.pdf(x_norm, mu, sigma) * len(residuals)
             * (residuals.max() - residuals.min()) / 20,
             color=PALETTE_ACCENT, linewidth=2, label="Normal Egri")
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel("Artik Deger")
axes[1].set_ylabel("Frekans")
axes[1].set_title(f"Artik Dagilimi\nmu={mu:.3f}, sigma={sigma:.3f}")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[2].scatter(osm, osr, color=PALETTE_SOFT, alpha=0.7, s=30)
x_qq = np.linspace(min(osm), max(osm), 200)
axes[2].plot(x_qq, slope * x_qq + intercept,
             color=PALETTE_ACCENT, linewidth=2, label=f"r={r:.3f}")
axes[2].set_xlabel("Teorik Kuantiller")
axes[2].set_ylabel("Gozlemlenen Kuantiller")
axes[2].set_title("Normal Q-Q Plot\n(Cizgiye yakin = normal dagilim)")
axes[2].legend()
axes[2].spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

# --- Tum modellerin tahminleri ---
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG_COLOR)
sira  = np.argsort(y_test.values)
x_idx = np.arange(len(sira))
ax.plot(x_idx, y_test.values[sira], color='black', linewidth=2.5,
        label='Gercek CO2', zorder=5, linestyle='solid')
for isim, model in en_iyi_modeller.items():
    y_pred = model.predict(X_test_sc)
    ax.plot(x_idx, y_pred[sira],
            color=RENKLER.get(isim, "#34495E"), linewidth=1.2,
            alpha=0.75, linestyle='--', label=isim)
ax.set_title("Tum Modellerin Test Seti Tahminleri - Gercek CO2 ile Karsilastirma",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("Gozlem Sirasi (CO2 degerine gore sirali)")
ax.set_ylabel("Kisi Basi CO2 (ton/kisi)")
ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.98, 0.05, "Siyah cizgiye en yakin model en basarili modeldir.",
        transform=ax.transAxes, ha='right', fontsize=8.5,
        color="#6B7A8D", style='italic')
watermark(fig)
plt.tight_layout()
plt.show()

# --- Gercek vs tahmin scatter ---
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(BG_COLOR)
ulkeler_test = df.loc[test_mask, 'Ulke'].values
for ulke in sorted(set(ulkeler_test)):
    mask_u = ulkeler_test == ulke
    ax.scatter(y_test.values[mask_u], y_pred_test_all[mask_u],
               color=ulke_renk_map[ulke], s=55, alpha=0.85,
               edgecolors="white", linewidth=0.4, label=ulke, zorder=3)
lo = min(y_test.min(), y_pred_test_all.min()) * 0.95
hi = max(y_test.max(), y_pred_test_all.max()) * 1.05
ax.plot([lo, hi], [lo, hi], "--", color="#AAAAAA", linewidth=1.5,
        label="Mukemmel Tahmin (y=x)", zorder=2)
ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                color=PALETTE_ACCENT, alpha=0.06, label="+-10% Hata Bandi")
r2_disp   = r2_score(y_test, y_pred_test_all)
rmse_disp = np.sqrt(mean_squared_error(y_test, y_pred_test_all))
ax.text(0.04, 0.95, f"R2 = {r2_disp:.4f}\nRMSE = {rmse_disp:.4f}",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        color=PALETTE_MAIN, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=PALETTE_SOFT, alpha=0.9))
ax.set_title(f"{en_iyi_model_ismi} - Gercek vs Tahmin CO2 (Test: 2016-2020)",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_xlabel("Gercek CO2 (ton/kisi)")
ax.set_ylabel("Tahmin Edilen CO2 (ton/kisi)")
ax.legend(loc="lower right", fontsize=7.5, ncol=2,
          framealpha=0.9, markerscale=1.2)
ax.spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

# --- Ulke bazli tahmin hatasi ---
tahmin_df = pd.DataFrame({
    'Ulke'                           : df.loc[test_mask, 'Ulke'].values,
    'Yil'                            : df.loc[test_mask, 'Yil'].values,
    'Gercek CO2'                     : y_test.values.round(3),
    f'{en_iyi_model_ismi} Tahmini'   : y_pred_test_all.round(3),
})
tahmin_df['Hata'] = (tahmin_df[f'{en_iyi_model_ismi} Tahmini']
                     - tahmin_df['Gercek CO2']).round(3)

hata_ulke = (tahmin_df.groupby("Ulke")["Hata"]
             .apply(lambda x: x.abs().mean())
             .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG_COLOR)
bar_colors_h = [PALETTE_ACCENT if v > hata_ulke.median() else PALETTE_SOFT
                for v in hata_ulke.values]
bars_h = ax.bar(hata_ulke.index, hata_ulke.values,
                color=bar_colors_h, alpha=0.88,
                edgecolor="white", width=0.6)
ax.axhline(hata_ulke.median(), color=PALETTE_GREEN, linestyle="--",
           linewidth=1.5)
for bar, val in zip(bars_h, hata_ulke.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003, f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#2C3E50")
ax.set_title(f"Ulke Bazli Ortalama Mutlak Tahmin Hatasi  |  {en_iyi_model_ismi}",
             fontsize=13, fontweight="bold", color=PALETTE_MAIN)
ax.set_ylabel("Ortalama Mutlak Hata (ton/kisi)")
ax.tick_params(axis="x", rotation=40, labelsize=9)
legend_patch = [
    mpatches.Patch(color=PALETTE_ACCENT, label="Medyan Ustu"),
    mpatches.Patch(color=PALETTE_SOFT,   label="Medyan Alti"),
    mpatches.Patch(color=PALETTE_GREEN,  label=f"Medyan MAE: {hata_ulke.median():.3f}"),
]
ax.legend(handles=legend_patch, framealpha=0.9, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
watermark(fig)
plt.tight_layout()
plt.show()

print(f"\n{en_iyi_model_ismi} tahminleri (2016-2020):")
for ulke in sorted(tahmin_df['Ulke'].unique()):
    print(f"\n[{ulke}]")
    print(tahmin_df[tahmin_df['Ulke'] == ulke].to_string(index=False))


# =============================================================================
# 9. LLM TABANLI POLITIKA URETIMI - Google Gemini
# =============================================================================
# Birincil model: gemini-2.5-flash
# Yedek: gemini-2.5-flash-lite -> gemini-flash-latest
# HTTP 429: 45sn bekleme, maks 3 deneme
# HTTP 404: sonraki modele gec
# Ardicil ulkeler arasi 15sn bekleme (ucretsiz kota)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "BURAYA_API_ANAHTARINIZI_GIRIN")
genai.configure(api_key=GEMINI_API_KEY)

BIRINCIL_MODEL = "gemini-2.5-flash"
YEDEK_MODELLER = ["gemini-2.5-flash-lite", "gemini-flash-latest"]

def gemini_politika_uret(ulke, surucu1, etki1, surucu2, etki2, max_deneme=3):
    prompt = f"""Sen bir uzman cevresel surdurulebilirlik politika yapicisisin.

{ulke} icin SHAP analizi bulgulari:
- En guclu emisyon surucusu: {surucu1} (ort. |SHAP| = {etki1:.3f})
- Ikinci guclu surucu: {surucu2} (ort. |SHAP| = {etki2:.3f})

Bu iki temel surucuyu hedef alan, ekonomik buyumeyi durdurmadan
kisi basi CO2 emisyonunu azaltacak 3 maddelik politika onerisi sun.

Her madde icin eylem plani ve beklenen sonucu belirt.
Format: numarali liste, Turkce, kisa ve uygulanabilir."""

    modeller = [BIRINCIL_MODEL] + YEDEK_MODELLER
    for model_adi in modeller:
        for deneme in range(1, max_deneme + 1):
            try:
                model = genai.GenerativeModel(model_adi)
                yanit = model.generate_content(prompt)
                return yanit.text
            except Exception as e:
                hata_str = str(e)
                if "404" in hata_str:
                    print(f"  [{model_adi}] bulunamadi, sonraki deneniyor.")
                    break
                elif "429" in hata_str:
                    if deneme < max_deneme:
                        print(f"  [{model_adi}] kota asimi, 45sn bekleniyor "
                              f"(deneme {deneme}/{max_deneme})...")
                        time.sleep(45)
                    else:
                        break
                else:
                    print(f"  [{model_adi}] hata: {hata_str[:80]}")
                    break
    return "Politika uretilemedi."


politikalar = {}
for ulke in sorted(ulke_listesi):
    if ulke not in ulke_shap_ort.index:
        continue

    ulke_shap_sirali = (ulke_shap_ort.loc[ulke, kritik_degiskenler]
                        .sort_values(ascending=False))
    surucu1, etki1 = ulke_shap_sirali.index[0], ulke_shap_sirali.iloc[0]
    surucu2, etki2 = ulke_shap_sirali.index[1], ulke_shap_sirali.iloc[1]

    print(f"\n{ulke}: {surucu1} ({etki1:.3f}), {surucu2} ({etki2:.3f})")
    oneri = gemini_politika_uret(ulke, surucu1, etki1, surucu2, etki2)

    politikalar[ulke] = {
        "surucu1": surucu1, "etki1": etki1,
        "surucu2": surucu2, "etki2": etki2,
        "politika": oneri,
    }
    print(oneri)
    time.sleep(15)

print("\n--- Politika uretimi tamamlandi ---")
