import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import streamlit as st

# 1. إعدادات الصفحة والتصميم
st.set_page_config(page_title="Mishrif NMR & Core Tool", layout="wide")

st.title("Mishrif Reservoir NMR & Core Integration Tool")
st.markdown("<b>برنامج تفاعلي لتقييم النفاذية ومعايرة سجلات NMR مع اللباب الصخري - مكمن المشرف</b>", unsafe_allow_html=True)

# 2. شريط التحكم الجانبي (Interactive Sidebar Controls)
st.sidebar.header("1. المعاملات البتروفيزيائية")
t2_cut = st.sidebar.slider("زمن القطع T2 Cutoff (ms)", min_value=10, max_value=200, value=92, step=1)
phi_cutoff = st.sidebar.slider("حد المسامية الأدنى (Porosity Cutoff %)", min_value=0.0, max_value=25.0, value=10.0, step=0.5) / 100.0
k_cutoff = st.sidebar.slider("حد النفاذية الأدنى (Permeability Cutoff mD)", min_value=0.1, max_value=50.0, value=1.0, step=0.5)

st.sidebar.header("2. معاملات نموذج كوتس (Coates)")
c_coates = st.sidebar.number_input("ثابت كوتس (C)", value=10.0)
a_coates = st.sidebar.number_input("أس المسامية (a)", value=4.0)
b_coates = st.sidebar.number_input("أس السوائل (b)", value=2.0)

# 3. إنشاء بيانات افتراضية لمكمن المشرف (Synthetic Data)
@st.cache_data
def generate_mishrif_data():
    np.random.seed(42)
    depth = np.linspace(2800, 2860, 120)  # العمق من 2800 إلى 2860 متر
    
    # المسامية phi
    phi = 0.08 + 0.16 * np.abs(np.sin(depth / 5) + np.random.normal(0, 0.03, len(depth)))
    phi = np.clip(phi, 0.05, 0.28)
    
    # توزيع السوائل FFI و BVI
    ffi = phi * (0.4 + 0.4 * np.abs(np.cos(depth / 4)))
    bvi = phi - ffi
    
    # المتوسط اللوغاريتمي T2LM
    t2lm = 10 + 150 * (ffi / (phi + 1e-5))
    
    # نفاذية اللباب الصخري الحقيقية مع ضوضاء جيولوجية (Core K)
    k_core = (phi / 0.10)**3.8 * (ffi / (bvi + 1e-5))**1.8 * np.random.lognormal(0, 0.25, len(depth))
    
    return pd.DataFrame({
        'Depth': depth,
        'Phi': phi,
        'FFI': ffi,
        'BVI': bvi,
        'T2LM': t2lm,
        'K_Core': k_core
    })

df = generate_mishrif_data()

# 4. دالات حساب النفاذية (Coates & SDR)
def calc_coates(phi, ffi, bvi, C, a, b):
    bvi_safe = np.where(bvi <= 0, 1e-5, bvi)
    return ((phi / C) ** a) * ((ffi / bvi_safe) ** b)

def calc_sdr(phi, t2lm, C=4.0, m=4.0, n=2.0):
    return C * (phi ** m) * (t2lm ** n)

# الحساب الأولي
df['K_Coates'] = calc_coates(df['Phi'], df['FFI'], df['BVI'], c_coates, a_coates, b_coates)
df['K_SDR'] = calc_sdr(df['Phi'], df['T2LM'])

# 5. المعايرة التلقائية باستخدام SciPy (Auto-Calibration)
st.sidebar.header("3. المعايرة التلقائية (Auto-Calibration)")
if st.sidebar.button("تشغيل المعايرة الرياضية (SciPy Optimizer)"):
    def loss_coates(params):
        C, a, b = params
        k_pred = calc_coates(df['Phi'], df['FFI'], df['BVI'], C, a, b)
        return np.sqrt(np.mean((np.log10(k_pred + 1e-3) - np.log10(df['K_Core'] + 1e-3))**2))
    
    res = optimize.minimize(loss_coates, [c_coates, a_coates, b_coates], bounds=[(1, 50), (1, 6), (0.5, 4)])
    c_opt, a_opt, b_opt = res.x
    
    df['K_Coates_Calibrated'] = calc_coates(df['Phi'], df['FFI'], df['BVI'], c_opt, a_opt, b_opt)
    st.sidebar.success(f"تمت المعايرة بنجاح!\n C={c_opt:.2f}, a={a_opt:.2f}, b={b_opt:.2f}")
else:
    df['K_Coates_Calibrated'] = df['K_Coates']

# 6. تحديد نطاقات الإنتاج الصافي (Net Pay Zones)
df['Net_Pay'] = (df['Phi'] >= phi_cutoff) & (df['K_Coates_Calibrated'] >= k_cutoff)

# 7. رسم المنحنيات المكمنية (Log Plotting)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 7), sharey=True)
fig.suptitle("Mishrif Reservoir Petrophysical & NMR Log Interpretation", fontsize=14, fontweight='bold')

# Track 1: Porosity & Net Pay Highlight
axes[0].plot(df['Phi'] * 100, df['Depth'], color='blue', label='Porosity (%)')
axes[0].axvline(x=phi_cutoff * 100, color='red', linestyle='--', label=f'Phi Cutoff ({phi_cutoff*100:.0f}%)')
axes[0].fill_betweenx(df['Depth'], 0, df['Phi'] * 100, where=df['Net_Pay'], color='green', alpha=0.4, label='Net Pay Zone')
axes[0].set_xlabel("Porosity (%)")
axes[0].set_ylabel("Depth (m)")
axes[0].set_xlim(0, 30)
axes[0].invert_yaxis()
axes[0].grid(True, linestyle=':')
axes[0].legend(loc='upper right', fontsize='small')

# Track 2: NMR Volumes (BVI vs FFI)
axes[1].plot(df['BVI'] * 100, df['Depth'], color='black', label='BVI (Bound Fluid)')
axes[1].plot(df['Phi'] * 100, df['Depth'], color='green', label='Total Porosity')
axes[1].fill_betweenx(df['Depth'], 0, df['BVI'] * 100, color='gray', alpha=0.5, label='BVI Zone')
axes[1].fill_betweenx(df['Depth'], df['BVI'] * 100, df['Phi'] * 100, color='cyan', alpha=0.5, label='FFI (Free Fluid)')
axes[1].set_xlabel("Volume Fraction (%)")
axes[1].set_xlim(0, 30)
axes[1].grid(True, linestyle=':')
axes[1].legend(loc='upper right', fontsize='small')

# Track 3: Permeability Comparison (Log Scale)
axes[2].scatter(df['K_Core'], df['Depth'], color='black', marker='o', s=20, label='Core K (mD)', zorder=5)
axes[2].plot(df['K_Coates_Calibrated'], df['Depth'], color='red', linewidth=2, label='Coates K (NMR)')
axes[2].plot(df['K_SDR'], df['Depth'], color='orange', linestyle='--', label='SDR K (NMR)')
axes[2].axvline(x=k_cutoff, color='darkgreen', linestyle=':', label=f'K Cutoff ({k_cutoff} mD)')
axes[2].set_xscale('log')
axes[2].set_xlabel("Permeability (mD)")
axes[2].set_xlim(0.01, 1000)
axes[2].grid(True, which='both', linestyle=':')
axes[2].legend(loc='upper right', fontsize='small')

plt.tight_layout()
st.pyplot(fig)

# 8. ملخص النتائج الرقمية
col1, col2, col3 = st.columns(3)
col1.metric("إجمالي المقطع المكمني", f"{df['Depth'].max() - df['Depth'].min():.1f} m")
col2.metric("سمك نطاق الإنتاج الصافي (Net Pay)", f"{df['Net_Pay'].sum() * (df['Depth'].iloc[1] - df['Depth'].iloc[0]):.1f} m")
col3.metric("نسبة الإنتاج الصافي (Net to Gross)", f"{(df['Net_Pay'].mean() * 100):.1f} %")
