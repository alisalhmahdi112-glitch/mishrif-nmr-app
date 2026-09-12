import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize
import streamlit as st

# 1. تهيئة الصفحة والتصميم العريض
st.set_page_config(page_title="Mishrif Petrophysics AI", layout="wide", page_icon="🛢️")

# تطبيق تنسيقات CSS احترافية (Dark Dashboard Style)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2e364f; }
    div[data-testid="metric-container"] label { color: #8892b0 !important; font-size: 16px; }
    div[data-testid="metric-container"] div { color: #64ffda !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. الهيدر الرئيسي
st.title("🛢️ Mishrif Reservoir NMR & Core Integration Platform")
st.caption("المنصة التفاعلية المتقدمة لمعايرة سجلات الرنين المغناطيسي وتحليل الإنتاج الصافي - مكمن المشرف")
st.markdown("---")

# 3. القائمة الجانبية المتقدمة
st.sidebar.image("https://img.icons8.com/fluency/96/oil-rig.png", width=70)
st.sidebar.title("🎛️ لوحة التحكم البتروفيزيائية")

st.sidebar.subheader("1. حدود القطع (Cutoffs)")
t2_cut = st.sidebar.slider("T2 Cutoff (ms)", 10, 200, 92)
phi_cutoff = st.sidebar.slider("Porosity Cutoff (%)", 0.0, 25.0, 10.0, 0.5) / 100.0
k_cutoff = st.sidebar.slider("Permeability Cutoff (mD)", 0.1, 50.0, 1.0, 0.5)

st.sidebar.subheader("2. معاملات نموذج كوتس (Coates)")
c_coates = st.sidebar.number_input("ثابت C", value=10.0, step=0.5)
a_coates = st.sidebar.number_input("أس المسامية (a)", value=4.0, step=0.1)
b_coates = st.sidebar.number_input("أس السوائل (b)", value=2.0, step=0.1)

# 4. توليد بيانات مكمن المشرف
@st.cache_data
def generate_mishrif_data():
    np.random.seed(42)
    depth = np.linspace(2800, 2860, 120)
    phi = 0.08 + 0.16 * np.abs(np.sin(depth / 5) + np.random.normal(0, 0.03, len(depth)))
    phi = np.clip(phi, 0.05, 0.28)
    ffi = phi * (0.4 + 0.4 * np.abs(np.cos(depth / 4)))
    bvi = phi - ffi
    t2lm = 10 + 150 * (ffi / (phi + 1e-5))
    k_core = (phi / 0.10)**3.8 * (ffi / (bvi + 1e-5))**1.8 * np.random.lognormal(0, 0.25, len(depth))
    return pd.DataFrame({'Depth': depth, 'Phi': phi, 'FFI': ffi, 'BVI': bvi, 'T2LM': t2lm, 'K_Core': k_core})

df = generate_mishrif_data()

def calc_coates(phi, ffi, bvi, C, a, b):
    bvi_safe = np.where(bvi <= 0, 1e-5, bvi)
    return ((phi / C) ** a) * ((ffi / bvi_safe) ** b)

def calc_sdr(phi, t2lm, C=4.0, m=4.0, n=2.0):
    return C * (phi ** m) * (t2lm ** n)

df['K_Coates'] = calc_coates(df['Phi'], df['FFI'], df['BVI'], c_coates, a_coates, b_coates)
df['K_SDR'] = calc_sdr(df['Phi'], df['T2LM'])

# 5. المعايرة التلقائية
st.sidebar.subheader("3. المعايرة والتحسين (SciPy)")
if st.sidebar.button("⚡ تشغيل المعايرة الرياضية"):
    def loss_coates(params):
        C, a, b = params
        k_pred = calc_coates(df['Phi'], df['FFI'], df['BVI'], C, a, b)
        return np.sqrt(np.mean((np.log10(k_pred + 1e-3) - np.log10(df['K_Core'] + 1e-3))**2))
    
    res = optimize.minimize(loss_coates, [c_coates, a_coates, b_coates], bounds=[(1, 50), (1, 6), (0.5, 4)])
    c_opt, a_opt, b_opt = res.x
    df['K_Coates_Calibrated'] = calc_coates(df['Phi'], df['FFI'], df['BVI'], c_opt, a_opt, b_opt)
    st.sidebar.success(f"تمت المعايرة!\nC={c_opt:.2f}, a={a_opt:.2f}, b={b_opt:.2f}")
else:
    df['K_Coates_Calibrated'] = df['K_Coates']

df['Net_Pay'] = (df['Phi'] >= phi_cutoff) & (df['K_Coates_Calibrated'] >= k_cutoff)

# 6. عرض المؤشرات الحيوية top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("إجمالي المقطع المكمني", f"{df['Depth'].max() - df['Depth'].min():.1f} m")
m2.metric("سمك Net Pay الصافي", f"{df['Net_Pay'].sum() * (df['Depth'].iloc[1] - df['Depth'].iloc[0]):.1f} m")
m3.metric("نسبة Net to Gross", f"{(df['Net_Pay'].mean() * 100):.1f} %")
m4.metric("معدل المسامية في الصافي", f"{(df[df['Net_Pay']]['Phi'].mean() * 100):.1f} %")

st.markdown("<br>", unsafe_allow_html=True)

# 7. رسم المنحنيات باستخدام Plotly التفاعلي (Interactive Plotly Logs)
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                    subplot_titles=("Porosity & Net Pay", "NMR Fluids (BVI vs FFI)", "Permeability Calibration"))

# Track 1
fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='Porosity (%)', line=dict(color='#00d2ff', width=2)), row=1, col=1)
fig.add_vline(x=phi_cutoff*100, line_dash="dash", line_color="red", row=1, col=1)

# Highlight Net Pay
pay_df = df[df['Net_Pay']]
fig.add_trace(go.Scatter(x=pay_df['Phi']*100, y=pay_df['Depth'], mode='markers', name='Net Pay Zone',
                         marker=dict(color='#00ff87', size=6, symbol='square')), row=1, col=1)

# Track 2
fig.add_trace(go.Scatter(x=df['BVI']*100, y=df['Depth'], name='BVI', line=dict(color='#795548'), fill='tozerox'), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='Total Phi', line=dict(color='#00e676'), fill='tonextx'), row=1, col=2)

# Track 3
fig.add_trace(go.Scatter(x=df['K_Core'], y=df['Depth'], mode='markers', name='Core K', marker=dict(color='#ffffff', size=6)), row=1, col=3)
fig.add_trace(go.Scatter(x=df['K_Coates_Calibrated'], y=df['Depth'], name='Coates K (Calibrated)', line=dict(color='#ff0055', width=2)), row=1, col=3)
fig.add_trace(go.Scatter(x=df['K_SDR'], y=df['Depth'], name='SDR K', line=dict(color='#ff9900', dash='dash')), row=1, col=3)
fig.add_vline(x=k_cutoff, line_dash="dot", line_color="#00ff87", row=1, col=3)

fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
fig.update_xaxes(type="log", range=[-2, 3], row=1, col=3)
fig.update_layout(height=700, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))

st.plotly_chart(fig, use_container_width=True)

# 8. خيار تنزيل البيانات
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 تحميل جدول النتائج (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='mishrif_petrophysics_results.csv',
    mime='text/csv'
)
