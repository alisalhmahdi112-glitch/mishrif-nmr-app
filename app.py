
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize
import streamlit as st

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Mishrif Petrophysical Suite & AI Calibration",
    layout="wide",
    page_icon="⚡"
)

# 2. تصميم CSS عالي الجودة (Executive Dashboard Theme)
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .kpi-title { color: #64748b; font-size: 13px; font-weight: 700; text-transform: uppercase; }
    .kpi-value { color: #0f172a; font-size: 26px; font-weight: 800; margin-top: 5px; }
    .kpi-unit { font-size: 14px; color: #64748b; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# 3. الهيدر الرئيسي
st.markdown("""
    <div class="main-header">
        <h2 style="margin:0;">🛢️ Mishrif Reservoir NMR & Core Integration Platform</h2>
        <p style="margin:5px 0 0 0; opacity: 0.9;">المنصة التفاعلية المتقدمة لمعايرة سجلات الرنين المغناطيسي وتحديد وحدات التدفق الهيدروليكية (HFU)</p>
    </div>
""", unsafe_allow_html=True)

# 4. الشريط الجانبي
st.sidebar.title("🎛️ لوحة التحكم البتروفيزيائية")

st.sidebar.subheader("1. حدود القطع (Cutoffs)")
phi_cutoff = st.sidebar.slider("Porosity Cutoff (%)", 0.0, 25.0, 8.0, 0.5) / 100.0
k_cutoff = st.sidebar.slider("Permeability Cutoff (mD)", 0.01, 20.0, 0.5, 0.1)

st.sidebar.subheader("2. معاملات نموذج كوتس (Coates)")
c_coates = st.sidebar.number_input("ثابت C", value=10.0, step=0.5)
a_coates = st.sidebar.number_input("أس المسامية (a)", value=4.0, step=0.1)
b_coates = st.sidebar.number_input("أس السوائل (b)", value=2.0, step=0.1)

# 5. البيانات والعمليات الحسابية
@st.cache_data
def generate_mishrif_data():
    np.random.seed(42)
    depth = np.linspace(2800, 2860, 120)
    phi = 0.08 + 0.16 * np.abs(np.sin(depth / 5) + np.random.normal(0, 0.03, len(depth)))
    phi = np.clip(phi, 0.05, 0.28)
    ffi = phi * (0.4 + 0.4 * np.abs(np.cos(depth / 4)))
    bvi = phi - ffi
    t2lm = 10 + 150 * (ffi / (phi + 1e-5))
    k_core = (phi / 0.10)**3.8 * (ffi / (bvi + 1e-5))**1.8 * np.random.lognormal(0, 0.2, len(depth))
    return pd.DataFrame({'Depth': depth, 'Phi': phi, 'FFI': ffi, 'BVI': bvi, 'T2LM': t2lm, 'K_Core': k_core})

df = generate_mishrif_data()

def calc_coates(phi, ffi, bvi, C, a, b):
    bvi_safe = np.where(bvi <= 0, 1e-5, bvi)
    return ((phi / C) ** a) * ((ffi / bvi_safe) ** b)

def calc_sdr(phi, t2lm, C=4.0, m=4.0, n=2.0):
    return C * (phi ** m) * (t2lm ** n)

# 6. المعايرة
st.sidebar.subheader("3. المعايرة والتحسين (SciPy Optimization)")
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
    df['K_Coates_Calibrated'] = calc_coates(df['Phi'], df['FFI'], df['BVI'], c_coates, a_coates, b_coates)

df['K_SDR'] = calc_sdr(df['Phi'], df['T2LM'])
df['Net_Pay'] = (df['Phi'] >= phi_cutoff) & (df['K_Coates_Calibrated'] >= k_cutoff)

# 7. حساب معامل التحديد R-Squared
log_k_core = np.log10(df['K_Core'])
log_k_pred = np.log10(df['K_Coates_Calibrated'])
r2_score = 1 - (np.sum((log_k_core - log_k_pred)**2) / np.sum((log_k_core - np.mean(log_k_core))**2))

# 8. بطاقات المؤشرات (KPI Dashboard)
c1, c2, c3, c4 = st.columns(4)

total_thick = df['Depth'].max() - df['Depth'].min()
net_pay_thick = df['Net_Pay'].sum() * (df['Depth'].iloc[1] - df['Depth'].iloc[0])
ntg = (net_pay_thick / total_thick) * 100
avg_phi_pay = df[df['Net_Pay']]['Phi'].mean() * 100 if df['Net_Pay'].sum() > 0 else 0.0

c1.markdown(f'<div class="kpi-card" style="border-color:#3b82f6;"><div class="kpi-title">المقطع المكمني الكلي</div><div class="kpi-value">{total_thick:.1f} <span class="kpi-unit">m</span></div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card" style="border-color:#10b981;"><div class="kpi-title">سمك Net Pay الصافي</div><div class="kpi-value" style="color:#10b981;">{net_pay_thick:.1f} <span class="kpi-unit">m</span></div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card" style="border-color:#0284c7;"><div class="kpi-title">نسبة Net to Gross</div><div class="kpi-value" style="color:#0284c7;">{ntg:.1f} <span class="kpi-unit">%</span></div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card" style="border-color:#8b5cf6;"><div class="kpi-title">معامل المطابقة (R²)</div><div class="kpi-value" style="color:#8b5cf6;">{r2_score:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 9. التبويبات التفاعلية (Tabs)
tab1, tab2, tab3 = st.tabs(["📊 السجلات التفسيرية (Petrophysical Logs)", "🎯 معايرة النفاذية (Crossplot & R²)", "📋 جدول البيانات والنتائج"])

with tab1:
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Porosity & Net Pay (%)", "NMR Volumes (BVI vs FFI)", "Permeability (mD)"))

    # Track 1: Porosity
    fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='Porosity (%)', line=dict(color='#0284c7', width=2)), row=1, col=1)
    fig.add_vline(x=phi_cutoff*100, line_dash="dash", line_color="#ef4444", annotation_text="Phi Cutoff", row=1, col=1)

    pay_df = df[df['Net_Pay']]
    if len(pay_df) > 0:
        fig.add_trace(go.Scatter(x=pay_df['Phi']*100, y=pay_df['Depth'], mode='markers', name='Net Pay Zone',
                                 marker=dict(color='#10b981', size=7, symbol='square')), row=1, col=1)

    # Track 2: NMR Fluids
    fig.add_trace(go.Scatter(x=df['BVI']*100, y=df['Depth'], name='BVI (Bound)', line=dict(color='#78350f'), fill='tozerox', fillcolor='rgba(120, 53, 15, 0.3)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='FFI (Free)', line=dict(color='#0284c7'), fill='tonextx', fillcolor='rgba(2, 132, 199, 0.3)'), row=1, col=2)

    # Track 3: Permeability
    fig.add_trace(go.Scatter(x=df['K_Core'], y=df['Depth'], mode='markers', name='Core K', marker=dict(color='#0f172a', size=6, symbol='circle')), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['K_Coates_Calibrated'], y=df['Depth'], name='Coates K (NMR)', line=dict(color='#dc2626', width=2)), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['K_SDR'], y=df['Depth'], name='SDR K', line=dict(color='#f59e0b', dash='dash')), row=1, col=3)
    fig.add_vline(x=k_cutoff, line_dash="dot", line_color="#10b981", annotation_text="K Cutoff", row=1, col=3)

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(gridcolor='#f1f5f9', row=1, col=2)
    fig.update_yaxes(gridcolor='#f1f5f9', row=1, col=3)

    fig.update_xaxes(range=[0, 30], gridcolor='#f1f5f9', row=1, col=1)
    fig.update_xaxes(range=[0, 30], gridcolor='#f1f5f9', row=1, col=2)
    fig.update_xaxes(type="log", range=[-2, 3], gridcolor='#f1f5f9', row=1, col=3)

    fig.update_layout(height=700, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20),
                       legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🎯 Crossplot: Core Permeability vs NMR Calculated Permeability")
    
    fig_cross = go.Figure()
    fig_cross.add_trace(go.Scatter(
        x=df['K_Core'],
        y=df['K_Coates_Calibrated'],
        mode='markers',
        marker=dict(size=9, color=df['Phi']*100, colorscale='Viridis', showscale=True, colorbar=dict(title="Phi (%)")),
        text=[f"Depth: {d:.1f}m" for d in df['Depth']],
        name="Data Points"
    ))
    
    # خط المطابقة المثالي 1:1
    line_range = np.logspace(-2, 3, 100)
    fig_cross.add_trace(go.Scatter(x=line_range, y=line_range, mode='lines', name='1:1 Perfect Match', line=dict(color='red', dash='dash')))

    fig_cross.update_xaxes(type="log", title="Core Permeability (mD)")
    fig_cross.update_yaxes(type="log", title="NMR Calibrated Coates Permeability (mD)")
    fig_cross.update_layout(height=550, template="plotly_white")
    
    st.plotly_chart(fig_cross, use_container_width=True)

with tab3:
    st.subheader("📋 Data Explorer")
    st.dataframe(df.style.highlight_max(axis=0, color='#dcfce7'), use_container_width=True)

# 10. زر التحميل
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 تحميل تقرير النتائج الكامل (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='mishrif_advanced_petrophysics.csv',
    mime='text/csv'
)
