import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize
import streamlit as st

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Mishrif Petrophysics Suite & AI Assistant",
    layout="wide",
    page_icon="🛢️"
)

# 2. تنسيق CSS مخصص وداكن فخم (Dark Glassmorphism Theme)
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19 !important; color: #e2e8f0 !important; }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 22px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; text-align: center;
    }
    .kpi-title { color: #94a3b8; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    .kpi-value { font-size: 24px; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# 3. الهيدر الرئيسي
st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; color:#f8fafc;">🛢️ Mishrif Reservoir NMR & Core Integration Platform</h2>
        <p style="margin:6px 0 0 0; color:#94a3b8;">منصة ذكية متكاملة لمعايرة سجلات NMR، تصنيف الصخور آلياً (HFU)، ومحاكي الإنتاجية التفاعلي</p>
    </div>
""", unsafe_allow_html=True)

# 4. الشريط الجانبي (Sidebar & Simulator Controls)
st.sidebar.title("🎛️ التحكم والبتروفيزياء")
phi_cutoff = st.sidebar.slider("Porosity Cutoff (%)", 0.0, 25.0, 8.0, 0.5) / 100.0
k_cutoff = st.sidebar.slider("Permeability Cutoff (mD)", 0.01, 20.0, 0.5, 0.1)

st.sidebar.subheader("⚙️ معاملات كوتس (Coates)")
c_coates = st.sidebar.number_input("ثابت C", value=10.0, step=0.5)
a_coates = st.sidebar.number_input("أس المسامية (a)", value=4.0, step=0.1)
b_coates = st.sidebar.number_input("أس السوائل (b)", value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 محاكي الإنتاجية (Production Simulator)")
oil_price = st.sidebar.slider("سعر البرميل ($)", 40, 120, 75)
well_type = st.sidebar.radio("نوع البئر", ["عمودي (Vertical)", "أفقي (Horizontal)"])

# 5. البيانات والعمليات البتروفيزيائية
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

# حساب Flow Zone Indicator (FZI) وتحديد وحدات التدفق الهيدروليكية (HFU)
RQI = 0.0314 * np.sqrt(df['K_Coates_Calibrated'] / (df['Phi'] + 1e-5))
Phi_z = df['Phi'] / (1 - df['Phi'])
df['FZI'] = RQI / (Phi_z + 1e-5)
df['HFU'] = pd.qcut(df['FZI'], q=3, labels=["HFU 1 (Low)", "HFU 2 (Medium)", "HFU 3 (High Risk/Prod)"])

# حسابات المحاكاة والإنتاجية (BOPD)
total_thick = df['Depth'].max() - df['Depth'].min()
net_pay_thick = df['Net_Pay'].sum() * (df['Depth'].iloc[1] - df['Depth'].iloc[0])
avg_k_pay = df[df['Net_Pay']]['K_Coates_Calibrated'].mean() if df['Net_Pay'].sum() > 0 else 0.1
h_kh = avg_k_pay * net_pay_thick
kh_mult = 3.5 if "Horizontal" in well_type else 1.0
est_bopd = int(h_kh * 12.5 * kh_mult)
daily_revenue = est_bopd * oil_price

# 7. بطاقات الإحصائيات (KPI Cards)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="kpi-card"><div class="kpi-title">سمك Net Pay الصافي</div><div class="kpi-value" style="color:#34d399;">{net_pay_thick:.1f} m</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="kpi-title">سعة التدفق K*h</div><div class="kpi-value" style="color:#38bdf8;">{h_kh:.1f} mD.m</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="kpi-title">الإنتاج المتوقع (BOPD)</div><div class="kpi-value" style="color:#fbbf24;">{est_bopd:,} bbl/d</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card"><div class="kpi-title">الإيراد اليومي التقديري</div><div class="kpi-value" style="color:#a78bfa;">${daily_revenue:,.0f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 8. التبويبات الرئيسية
tab1, tab2, tab3, tab4 = st.tabs(["📊 السجلات وتصنيف الصخور", "🎯 Crossplots & HFU", "💬 المساعد الذكي (AI Petrophysicist)", "📋 جدول البيانات"])

with tab1:
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=("Porosity & Net Pay", "NMR Fluids (BVI/FFI)", "Permeability (mD)", "Rock Typing (HFU)"))

    fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='Porosity (%)', line=dict(color='#38bdf8', width=2)), row=1, col=1)
    fig.add_vline(x=phi_cutoff*100, line_dash="dash", line_color="#f87171", row=1, col=1)

    fig.add_trace(go.Scatter(x=df['BVI']*100, y=df['Depth'], name='BVI', line=dict(color='#a16207'), fill='tozerox', fillcolor='rgba(161, 98, 7, 0.4)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['Phi']*100, y=df['Depth'], name='FFI', line=dict(color='#38bdf8'), fill='tonextx', fillcolor='rgba(56, 189, 248, 0.4)'), row=1, col=2)

    fig.add_trace(go.Scatter(x=df['K_Core'], y=df['Depth'], mode='markers', name='Core K', marker=dict(color='#f8fafc', size=5)), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['K_Coates_Calibrated'], y=df['Depth'], name='Coates K', line=dict(color='#f87171', width=2)), row=1, col=3)

    # Track 4: HFU Classification
    hfu_colors = {'HFU 1 (Low)': '#ef4444', 'HFU 2 (Medium)': '#f59e0b', 'HFU 3 (High Risk/Prod)': '#10b981'}
    for hfu_name, color in hfu_colors.items():
        sub_df = df[df['HFU'] == hfu_name]
        fig.add_trace(go.Scatter(x=[1]*len(sub_df), y=sub_df['Depth'], mode='markers', name=hfu_name, marker=dict(color=color, size=8, symbol='square')), row=1, col=4)

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1, gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155', row=1, col=2); fig.update_yaxes(gridcolor='#334155', row=1, col=3); fig.update_yaxes(gridcolor='#334155', row=1, col=4)
    fig.update_xaxes(type="log", range=[-2, 3], gridcolor='#334155', row=1, col=3)
    fig.update_layout(height=700, template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19')

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🎯 Flow Zone Indicator (FZI) & Crossplots")
    fig_cross = go.Figure()
    fig_cross.add_trace(go.Scatter(
        x=df['Phi']*100, y=df['K_Coates_Calibrated'], mode='markers',
        marker=dict(size=9, color=df['FZI'], colorscale='Viridis', showscale=True, colorbar=dict(title="FZI Index")),
        text=[f"Depth: {d:.1f}m | HFU: {h}" for d, h in zip(df['Depth'], df['HFU'])]
    ))
    fig_cross.update_xaxes(title="Porosity (%)", gridcolor='#334155')
    fig_cross.update_yaxes(type="log", title="Permeability (mD)", gridcolor='#334155')
    fig_cross.update_layout(height=500, template="plotly_dark", paper_bgcolor='#0b0f19', plot_bgcolor='#0b0f19')
    st.plotly_chart(fig_cross, use_container_width=True)

with tab3:
    st.subheader("💬 AI Petrophysical Assistant")
    st.caption("اسأل الذكاء الاصطناعي عن تفسير الطبقات، التثقيب، أو جودة المعايرة لنتائج مكمن المشرف:")
    user_query = st.text_input("اكتب سؤالك الهندسي هنا:", value="ما هي أفضل أعماق ومناطق تثقيب الإنتاج (Perforation Intervals)؟")
    
    if st.button("تحليل السؤال بواسطة AI"):
        best_zone = df[df['Net_Pay']].sort_values(by='K_Coates_Calibrated', ascending=False).iloc[:5]
        min_d, max_d = best_zone['Depth'].min(), best_zone['Depth'].max()
        st.success(f"🤖 **إجابة المساعد الذكي:** بناءً على البيانات المحسوبة لمكمن المشرف، أفضل نطاق إنتاجي محتمل للتثقيب هو بين العمق **{min_d:.1f}m** و **{max_d:.1f}m**، حيث بلغ متوسط النفاذية في هذا النطاق **{best_zone['K_Coates_Calibrated'].mean():.1f} mD** مع وحدات تدفق هيدروليكية عالية (HFU 3). يُنصح بتركيز عمليات الإكمال (Completion) في هذا النطاق.")

with tab4:
    st.dataframe(df, use_container_width=True)
