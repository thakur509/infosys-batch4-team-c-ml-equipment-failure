"""Custom styling, design system, and theme definitions for the Predictive Maintenance Dashboard.

Implements the Obsidian & Slate Industrial Analytics Design System.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base application styling */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #F9FAFB;
}

.stApp {
    background-color: #0B0F19;
    color: #F9FAFB;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0B0F19 !important;
    border-right: 1px solid #1F2937 !important;
}

section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #94A3B8 !important;
}

/* Standardized Hero Bento Section */
.hero-bento {
    display: grid;
    grid-template-columns: 1.6fr 1fr;
    gap: 1.25rem;
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #172554 100%);
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
    color: #FFFFFF;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
    border: 1px solid #312E81;
}

@media (max-width: 900px) {
    .hero-bento { grid-template-columns: 1fr; }
}

.hero-main h1 {
    font-size: 1.95rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin-bottom: 0.5rem;
}

.hero-main p {
    font-size: 0.95rem;
    color: #CBD5E1;
    line-height: 1.55;
    margin-bottom: 1.25rem;
}

.hero-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
}

.bento-stat {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    backdrop-filter: blur(8px);
}

.bento-stat .stat-label {
    font-size: 0.74rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.bento-stat-val {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
}

.bento-stat .muted {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.2rem;
}

/* Monitoring Grid (Module cards) */
.monitoring-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 1.8rem;
}

.module-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    transition: all 0.2s ease;
    text-decoration: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.module-card:hover {
    border-color: #6366F1;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}

.module-card h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.module-card p {
    font-size: 0.84rem;
    color: #94A3B8;
    margin: 0;
    line-height: 1.45;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.25rem 0.75rem;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #A5B4FC;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* Metric Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: transform 0.18s ease, border-color 0.18s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #374151;
}

.metric-label {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.15;
    margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}

.metric-subtext {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

/* Dark Slate Content Cards */
.content-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.5rem;
}

.card-header-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.25rem;
}

.card-header-desc {
    font-size: 0.86rem;
    color: #94A3B8;
    margin-bottom: 1rem;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.badge-critical {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-healthy {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-neutral {
    background-color: rgba(148, 163, 184, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

/* Forms, Inputs, Buttons */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.18s ease;
    border: 1px solid #374151;
}

.stButton button[kind="primary"] {
    background: #3B82F6 !important;
    color: #FFFFFF !important;
    border: none !important;
}

.stButton button[kind="primary"]:hover {
    background: #2563EB !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
}

.stButton button[kind="secondary"] {
    background: #1F2937 !important;
    color: #E2E8F0 !important;
}

.stButton button[kind="secondary"]:hover {
    background: #374151 !important;
    border-color: #4B5563 !important;
}

/* Sidebar telemetry status indicator */
.sidebar-panel {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1.5rem;
}

.sidebar-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}

.sidebar-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
}

/* Streamlit Tabs */
div[data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #1F2937;
    padding-bottom: 0.5rem;
}

button[data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
}

/* Hide Streamlit default hamburger menu and footer for clean app feel */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background-color: transparent;}
</style>
"""


def apply_styles():
    """Inject global CSS rules into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_chart_theme(fig, height=None, title=None):
    """Apply professional Obsidian & Slate styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0", size=12),
        title_font=dict(family="Inter, -apple-system, sans-serif", color="#F8FAFC", size=14, weight=700),
        xaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        legend=dict(
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(17,24,39,0.7)",
            bordercolor="#1F2937",
            borderwidth=1
        ),
        margin=dict(t=40 if (title or (hasattr(fig.layout, 'title') and fig.layout.title.text)) else 20, b=25, l=25, r=25)
    )
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=title)
    return fig


def render_hero(title: str, subtitle: str, tag: str = "Industrial AI Analytics"):
    """Render a standard page title banner."""
    html = f"""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%); border: 1px solid #312E81; border-radius: 12px; padding: 1.6rem 2rem; margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <span class="hero-tag">⚙️ {tag}</span>
        <h1 style="font-size: 1.75rem; font-weight: 800; color: #FFFFFF; margin: 0.2rem 0 0.4rem 0; letter-spacing: -0.02em;">{title}</h1>
        <p style="font-size: 0.92rem; color: #94A3B8; margin: 0; line-height: 1.5;">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_hero_bento(
    title: str,
    subtitle: str,
    accuracy: str = "99.05%",
    roc_auc: str = "97.50%",
    f1_score: str = "85.27%",
    fleet_health: str = "96.6",
    tag: str = "AI4I 2020 Predictive Intelligence"
):
    """Render the 2-column Bento Hero matching the reference platform."""
    html = f"""
    <div class="hero-bento">
        <div class="hero-main">
            <span class="hero-tag">⚙️ {tag}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        <div class="hero-stats">
            <div class="bento-stat">
                <span class="stat-label">Held-Out Test Accuracy</span>
                <p class="bento-stat-val" style="color: #60A5FA;">{accuracy}</p>
                <span class="muted">2,000 verified test units</span>
            </div>
            <div class="bento-stat">
                <span class="stat-label">ROC-AUC Discrimination</span>
                <p class="bento-stat-val" style="color: #2DD4BF;">{roc_auc}</p>
                <span class="muted">F1-Score: {f1_score}</span>
            </div>
            <div class="bento-stat">
                <span class="stat-label">Fleet Health Avg</span>
                <p class="bento-stat-val" style="color: #34D399;">{fleet_health}</p>
                <span class="muted">0–100 Continuous Index</span>
            </div>
            <div class="bento-stat">
                <span class="stat-label">Model Engine Architecture</span>
                <p class="bento-stat-val" style="font-size: 1.25rem; color: #E2E8F0; margin-top: 4px;">Random Forest</p>
                <span class="muted">150 Trees · Imbalance Tuned</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_monitoring_modules_grid():
    """Render the 6 interactive module cards matching the reference site."""
    html = """
    <div style="margin-bottom: 0.6rem;">
        <h2 style="font-size: 1.15rem; font-weight: 700; margin: 0; color: #F8FAFC;">Interactive Monitoring Modules</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0.2rem 0 1rem 0;">Explore real-time risk classifications, sensor anomalies, alerts, and historical trajectories.</p>
    </div>
    <div class="monitoring-grid">
        <div class="module-card">
            <h3>Failure Prediction Studio <span style="color:#60A5FA;">→</span></h3>
            <p>Interactive telemetry inference with real-time probability gauge and Celsius/Kelvin conversion.</p>
        </div>
        <div class="module-card">
            <h3>Equipment Fleet <span style="color:#60A5FA;">→</span></h3>
            <p>10,000-unit asset registry with telemetry drilldown and radar comparison against fleet baselines.</p>
        </div>
        <div class="module-card">
            <h3>Anomaly Detection <span style="color:#60A5FA;">→</span></h3>
            <p>Review sensor readings flagged outside statistical IQR operating thresholds & Isolation Forest.</p>
        </div>
        <div class="module-card">
            <h3>Equipment Alerts <span style="color:#60A5FA;">→</span></h3>
            <p>SCADA-inspired alert dispatch with New, Acknowledged, and Resolved lifecycle management.</p>
        </div>
        <div class="module-card">
            <h3>Maintenance Recommendations <span style="color:#60A5FA;">→</span></h3>
            <p>Coordinated work order registry with certified technician assignment and schedule tracking.</p>
        </div>
        <div class="module-card">
            <h3>Advanced ML Tasks (T38–T45) <span style="color:#60A5FA;">→</span></h3>
            <p>6-Class specific failure model, Isolation Forest benchmark, TreeSHAP, and 20-tick streaming pipeline.</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", delta: str = None, delta_color: str = "normal"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        color = "#34D399" if delta_color == "positive" else ("#F87171" if delta_color == "negative" else "#94A3B8")
        delta_html = f"<span style='color:{color}; font-size:0.8rem; font-weight:600;'>{delta}</span>"
    
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext} {delta_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_status():
    """Render real-time system and model status in the sidebar."""
    html = """
    <div class="sidebar-panel">
        <div class="sidebar-panel-title">System Status</div>
        <div class="sidebar-status-row">
            <span>Model Engine</span>
            <span style="color:#34D399; font-weight:600;">● Active</span>
        </div>
        <div class="sidebar-status-row">
            <span>Algorithm</span>
            <span style="color:#A5B4FC;">Random Forest</span>
        </div>
        <div class="sidebar-status-row">
            <span>Database</span>
            <span style="color:#34D399; font-weight:600;">● Connected</span>
        </div>
        <div class="sidebar-status-row">
            <span>Telemetry Mode</span>
            <span style="color:#94A3B8;">AI4I 2020 Dataset</span>
        </div>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)
