"""FinSight AI - Gen Z Fintech Presentation Layer & Design System.

Inspired by modern neo-fintech apps like Cred, Jupiter, and Fi Money.
Features dark glassmorphism, animated gradient glow mesh, Plus Jakarta Sans
typography, pill-shaped buttons, and responsive micro-interactions.
"""

from typing import Optional
import textwrap
import plotly.graph_objects as go
import streamlit as st


def render_html(html: str, sidebar: bool = False) -> None:
    """Render HTML safely without triggering Markdown indented code block formatting.

    Dedents the string, strips leading and trailing whitespace from each line,
    collapses blank lines, and then calls st.markdown (or st.sidebar.markdown)
    with unsafe_allow_html=True.
    """
    if not html:
        return
    dedented = textwrap.dedent(html).strip()
    cleaned_lines = [line.strip() for line in dedented.splitlines() if line.strip()]
    cleaned_html = "".join(cleaned_lines)
    if sidebar:
        st.sidebar.markdown(cleaned_html, unsafe_allow_html=True)
    else:
        st.markdown(cleaned_html, unsafe_allow_html=True)


GENZ_CSS = r"""
<style>
/* 1. Google Font Import - Plus Jakarta Sans */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Font: applied only to text elements, never to icons */
html, body, p, label, h1, h2, h3, h4, h5, h6, input, textarea, button, select {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.015em;
}

/* Explicitly preserve and restore Streamlit icon glyphs */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons,
span[class*="material"],
span[data-testid*="Icon"],
div[data-testid="stExpanderToggleIcon"] span,
summary span[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
}

/* 2. Dark Animated Gradient Mesh Background */
.stApp {
    background-color: #0B0B14 !important;
    background-image: 
        radial-gradient(at 15% 15%, rgba(139, 92, 246, 0.16) 0px, transparent 45%),
        radial-gradient(at 85% 12%, rgba(59, 130, 246, 0.14) 0px, transparent 45%),
        radial-gradient(at 30% 85%, rgba(236, 72, 153, 0.13) 0px, transparent 45%),
        radial-gradient(at 80% 85%, rgba(16, 240, 176, 0.09) 0px, transparent 45%) !important;
    background-attachment: fixed !important;
    color: #F4F4FB !important;
}

/* Page Entry Fade-In Animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.main .block-container {
    animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* 3. Hide Default Streamlit Menus & Branding */
#MainMenu, footer, header {
    visibility: hidden !important;
    height: 0px !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}

/* 4. Glassmorphism Card System */
.genz-glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(20, 20, 43, 0.72) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 22px;
    padding: 24px 28px;
    box-shadow: 0 12px 32px -4px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.genz-glass-card:hover {
    transform: translateY(-4px) scale(1.006);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 20px 44px -8px rgba(139, 92, 246, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}


/* 5. Custom Metric Cards */
.genz-metric-box {
    background: rgba(20, 20, 43, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
}

.genz-metric-box:hover {
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 16px 36px -6px rgba(139, 92, 246, 0.25);
}

.genz-metric-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8B5CF6, #3B82F6, #EC4899);
    opacity: 0.85;
}

.genz-metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.genz-metric-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
}

.genz-metric-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.genz-metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #FFFFFF 30%, #E2E8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.genz-metric-delta {
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}

.delta-positive {
    background: rgba(16, 240, 176, 0.15);
    color: #10F0B0;
    border: 1px solid rgba(16, 240, 176, 0.3);
}

.delta-negative {
    background: rgba(236, 72, 153, 0.15);
    color: #F43F5E;
    border: 1px solid rgba(236, 72, 153, 0.3);
}

.delta-neutral {
    background: rgba(148, 163, 184, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

/* 6. Section Headers with Gradient Accents */
.genz-section-header {
    margin-top: 10px;
    margin-bottom: 22px;
}

.genz-gradient-text {
    background: linear-gradient(135deg, #C084FC 0%, #8B5CF6 35%, #3B82F6 70%, #10F0B0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.genz-header-title {
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.25;
}

.genz-header-subtitle {
    font-size: 0.95rem;
    color: #94A3B8;
    margin-top: 4px;
    font-weight: 400;
}

/* 7. Hero Banner */
.genz-hero-banner {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.16) 0%, rgba(59, 130, 246, 0.12) 50%, rgba(20, 20, 43, 0.75) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5);
}

.genz-hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #8B5CF6, #3B82F6);
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 12px;
}

/* 8. Modern Pill-Shaped Gradient Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 50%, #3B82F6 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.65rem 1.8rem !important;
    box-shadow: 0 4px 18px rgba(139, 92, 246, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(139, 92, 246, 0.55) !important;
    border: none !important;
}

div.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* 9. Sidebar Neo-Styling */
div[data-testid="stSidebar"] {
    background: rgba(11, 11, 20, 0.88) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

div[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}

/* Sidebar Radio Styling */
div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(20, 20, 43, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(139, 92, 246, 0.12);
    border-color: rgba(139, 92, 246, 0.3);
}

/* 10. Form Controls: Selectbox, Inputs, Sliders */
div[data-baseweb="select"] > div {
    background: rgba(20, 20, 43, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #F4F4FB !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(139, 92, 246, 0.4) !important;
}

input, textarea {
    background: rgba(20, 20, 43, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #F4F4FB !important;
}

input:focus, textarea:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
}

/* 11. Sliders */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #8B5CF6 !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.75), 0 0 4px rgba(16, 240, 176, 0.4) !important;
    width: 20px !important;
    height: 20px !important;
}

div[data-testid="stSlider"] > div {
    padding-top: 6px !important;
}

div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
}

/* 11b. Modern Glass Expanders */
div[data-testid="stExpander"] {
    background: rgba(20, 20, 43, 0.65) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35) !important;
    margin-bottom: 20px !important;
    overflow: hidden !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(139, 92, 246, 0.35) !important;
    box-shadow: 0 12px 30px -4px rgba(139, 92, 246, 0.2) !important;
}

div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #FFFFFF !important;
    font-size: 0.96rem !important;
    padding: 14px 18px !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #C084FC !important;
}

/* 11c. Modern Glass Alerts */
div[data-testid="stAlert"] {
    background: rgba(20, 20, 43, 0.78) !important;
    backdrop-filter: blur(18px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35) !important;
    padding: 16px 20px !important;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlert-info"]) {
    border-left: 4px solid #8B5CF6 !important;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlert-warning"]) {
    border-left: 4px solid #FB923C !important;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlert-success"]) {
    border-left: 4px solid #10F0B0 !important;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlert-error"]) {
    border-left: 4px solid #F43F5E !important;
}


/* 12. Pill Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-high {
    background: rgba(236, 72, 153, 0.15);
    color: #F43F5E;
    border: 1px solid rgba(236, 72, 153, 0.35);
    box-shadow: 0 0 12px rgba(236, 72, 153, 0.2);
}

.badge-medium {
    background: rgba(251, 146, 60, 0.15);
    color: #FB923C;
    border: 1px solid rgba(251, 146, 60, 0.35);
}

.badge-low {
    background: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.badge-success {
    background: rgba(16, 240, 176, 0.15);
    color: #10F0B0;
    border: 1px solid rgba(16, 240, 176, 0.35);
    box-shadow: 0 0 12px rgba(16, 240, 176, 0.2);
}

.badge-purple {
    background: rgba(139, 92, 246, 0.15);
    color: #C084FC;
    border: 1px solid rgba(139, 92, 246, 0.35);
}

/* 13. Recommendation Cards */
.genz-rec-card {
    background: rgba(20, 20, 43, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
}

.genz-rec-card:hover {
    transform: translateY(-3px);
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 16px 36px -8px rgba(139, 92, 246, 0.2);
}

/* 14. Custom Smooth Scrollbars */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #0B0B14;
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.35);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.65);
}

/* 15. Responsive Mobile Tweaks */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }
    .genz-hero-banner {
        padding: 20px;
    }
    .genz-header-title {
        font-size: 1.45rem;
    }
    .genz-metric-value {
        font-size: 1.5rem;
    }
}
/* 16. Animated Count-Up & Typography */
@keyframes countUpFade {
    0% {
        opacity: 0;
        transform: translateY(6px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.count-up-val {
    animation: countUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* 17. Spotify Wrapped Insights Ribbon */
.wrapped-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 18px 0 24px 0;
}

.wrapped-card {
    border-radius: 20px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}

.wrapped-card:hover {
    transform: translateY(-4px) scale(1.02);
}

.wrapped-card-1 {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28) 0%, rgba(236, 72, 153, 0.2) 100%);
    border: 1px solid rgba(139, 92, 246, 0.4);
    box-shadow: 0 12px 28px -6px rgba(139, 92, 246, 0.25);
}

.wrapped-card-2 {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.28) 0%, rgba(245, 158, 11, 0.2) 100%);
    border: 1px solid rgba(236, 72, 153, 0.4);
    box-shadow: 0 12px 28px -6px rgba(236, 72, 153, 0.25);
}

.wrapped-card-3 {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.28) 0%, rgba(16, 240, 176, 0.2) 100%);
    border: 1px solid rgba(59, 130, 246, 0.4);
    box-shadow: 0 12px 28px -6px rgba(59, 130, 246, 0.25);
}

.wrapped-card-4 {
    background: linear-gradient(135deg, rgba(16, 240, 176, 0.28) 0%, rgba(139, 92, 246, 0.2) 100%);
    border: 1px solid rgba(16, 240, 176, 0.4);
    box-shadow: 0 12px 28px -6px rgba(16, 240, 176, 0.25);
}

.wrapped-pill {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #E2E8F0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.wrapped-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.wrapped-sub {
    font-size: 0.82rem;
    color: #CBD5E1;
    font-weight: 500;
}

/* 18. Savings Goal Progress Bar */
.savings-goal-box {
    background: rgba(20, 20, 43, 0.72);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(139, 92, 246, 0.28);
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
}

.savings-goal-bar-bg {
    width: 100%;
    height: 12px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
    margin: 12px 0 10px 0;
}

.savings-goal-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #8B5CF6 0%, #3B82F6 50%, #10F0B0 100%);
    box-shadow: 0 0 14px rgba(16, 240, 176, 0.5);
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

@media (max-width: 900px) {
    .wrapped-strip {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 600px) {
    .wrapped-strip {
        grid-template-columns: 1fr;
    }
}

/* 19. Anomaly Alert Cards */
.genz-anomaly-card {
    background: rgba(20, 20, 43, 0.72);
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.genz-anomaly-card:hover {
    transform: translateY(-3px);
}

.anomaly-card-red {
    border: 1px solid rgba(244, 63, 94, 0.38);
    box-shadow: 0 8px 24px -4px rgba(244, 63, 94, 0.22);
}

.anomaly-card-red:hover {
    border-color: rgba(244, 63, 94, 0.6);
    box-shadow: 0 14px 32px -4px rgba(244, 63, 94, 0.35);
}

.anomaly-card-orange {
    border: 1px solid rgba(251, 146, 60, 0.38);
    box-shadow: 0 8px 24px -4px rgba(251, 146, 60, 0.22);
}

.anomaly-card-orange:hover {
    border-color: rgba(251, 146, 60, 0.6);
    box-shadow: 0 14px 32px -4px rgba(251, 146, 60, 0.35);
}

/* 20. Confidence Visual Bars */
.conf-bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
}

.conf-bar-track {
    background: rgba(255, 255, 255, 0.08);
    width: 90px;
    height: 8px;
    border-radius: 9999px;
    overflow: hidden;
}

.conf-bar-green {
    background: #10F0B0;
    box-shadow: 0 0 8px rgba(16, 240, 176, 0.4);
}

.conf-bar-yellow {
    background: #F59E0B;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
}

.conf-bar-red {
    background: #F43F5E;
    box-shadow: 0 0 8px rgba(244, 63, 94, 0.4);
}

/* 21. Friendly Empty State Card */
.genz-empty-state {
    text-align: center;
    padding: 42px 24px;
    background: rgba(20, 20, 43, 0.6);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    margin: 18px 0;
}

/* 22. AI Coach Recommendation Cards */
.genz-rec-card {
    background: rgba(20, 20, 43, 0.72);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 190px;
}

.genz-rec-card:hover {
    transform: translateY(-3px) scale(1.01);
}

.rec-border-high {
    border-left: 5px solid #F43F5E !important;
    box-shadow: 0 8px 24px -4px rgba(244, 63, 94, 0.2);
}

.rec-border-medium {
    border-left: 5px solid #FB923C !important;
    box-shadow: 0 8px 24px -4px rgba(251, 146, 60, 0.2);
}

.rec-border-low {
    border-left: 5px solid #3B82F6 !important;
    box-shadow: 0 8px 24px -4px rgba(59, 130, 246, 0.2);
}

.glow-save-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.88rem;
    font-weight: 800;
    color: #10F0B0;
    background: rgba(16, 240, 176, 0.12);
    border: 1px solid rgba(16, 240, 176, 0.35);
    padding: 6px 14px;
    border-radius: 9999px;
    box-shadow: 0 0 14px rgba(16, 240, 176, 0.35);
    letter-spacing: -0.01em;
}

/* 23. Messaging App Style Chat Interface */
div[data-testid="stChatMessage"] {
    background: rgba(20, 20, 43, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15)) !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
}

/* 24. Global Footer */
.genz-footer {
    text-align: center;
    padding: 36px 0 16px 0;
    margin-top: 56px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #94A3B8;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

.genz-footer b {
    color: #FFFFFF;
}

.genz-footer .heart {
    color: #EC4899;
    display: inline-block;
    animation: heartPulse 1.5s infinite;
}

@keyframes heartPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.25); }
}
</style>
"""


def inject_css() -> None:
    """Inject the complete Gen Z fintech CSS stylesheet into the Streamlit app."""
    st.markdown(GENZ_CSS, unsafe_allow_html=True)


def metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "💰",
    color: str = "purple",
) -> None:
    """Render a premium glassmorphic metric card with count-up styling and delta arrows."""
    delta_html = ""
    if delta:
        d_lower = delta.lower()
        if any(w in d_lower for w in ["+", "save", "income", "surplus"]):
            arrow = "↑ " if not any(s in delta for s in ["↑", "▲", "↓", "▼"]) else ""
            delta_class = "delta-positive"
            formatted_delta = f"{arrow}{delta}"
        elif any(w in d_lower for w in ["-", "expense", "deficit", "burn"]):
            arrow = "↓ " if not any(s in delta for s in ["↑", "▲", "↓", "▼"]) else ""
            delta_class = "delta-negative"
            formatted_delta = f"{arrow}{delta}"
        else:
            delta_class = "delta-neutral"
            formatted_delta = delta
        delta_html = f'<span class="genz-metric-delta {delta_class}">{formatted_delta}</span>'

    card_html = f"""
    <div class="genz-metric-box">
        <div class="genz-metric-header">
            <span class="genz-metric-title">{title}</span>
            <div class="genz-metric-icon">{icon}</div>
        </div>
        <div class="genz-metric-value count-up-val">{value}</div>
        {delta_html}
    </div>
    """
    render_html(card_html)


GENZ_COLOR_SEQUENCE = ["#8B5CF6", "#3B82F6", "#EC4899", "#10F0B0", "#F59E0B", "#06B6D4"]


def style_fig(
    fig: go.Figure,
    is_donut: bool = False,
    donut_total_text: Optional[str] = None,
) -> go.Figure:
    """Apply the unified Gen Z Fintech styling to any Plotly figure.

    - Transparent paper & plot background
    - Very faint grid lines (rgba(255, 255, 255, 0.05))
    - 'Plus Jakarta Sans' font hierarchy
    - Custom color sequence: #8B5CF6, #3B82F6, #EC4899, #10F0B0, #F59E0B, #06B6D4
    - Rounded bars with cornerradius
    - Spline smoothed lines with subtle gradient area fill
    - Donut charts with hole=0.65 and center total text
    - INR formatted hover templates and smooth transitions
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=GENZ_COLOR_SEQUENCE,
        font=dict(
            family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#CBD5E1",
            size=12,
        ),
        transition=dict(
            duration=500,
            easing="cubic-in-out",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                family="Plus Jakarta Sans, sans-serif",
                color="#94A3B8",
                size=11,
            ),
        ),
    )

    # Update Cartesian axes if present
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        showline=False,
        tickfont=dict(
            family="Plus Jakarta Sans, sans-serif",
            color="#94A3B8",
            size=11,
        ),
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.08)",
        showline=False,
        tickfont=dict(
            family="Plus Jakarta Sans, sans-serif",
            color="#94A3B8",
            size=11,
        ),
    )

    # Style traces according to trace type
    for trace in fig.data:
        # Bar traces: rounded corners
        if trace.type == "bar":
            trace.marker.cornerradius = 8
            if trace.marker.line.width is None or trace.marker.line.width == 0:
                trace.marker.line.width = 0

        # Scatter line traces: spline smoothing + gradient area fill
        elif trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.shape = "spline"
            trace.line.smoothing = 1.3
            # If not already specified, add subtle area fill for primary lines
            if not getattr(trace, "fill", None) and getattr(trace, "name", "") in ["Daily Spend", "Spend"]:
                trace.fill = "tozeroy"
                trace.fillcolor = "rgba(59, 130, 246, 0.08)"

        # Donut / Pie traces: 0.65 hole and border separation
        elif trace.type == "pie" or is_donut:
            trace.hole = 0.65
            trace.marker.line = dict(color="#0B0B14", width=2)
            trace.textfont = dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF")

    # Add center total text for donut charts if provided
    if (is_donut or any(t.type == "pie" for t in fig.data)) and donut_total_text:
        fig.update_layout(
            annotations=[
                dict(
                    text=f"<span style='font-size:10px;letter-spacing:0.08em;color:#94A3B8;font-weight:700;'>TOTAL SPEND</span><br><b style='font-size:17px;color:#FFFFFF;letter-spacing:-0.02em;'>{donut_total_text}</b>",
                    x=0.5,
                    y=0.5,
                    font=dict(family="Plus Jakarta Sans, sans-serif"),
                    showarrow=False,
                )
            ]
        )

    return fig


def spotify_insights_strip(
    top_category: str,
    top_cat_amount: str,
    top_cat_share: str,
    most_expensive_day: str,
    most_expensive_day_amt: str,
    biggest_merchant: str,
    biggest_merchant_amt: str,
    no_spend_days: int,
) -> None:
    """Render a Spotify Wrapped-style spending insights ribbon with vibrant gradient cards."""
    strip_html = f"""
    <div class="wrapped-strip">
        <div class="wrapped-card wrapped-card-1">
            <div class="wrapped-pill"><span>🍔</span> Top Spending Category</div>
            <div class="wrapped-val">{top_category}</div>
            <div class="wrapped-sub">{top_cat_amount} ({top_cat_share} of expenses)</div>
        </div>
        <div class="wrapped-card wrapped-card-2">
            <div class="wrapped-pill"><span>⚡</span> Most Expensive Day</div>
            <div class="wrapped-val">{most_expensive_day}</div>
            <div class="wrapped-sub">{most_expensive_day_amt} single-day burn</div>
        </div>
        <div class="wrapped-card wrapped-card-3">
            <div class="wrapped-pill"><span>🛍️</span> Biggest Merchant</div>
            <div class="wrapped-val">{biggest_merchant}</div>
            <div class="wrapped-sub">{biggest_merchant_amt} total outlays</div>
        </div>
        <div class="wrapped-card wrapped-card-4">
            <div class="wrapped-pill"><span>✨</span> Zero-Spend Days</div>
            <div class="wrapped-val">{no_spend_days} Days</div>
            <div class="wrapped-sub">Financial discipline streak</div>
        </div>
    </div>
    """
    render_html(strip_html)


def savings_goal_progress(current_savings_rate: float, target_rate: float = 20.0) -> None:
    """Render an aesthetic glassmorphic savings goal progress tracker with motivating message."""
    pct_towards_goal = min(100.0, max(0.0, (current_savings_rate / target_rate) * 100.0)) if target_rate > 0 else 0.0

    if current_savings_rate >= 20.0:
        icon = "🎉"
        headline = "Elite 20%+ Savings Club Unlocked!"
        message = f"You're currently saving <b>{current_savings_rate:.1f}%</b> of your income. Future you is celebrating — keep compounding this wealth snowball!"
        bar_color = "linear-gradient(90deg, #8B5CF6 0%, #3B82F6 50%, #10F0B0 100%)"
    elif current_savings_rate >= 10.0:
        icon = "⚡"
        headline = "Strong Inflow Momentum"
        message = f"You're at <b>{current_savings_rate:.1f}%</b> savings rate ({pct_towards_goal:.0f}% of your 20% target). A slight trim in dining or shopping will put you into the golden tier!"
        bar_color = "linear-gradient(90deg, #8B5CF6 0%, #3B82F6 100%)"
    else:
        icon = "💪"
        headline = "Building Your Cash Cushion"
        message = f"Current savings rate is <b>{current_savings_rate:.1f}%</b>. Use the AI Coach recommendations below to close discretionary spending leaks and reach your 20% milestone."
        bar_color = "linear-gradient(90deg, #EC4899 0%, #F59E0B 100%)"

    box_html = f"""
    <div class="savings-goal-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.25rem;">{icon}</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">{headline}</span>
            </div>
            <span style="font-size: 0.88rem; font-weight: 700; color: #10F0B0; background: rgba(16,240,176,0.12); padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(16,240,176,0.3);">
                {current_savings_rate:.1f}% / {target_rate:.0f}% Target
            </span>
        </div>
        <div class="savings-goal-bar-bg">
            <div class="savings-goal-bar-fill" style="width: {pct_towards_goal:.1f}%; background: {bar_color};"></div>
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.4;">
            {message}
        </div>
    </div>
    """
    render_html(box_html)


def section_header(title: str, subtitle: Optional[str] = None, emoji: str = "✨") -> None:
    """Render an expressive section header with a gradient title."""
    sub_html = f'<div class="genz-header-subtitle">{subtitle}</div>' if subtitle else ""
    header_html = f"""
    <div class="genz-section-header">
        <h2 class="genz-header-title">
            <span style="margin-right: 8px;">{emoji}</span>
            <span class="genz-gradient-text">{title}</span>
        </h2>
        {sub_html}
    </div>
    """
    render_html(header_html)


def glass_card(content_html: str, class_name: str = "") -> None:
    """Render arbitrary HTML content enclosed inside a frosted glass card container."""
    render_html(f'<div class="genz-glass-card {class_name}">{content_html}</div>')


def badge(text: str, badge_type: str = "medium") -> str:
    """Generate inline HTML string for a pill-shaped status or priority badge."""
    type_map = {
        "high": "badge-high",
        "medium": "badge-medium",
        "low": "badge-low",
        "success": "badge-success",
        "warning": "badge-medium",
        "purple": "badge-purple",
    }
    css_class = type_map.get(badge_type.lower(), "badge-medium")
    return f'<span class="badge-pill {css_class}">{text}</span>'


def hero_banner(title: str, tagline: str, badge_text: Optional[str] = "⚡ FinSight AI 2.0") -> None:
    """Render a top-level glassmorphic hero banner with glowing gradient accents."""
    badge_html = f'<div class="genz-hero-badge">{badge_text}</div>' if badge_text else ""
    banner_html = f"""
    <div class="genz-hero-banner">
        {badge_html}
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.03em;">
            <span class="genz-gradient-text">{title}</span>
        </h1>
        <div style="font-size: 1.05rem; color: #CBD5E1; font-weight: 400; max-width: 800px; line-height: 1.5;">
            {tagline}
        </div>
    </div>
    """
    render_html(banner_html)


def empty_state(emoji: str, title: str, message: str) -> None:
    """Render a friendly Gen Z styled empty state card with emoji and helpful guidance."""
    html = f"""
    <div class="genz-empty-state">
        <div style="font-size: 3.2rem; margin-bottom: 12px;">{emoji}</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-bottom: 6px; letter-spacing: -0.02em;">
            {title}
        </div>
        <div style="font-size: 0.95rem; color: #94A3B8; max-width: 520px; margin: 0 auto; line-height: 1.5;">
            {message}
        </div>
    </div>
    """
    render_html(html)


def unlock_hint_card(
    title: str,
    current: int,
    required: int,
    item_label: str = "transactions",
    unlocked_msg: str = "Feature fully unlocked!",
    locked_msg: Optional[str] = None,
    icon: str = "🔓",
) -> bool:
    """Render a Gen Z progress hint card showing feature unlock status. Returns True if unlocked."""
    is_unlocked = current >= required
    pct = min(100.0, max(0.0, (current / required * 100.0))) if required > 0 else 100.0
    bar_color = (
        "linear-gradient(90deg, #10F0B0 0%, #3B82F6 100%)"
        if is_unlocked
        else "linear-gradient(90deg, #8B5CF6 0%, #EC4899 100%)"
    )
    badge_label = "UNLOCKED" if is_unlocked else f"{current}/{required} {item_label.upper()}"
    badge_color = "#10F0B0" if is_unlocked else "#C084FC"
    badge_bg = "rgba(16,240,176,0.15)" if is_unlocked else "rgba(139,92,246,0.15)"
    badge_border = "rgba(16,240,176,0.3)" if is_unlocked else "rgba(139,92,246,0.3)"

    desc = (
        unlocked_msg
        if is_unlocked
        else (locked_msg or f"Add {max(0, required - current)} more {item_label} to unlock full capability.")
    )

    html = f"""
    <div class="genz-glass-card" style="padding: 16px 20px; margin-bottom: 20px; border-left: 4px solid {'#10F0B0' if is_unlocked else '#8B5CF6'};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.25rem;">{icon}</span>
                <span style="font-size: 0.98rem; font-weight: 700; color: #FFFFFF;">{title}</span>
            </div>
            <span style="font-size: 0.78rem; font-weight: 700; color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_border}; padding: 3px 10px; border-radius: 9999px;">
                {badge_label}
            </span>
        </div>
        <div style="background: rgba(255,255,255,0.06); height: 7px; border-radius: 9999px; overflow: hidden; margin: 10px 0 8px 0;">
            <div style="width: {pct:.1f}%; height: 100%; border-radius: 9999px; background: {bar_color};"></div>
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            {desc}
        </div>
    </div>
    """
    render_html(html)
    return is_unlocked


def render_footer() -> None:
    """Render global Gen Z fintech footer line."""
    footer_html = """
    <div class="genz-footer">
        Built with <span class="heart">❤️</span> by <b>Sainath</b> &nbsp;|&nbsp; <span>AI for Finance</span>
    </div>
    """
    render_html(footer_html)

