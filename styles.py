import streamlit as st

def apply_styles():
    st.set_page_config(
        page_title="DocMind – Ask Your PDFs",
        page_icon="📄",
        layout="wide"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Force light theme on EVERYTHING ── */
    html, body,
    [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stTabPanel"],
    .main, .block-container,
    section[data-testid="stSidebar"] {
        background-color: #f8f9fb !important;
        color: #111827 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Override Streamlit's dark mode variables */
    :root {
        --background-color: #f8f9fb !important;
        --text-color: #111827 !important;
        --secondary-background-color: #ffffff !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 1.5rem 3rem 120px 3rem !important; /* bottom padding for pinned input */
        max-width: 1100px !important;
    }

    /* ── App title ── */
    .app-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #111827 !important;
        margin-bottom: 4px !important;
    }
    .app-title span { color: #2563eb !important; }
    .app-subtitle {
        color: #6b7280 !important;
        font-size: 14px !important;
        margin-bottom: 20px !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        background: transparent !important;
        border-bottom: 2px solid #e5e7eb !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 22px !important;
        margin-bottom: -2px !important;
        transition: color 0.15s !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #111827 !important;
        border-bottom-color: #111827 !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: #111827 !important;
        background: transparent !important;
    }
    [data-testid="stTabPanel"] {
        padding-top: 24px !important;
        background: transparent !important;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1.5px dashed #d1d5db !important;
        border-radius: 14px !important;
        padding: 28px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb !important;
        background: #f0f6ff !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #374151 !important;
        font-size: 14px !important;
    }

    /* ── Document table ── */
    .doc-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border: 1px solid #e5e7eb;
    }
    .doc-table th {
        text-align: left;
        padding: 11px 20px;
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        border-bottom: 1px solid #f0f0f0;
        background: #fafafa;
    }
    .doc-table td {
        padding: 14px 20px;
        font-size: 14px;
        color: #111827;
        border-bottom: 1px solid #f3f4f6;
        background: #ffffff;
    }
    .doc-table tr:last-child td { border-bottom: none; }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #16a34a;
        font-size: 13px;
        font-weight: 500;
    }
    .status-dot {
        width: 7px; height: 7px;
        background: #16a34a;
        border-radius: 50%;
        display: inline-block;
    }

    /* ── Chat messages area — scrollable ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
        color: #111827 !important;
    }
    /* Force message text to be dark */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {
        color: #111827 !important;
    }

    /* ── PIN CHAT INPUT TO BOTTOM ── */
    /* This is the key fix — lift the input bar out of the flow
       and anchor it to the bottom of the viewport */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background: #f8f9fb !important;
        border-top: 1px solid #e5e7eb !important;
        padding: 12px 3rem !important;
        backdrop-filter: blur(8px) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 28px !important;
        color: #111827 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }

    /* ── Source expander ── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        margin-top: 6px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #6b7280 !important;
        font-size: 13px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stMetricLabel"] p { color: #6b7280 !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        color: #111827 !important;
    }

    /* ── Status / info / success ── */
    [data-testid="stInfo"],
    [data-testid="stInfo"] p {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 10px !important;
        color: #1d4ed8 !important;
    }
    [data-testid="stSuccess"],
    [data-testid="stSuccess"] p {
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 10px !important;
        color: #15803d !important;
    }

    /* ── Spinner text ── */
    [data-testid="stSpinner"] p { color: #6b7280 !important; }

    /* ── General text overrides ── */
    p, span, label, div { color: #111827; }
    h1, h2, h3 {
        color: #111827 !important;
        font-family: 'Syne', sans-serif !important;
    }
    hr { border-color: #e5e7eb !important; }

    /* ── Buttons ── */
    [data-testid="stButton"] button {
        background: #111827 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
    }
    [data-testid="stButton"] button:hover { opacity: 0.85 !important; }

    /* ── Caption / small text ── */
    [data-testid="stCaptionContainer"] p,
    small { color: #6b7280 !important; }
    </style>
    """, unsafe_allow_html=True)