import streamlit as st

def apply_styles():
    st.set_page_config(
        page_title="DocMind – AI Document Intelligence",
        page_icon="📄",
        layout="wide"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

    /* ═══════════════════════════════════════════════
       GLOBAL RESET & LIGHT THEME ENFORCEMENT
    ═══════════════════════════════════════════════ */
    html, body,
    [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stTabPanel"],
    .main, .block-container,
    section[data-testid="stSidebar"] {
        background-color: #f5f4f0 !important;
        color: #1a1a2e !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    :root {
        --navy:    #1a1a2e;
        --ink:     #2d2d44;
        --steel:   #4a4a6a;
        --mist:    #8a8aaa;
        --cloud:   #e8e7f0;
        --paper:   #f5f4f0;
        --white:   #ffffff;
        --accent:  #4f46e5;
        --accent2: #7c3aed;
        --gold:    #d97706;
        --green:   #059669;
        --border:  rgba(26,26,46,0.10);
        --border2: rgba(26,26,46,0.18);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 0 2.5rem 140px 2.5rem !important;
        max-width: 1200px !important;
    }

    /* ═══════════════════════════════════════════════
       NAVBAR
    ═══════════════════════════════════════════════ */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 20px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(26,26,46,0.10);
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .navbar-logo {
        width: 34px; height: 34px;
        background: #1a1a2e;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 15px;
    }
    .navbar-name {
        font-family: 'Playfair Display', serif;
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.3px;
    }
    .navbar-accent { color: #4f46e5; }
    .navbar-tag {
        background: rgba(79,70,229,0.08);
        border: 1px solid rgba(79,70,229,0.18);
        border-radius: 100px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
        color: #4f46e5;
        letter-spacing: 0.3px;
    }
    .navbar-right { display: flex; align-items: center; gap: 12px; }
    .navbar-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #8a8aaa;
        font-weight: 500;
    }
    .navbar-online {
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 5px #22c55e;
        display: inline-block;
    }

    /* ═══════════════════════════════════════════════
       HERO HEADER — editorial magazine style
    ═══════════════════════════════════════════════ */
    .hero-wrapper {
        background: var(--navy);
        border-radius: 20px;
        padding: 48px 56px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(79,70,229,0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-wrapper::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 200px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 100px;
        padding: 5px 14px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.65);
        margin-bottom: 20px;
    }
    .hero-dot {
        width: 6px; height: 6px;
        background: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 6px #4ade80;
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 48px !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.5px !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        margin: 0 0 14px !important;
    }
    .hero-title em {
        font-style: italic;
        color: rgba(255,255,255,0.5) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.5) !important;
    }
    .hero-sub {
        font-size: 16px;
        color: rgba(255,255,255,0.55);
        font-weight: 300;
        line-height: 1.6;
        max-width: 480px;
        margin: 0 0 36px;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 500;
        color: rgba(255,255,255,0.7);
    }
    .hero-pill-icon {
        font-size: 13px;
    }

    /* ═══════════════════════════════════════════════
       STATS STRIP
    ═══════════════════════════════════════════════ */
    .stats-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    .stat-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--accent);
        border-radius: 3px 0 0 3px;
    }
    .stat-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: var(--mist);
        margin-bottom: 6px;
    }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        font-weight: 600;
        color: var(--navy);
        line-height: 1;
    }
    .stat-sub {
        font-size: 11px;
        color: var(--mist);
        margin-top: 4px;
    }

    /* ═══════════════════════════════════════════════
       TABS
    ═══════════════════════════════════════════════ */
    [data-testid="stTabs"] [role="tablist"] {
        background: transparent !important;
        border-bottom: 1.5px solid var(--border2) !important;
        gap: 0 !important;
        padding: 0 !important;
        margin-bottom: 28px !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--mist) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 12px 24px !important;
        margin-bottom: -1.5px !important;
        letter-spacing: 0.2px !important;
        transition: color 0.15s !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--navy) !important;
        border-bottom-color: var(--navy) !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--ink) !important;
        background: transparent !important;
    }
    [data-testid="stTabPanel"] {
        padding-top: 8px !important;
        background: transparent !important;
    }

    /* ═══════════════════════════════════════════════
       UPLOAD ZONE
    ═══════════════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        background: var(--white) !important;
        border: 1.5px dashed var(--border2) !important;
        border-radius: 16px !important;
        padding: 24px 32px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: rgba(79,70,229,0.02) !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: var(--steel) !important;
        font-size: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    /* Fix dark background on the inner drop zone */
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] section > div {
        background: transparent !important;
    }
    /* Upload button — hide ALL text inside it, replace via pseudo-element */
    [data-testid="stFileUploader"] button {
        background: var(--navy) !important;
        color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0 !important;
        font-weight: 500 !important;
        padding: 7px 16px !important;
        position: relative !important;
        min-width: 100px !important;
    }
    [data-testid="stFileUploader"] button::after {
        content: 'Browse Files' !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
        white-space: nowrap !important;
    }
    [data-testid="stFileUploader"] button span {
        display: none !important;
    }
    /* Remove dark dropzone background */
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }

    /* ═══════════════════════════════════════════════
       DOCUMENT TABLE
    ═══════════════════════════════════════════════ */
    .doc-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 24px;
        background: var(--white);
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .doc-table th {
        text-align: left;
        padding: 12px 20px;
        font-size: 10px;
        font-weight: 700;
        color: var(--mist);
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid var(--border);
        background: #fafafa;
    }
    .doc-table td {
        padding: 15px 20px;
        font-size: 14px;
        color: var(--ink);
        border-bottom: 1px solid rgba(26,26,46,0.05);
        background: var(--white);
    }
    .doc-table tr:last-child td { border-bottom: none; }
    .doc-table tr:hover td { background: rgba(79,70,229,0.02); }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(5,150,105,0.08);
        border: 1px solid rgba(5,150,105,0.2);
        border-radius: 100px;
        padding: 3px 12px;
        color: var(--green);
        font-size: 12px;
        font-weight: 600;
    }
    .status-dot {
        width: 6px; height: 6px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
    }

    /* ═══════════════════════════════════════════════
       CHAT MESSAGES
    ═══════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ═══════════════════════════════════════════════
       PINNED CHAT INPUT
    ═══════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background: rgba(245,244,240,0.95) !important;
        border-top: 1px solid var(--border2) !important;
        padding: 14px 3rem !important;
        backdrop-filter: blur(12px) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: var(--white) !important;
        border: 1.5px solid var(--border2) !important;
        border-radius: 30px !important;
        color: var(--navy) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 13px 22px !important;
        box-shadow: 0 2px 12px rgba(26,26,46,0.07) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.1), 0 2px 12px rgba(26,26,46,0.07) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--mist) !important;
        font-style: italic !important;
    }

    /* ═══════════════════════════════════════════════
       EXPANDER / SOURCES
    ═══════════════════════════════════════════════ */
    [data-testid="stExpander"] {
        background: var(--white) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: var(--steel) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* ═══════════════════════════════════════════════
       STATUS / INFO / SUCCESS BOXES
    ═══════════════════════════════════════════════ */
    [data-testid="stInfo"],
    [data-testid="stInfo"] p {
        background: rgba(79,70,229,0.06) !important;
        border: 1px solid rgba(79,70,229,0.18) !important;
        border-radius: 12px !important;
        color: var(--accent) !important;
    }
    [data-testid="stSuccess"],
    [data-testid="stSuccess"] p {
        background: rgba(5,150,105,0.06) !important;
        border: 1px solid rgba(5,150,105,0.18) !important;
        border-radius: 12px !important;
        color: var(--green) !important;
    }

    /* ═══════════════════════════════════════════════
       METRICS (streamlit native)
    ═══════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: var(--white) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--mist) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: var(--navy) !important;
        font-size: 24px !important;
    }

    /* ═══════════════════════════════════════════════
       GENERAL OVERRIDES
    ═══════════════════════════════════════════════ */
    p, span, label, li { color: var(--ink); }
    h1, h2, h3 {
        color: var(--navy) !important;
        font-family: 'Playfair Display', serif !important;
    }
    hr { border-color: var(--border) !important; }

    /* Inline code rendered by st.write_stream — make it look like normal text */
    code {
        background: transparent !important;
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: inherit !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
    }
    [data-testid="stSpinner"] p { color: var(--steel) !important; }
    [data-testid="stCaptionContainer"] p, small { color: var(--mist) !important; }
    [data-testid="stButton"] button {
        background: var(--navy) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
    }
    [data-testid="stButton"] button:hover { opacity: 0.85 !important; }
    </style>
    """, unsafe_allow_html=True)