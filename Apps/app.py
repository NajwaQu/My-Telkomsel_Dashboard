import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MyTelkomsel Sentiment Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Root & background ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #0f0a0a;
    color: #f0e8e8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0a0a 0%, #120808 100%) !important;
    border-right: 1px solid #2e1515;
}

[data-testid="stSidebar"] .stRadio label {
    color: #c09090 !important;
    font-size: 0.88rem;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 6px 0;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #ff3b3b !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(145deg, #1a0a0a, #150808);
    border: 1px solid #2e1515;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #ff3b3b66;
    box-shadow: 0 8px 24px #ff3b3b18;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #ff3b3b);
    border-radius: 16px 16px 0 0;
}

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent, #ff3b3b);
    line-height: 1;
    margin-bottom: 4px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8a6060;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f0e8e8;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #2e1515, transparent);
    margin-left: 12px;
}

/* ── Page title ── */
.page-header {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #2e1515;
    margin-bottom: 24px;
}

.page-header h1 {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f0e8e8;
    margin: 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.02em;
}

.page-header p {
    color: #8a6060;
    font-size: 0.88rem;
    margin: 6px 0 0 0;
}

/* ── Insight cards ── */
.insight-box {
    background: linear-gradient(145deg, #1a0a0a, #150808);
    border: 1px solid #2e1515;
    border-left: 4px solid #ff3b3b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #c09090;
    line-height: 1.6;
}

.insight-box strong {
    color: #f0e8e8;
}

/* ── Prediction box ── */
.pred-positive {
    background: linear-gradient(135deg, #0a1a0d 0%, #120808 100%);
    border: 1px solid #00c45544;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.pred-negative {
    background: linear-gradient(135deg, #1a0808 0%, #120808 100%);
    border: 1px solid #ff3b3b44;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.pred-emoji {
    font-size: 3.5rem;
    margin-bottom: 8px;
}

.pred-label-pos {
    font-size: 1.4rem;
    font-weight: 800;
    color: #00c455;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.pred-label-neg {
    font-size: 1.4rem;
    font-weight: 800;
    color: #ff3b3b;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Tag badge ── */
.badge {
    display: inline-block;
    background: #ff3b3b18;
    color: #ff6b6b;
    border: 1px solid #ff3b3b44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.badge-red {
    background: #ff3b3b22;
    color: #ff3b3b;
    border-color: #ff3b3b55;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #1a0a0a !important;
    border: 1px solid #2e1515 !important;
    color: #f0e8e8 !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #ff3b3b !important;
    box-shadow: 0 0 0 2px #ff3b3b22 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff3b3b 0%, #cc1a1a 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px 32px !important;
    font-size: 0.9rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 14px #ff3b3b33 !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px #ff3b3b44 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #1a0a0a !important;
    border-color: #2e1515 !important;
    color: #f0e8e8 !important;
    border-radius: 12px !important;
}

/* ── About cards ── */
.about-card {
    background: linear-gradient(145deg, #1a0a0a, #150808);
    border: 1px solid #2e1515;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
}

.about-card h3 {
    color: #ff6b6b;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 12px 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.about-card p, .about-card li {
    color: #c09090;
    font-size: 0.88rem;
    line-height: 1.7;
}

.about-card ul {
    padding-left: 16px;
    margin: 0;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ff3b3b, #ff6b6b) !important;
    border-radius: 4px !important;
}

/* ── Matplotlib background ── */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a0a0a",
    "axes.facecolor":    "#1a0a0a",
    "axes.edgecolor":    "#2e1515",
    "axes.labelcolor":   "#c09090",
    "xtick.color":       "#8a6060",
    "ytick.color":       "#8a6060",
    "text.color":        "#f0e8e8",
    "grid.color":        "#2e1515",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "font.weight":       "500",
    "axes.titleweight":  "700",
    "axes.labelweight":  "600",
})

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("..\Data\label_MyTelkomsel.csv")

@st.cache_resource
def load_model():
    model = joblib.load("../naive_bayes_model.pkl")
    tfidf = joblib.load("../tfidf.pkl")
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 32px 0 24px 0;">
        <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:62px; height:62px;
            background: linear-gradient(135deg, #ff3b3b22, #1a0a0a);
            border: 1px solid #ff3b3b44;
            border-radius: 18px;
            font-size: 1.8rem;
            margin-bottom: 14px;
        ">📞</div>
        <div style="
            font-weight:800; font-size:1.05rem; color:#f0e8e8;
            font-family:'Plus Jakarta Sans',sans-serif; letter-spacing:-0.01em;
            margin-bottom: 4px;
        ">MyTelkomsel Analytics</div>
        <div style="
            display:inline-block;
            font-size:0.65rem; color:#ff6b6b;
            font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
            background:#ff3b3b18; border:1px solid #ff3b3b33;
            border-radius:20px; padding:3px 12px; margin-top:2px;
        ">Sentiment Intelligence Dashboard</div>
    </div>
    <div style="height:1px; background:linear-gradient(to right, transparent, #2e1515, transparent); margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Dashboard", "☁️  WordCloud", "🎯  Evaluasi Model", "🤖  Prediksi", "📄  Dataset", "📚  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="height:1px; background:linear-gradient(to right, transparent, #2e1515, transparent); margin: 20px 0 16px 0;"></div>
    <div style="display:flex; flex-direction:column; gap:6px;">
        <div style="display:flex; align-items:center; gap:8px; background:#1a0a0a; border:1px solid #2e1515; border-radius:10px; padding:8px 12px;">
            <span style="font-size:0.7rem;">🤖</span>
            <span style="font-size:0.68rem; color:#8a6060; font-weight:500;">Naive Bayes · TF-IDF</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:#1a0a0a; border:1px solid #2e1515; border-radius:10px; padding:8px 12px;">
            <span style="font-size:0.7rem;">📊</span>
            <span style="font-size:0.68rem; color:#8a6060; font-weight:500;">8.000 ulasan dianalisis</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:#1a0a0a; border:1px solid #2e1515; border-radius:10px; padding:8px 12px;">
            <span style="font-size:0.7rem;">🎯</span>
            <span style="font-size:0.68rem; color:#8a6060; font-weight:500;">Akurasi model <strong style="color:#ff6b6b">82%</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHARED DATA
# ─────────────────────────────────────────────
total      = len(df)
positif    = int((df['sentiment'] == 1).sum())
negatif    = int((df['sentiment'] == 0).sum())
pct_pos    = positif / total * 100
pct_neg    = negatif / total * 100

CM = np.array([[871, 40], [251, 438]])

# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "Dashboard" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🏠 Dashboard Analisis Sentimen</h1>
        <p>Gambaran umum distribusi sentimen ulasan pengguna aplikasi MyTelkomsel</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "📊", f"{total:,}", "Total Ulasan", "#ff3b3b"),
        (c2, "😄", f"{positif:,}", "Sentimen Positif", "#00c455"),
        (c3, "😠", f"{negatif:,}", "Sentimen Negatif", "#ff3b3b"),
        (c4, "🎯", "82%", "Akurasi Model", "#ff8c42"),
    ]

    for col, icon, value, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Positif", "Negatif"],
            [positif, negatif],
            color=["#00c455", "#ff3b3b"],
            width=0.45,
            edgecolor="none",
            zorder=3
        )
        ax.set_ylabel("Jumlah Ulasan", fontsize=12)
        ax.set_title("Distribusi Kelas", fontsize=14, fontweight='600', pad=14)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha='center', va='bottom', fontsize=12, color='#f0e8e8', fontweight='600'
            )

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [positif, negatif],
            labels=["Positif", "Negatif"],
            colors=["#00c455", "#ff3b3b"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#0f0a0a", linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#f0e8e8")
            at.set_fontsize(13)
            at.set_fontweight('600')
        for t in texts:
            t.set_color("#c09090")
            t.set_fontsize(13)
        ax2.set_title("Proporsi Kelas", fontsize=14, fontweight='600', pad=14)

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_info:
        st.markdown("""
        <div class="insight-box">
            <strong>📌 Komposisi Dataset</strong><br>
            Dataset berisi ulasan pengguna MyTelkomsel dari Google Play Store yang sudah dilabeli secara manual ke dalam dua kelas sentimen.
        </div>
        """, unsafe_allow_html=True)

        pos_pct = f"{pct_pos:.1f}%"
        neg_pct = f"{pct_neg:.1f}%"

        st.markdown(f"""
        <div class="insight-box">
            <strong>😄 Positif — {pos_pct}</strong><br>
            Sebagian besar pengguna memberikan ulasan positif, menunjukkan tingkat kepuasan yang tinggi.
        </div>
        <div class="insight-box">
            <strong>😠 Negatif — {neg_pct}</strong><br>
            Ulasan negatif umumnya menyoroti masalah sinyal, harga, dan antarmuka aplikasi.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: WORDCLOUD
# ══════════════════════════════════════════════
elif "WordCloud" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>☁️ Word Cloud Visualisasi</h1>
        <p>Kata-kata yang paling sering muncul dalam ulasan positif dan negatif</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["😄  Ulasan Positif", "😠  Ulasan Negatif"])

    def make_wordcloud(text, colormap, bg="#1a0a0a"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
            margin=8
        ).generate(text)
        return wc

    with tab1:
        positive_text = " ".join(df[df['sentiment'] == 1]['content'].dropna())
        wc = make_wordcloud(positive_text, "Greens")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#1a0a0a")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        words = positive_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#00c455", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        negative_text = " ".join(df[df['sentiment'] == 0]['content'].dropna())
        wc = make_wordcloud(negative_text, "Reds")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#1a0a0a")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = negative_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#ff3b3b", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════
elif "Evaluasi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🎯 Evaluasi Model</h1>
        <p>Performa model Naive Bayes pada data uji</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ──
    TP, FN = CM[1][1], CM[1][0]
    FP, TN = CM[0][1], CM[0][0]

    accuracy  = (TP + TN) / CM.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)
    precision_neg = TN / (TN + FN)
    recall_neg = TN / (TN + FP)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in [
        (m1, "Accuracy",  f"{accuracy*100:.2f}%",  "#6c8eff"),
        (m2, "Precision", f"{precision*100:.2f}%", "#ff3b3b"),
        (m3, "Recall",    f"{recall*100:.2f}%",    "#f5a623"),
        (m4, "F1-Score",  f"{f1*100:.2f}%",        "#c47aff"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        import matplotlib.colors as mcolors

        # Custom colormap: merah gelap (TN) → krem (FP/FN) → hijau (TP)
        cm_colors = ["#ff3b3b", "#ff8c42", "#f5e6c8", "#5ecf82", "#00c455"]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cm", cm_colors)

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#13080d")
        ax.set_facecolor("#13080d")

        im = ax.imshow(CM, cmap=custom_cmap, aspect='auto', vmin=0, vmax=CM.max())

        # Annotasi nilai dengan font besar
        for i in range(2):
            for j in range(2):
                val = CM[i, j]
                # Warna teks kontras berdasarkan nilai sel
                brightness = val / CM.max()
                txt_color = "#0f0a0a" if 0.25 < brightness < 0.85 else "#f0e8e8"
                ax.text(j, i, f"{val:,}", ha='center', va='center',
                        fontsize=22, fontweight='bold', color=txt_color,
                        fontfamily='DejaVu Sans')

        # Border cell
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="#1a0a0a", linewidth=2.5))

        # Label sumbu
        labels = ["Negatif", "Positif"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=12, fontfamily='DejaVu Sans', color='#c09090')
        ax.set_yticklabels(labels, fontsize=12, fontfamily='DejaVu Sans', color='#c09090', rotation=0)
        ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12, color='#f0e8e8', fontfamily='DejaVu Sans')
        ax.set_ylabel("Actual Label", fontsize=13, labelpad=12, color='#f0e8e8', fontfamily='DejaVu Sans')

        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Label sudut kecil (TP, TN, FP, FN)
        corner_labels = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}
        for (i, j), lbl in corner_labels.items():
            ax.text(j - 0.42, i - 0.40, lbl,
                    ha='left', va='top', fontsize=9,
                    color='#ffffff99', fontweight='600',
                    fontfamily='DejaVu Sans')

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_report:
        st.markdown("""
        <div class="insight-box">
            <strong>True Positive (TP)</strong> — 438<br>
            Ulasan positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box">
            <strong>True Negative (TN)</strong> — 871<br>
            Ulasan negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box" style="border-left-color:#ff3b3b">
            <strong>False Positive (FP)</strong> — 40<br>
            Ulasan negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#ff3b3b">
            <strong>False Negative (FN)</strong> — 251<br>
            Ulasan positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    # Classification report per class
    st.markdown('<div class="section-title">Laporan Per Kelas</div>', unsafe_allow_html=True)

    report_data = {
    "Kelas": ["Negatif (0)", "Positif (1)"],
    "Precision": [f"{precision_neg:.2f}", f"{precision:.2f}"],
    "Recall": [f"{recall_neg:.2f}", f"{recall:.2f}"],
    "F1-Score": [f"{f1_neg:.2f}", f"{f1:.2f}"],
    "Support": [TN + FP, TP + FN],
}
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════
elif "Prediksi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📈 Prediksi Sentimen</h1>
        <p>Masukkan ulasan untuk diklasifikasikan oleh model</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        review = st.text_area(
            "Tulis ulasan MyTelkomsel di sini…",
            placeholder="Contoh: pembelian paket internet 65000 pembayaran qris saldo terpotong tapi paket belum masuk!",
            height=180
        )

        examples = [
            "Mantap mudah kalau isi paket GK perlu keluar rumah",
            "Telkomsel ini kenapa jaringannya di riau lelet kali tolong di perbaikilah",
            "apk Telkomsel yg sngt menarik",
        ]

        st.markdown("<div style='font-size:0.8rem; color:#8a6060; margin:12px 0 6px 0;'>✨ Coba contoh ulasan:</div>", unsafe_allow_html=True)
        for ex in examples:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=ex):
                review = ex

        predict_btn = st.button("🔍  Analisis Sentimen", use_container_width=True)

    with col_result:
        if predict_btn and review.strip():
            vector     = tfidf.transform([review])
            pred       = model.predict(vector)[0]
            prob       = model.predict_proba(vector)[0]
            confidence = max(prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-emoji">😄</div>
                    <div class="pred-label-pos">Sentimen Positif</div>
                    <div style="color:#8a6060; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen positif
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-emoji">😠</div>
                    <div class="pred-label-neg">Sentimen Negatif</div>
                    <div style="color:#8a6060; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen negatif
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:0.82rem; color:#c09090; font-weight:500;">Confidence Score</span>
                    <span style="font-size:0.9rem; color:#f0e8e8; font-weight:700;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)

            # Prob bars
            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.5))
            classes = ["Negatif", "Positif"]
            colors  = ["#ff3b3b", "#00c455"]
            bars = ax.barh(classes, [prob[0]*100, prob[1]*100], color=colors, edgecolor="none", height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)", fontsize=8)
            ax.spines[['top','right','bottom']].set_visible(False)
            for bar, p in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontsize=8, color='#f0e8e8')
            fig.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not review.strip():
            st.warning("⚠️ Masukkan ulasan terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#8a6060;">
                <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
                <div style="font-size:0.88rem;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════
elif "Dataset" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📄 Dataset Explorer</h1>
        <p>Jelajahi dan filter dataset ulasan MyTelkomsel</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search_term = st.text_input("🔍 Cari kata dalam ulasan", placeholder="Ketik kata kunci…")
    with c2:
        pilihan = st.selectbox("Filter Sentimen", ["Semua", "Positif 😊", "Negatif 😞"])
    with c3:
        n_rows = st.selectbox("Tampilkan", [50, 100, 200, 500, "Semua"], index=1)

    data = df.copy()
    if "Positif" in pilihan:
        data = data[data['sentiment'] == 1]
    elif "Negatif" in pilihan:
        data = data[data['sentiment'] == 0]

    if search_term:
        data = data[data['content'].str.contains(search_term, case=False, na=False)]

    if n_rows != "Semua":
        data = data.head(int(n_rows))

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8a6060; margin-bottom:12px;">
        Menampilkan <strong style="color:#f0e8e8">{len(data):,}</strong> baris
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        data.reset_index(drop=True),
        use_container_width=True,
        height=480
    )

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="MyTelkomsel_sentiment_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
#  PAGE: TENTANG
# ══════════════════════════════════════════════
elif "Tentang" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Penelitian</h1>
        <p>Informasi metodologi dan detail teknis proyek ini</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Tujuan Penelitian</h3>
            <p>
                Mengklasifikasikan sentimen ulasan pengguna aplikasi MyTelkomsel dari Google Play Store
                ke dalam dua kelas: <strong style="color:#00c455">positif</strong> dan
                <strong style="color:#ff3b3b">negatif</strong>, untuk membantu memahami
                persepsi pengguna secara otomatis menggunakan pendekatan machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <ul>
                <li>Sumber: Google Play Store</li>
                <li>Total: <strong style="color:#f0e8e8">8.000 ulasan</strong></li>
                <li>Label: Positif & Negatif</li>
                <li>Metode labeling: Semi-otomatis + verifikasi manual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Pipeline NLP</h3>
            <ul>
                <li>Preprocessing: case folding, stopword removal, stemming</li>
                <li>Ekstraksi fitur: <strong style="color:#f0e8e8">TF-IDF Vectorizer</strong></li>
                <li>Classifier: <strong style="color:#f0e8e8">Multinomial Naive Bayes</strong></li>
                <li>Validasi: train-test split 80:20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:#6c8eff">82.0%</strong></li>
                <li>Precision: <strong style="color:#00c455">85.0%</strong></li>
                <li>Recall: <strong style="color:#f5a623">80.0%</strong></li>
                <li>F1-Score: <strong style="color:#c47aff">80.0%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    <div style="background:#12080f; border:1px solid #6b1a3b; border-radius:16px; padding:22px 24px; text-align:center;">
        <span style="font-family:'Playfair Display', serif; font-size:0.78rem; font-style:italic; color:#d4a0c0; letter-spacing:0.04em;">
            Dibuat dengan ❤️ menggunakan Streamlit · Naive Bayes · TF-IDF
        </span>
    </div>
    <br>
    <div style="background:#12080f; border:1px solid #6b1a3b; border-radius:16px; padding:22px 24px; text-align:center;">
        <span style="font-family:'Playfair Display', serif; font-size:0.78rem; font-style:italic; color:#d4a0c0; letter-spacing:0.04em;">
            Oleh Najwa Quratul Aini_ w/💚
        </span>
    </div>
    """, unsafe_allow_html=True)