"""
app.py — AI Survey GPA Predictor
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Survey — GPA Predictor",
    page_icon="🤖",
    layout="wide",
)

# ─── Theme colours ────────────────────────────────────────────────────────────
LIME  = "#C9F01E"
DARK  = "#1A1A2E"
TEAL  = "#028090"
RED   = "#F96167"

st.markdown(f"""
<style>
  .main {{ background-color: #F5F5F0; }}
  h1, h2, h3 {{ color: {DARK}; }}
  .metric-card {{
      background: {LIME}; border-radius: 12px;
      padding: 18px 22px; text-align: center;
      box-shadow: 3px 3px 8px rgba(0,0,0,.15);
  }}
  .metric-val {{ font-size: 2.4rem; font-weight: 900; color: {DARK}; margin: 0; }}
  .metric-lbl {{ font-size: .85rem; font-weight: 700; color: #333; margin: 0; }}
  .result-high {{
      background: {LIME}; color: {DARK}; border-radius: 12px;
      padding: 20px; font-size: 1.4rem; font-weight: 900; text-align: center;
  }}
  .result-low {{
      background: {RED}; color: white; border-radius: 12px;
      padding: 20px; font-size: 1.4rem; font-weight: 900; text-align: center;
  }}
  .stSlider > div > div > div {{ background: {LIME} !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Data & mappings ──────────────────────────────────────────────────────────
FREQ_MAP  = {"Never": 0, "Sometimes": 1, "Often": 2, "Daily": 3}
COPY_MAP  = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
HOURS_MAP = {"0 hours": 0, "less than 1 hours": 0.5,
             "1-3 hours": 2, "4-6 hours": 5, "More than 6 hours": 7}

FEATURES = [
    "ai_frequency_enc", "hours_per_week_enc", "copy_behavior_enc",
    "improved_grades",  "cannot_complete_without_ai", "critical_thinking",
]
FEATURE_LABELS = [
    "AI Frequency", "Hours / Week", "Copy Behavior",
    "Improved Grades", "AI Dependence", "Critical Thinking",
]

@st.cache_data
def load_and_train():
    """Load dataset, encode, fit model — cached so it only runs once."""
    raw = pd.read_csv("AI_Survey_Dataset_tp3.csv", skiprows=2)
    raw.columns = [
        "No", "ai_frequency", "hours_per_week", "copy_behavior",
        "improved_grades", "cannot_complete_without_ai", "critical_thinking", "gpa"
    ]
    df = raw.drop(columns=["No"]).dropna(subset=["gpa"]).copy()
    df["ai_frequency_enc"]   = df["ai_frequency"].map(FREQ_MAP)
    df["hours_per_week_enc"] = df["hours_per_week"].map(HOURS_MAP)
    df["copy_behavior_enc"]  = df["copy_behavior"].map(COPY_MAP)
    df["gpa_high"] = (df["gpa"] >= 3.5).astype(int)
    X = df[FEATURES]
    y = df["gpa_high"]
    mdl = LogisticRegression(random_state=42, max_iter=1000)
    mdl.fit(X, y)
    return df, mdl

df, model = load_and_train()
y      = df["gpa_high"]
y_pred = model.predict(df[FEATURES])
acc    = accuracy_score(y, y_pred)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:{DARK}; border-radius:14px; padding:28px 36px; margin-bottom:24px;">
  <h1 style="color:{LIME}; margin:0; font-size:2.1rem;">🤖 AI Tools Usage — GPA Predictor</h1>
  <p style="color:#ccc; margin:6px 0 0 0; font-size:1rem;">
    Logistic Regression on AI Survey Dataset &nbsp;|&nbsp; 57 students &nbsp;|&nbsp;
    Target: GPA ≥ 3.5 (High)
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict GPA",
    "📊 Model Results",
    "📈 EDA Charts",
    "⚙️ Model Settings",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🎯 Enter Student Parameters to Predict GPA Class")
    st.markdown("Adjust the sliders and dropdowns below, then click **Predict**.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📱 AI Usage Behaviour")
        ai_freq = st.selectbox(
            "How often do you use AI tools for academic purposes?",
            options=list(FREQ_MAP.keys()), index=3,
            help="Never / Sometimes / Often / Daily"
        )
        hours = st.selectbox(
            "How many hours per week do you use AI for studying?",
            options=list(HOURS_MAP.keys()), index=2,
            help="Select your weekly AI study hours"
        )
        copy = st.selectbox(
            "How often do you copy AI content directly into assignments?",
            options=list(COPY_MAP.keys()), index=2,
            help="Never → Always"
        )

    with col2:
        st.markdown("#### 🧠 Perception & Self-Assessment")
        improved = st.slider(
            "AI has improved my academic grades",
            min_value=1, max_value=5, value=3,
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )
        dependence = st.slider(
            "I cannot complete assignments without AI tools",
            min_value=1, max_value=5, value=3,
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )
        critical = st.slider(
            "AI has improved my critical thinking & problem solving",
            min_value=1, max_value=5, value=3,
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict GPA Class", use_container_width=True, type="primary")

    if predict_btn:
        input_vec = np.array([[
            FREQ_MAP[ai_freq],
            HOURS_MAP[hours],
            COPY_MAP[copy],
            improved,
            dependence,
            critical,
        ]])
        pred       = model.predict(input_vec)[0]
        prob_high  = model.predict_proba(input_vec)[0][1]
        prob_low   = 1 - prob_high

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            if pred == 1:
                st.markdown(f'<div class="result-high">🎓 High GPA Predicted<br>(≥ 3.5)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-low">📉 Low GPA Predicted<br>(< 3.5)</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
              <p class="metric-val">{prob_high*100:.1f}%</p>
              <p class="metric-lbl">Probability — High GPA</p>
            </div>""", unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card" style="background:{RED if prob_low > 0.5 else '#eee'}">
              <p class="metric-val" style="color:{'white' if prob_low>0.5 else DARK}">{prob_low*100:.1f}%</p>
              <p class="metric-lbl" style="color:{'white' if prob_low>0.5 else '#333'}">Probability — Low GPA</p>
            </div>""", unsafe_allow_html=True)

        # Gauge bar
        st.markdown("#### Confidence Bar")
        fig_g, ax_g = plt.subplots(figsize=(9, 1.2))
        ax_g.barh([""], [prob_high], color=LIME, height=0.5, label="High GPA")
        ax_g.barh([""], [prob_low], left=[prob_high], color=RED, height=0.5, label="Low GPA")
        ax_g.axvline(0.5, color="white", linewidth=2, linestyle="--")
        ax_g.set_xlim(0, 1)
        ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_g.set_xticklabels(["0%","25%","50%","75%","100%"])
        ax_g.set_facecolor(DARK)
        fig_g.patch.set_facecolor(DARK)
        ax_g.tick_params(colors="white")
        ax_g.legend(loc="upper right", fontsize=9, facecolor=DARK, labelcolor="white")
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        # Input summary
        with st.expander("📋 Your Input Summary"):
            summary = pd.DataFrame({
                "Feature": FEATURE_LABELS,
                "Raw Input": [ai_freq, hours, copy, improved, dependence, critical],
                "Encoded Value": input_vec[0].tolist(),
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL RESULTS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Logistic Regression — Full Dataset Results")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("66.7%", "Accuracy"),
        ("57", "Students"),
        ("44%", "High GPA"),
        ("6", "Features"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <p class="metric-val">{val}</p>
          <p class="metric-lbl">{lbl}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Pred Low","Pred High"],
                    yticklabels=["Actual Low","Actual High"],
                    annot_kws={"size": 16, "weight": "bold"}, linewidths=1)
        ax_cm.set_title(f"Accuracy = {acc:.1%}", fontweight="bold")
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with r2:
        st.markdown("#### Feature Coefficients")
        coef_s = pd.Series(model.coef_[0], index=FEATURE_LABELS).sort_values()
        fig_c, ax_c = plt.subplots(figsize=(5, 4))
        bar_cols = [RED if v < 0 else LIME for v in coef_s]
        coef_s.plot(kind="barh", ax=ax_c, color=bar_cols, edgecolor=DARK, linewidth=1)
        ax_c.axvline(0, color=DARK, linewidth=1.5, linestyle="--")
        ax_c.set_title("Coefficients (Impact on High GPA)", fontweight="bold")
        ax_c.set_xlabel("Coefficient Value")
        for i, v in enumerate(coef_s.values):
            offset = 0.005 if v >= 0 else -0.005
            ha = "left" if v >= 0 else "right"
            ax_c.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    st.markdown("#### Classification Report")
    report = classification_report(y, y_pred, target_names=["Low GPA","High GPA"], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 Exploratory Data Analysis")

    e1, e2 = st.columns(2, gap="large")

    with e1:
        st.markdown("#### GPA Distribution")
        fig_gpa, ax_gpa = plt.subplots(figsize=(5.5, 3.5))
        ax_gpa.hist(df["gpa"], bins=8, color=LIME, edgecolor=DARK, linewidth=1.5)
        ax_gpa.axvline(df["gpa"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={df['gpa'].mean():.2f}")
        ax_gpa.axvline(3.5, color="orange", linestyle=":", linewidth=2, label="Threshold=3.5")
        ax_gpa.set_xlabel("GPA"); ax_gpa.set_ylabel("Count")
        ax_gpa.legend(); ax_gpa.set_title("GPA Distribution", fontweight="bold")
        st.pyplot(fig_gpa, use_container_width=True)
        plt.close(fig_gpa)

    with e2:
        st.markdown("#### Avg GPA by AI Frequency")
        freq_order = ["Never","Sometimes","Often","Daily"]
        gpa_freq = df.groupby("ai_frequency")["gpa"].mean().reindex(freq_order)
        fig_f, ax_f = plt.subplots(figsize=(5.5, 3.5))
        ax_f.bar(gpa_freq.index, gpa_freq.values, color=DARK, edgecolor=LIME, linewidth=2)
        ax_f.set_ylim(2.5, 4.0); ax_f.set_ylabel("Avg GPA")
        ax_f.set_title("Avg GPA by AI Frequency", fontweight="bold")
        for i, v in enumerate(gpa_freq.values):
            ax_f.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", color=DARK)
        st.pyplot(fig_f, use_container_width=True)
        plt.close(fig_f)

    e3, e4 = st.columns(2, gap="large")

    with e3:
        st.markdown("#### Correlation Heatmap")
        num_cols = ["ai_frequency_enc","hours_per_week_enc","copy_behavior_enc",
                    "improved_grades","cannot_complete_without_ai","critical_thinking","gpa"]
        lbls = ["AI Freq","Hours","Copy","Grades","Depend","Critical","GPA"]
        fig_h, ax_h = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=lbls, yticklabels=lbls, ax=ax_h, annot_kws={"size":8})
        ax_h.set_title("Correlation Heatmap", fontweight="bold")
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

    with e4:
        st.markdown("#### Target Class Distribution")
        counts = df["gpa_high"].value_counts().rename({0:"Low GPA (<3.5)",1:"High GPA (≥3.5)"})
        fig_p, ax_p = plt.subplots(figsize=(5.5, 4.5))
        ax_p.pie(counts.values, labels=counts.index,
                 colors=[RED, LIME], autopct="%1.1f%%", startangle=90,
                 wedgeprops={"edgecolor":"white","linewidth":2},
                 textprops={"fontsize":11,"fontweight":"bold"})
        ax_p.set_title("GPA Target Distribution", fontweight="bold")
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

    st.markdown("#### Raw Data Preview")
    st.dataframe(df[["ai_frequency","hours_per_week","copy_behavior",
                      "improved_grades","cannot_complete_without_ai",
                      "critical_thinking","gpa","gpa_high"]].head(15),
                 use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ⚙️ Retrain Model with Custom Parameters")
    st.info("Adjust hyperparameters and threshold below, then click **Retrain**.")

    s1, s2 = st.columns(2, gap="large")

    with s1:
        st.markdown("#### Logistic Regression Hyperparameters")
        c_val       = st.select_slider("Regularization C (higher = less regularization)",
                                       options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0], value=1.0)
        solver      = st.selectbox("Solver", ["lbfgs","liblinear","saga","newton-cg"], index=0)
        max_iter    = st.slider("Max Iterations", 100, 2000, 1000, 100)
        penalty     = st.selectbox("Penalty", ["l2","l1","none"],
                                   help="l1 requires liblinear/saga solver")

    with s2:
        st.markdown("#### GPA Threshold & Features")
        threshold   = st.slider("GPA classification threshold (High ≥ X)", 2.5, 4.0, 3.5, 0.1)
        selected_feats = st.multiselect(
            "Select features to include",
            options=FEATURE_LABELS, default=FEATURE_LABELS,
        )

    retrain_btn = st.button("🔄 Retrain Model", use_container_width=True, type="primary")

    if retrain_btn:
        if len(selected_feats) == 0:
            st.error("Please select at least one feature.")
        else:
            # Map selected labels back to column names
            label_to_col = dict(zip(FEATURE_LABELS, FEATURES))
            sel_cols = [label_to_col[l] for l in selected_feats]

            y_custom = (df["gpa"] >= threshold).astype(int)
            X_custom = df[sel_cols]

            try:
                penalty_val = None if penalty == "none" else penalty
                mdl2 = LogisticRegression(
                    C=c_val, solver=solver, max_iter=max_iter,
                    penalty=penalty_val, random_state=42
                )
                mdl2.fit(X_custom, y_custom)
                pred2 = mdl2.predict(X_custom)
                acc2  = accuracy_score(y_custom, pred2)

                st.success(f"✅ Retrained! Accuracy = **{acc2:.1%}**  |  Threshold = GPA ≥ {threshold}  |  Features = {len(sel_cols)}")

                rc1, rc2 = st.columns(2, gap="large")
                with rc1:
                    cm2 = confusion_matrix(y_custom, pred2)
                    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=ax2,
                                xticklabels=["Pred Low","Pred High"],
                                yticklabels=["Actual Low","Actual High"],
                                annot_kws={"size":14,"weight":"bold"})
                    ax2.set_title(f"Confusion Matrix (Acc={acc2:.1%})", fontweight="bold")
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                with rc2:
                    coef2 = pd.Series(mdl2.coef_[0], index=selected_feats).sort_values()
                    fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
                    bc = [RED if v < 0 else LIME for v in coef2]
                    coef2.plot(kind="barh", ax=ax3, color=bc, edgecolor=DARK)
                    ax3.axvline(0, color=DARK, linewidth=1.5, linestyle="--")
                    ax3.set_title("Coefficients (Custom Model)", fontweight="bold")
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

                st.markdown("**Classification Report:**")
                rep2 = classification_report(y_custom, pred2,
                                             target_names=["Low GPA","High GPA"],
                                             output_dict=True)
                st.dataframe(pd.DataFrame(rep2).T.round(3), use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}. Try changing the solver (e.g. l1 requires liblinear).")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<p style="text-align:center; color:{"#64748B"}; font-size:.85rem;">
  AI Survey GPA Predictor &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp;
  57 students &nbsp;|&nbsp; Built with Streamlit
</p>
""", unsafe_allow_html=True)
