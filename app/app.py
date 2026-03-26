import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import json

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]

SEVERITY = {
    "Blackheads":("Mild",     "#27AE60", "🟢"),
    "Whiteheads":("Mild",     "#27AE60", "🟢"),
    "Papules":   ("Moderate", "#F39C12", "🟡"),
    "Pustules":  ("Moderate", "#F39C12", "🟡"),
    "Cyst":      ("Severe",   "#E74C3C", "🔴"),
}
ADVICE = {
    "Blackheads": "Use salicylic acid cleanser daily. Avoid heavy face oils.",
    "Whiteheads": "Non-comedogenic moisturizer. Gentle cleanser. Try retinoids.",
    "Papules":    "Apply benzoyl peroxide. Do NOT squeeze or pop.",
    "Pustules":   "Topical antibiotics. Keep hands off. See dermatologist.",
    "Cyst":       "See dermatologist immediately. May need professional drainage."
}
DESCRIPTION = {
    "Blackheads": "Open comedones. Dark spots caused by oxidized melanin.",
    "Whiteheads": "Closed comedones. Small white bumps under skin.",
    "Papules":    "Inflamed red bumps without pus. Do not squeeze.",
    "Pustules":   "Pus-filled red bumps. Classic pimples.",
    "Cyst":       "Deep pus-filled lesion. Most severe type. Can cause scarring."
}

st.set_page_config(page_title="AcneAI", page_icon="🧴", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2.5rem;font-weight:bold;text-align:center;
            background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub{text-align:center;color:#888;margin-bottom:1.5rem;}
.advice{background:#EBF5FB;border-left:4px solid #3498DB;
        padding:1rem;border-radius:0 10px 10px 0;margin:0.8rem 0;}
.card{background:white;padding:1rem;border-radius:10px;text-align:center;
      box-shadow:0 2px 8px rgba(0,0,0,0.08);}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧴 AcneAI")
    st.markdown("---")
    threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Backbone: EfficientNetB0")
    st.markdown("- Classes: 5 acne types")
    st.markdown("- Input: 224×224 px")
    st.markdown("- Method: Transfer Learning")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only.")

st.markdown('<div class="main-title">🧴 AcneAI Skin Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Real Dataset · EfficientNetB0 · Transfer Learning</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_acne_model.keras")

uploaded = st.file_uploader("📤 Upload a skin image", type=["jpg","jpeg","png"])

if uploaded:
    model     = load_model()
    image     = Image.open(uploaded).convert("RGB")
    img_r     = image.resize((224, 224))
    # Use EfficientNet preprocessing (same as training)
    img_arr   = np.array(img_r, dtype=np.float32)
    img_arr   = tf.keras.applications.efficientnet.preprocess_input(img_arr)
    img_batch = np.expand_dims(img_arr, 0)

    with st.spinner("🔍 Analyzing skin image..."):
        probs = model.predict(img_batch, verbose=0)[0]

    pred_idx   = np.argmax(probs)
    pred_cls   = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    sev, color, emoji = SEVERITY[pred_cls]

    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown(f"<h2 style='color:{color};'>{emoji} {pred_cls}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.progress(int(confidence))
        st.markdown(
            f"**Severity:** <span style='color:{color};font-weight:bold;'>{sev}</span>",
            unsafe_allow_html=True
        )
        st.info(f"📋 {DESCRIPTION[pred_cls]}")
        st.markdown(
            f'<div class="advice">💡 <b>Recommended Action:</b><br>{ADVICE[pred_cls]}</div>',
            unsafe_allow_html=True
        )
        if confidence < threshold * 100:
            st.warning("⚠️ Low confidence. Please consult a dermatologist.")

    st.markdown("---")
    st.subheader("📊 All Class Probabilities")

    bar_colors = [color if i == pred_idx else "#BDC3C7" for i in range(len(CLASS_NAMES))]
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=[float(p)*100 for p in probs],
        marker_color=bar_colors,
        text=[f"{float(p)*100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        yaxis=dict(range=[0,115], title="Probability (%)"),
        xaxis_title="Acne Type",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 3 Predictions")
    top3    = np.argsort(probs)[-3:][::-1]
    medals  = ["🥇","🥈","🥉"]
    c1,c2,c3 = st.columns(3)
    for col, idx, medal in zip([c1,c2,c3], top3, medals):
        s, sc, em = SEVERITY[CLASS_NAMES[idx]]
        with col:
            st.metric(
                label=f"{medal} {CLASS_NAMES[idx]}",
                value=f"{float(probs[idx])*100:.1f}%",
                delta=f"{s} severity"
            )

    result = {
        "predicted_class": pred_cls,
        "confidence":      round(confidence, 2),
        "severity":        sev,
        "probabilities":   {CLASS_NAMES[i]: round(float(p)*100,2) for i,p in enumerate(probs)}
    }
    st.download_button("📥 Download Result (JSON)",
                       json.dumps(result, indent=2),
                       "acne_result.json", "application/json")

else:
    st.info("👆 Upload a skin image above to get started")
    st.markdown("---")
    st.subheader("📚 Acne Type Reference Guide")
    cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        sev, color, emoji = SEVERITY[cls]
        with cols[i]:
            st.markdown(f"""
            <div style="background:#fafafa;padding:1rem;border-radius:10px;
                        border-top:4px solid {color};text-align:center;">
                <div style="font-size:1.5rem">{emoji}</div>
                <b style="color:{color};">{cls}</b><br>
                <small style="color:#888;">{sev}</small><br>
                <small>{DESCRIPTION[cls]}</small>
            </div>
            """, unsafe_allow_html=True)
