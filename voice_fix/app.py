"""
Smart Customer Analytics Assistant
====================================
A business intelligence tool for retail store managers.

Pages
-----
  1. Ask the Assistant  — voice + text chatbot (voice auto-sends, zero copy-paste)
  2. Sentiment Monitor  — BERT review sentiment with live alert
  3. Sales Dashboard    — product analytics with AI insight captions
  4. Sales Prediction   — Random Forest demand forecast
  5. Customer Intel     — customer tier analysis + GOLD/SILVER/BRONZE predictor
  6. Upload Data        — replace any dataset without touching code
"""

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
sys.path.append(os.path.dirname(__file__))

from modules.chatbot    import chatbot_response
from modules.sentiment  import get_sentiment_summary, load_sentiment_model, analyze_single
from modules.prediction import (
    get_customer_distribution, get_location_analysis, get_top_customers,
    get_occupation_analysis,   get_accuracy,           train_and_save,
    predict_single_customer,   get_predictions,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Customer Analytics Assistant",
    page_icon="",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .insight-box {
    background: #f0faf6;
    border-left: 4px solid #1D9E75;
    border-radius: 6px;
    padding: 10px 16px;
    margin-top: -8px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #1a4a37;
  }
  .alert-banner {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #856404;
  }
  .page-sub {
    color: #888;
    font-size: 14px;
    margin-top: -12px;
    margin-bottom: 20px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
if "data_paths" not in st.session_state:
    st.session_state.data_paths = {
        "products":  "data/products.csv",
        "customers": "data/customers.xlsx",
        "sales":     "data/sales.csv",
        "reviews":   "data/reviews.csv",
    }

def get_products():   return pd.read_csv(st.session_state.data_paths["products"])
def get_customers():  return pd.read_excel(st.session_state.data_paths["customers"])
def get_sales():      return pd.read_csv(st.session_state.data_paths["sales"])

# ─────────────────────────────────────────────────────────────────────────────
# AI INSIGHT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def insight(text: str):
    st.markdown(f'<div class="insight-box">🤖 <b>AI Insight:</b> {text}</div>',
                unsafe_allow_html=True)

def generate_sales_insight(df, cat_col, price_col, disc_col):
    top_cat  = df[cat_col].value_counts().index[0]
    top_disc = df.groupby(cat_col)[disc_col].mean().idxmax()
    avg_p    = round(df[price_col].mean(), 0)
    return (f"<b>{top_cat}</b> has the highest transaction volume. "
            f"<b>{top_disc}</b> offers the best average discount. "
            f"Overall average price is ₹{avg_p}.")

def generate_sentiment_insight(pos, neg, total):
    pct = round(pos / total * 100) if total else 0
    if pct >= 70:
        return f"Customer satisfaction is strong at <b>{pct}%</b> positive. Keep up the product quality."
    elif pct >= 50:
        return f"Sentiment is moderate at <b>{pct}%</b> positive. Review negative feedback to find improvement areas."
    else:
        return f" Negative sentiment dominates at <b>{100-pct}%</b>. Immediate product/service review recommended."

def generate_prediction_insight(df):
    top = df.sort_values("Demand Score", ascending=False).iloc[0]
    low = df.sort_values("Demand Score").iloc[0]
    return (f"<b>{top['Category']}</b> has the highest predicted demand ({top['Demand Score']}%). "
            f"Consider stocking more. <b>{low['Category']}</b> is slowest — review pricing.")

def generate_customer_insight(df_cust):
    gold_pct = round((df_cust["Customer_value"] == "GOLD").mean() * 100, 1)
    top_city = df_cust.groupby("location")["total_transaction_amount"].mean().idxmax()
    top_occ  = df_cust.groupby("occupation")["total_transaction_amount"].mean().idxmax()
    return (f"<b>{gold_pct}%</b> of customers are GOLD tier. "
            f"<b>{top_city}</b> generates the highest average spend. "
            f"<b>{top_occ}</b> is the highest-spending occupation.")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Smart Customer\nAnalytics Assistant")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            " Ask the Assistant",
            " Sentiment Monitor",
            " Sales Dashboard",
            " Sales Prediction",
            " Customer Intel",
            " Upload Data",
        ],
    )
    st.markdown("---")



# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ASK THE ASSISTANT  (voice auto-sends, zero copy-paste)
# ═════════════════════════════════════════════════════════════════════════════
if page == " Ask the Assistant":
    st.title(" Ask the Assistant")
    st.markdown('<p class="page-sub">Voice or text — ask anything about your store data</p>',
                unsafe_allow_html=True)

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "bot", "text": (
                "Hello! I am your Smart Analytics Assistant. 👋\n\n"
                "You can **speak** using the mic button or **type** below.\n\n"
                "Ask me about:\n"
                "- Products, prices and deals\n"
                "- Store policies\n"
                "- Cheapest or most popular items\n"
                "- Payment methods\n\n"
                "What would you like to know?"
            )}
        ]

    # ── Voice → auto-fills & submits the chat_input, zero copy-paste ─────────
    # How it works:
    #   1. components.html() renders the mic button inside an iframe.
    #   2. On speech result, JS walks UP to window.parent.document (the main
    #      Streamlit page) and finds the <textarea> that backs st.chat_input.
    #   3. It injects the transcript via React's native setter (so React sees
    #      the change), fires an 'input' event, waits 150ms, then fires Enter.
    #   4. Streamlit processes the Enter exactly as if the user typed + pressed
    #      Enter — no Python-side polling needed at all.

    st.markdown("** Voice Input** ")
    st.caption("Click mic → speak → query fills the chat box and sends instantly.")

    VOICE_COMPONENT = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: transparent;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0;
  }
  #mic-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 22px;
    border-radius: 8px;
    border: 1.8px solid #1D9E75;
    background: transparent;
    color: #1D9E75;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.18s;
    white-space: nowrap;
  }
  #mic-btn:hover:not(:disabled) { background: #1D9E75; color: #fff; }
  #mic-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #mic-btn.listening {
    background: #E24B4A !important;
    border-color: #E24B4A !important;
    color: #fff !important;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0   rgba(226,75,74,.45); }
    50%      { box-shadow: 0 0 0 8px rgba(226,75,74,0);   }
  }
  #status        { font-size: 13px; color: #888; }
  #status.active { color: #E24B4A; font-weight: 600; }
  #status.done   { color: #1D9E75; font-weight: 600; }
  #status.error  { color: #cc6600; }
</style>
</head>
<body>
  <button id="mic-btn" onclick="toggleMic()">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.2"
         stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="2" width="6" height="12" rx="3"/>
      <path d="M5 10a7 7 0 0 0 14 0"/>
      <line x1="12" y1="19" x2="12" y2="22"/>
      <line x1="8"  y1="22" x2="16" y2="22"/>
    </svg>
    <span id="btn-label">🎙 Click to speak</span>
  </button>
  <span id="status">Speak — auto-fills and sends the chat box below</span>

<script>
(function(){
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    document.getElementById('status').className = 'error';
    document.getElementById('status').textContent = '⚠️ Voice needs Chrome or Edge.';
    document.getElementById('mic-btn').disabled = true;
    return;
  }

  var rec = new SR();
  rec.lang            = 'en-IN';
  rec.continuous      = false;
  rec.interimResults  = false;
  rec.maxAlternatives = 1;
  var listening = false;

  window.toggleMic = function(){
    if (listening){ rec.stop(); return; }
    try { rec.start(); } catch(e){ setStatus('error','Mic error: '+e.message); }
  };

  rec.onstart = function(){
    listening = true;
    document.getElementById('mic-btn').classList.add('listening');
    document.getElementById('btn-label').textContent = '⏹ Stop';
    setStatus('active','🎤 Listening… speak now');
  };

  rec.onend = function(){
    listening = false;
    document.getElementById('mic-btn').classList.remove('listening');
    document.getElementById('btn-label').textContent = '🎙 Click to speak';
  };

  rec.onerror = function(e){
    listening = false;
    document.getElementById('mic-btn').classList.remove('listening');
    document.getElementById('btn-label').textContent = '🎙 Click to speak';
    var msgs = {
      'not-allowed'  : ' Mic denied — allow microphone in browser settings.',
      'no-speech'    : ' No speech detected. Try again.',
      'network'      : ' Network error.',
      'audio-capture': ' No microphone found.',
    };
    setStatus('error', msgs[e.error] || 'Error: ' + e.error);
  };

  rec.onresult = function(e){
    var text = e.results[0][0].transcript.trim();
    setStatus('done', ' Heard: "' + text + '" — sending…');
    injectAndSend(text);
  };

  function injectAndSend(text) {
    // The mic iframe is inside Streamlit's main page.
    // window.parent.document IS the Streamlit app document.
    var parentDoc = window.parent.document;

    // st.chat_input renders a <textarea> with this test-id
    var ta = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if (!ta) {
      // fallback: grab any visible textarea in the page
      var all = parentDoc.querySelectorAll('textarea');
      for (var i = 0; i < all.length; i++) {
        if (all[i].offsetParent !== null) { ta = all[i]; break; }
      }
    }

    if (!ta) {
      setStatus('error', ' Chat box not found. Scroll down and try again.');
      return;
    }

    // Focus the textarea first
    ta.focus();

    // Use React internal setter so onChange fires correctly
    var setter = Object.getOwnPropertyDescriptor(
      window.parent.HTMLTextAreaElement.prototype, 'value'
    ).set;
    setter.call(ta, text);

    // Fire React's synthetic change event
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.dispatchEvent(new Event('change', { bubbles: true }));

    // Wait a tick for React to register the value, then press Enter
    setTimeout(function(){
      var enterOpts = {
        key: 'Enter', code: 'Enter', keyCode: 13,
        which: 13, bubbles: true, cancelable: true
      };
      ta.dispatchEvent(new KeyboardEvent('keydown',  enterOpts));
      ta.dispatchEvent(new KeyboardEvent('keypress', enterOpts));
      ta.dispatchEvent(new KeyboardEvent('keyup',    enterOpts));
      setStatus('done', ' Sent: "' + text + '" — ready for next query');
    }, 150);
  }

  function setStatus(cls, msg){
    var el = document.getElementById('status');
    el.className   = cls;
    el.textContent = msg;
  }
})();
</script>
</body>
</html>"""

    components.html(VOICE_COMPONENT, height=55)

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "bot" else "user"
        st.chat_message(role).write(msg["text"])

    # ── Typed input (also works perfectly as before) ──────────────────────────
    user_input = st.chat_input("Type your question here…")
    if user_input:
        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.messages.append({"role": "bot",  "text": chatbot_response(user_input)})
        st.rerun()

    # ── Quick-fire buttons ────────────────────────────────────────────────────
    st.markdown("**Quick queries:**")
    qcols = st.columns(4)
    quick = [
        (" Best deals",      "best deals"),
        (" All categories",  "what do you sell"),
        (" Contact info",    "contact details"),
        (" Return policy",   "what is your return policy"),
    ]
    for i, (label, query) in enumerate(quick):
        if qcols[i].button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "text": query})
            st.session_state.messages.append({"role": "bot",  "text": chatbot_response(query)})
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT MONITOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Sentiment Monitor":
    st.title(" Sentiment Monitor")
    st.markdown('<p class="page-sub">real-time customer review analysis</p>',
                unsafe_allow_html=True)

    with st.spinner("Loading sentiment data…"):
        counts = get_sentiment_summary()

    pos   = counts.get("POSITIVE", 0)
    neg   = counts.get("NEGATIVE", 0)
    total = pos + neg

    neg_pct = round(neg / total * 100) if total else 0
    if neg_pct >= 40:
        st.markdown(
            f'<div class="alert-banner"> <b>Alert:</b> Negative sentiment is at <b>{neg_pct}%</b>. '
            f'Review recent customer feedback immediately.</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric(" Positive Reviews", f"{round(pos/total*100)}%" if total else "0%", f"{pos} reviews")
    c2.metric(" Negative Reviews", f"{round(neg/total*100)}%" if total else "0%", f"{neg} reviews")
    c3.metric(" Total Analyzed",   str(total))

    insight(generate_sentiment_insight(pos, neg, total))
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.pie(
            values=counts.values, names=counts.index,
            title="Sentiment Distribution",
            color_discrete_map={"POSITIVE": "#1D9E75", "NEGATIVE": "#E24B4A"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            x=counts.index, y=counts.values,
            title="Review Count by Sentiment",
            color=counts.index,
            color_discrete_map={"POSITIVE": "#1D9E75", "NEGATIVE": "#E24B4A"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader(" Test any review live")
    test_text = st.text_input("Paste or type any customer review:")
    if test_text:
        with st.spinner(""):
            model        = load_sentiment_model()
            label, score = analyze_single(test_text, model)
        if label == "POSITIVE":
            st.success(f" POSITIVE — {score}% confidence")
        else:
            st.error(f" NEGATIVE — {score}% confidence")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SALES DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Sales Dashboard":
    st.title(" Sales Dashboard")
    st.markdown('<p class="page-sub">Product transaction insights with AI-written analysis</p>',
                unsafe_allow_html=True)

    df        = get_products()
    CAT_COL   = "Category"
    PRICE_COL = "Price (Rs.)"
    DISC_COL  = "Discount (%)"
    PAY_COL   = "Payment_Method"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", len(df))
    c2.metric("Categories",          df[CAT_COL].nunique())
    c3.metric("Avg Price",           f"₹{round(df[PRICE_COL].mean(), 2)}")
    c4.metric("Avg Discount",        f"{round(df[DISC_COL].mean(), 1)}%")

    insight(generate_sales_insight(df, CAT_COL, PRICE_COL, DISC_COL))
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        cat_counts = df[CAT_COL].value_counts()
        fig1 = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            title="Transactions by Category",
            color=cat_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        avg_disc = df.groupby(CAT_COL)[DISC_COL].mean().reset_index()
        fig2 = px.bar(
            avg_disc, x=CAT_COL, y=DISC_COL,
            title="Average Discount by Category",
            color=DISC_COL, color_continuous_scale="Teal",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        pay_counts = df[PAY_COL].value_counts()
        fig3 = px.pie(
            values=pay_counts.values, names=pay_counts.index,
            title="Payment Method Distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        avg_price = df.groupby(CAT_COL)[PRICE_COL].mean().reset_index()
        fig4 = px.bar(
            avg_price, x=CAT_COL, y=PRICE_COL,
            title="Average Price by Category",
            color=PRICE_COL, color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader(" Category Deep Dive")
    selected_cat = st.selectbox("Select a category:", df[CAT_COL].unique().tolist())
    sub = df[df[CAT_COL] == selected_cat]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Items",        len(sub))
    s2.metric("Min Price",    f"₹{sub[PRICE_COL].min()}")
    s3.metric("Max Price",    f"₹{sub[PRICE_COL].max()}")
    s4.metric("Avg Discount", f"{round(sub[DISC_COL].mean(), 1)}%")
    insight(
        f"<b>{selected_cat}</b> has {len(sub)} products ranging from "
        f"₹{sub[PRICE_COL].min()} to ₹{sub[PRICE_COL].max()} "
        f"with an average discount of {round(sub[DISC_COL].mean(),1)}%."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SALES PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Sales Prediction":
    st.title(" Sales Prediction")
    st.markdown('<p class="page-sub"> forecasting by product category</p>',
                unsafe_allow_html=True)

    if st.button(" Train / Retrain Model"):
        with st.spinner(""):
            acc = train_and_save()
        st.success(f" Model trained! Accuracy: {acc}%")

    with st.spinner("Loading predictions…"):
        df_pred = get_predictions()

    c1, c2, c3 = st.columns(3)
    c1.metric("Categories Analyzed", len(df_pred))
    c2.metric("High Demand (≥50%)",  len(df_pred[df_pred["Demand Score"] >= 50]))
    c3.metric("Model",               "")

    insight(generate_prediction_insight(df_pred))
    st.markdown("---")

    fig = px.bar(
        df_pred, x="Category", y="Demand Score",
        title="Predicted Demand Score by Category (%)",
        color="Demand Score", color_continuous_scale="Teal",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(" Full Prediction Table")
    st.dataframe(
        df_pred.sort_values("Demand Score", ascending=False),
        use_container_width=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER INTEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Customer Intel":
    st.title(" Customer Intelligence")
    st.markdown('<p class="page-sub">GOLD / SILVER / BRONZE tier analysis + value predictor</p>',
                unsafe_allow_html=True)

    if st.button(" Train / Retrain Model"):
        with st.spinner("Training…"):
            acc = train_and_save()
        st.success(f" Model trained! Accuracy: {acc}%")

    acc     = get_accuracy()
    df_cust = get_customers()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy",  f"{acc}%")
    c2.metric("Total Customers", len(df_cust))
    c3.metric(" Gold Customers", len(df_cust[df_cust["Customer_value"] == "GOLD"]))
    c4.metric("Cities Covered",  df_cust["location"].nunique())

    insight(generate_customer_insight(df_cust))
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        dist = get_customer_distribution()
        fig1 = px.pie(
            dist, names="Customer Value", values="Count",
            title="Customer Tier Distribution",
            color="Customer Value",
            color_discrete_map={"GOLD": "#EF9F27", "SILVER": "#888780", "BRONZE": "#D85A30"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        loc = get_location_analysis()
        fig2 = px.bar(
            loc, x="location", y="avg_transaction",
            title="Avg Spend by City",
            color="avg_transaction", color_continuous_scale="Teal",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        occ = get_occupation_analysis()
        fig3 = px.bar(
            occ, x="occupation", y="gold_percentage",
            title="Gold Tier % by Occupation",
            color="gold_percentage", color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader(" Top 5 Customers by Spend")
        top = get_top_customers()
        st.dataframe(top, use_container_width=True)

    st.markdown("---")
    st.subheader(" Predict a Customer's Tier")
    st.caption("Enter details below — the model predicts GOLD / SILVER / BRONZE instantly.")

    p1, p2, p3 = st.columns(3)
    age_in = p1.number_input("Age",    18, 70, 30)
    gen_in = p2.selectbox("Gender",   ["Female", "Male"])
    loc_in = p3.selectbox("City",     sorted(df_cust["location"].unique().tolist()))

    p4, p5, p6 = st.columns(3)
    occ_in = p4.selectbox("Occupation", sorted(df_cust["occupation"].unique().tolist()))
    amt_in = p5.number_input("Total Spent (₹)", 0, 200000, 10000)
    cnt_in = p6.number_input("No. of Transactions", 1, 100, 10)

    p7, _, __ = st.columns(3)
    login_in = p7.slider("Login Days / Month", 1, 60, 15)

    if st.button("🔮 Predict Now"):
        with st.spinner("Running model…"):
            result = predict_single_customer(
                age_in, gen_in, loc_in, occ_in, amt_in, cnt_in, login_in
            )
        if   "GOLD"   in result: st.success(f"🥇 Prediction: {result}")
        elif "SILVER" in result: st.warning(f"🥈 Prediction: {result}")
        else:                    st.info(   f"🥉 Prediction: {result}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — UPLOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Upload Data":
    st.title(" Upload Your Data")
    st.markdown('<p class="page-sub">Replace any dataset without touching the code</p>',
                unsafe_allow_html=True)

    st.info(
        " Upload your own CSV / Excel files here. "
        "All dashboards, predictions and the chatbot will update automatically. "
        "Your files must have the same column names as the original data.",
    )
    st.markdown("---")

    st.subheader(" Products Data")
    st.caption("Required columns: Category, Price (Rs.), Discount (%), Final_Price(Rs.), Payment_Method")
    prod_file = st.file_uploader("Upload products CSV", type=["csv"], key="prod")
    if prod_file:
        save_path = "data/uploaded_products.csv"
        with open(save_path, "wb") as f:
            f.write(prod_file.getbuffer())
        st.session_state.data_paths["products"] = save_path
        df_check = pd.read_csv(save_path)
        st.success(f" Products file uploaded — {len(df_check)} rows, {df_check.shape[1]} columns")
        st.dataframe(df_check.head(3), use_container_width=True)

    st.markdown("---")

    st.subheader(" Customers Data")
    st.caption("Required columns: name, age, gender, location, occupation, total_transaction_amount, total_transaction_count, login_days, Customer_value")
    cust_file = st.file_uploader("Upload customers Excel", type=["xlsx", "xls"], key="cust")
    if cust_file:
        save_path = "data/uploaded_customers.xlsx"
        with open(save_path, "wb") as f:
            f.write(cust_file.getbuffer())
        st.session_state.data_paths["customers"] = save_path
        df_check = pd.read_excel(save_path)
        st.success(f" Customers file uploaded — {len(df_check)} rows, {df_check.shape[1]} columns")
        st.dataframe(df_check.head(3), use_container_width=True)

    st.markdown("---")

    st.subheader(" Sales / Demand Data")
    st.caption("Required columns: Category, Sales")
    sales_file = st.file_uploader("Upload sales CSV", type=["csv"], key="sales")
    if sales_file:
        save_path = "data/uploaded_sales.csv"
        with open(save_path, "wb") as f:
            f.write(sales_file.getbuffer())
        st.session_state.data_paths["sales"] = save_path
        df_check = pd.read_csv(save_path)
        st.success(f" Sales file uploaded — {len(df_check)} rows")
        st.dataframe(df_check.head(3), use_container_width=True)

    st.markdown("---")

    st.subheader(" Retrain Models on New Data")
    st.caption("After uploading customers data, click below to retrain the prediction model.")
    if st.button(" Retrain All Models Now", use_container_width=True):
        with st.spinner("Retraining Random Forest on uploaded data…"):
            try:
                acc = train_and_save()
                st.success(f" Models retrained successfully! Accuracy: {acc}%")
            except Exception as e:
                st.error(f" Retraining failed: {e}. Check that your uploaded files have the correct columns.")

    st.markdown("---")
    st.subheader(" Currently Active Data Files")
    for key, path in st.session_state.data_paths.items():
        st.markdown(f"- **{key.capitalize()}:** `{path}`")
