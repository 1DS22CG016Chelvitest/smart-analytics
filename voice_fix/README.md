# 📊 Smart Customer Analytics Assistant

A business intelligence web application built for retail store managers.  
Combines **NLP (BERT)**, **Machine Learning (Random Forest)**, and a **voice-powered chatbot** into a single Streamlit dashboard.

---

## 🚀 How to Run

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Launch the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
Streamlit will print a local URL like:
```
Local URL:  http://localhost:8501
Network URL: http://192.168.x.x:8501
```
Open the **Local URL** in **Google Chrome** or **Microsoft Edge** (required for voice input).

---

## 🗂️ Project Structure

```
smart-analytics/
├── app.py                  ← Main Streamlit application (all 6 pages)
├── requirements.txt        ← Python dependencies
├── README.md
│
├── modules/
│   ├── chatbot.py          ← Rule-based + fuzzy FAQ chatbot
│   ├── sentiment.py        ← DistilBERT sentiment analysis
│   └── prediction.py       ← Random Forest classifier + regressor
│
├── data/
│   ├── products.csv        ← 3,660+ product transactions
│   ├── customers.xlsx      ← Customer behavioral data
│   ├── sales.csv           ← Historical sales by category
│   ├── reviews.csv         ← Raw customer review text
│   ├── reviews_labeled.csv ← BERT-labeled sentiment output
│   └── chatbot.json        ← FAQ question-answer pairs
│
└── models/
    └── sales_model.pkl     ← Pre-trained Random Forest model
```

---

## 🧠 Features & Pages

### 🎙️ Page 1 — Ask the Assistant (Voice + Text Chatbot)
- **Voice input**: Click mic → speak → query is sent **automatically** (zero copy-paste)
- **Text input**: Normal chat box also works
- Answers questions about products, prices, deals, policies, contact info
- Fuzzy matching (FuzzyWuzzy) for FAQ lookup
- Quick-action buttons for common queries

### 😊 Page 2 — Sentiment Monitor
- Loads pre-labeled review sentiment from `reviews_labeled.csv`
- Shows positive/negative split as pie + bar chart
- Auto-alert banner if negative sentiment exceeds 40%
- Live tester: type any review → BERT classifies it in real time

### 📈 Page 3 — Sales Dashboard
- Transactions by category, average discount, payment method breakdown
- Average price per category
- Category deep-dive with price distribution histogram
- AI Insight caption auto-generated from the data

### 🔮 Page 4 — Sales Prediction
- Random Forest Regressor predicts demand score (%) per category
- Retrain button to update model with fresh data
- Progress bar column in results table

### 👥 Page 5 — Customer Intelligence
- Classifies customers as GOLD / SILVER / BRONZE using Random Forest
- City-wise average spend, occupation-wise gold %
- Top 5 customers by total spend
- Live prediction form: enter any customer's details → instant tier prediction

### 📂 Page 6 — Upload Data
- Upload your own CSV / Excel files to replace the default datasets
- All dashboards update automatically
- Retrain models on new data with one click

---

## 🎙️ How Voice Input Works (Technical)

The voice feature uses the **Web Speech API** (built into Chrome/Edge — no API key needed).

**Flow:**
1. User clicks the 🎙 mic button
2. Browser activates microphone
3. Speech is transcribed locally in the browser (no server involved)
4. On recognition, JavaScript fires `window.parent.postMessage()` to send the transcript to the Streamlit parent frame
5. Streamlit's `components.html()` receives the value
6. Python immediately appends it as a user message and calls `chatbot_response()`
7. `st.rerun()` refreshes the page — the chatbot reply appears in the chat

**Result: speak → reply appears. Zero copy-paste. Zero extra clicks.**

> ⚠️ Voice only works in **Google Chrome** or **Microsoft Edge**. Firefox does not support the Web Speech API.

---

## 🛠️ Commands Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run the app | `streamlit run app.py` |
| Run on a specific port | `streamlit run app.py --server.port 8080` |
| Run accessible on network | `streamlit run app.py --server.address 0.0.0.0` |

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Streamlit |
| Sentiment NLP | HuggingFace DistilBERT (SST-2 fine-tuned) |
| Prediction model | Scikit-learn Random Forest |
| FAQ matching | FuzzyWuzzy |
| Charts | Plotly Express |
| Voice input | Web Speech API (browser-native) |
| Data | Pandas, OpenPyXL |

---

## 📦 Requirements

- Python 3.9 or higher  
- Google Chrome or Microsoft Edge (for voice input)  
- Internet connection (first run downloads the DistilBERT model ~250MB)
