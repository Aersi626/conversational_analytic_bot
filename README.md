# README.md

# 🏀 NBA Conversational Analytics Bot

An interactive chatbot built with **Streamlit**, **LangChain**, **OpenAI**, and **FAISS** to answer natural language questions about NBA games. Live game data is retrieved and embedded into a vector store for semantic search and conversational Q&A.

---

## 🚀 Features

- ✅ Ask questions like:
  - "Who won the Lakers game yesterday?"
  - "What was the score between the Heat and Celtics?"
- 🔍 Uses FAISS for fast semantic search over recent NBA games
- 💬 Powered by OpenAI’s GPT-3.5 for natural responses
- 📊 Pulls live game data via public NBA API
- 🎯 Simple Streamlit UI for seamless interaction

---

## 🧱 Tech Stack

- **LangChain** (retrieval-based QA)
- **OpenAI GPT-3.5** (LLM backend)
- **FAISS** (vector search)
- **Streamlit** (frontend)
- **SQLite** (optional local caching)
- **Python + Pandas + Requests**

---

## 📦 Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/nba-chatbot.git
cd nba-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Copy `.env.example` to `.env` and add your OpenAI key:

```bash
cp .env.example .env
```

4. **Run the app**

```bash
streamlit run nba_chatbot_app.py
```

Go to [http://localhost:8501](http://localhost:8501) to chat!

---

## 📁 Project Structure

```
nba-chatbot/
├── nba_chatbot_app.py       # Main app script
├── requirements.txt         # Python dependencies
├── .env.example             # API key placeholder
├── .gitignore               # Excludes .env, cache, etc.
└── README.md                # Project overview
```

---

## 📌 Notes
- API source: [https://www.balldontlie.io](https://www.balldontlie.io)
- You may customize question handling or support multi-season queries.

---

## 🧠 Future Ideas
- Vector cache via SQLite or DuckDB
- Player-specific performance summaries
- Add playoff filters and trends

---

## 📜 License
MIT — feel free to fork, adapt, and build on it!

---
