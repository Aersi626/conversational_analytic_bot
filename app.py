# nba_chatbot_app.py
import os
import streamlit as st
import pandas as pd
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import requests
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")

st.set_page_config(page_title="🏀 NBA Chatbot", layout="centered")
st.title("🏀 NBA Conversational Game Bot")
st.markdown("Ask about recent NBA games, player stats, or match outcomes.")

@st.cache_resource

def load_nba_data():
    url = "https://api.balldontlie.io/v1/games?seasons[]=2024&per_page=100"
    headers = {"Authorization": f"{BALLDONTLIE_API_KEY}"}
    response = requests.get(url, headers=headers)

    # Safely check the response
    if response.status_code != 200:
        print("API call failed:", response.status_code)
        print(response.text)  # Optional: see what was returned
    else:
        try:
            data = response.json()['data']
            df = pd.DataFrame([{
                "date": d["date"],
                "home_team": d["home_team"]["full_name"],
                "visitor_team": d["visitor_team"]["full_name"],
                "home_score": d["home_team_score"],
                "visitor_score": d["visitor_team_score"]
            } for d in data])
            return df
        
        except Exception as e:
            print("Failed to parse JSON:", str(e))
            print("Raw response:", response.text)
    

def create_faiss_index(df):
    text = "\n".join([
        f"{row['date'][:10]}: {row['home_team']} ({row['home_score']}) vs {row['visitor_team']} ({row['visitor_score']})"
        for _, row in df.iterrows()
    ])
    docs = [Document(page_content=text)]
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Load and index data
nba_df = load_nba_data()
vectorstore = create_faiss_index(nba_df)
qa_chain = build_qa_chain(vectorstore)

# User interaction
user_input = st.text_input("Ask a question about NBA games:")
if user_input:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke(user_input)
        st.markdown(f"**Answer:** {answer}")

st.write("Made with Heart using Streamlit and OpenAI")