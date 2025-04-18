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
from src.db import table_exists, load_df, save_df
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")

st.set_page_config(page_title="🏀 NBA Chatbot", layout="centered")
st.title("🏀 NBA Conversational Game Bot")
st.markdown("Ask about recent NBA games, player stats, or match outcomes.")

@st.cache_resource

def load_games_data(use_cache=True, max_pages=10):
    logging.info("🔍 load_games_data() called")
    if use_cache and table_exists("games"):
        return load_df("games")

    # else, pull from API
    all_games = []
    for page in range(1, max_pages + 1):
        url = f"https://api.balldontlie.io/v1/games?seasons[]=2023&per_page=100&page={page}"
        headers = {"Authorization": f"{BALLDONTLIE_API_KEY}"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print("API call failed:", response.status_code)
        data = response.json().get("data", [])
        all_games.extend(data)
        if not data:
            print("Data is empty.")

    df = pd.DataFrame([{
        "date": g["date"],
        "home_team": g["home_team"]["full_name"],
        "visitor_team": g["visitor_team"]["full_name"],
        "home_score": g["home_team_score"],
        "visitor_score": g["visitor_team_score"]
    } for g in all_games])

    save_df(df, "games")
    return df
    
def load_teams_data(use_cache=True):
    logging.info("🔍 load_teams_data() called")
    if use_cache and table_exists("teams"):
        return load_df("teams")

    url = "https://api.balldontlie.io/v1/teams?per_page=100"
    headers = {"Authorization": f"{BALLDONTLIE_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        teams = response.json()["data"]
        df = pd.DataFrame(teams)
        save_df(df, "teams")
        return df
    else:
        print("Failed to fetch teams:", response.status_code)
        return pd.DataFrame()

def load_players_data(use_cache=True, max_pages=10):
    logging.info("🔍 load_players_data() called")
    if use_cache and table_exists("players"):
        return load_df("players")

    logging.info("🌐 Fetching players from API...")
    all_players = []
    for page in range(1, max_pages + 1):
        url = f"https://api.balldontlie.io/v1/players?page={page}&per_page=100"
        headers = {"Authorization": f"{BALLDONTLIE_API_KEY}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get("data", [])
            all_players.extend(data)
        else:
            print(f"Failed to fetch players page {page}: {response.status_code}")
            break

    df = pd.json_normalize(all_players)
    save_df(df, "players")
    return df

def create_faiss_index(games_df, teams_df, players_df):
    game_lines = [
        f"{row['date'][:10]}: {row['home_team']} ({row['home_score']}) vs {row['visitor_team']} ({row['visitor_score']})"
        for _, row in games_df.iterrows()
    ]

    team_lines = [
        f"{row['full_name']} is based in {row['city']} and plays in the {row['division']} division of the {row['conference']} conference."
        for _, row in teams_df.iterrows()
    ]

    player_lines = [
        f"{row['first_name']} {row['last_name']} plays for {row['team.full_name']} as a {row['position']}."
        for _, row in players_df.iterrows()
    ]
    
    text = "\n".join(game_lines + team_lines + player_lines)
    docs = [Document(page_content=text)]
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Load and index data
st.info("Loading games data...")
games_df = load_games_data()
st.success(f"Loaded {len(games_df)} games")

st.info("Loading teams data...")
teams_df = load_teams_data()
st.success(f"Loaded {len(teams_df)} teams")

st.info("Loading players data...")
players_df = load_players_data()
st.success(f"Loaded {len(players_df)} players")

vectorstore = create_faiss_index(games_df, teams_df, players_df)
qa_chain = build_qa_chain(vectorstore)

# User interaction
user_input = st.text_input("Ask a question about NBA games:")
if user_input:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke(user_input)
        st.markdown(f"**Answer:** {answer}")

st.write("Made with Heart using Streamlit and OpenAI")