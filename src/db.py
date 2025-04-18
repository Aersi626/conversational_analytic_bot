import sqlite3
import pandas as pd

DB_PATH = "nba_data.db"

def connect_db():
    return sqlite3.connect(DB_PATH)

def save_df(df: pd.DataFrame, table_name: str):
    conn = connect_db()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def load_df(table_name: str):
    conn = connect_db()
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def table_exists(table_name: str):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    result = cursor.fetchone()
    conn.close()
    return result is not None