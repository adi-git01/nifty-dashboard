import sqlite3
import pandas as pd
import os
from datetime import datetime

class TradingDatabase:
    """
    In-Memory SQLite backed by CSVs for native Git version control.
    Loads CSVs -> SQLite on init.
    Dumps SQLite -> CSVs on commit/close.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        
        # File paths
        self.portfolio_csv = os.path.join(self.data_dir, "active_portfolio.csv")
        self.ledger_csv = os.path.join(self.data_dir, "trade_ledger.csv")
        self.watchlist_csv = os.path.join(self.data_dir, "watchlist_history.csv")
        self.alerts_csv = os.path.join(self.data_dir, "alerts_log.csv")
        self.rotation_csv = os.path.join(self.data_dir, "sub_industry_rotation.csv")
        
        self._init_schema()
        self._load_from_csv()

    def _init_schema(self):
        self.cursor.executescript("""
            CREATE TABLE portfolio (
                ticker TEXT PRIMARY KEY,
                buy_date TEXT,
                entry_price REAL,
                peak_price REAL,
                position_size REAL,
                stop_loss REAL,
                strategy_tag TEXT
            );

            CREATE TABLE ledger (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                buy_date TEXT,
                sell_date TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL,
                reason TEXT,
                strategy_tag TEXT
            );

            CREATE TABLE watchlist (
                ticker TEXT,
                added_date TEXT,
                v3_score REAL,
                rs_score REAL,
                sector TEXT,
                status TEXT,
                PRIMARY KEY (ticker, added_date)
            );

            CREATE TABLE alerts_log (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_date TEXT,
                ticker TEXT,
                alert_type TEXT,
                message TEXT
            );
            
            CREATE TABLE sub_industry_rotation (
                record_date TEXT,
                sub_industry TEXT,
                rs_momentum REAL,
                top_components TEXT,
                PRIMARY KEY (record_date, sub_industry)
            );
        """)
        self.conn.commit()

    def _load_from_csv(self):
        """Load CSV data into in-memory SQLite"""
        tables = {
            'portfolio': self.portfolio_csv,
            'ledger': self.ledger_csv,
            'watchlist': self.watchlist_csv,
            'alerts_log': self.alerts_csv,
            'sub_industry_rotation': self.rotation_csv
        }
        
        for table_name, csv_path in tables.items():
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        df.to_sql(table_name, self.conn, if_exists='append', index=False)
                except Exception as e:
                    print(f"Error loading {table_name}: {e}")

    def commit_to_csv(self):
        """Dump in-memory SQLite tables back to persistent CSVs"""
        tables = {
            'portfolio': self.portfolio_csv,
            'ledger': self.ledger_csv,
            'watchlist': self.watchlist_csv,
            'alerts_log': self.alerts_csv,
            'sub_industry_rotation': self.rotation_csv
        }
        
        for table_name, csv_path in tables.items():
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Error saving {table_name}: {e}")

    def close(self):
        self.commit_to_csv()
        self.conn.close()

if __name__ == "__main__":
    db = TradingDatabase()
    # Test insertion
    db.cursor.execute("INSERT OR REPLACE INTO alerts_log (alert_date, ticker, alert_type, message) VALUES (?, ?, ?, ?)", 
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SYSTEM", "INFO", "Database Initialized"))
    db.close()
    print("Database synced to CSVs successfully.")
