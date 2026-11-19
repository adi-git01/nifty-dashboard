import sqlite3
import pandas as pd
import os
from datetime import datetime
from utils.atomic_io import atomic_to_csv

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
                score_0_100 INTEGER DEFAULT 50,
                members INTEGER,
                breadth_pct REAL,
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
        
        self._loaded = {}
        for table_name, csv_path in tables.items():
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        # one duplicate key used to fail the whole insert, leave the table empty, and close() then
                        # wrote the empty table over the CSV. Keep the last copy of each key; never insert all-or-nothing.
                        cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table_name})")]
                        pk = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table_name})") if r[5]]
                        df = df[[c for c in df.columns if c in cols]]
                        if pk and set(pk) <= set(df.columns):
                            df = df.drop_duplicates(pk, keep="last")
                        df.to_sql(table_name, self.conn, if_exists='append', index=False)
                    self._loaded[table_name] = len(df)
                except Exception as e:
                    print(f"Error loading {table_name}: {e}")
                    self._loaded[table_name] = None          # unreadable: never write this table back

    def commit_to_csv(self):
        """Dump in-memory SQLite tables back to persistent CSVs"""
        tables = {
            'portfolio': self.portfolio_csv,
            'ledger': self.ledger_csv,
            'watchlist': self.watchlist_csv,
            'alerts_log': self.alerts_csv,
            'sub_industry_rotation': self.rotation_csv
        }
        
        loaded = getattr(self, "_loaded", {})
        for table_name, csv_path in tables.items():
            try:
                if table_name in loaded and loaded[table_name] is None:
                    print(f"Not saving {table_name}: it failed to load, and writing would replace {csv_path}")
                    continue
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
                before = loaded.get(table_name) or 0
                if before >= 50 and len(df) < 0.5 * before:
                    print(f"Not saving {table_name}: {len(df)} rows now vs {before} loaded -- refusing to shrink it by half")
                    continue
                atomic_to_csv(df, csv_path, index=False)
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
