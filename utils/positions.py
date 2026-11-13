"""
Unified Positions System
========================
Consolidates: Watchlist + Tracker + Alerts + Notes into ONE system.

Position Schema:
{
    "id": "uuid",
    "ticker": "HDFCBANK.NS",
    "name": "HDFC Bank",
    "sector": "Banking",
    "status": "active" | "watching" | "closed",
    "entry_price": 1650.0,
    "entry_date": "2024-01-15",
    "stop_loss": 1550.0,
    "target": 1850.0,
    "quantity": 10,
    "notes": "Investment thesis...",
    "alerts_enabled": true,
    "entry_signal": "STRONG UPTREND",
    "entry_score": 85,
    "exit_price": null,
    "exit_date": null,
    "created_at": "2024-01-15T10:30:00"
}
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from utils.alert_store import AlertStore

POSITIONS_FILE = "positions.json"

# Positions and the watchlist used to be written only to the dashboard's own disk. On Streamlit
# Cloud that disk is replaced on every redeploy (daily, after the EOD commit), so anything added
# since the last git commit of positions.json (5 May 2026) was lost, and the EOD alert daemon
# only ever saw the committed copy. They now live in the same GitHub-backed store as the RS and
# industry alerts (utils/alert_store.py): one operation per change, on the alert-state branch.
STORE = AlertStore(POSITIONS_FILE, "Positions")
_BG = not os.environ.get("GITHUB_ACTIONS")       # dashboard: save in the background; CI: save now


def storage_status() -> Dict:
    return STORE.status()


def sync_positions() -> bool:
    return STORE.flush("retry sync")


def _load_positions() -> List[Dict]:
    """All positions: last synced state plus anything still waiting to sync."""
    out = STORE.load()
    for p in out:                                 # legacy rows without an id cannot be edited by id
        if not p.get("id"):
            p["id"] = f"legacy-{p.get('ticker', '')}"
    return out


def resolve_ticker(ticker: str, known) -> Optional[str]:
    """
    The symbol market data is filed under. A position saved as FEDERALBANK.NS (the NSE symbol is
    FEDERALBNK) had no price for ten months, so its stop-loss and target were never checked.
    Returns `ticker` when it is known, a single near-identical known symbol otherwise, else None.
    """
    known = set(known)
    if ticker in known:
        return ticker
    import difflib
    m = difflib.get_close_matches(ticker, list(known), n=2, cutoff=0.9)
    if len(m) == 1 or (len(m) == 2 and difflib.SequenceMatcher(None, ticker, m[0]).ratio()
                       - difflib.SequenceMatcher(None, ticker, m[1]).ratio() > 0.03):
        return m[0]
    return None


def get_all_positions(status: Optional[str] = None) -> List[Dict]:
    """Get all positions, optionally filtered by status."""
    positions = _load_positions()
    if status:
        positions = [p for p in positions if p.get('status') == status]
    return positions

def get_position(ticker: str) -> Optional[Dict]:
    """Get a specific position by ticker."""
    positions = _load_positions()
    for p in positions:
        if p.get('ticker') == ticker:
            return p
    return None

def add_position(
    ticker: str,
    name: str = None,
    sector: str = "Unknown",
    status: str = "active",
    entry_price: float = None,
    stop_loss: float = None,
    target: float = None,
    quantity: int = 1,
    notes: str = "",
    entry_signal: str = "N/A",
    entry_score: float = 0,
    alerts_enabled: bool = True
) -> Dict:
    """Add a new position or update existing."""
    positions = _load_positions()
    existing = next((p for p in positions if p.get('ticker') == ticker), None)
    now = datetime.now().isoformat()

    if existing:
        fields = {'status': status, 'alerts_enabled': alerts_enabled, 'updated_at': now}
        if entry_price: fields['entry_price'] = entry_price
        if stop_loss: fields['stop_loss'] = stop_loss
        if target: fields['target'] = target
        if notes: fields['notes'] = notes
        if quantity: fields['quantity'] = quantity
        STORE.commit({"patches": {existing['id']: fields}}, f"update {ticker}", background=_BG)
        return {**existing, **fields}

    position = {
        'id': str(uuid.uuid4()),
        'ticker': ticker,
        'name': name or ticker.replace('.NS', ''),
        'sector': sector,
        'status': status,
        'entry_price': entry_price,
        'entry_date': datetime.now().strftime('%Y-%m-%d'),
        'stop_loss': stop_loss,
        'target': target,
        'quantity': quantity,
        'notes': notes,
        'alerts_enabled': alerts_enabled,
        'entry_signal': entry_signal,
        'entry_score': entry_score,
        'exit_price': None,
        'exit_date': None,
        'created_at': now
    }
    STORE.commit({"adds": [position]}, f"add {ticker} ({status})", background=_BG)
    return position

def update_position(ticker: str, **updates) -> Optional[Dict]:
    """Update a position's fields (only those fields: other positions are untouched)."""
    pos = get_position(ticker)
    if not pos:
        return None
    updates['updated_at'] = datetime.now().isoformat()
    STORE.commit({"patches": {pos['id']: updates}}, f"update {ticker}", background=_BG)
    return {**pos, **updates}

def close_position(ticker: str, exit_price: float) -> Optional[Dict]:
    """Close a position with exit price."""
    return update_position(
        ticker,
        status='closed',
        exit_price=exit_price,
        exit_date=datetime.now().strftime('%Y-%m-%d')
    )

def remove_position(ticker: str) -> bool:
    """Remove a position completely."""
    ids = [p['id'] for p in _load_positions() if p.get('ticker') == ticker]
    if not ids:
        return False
    STORE.commit({"removes": ids}, f"remove {ticker}", background=_BG)
    return True

def add_to_watchlist(ticker: str, name: str = None, sector: str = "Unknown") -> Dict:
    """Quick add to watchlist (no price info needed)."""
    return add_position(ticker, name, sector, status='watching')

def is_position_exists(ticker: str) -> bool:
    """Check if position exists."""
    return get_position(ticker) is not None

def get_positions_with_pnl(market_df) -> List[Dict]:
    """Calculate P&L for all active positions using current market data."""
    positions = get_all_positions()
    result = []
    
    known = set(market_df['ticker']) if market_df is not None and 'ticker' in market_df else set()
    by_ticker = market_df.drop_duplicates('ticker').set_index('ticker') if known else None
    try:
        from utils import corp_actions
        ca = corp_actions.load()
    except Exception:
        ca = None
    pcol = 'price' if (by_ticker is not None and 'price' in by_ticker.columns) else 'currentPrice'

    for pos in positions:
        ticker = pos.get('ticker')
        use = resolve_ticker(ticker, known) if known else None
        pos['has_data'] = use is not None
        if use and use != ticker:
            pos['resolved_ticker'] = use
            pos['ticker_note'] = f"priced as {use.replace('.NS', '')}"

        # Splits and bonuses since the entry date: put entry, stop and target on today's basis
        fac = 1.0
        if ca is not None and use and pos.get('entry_date'):
            try:
                fac = corp_actions.cumulative_factor(ca, use, pos['entry_date'])
            except Exception:
                fac = 1.0
        entry = pos.get('entry_price') or 0
        sl, tgt = pos.get('stop_loss'), pos.get('target')
        if fac != 1.0:
            pos['adj_factor'] = fac
            pos['adj_note'] = f"entry, stop and target adjusted ×{fac:.3g} for a split or bonus since {pos['entry_date']}"
            entry = entry * fac
            sl = sl * fac if sl else sl
            tgt = tgt * fac if tgt else tgt
            pos['entry_price_adj'], pos['stop_loss_adj'], pos['target_adj'] = entry, sl, tgt

        if use:
            row = by_ticker.loc[use]
            current_price = row.get(pcol, 0)
            pos['current_price'] = current_price

            if entry and entry > 0:
                pos['pnl_pct'] = ((current_price - entry) / entry) * 100
                pos['pnl_abs'] = (current_price - entry) * (pos.get('quantity', 1) / fac if fac else pos.get('quantity', 1))
            else:
                pos['pnl_pct'] = 0
                pos['pnl_abs'] = 0

            # Check alert triggers
            pos['sl_triggered'] = bool(sl and current_price <= sl)
            pos['target_triggered'] = bool(tgt and current_price >= tgt)

            # Current trend
            pos['current_signal'] = row.get('trend_signal', 'N/A')
            pos['current_score'] = row.get('trend_score', 0)

        result.append(pos)
    
    return result

def check_position_alerts(market_df) -> List[Dict]:
    """Check all positions for triggered alerts."""
    positions = get_positions_with_pnl(market_df)
    triggered = []
    
    for pos in positions:
        if not pos.get('alerts_enabled'):
            continue
        if pos.get('status') != 'active':
            continue
            
        if pos.get('sl_triggered'):
            triggered.append({**pos, 'alert_type': 'STOP_LOSS'})
        elif pos.get('target_triggered'):
            triggered.append({**pos, 'alert_type': 'TARGET'})
    
    return triggered

def migrate_from_legacy():
    """
    Migrate data from old tracker_data.json and alerts.json to new positions.json.
    Call this once during upgrade.
    """
    migrated = 0
    
    # Migrate from tracker_data.json
    if os.path.exists('tracker_data.json'):
        try:
            with open('tracker_data.json', 'r') as f:
                tracker_data = json.load(f)
            
            for item in tracker_data.get('positions', []):
                ticker = item.get('ticker')
                if ticker and not is_position_exists(ticker):
                    add_position(
                        ticker=ticker,
                        name=item.get('name'),
                        sector=item.get('sector', 'Unknown'),
                        status='active',
                        entry_price=item.get('entry_price'),
                        entry_signal=item.get('entry_signal', 'N/A'),
                        entry_score=item.get('entry_score', 0)
                    )
                    migrated += 1
        except Exception as e:
            print(f"Error migrating tracker: {e}")
    
    # Migrate from alerts.json
    if os.path.exists('alerts.json'):
        try:
            with open('alerts.json', 'r') as f:
                alerts = json.load(f)
            
            for alert in alerts:
                ticker = alert.get('ticker')
                if ticker:
                    existing = get_position(ticker)
                    if existing:
                        # Update with alert info
                        update_position(
                            ticker,
                            stop_loss=alert.get('stop_loss') or existing.get('stop_loss'),
                            target=alert.get('target') or existing.get('target'),
                            entry_price=alert.get('entry_price') or existing.get('entry_price'),
                            notes=alert.get('notes') or existing.get('notes')
                        )
                    else:
                        add_position(
                            ticker=ticker,
                            status='active',
                            entry_price=alert.get('entry_price'),
                            stop_loss=alert.get('stop_loss'),
                            target=alert.get('target'),
                            notes=alert.get('notes', '')
                        )
                        migrated += 1
        except Exception as e:
            print(f"Error migrating alerts: {e}")
    
    return migrated

def get_summary(market_df) -> Dict:
    """Get position summary stats."""
    positions = get_positions_with_pnl(market_df)
    
    active = [p for p in positions if p.get('status') == 'active']
    watching = [p for p in positions if p.get('status') == 'watching']
    closed = [p for p in positions if p.get('status') == 'closed']
    
    total_pnl = sum(p.get('pnl_abs', 0) for p in active)
    avg_pnl_pct = sum(p.get('pnl_pct', 0) for p in active) / len(active) if active else 0
    
    winners = len([p for p in active if p.get('pnl_pct', 0) > 0])
    losers = len([p for p in active if p.get('pnl_pct', 0) < 0])
    
    return {
        'total_active': len(active),
        'total_watching': len(watching),
        'total_closed': len(closed),
        'total_pnl': total_pnl,
        'avg_pnl_pct': avg_pnl_pct,
        'winners': winners,
        'losers': losers,
        'win_rate': (winners / len(active) * 100) if active else 0
    }
