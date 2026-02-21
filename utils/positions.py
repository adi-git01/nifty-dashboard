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

POSITIONS_FILE = "positions.json"

def _load_positions() -> List[Dict]:
    """Load all positions from file."""
    if not os.path.exists(POSITIONS_FILE):
        return []
    try:
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def _save_positions(positions: List[Dict]):
    """Save positions to file."""
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2, default=str)

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
    
    # Check if position exists
    existing = None
    for i, p in enumerate(positions):
        if p.get('ticker') == ticker:
            existing = (i, p)
            break
    
    now = datetime.now().isoformat()
    
    if existing:
        # Update existing
        idx, pos = existing
        if entry_price: pos['entry_price'] = entry_price
        if stop_loss: pos['stop_loss'] = stop_loss
        if target: pos['target'] = target
        if notes: pos['notes'] = notes
        if quantity: pos['quantity'] = quantity
        pos['status'] = status
        pos['alerts_enabled'] = alerts_enabled
        pos['updated_at'] = now
        positions[idx] = pos
        _save_positions(positions)
        return pos
    else:
        # Create new
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
        positions.append(position)
        _save_positions(positions)
        return position

def update_position(ticker: str, **updates) -> Optional[Dict]:
    """Update a position's fields."""
    positions = _load_positions()
    for i, p in enumerate(positions):
        if p.get('ticker') == ticker:
            p.update(updates)
            p['updated_at'] = datetime.now().isoformat()
            positions[i] = p
            _save_positions(positions)
            return p
    return None

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
    positions = _load_positions()
    initial_len = len(positions)
    positions = [p for p in positions if p.get('ticker') != ticker]
    if len(positions) < initial_len:
        _save_positions(positions)
        return True
    return False

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
    
    for pos in positions:
        ticker = pos.get('ticker')
        entry = pos.get('entry_price', 0)
        
        # Get current price from market data
        row = market_df[market_df['ticker'] == ticker]
        if not row.empty:
            current_price = row.iloc[0].get('price', 0)
            pos['current_price'] = current_price
            
            if entry and entry > 0:
                pos['pnl_pct'] = ((current_price - entry) / entry) * 100
                pos['pnl_abs'] = (current_price - entry) * pos.get('quantity', 1)
            else:
                pos['pnl_pct'] = 0
                pos['pnl_abs'] = 0
                
            # Check alert triggers
            sl = pos.get('stop_loss')
            tgt = pos.get('target')
            pos['sl_triggered'] = sl and current_price <= sl
            pos['target_triggered'] = tgt and current_price >= tgt
            
            # Current trend
            pos['current_signal'] = row.iloc[0].get('trend_signal', 'N/A')
            pos['current_score'] = row.iloc[0].get('trend_score', 0)
        
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
