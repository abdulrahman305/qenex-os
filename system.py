#!/usr/bin/env python3
"""
Unified System Implementation
"""

import os
import sys
import json
import time
import hashlib
import secrets
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
from decimal import Decimal, getcontext
from collections import defaultdict
from queue import Queue, PriorityQueue, Empty

# Set precision
getcontext().prec = 28

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('System')


# ============================================================================
# Core Database Layer
# ============================================================================

class Database:
    """Thread-safe database manager"""
    
    def __init__(self, db_path: Path = DATA_DIR / 'system.db'):
        self.db_path = db_path
        self.local = threading.local()
        self._init_schema()
    
    def _get_conn(self):
        """Get thread-local connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False
            )
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    @contextmanager
    def transaction(self):
        """Transaction context manager"""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.transaction() as conn:
            conn.executescript('''
                -- Users table
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                );
                
                -- Sessions table
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Tokens table
                CREATE TABLE IF NOT EXISTS tokens (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    decimals INTEGER DEFAULT 18,
                    total_supply TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Balances table
                CREATE TABLE IF NOT EXISTS balances (
                    user_id INTEGER NOT NULL,
                    token_symbol TEXT NOT NULL,
                    balance TEXT NOT NULL DEFAULT '0',
                    locked_balance TEXT DEFAULT '0',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, token_symbol),
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (token_symbol) REFERENCES tokens(symbol)
                );
                
                -- Transactions table
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    from_user_id INTEGER,
                    to_user_id INTEGER,
                    token_symbol TEXT,
                    amount TEXT,
                    fee TEXT DEFAULT '0',
                    status TEXT DEFAULT 'pending',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confirmed_at TIMESTAMP,
                    FOREIGN KEY (from_user_id) REFERENCES users(id),
                    FOREIGN KEY (to_user_id) REFERENCES users(id),
                    FOREIGN KEY (token_symbol) REFERENCES tokens(symbol)
                );
                
                -- Pools table
                CREATE TABLE IF NOT EXISTS pools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_a TEXT NOT NULL,
                    token_b TEXT NOT NULL,
                    reserve_a TEXT NOT NULL,
                    reserve_b TEXT NOT NULL,
                    fee_rate TEXT DEFAULT '0.003',
                    total_shares TEXT DEFAULT '0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token_a, token_b),
                    FOREIGN KEY (token_a) REFERENCES tokens(symbol),
                    FOREIGN KEY (token_b) REFERENCES tokens(symbol)
                );
                
                -- Liquidity positions
                CREATE TABLE IF NOT EXISTS liquidity (
                    user_id INTEGER NOT NULL,
                    pool_id INTEGER NOT NULL,
                    shares TEXT NOT NULL DEFAULT '0',
                    PRIMARY KEY (user_id, pool_id),
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (pool_id) REFERENCES pools(id)
                );
                
                -- System metrics
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Audit log
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                -- Create indices
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
                CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(from_user_id, to_user_id);
                CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id, timestamp);
            ''')


# ============================================================================
# Authentication System
# ============================================================================

class Auth:
    """Authentication and session management"""
    
    def __init__(self, db: Database):
        self.db = db
        self.sessions = {}
        self._cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def hash_password(self, password: str) -> str:
        """Securely hash password"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        salt = secrets.token_hex(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 200000)
        return f"{salt}${key.hex()}"
    
    def verify_password(self, password: str, hash_str: str) -> bool:
        """Verify password against hash"""
        try:
            salt, key_hex = hash_str.split('$')
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 200000)
            return key.hex() == key_hex
        except:
            return False
    
    def register(self, username: str, password: str, email: Optional[str] = None) -> Optional[int]:
        """Register new user"""
        password_hash = self.hash_password(password)
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                    (username, email, password_hash)
                )
                
                user_id = cursor.lastrowid
                
                # Log registration
                conn.execute(
                    'INSERT INTO audit_log (user_id, action, details) VALUES (?, ?, ?)',
                    (user_id, 'REGISTER', json.dumps({'username': username}))
                )
                
                logger.info(f"User registered: {username} (ID: {user_id})")
                return user_id
                
        except sqlite3.IntegrityError:
            logger.warning(f"Registration failed - user exists: {username}")
            return None
    
    def login(self, username: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user and create session"""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                'SELECT id, password_hash, is_active FROM users WHERE username = ?',
                (username,)
            )
            user = cursor.fetchone()
            
            if not user or not user['is_active']:
                logger.warning(f"Login failed - invalid user: {username}")
                return None
            
            if not self.verify_password(password, user['password_hash']):
                logger.warning(f"Login failed - invalid password: {username}")
                conn.execute(
                    'INSERT INTO audit_log (user_id, action, details, ip_address) VALUES (?, ?, ?, ?)',
                    (user['id'], 'LOGIN_FAILED', 'Invalid password', ip_address)
                )
                return None
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            conn.execute(
                'INSERT INTO sessions (id, user_id, expires_at, ip_address) VALUES (?, ?, ?, ?)',
                (session_id, user['id'], expires_at, ip_address)
            )
            
            conn.execute(
                'INSERT INTO audit_log (user_id, action, details, ip_address) VALUES (?, ?, ?, ?)',
                (user['id'], 'LOGIN_SUCCESS', None, ip_address)
            )
            
            self.sessions[session_id] = {
                'user_id': user['id'],
                'username': username,
                'expires_at': expires_at
            }
            
            logger.info(f"User logged in: {username}")
            return session_id
    
    def verify_session(self, session_id: str) -> Optional[Dict]:
        """Verify session and return user info"""
        # Check cache
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() < session['expires_at']:
                return session
            del self.sessions[session_id]
        
        # Check database
        with self.db.transaction() as conn:
            cursor = conn.execute(
                '''SELECT s.user_id, u.username, s.expires_at 
                   FROM sessions s 
                   JOIN users u ON s.user_id = u.id 
                   WHERE s.id = ? AND s.expires_at > ?''',
                (session_id, datetime.now())
            )
            session = cursor.fetchone()
            
            if session:
                self.sessions[session_id] = {
                    'user_id': session['user_id'],
                    'username': session['username'],
                    'expires_at': session['expires_at']
                }
                return dict(session)
        
        return None
    
    def logout(self, session_id: str):
        """Logout user"""
        if session_id in self.sessions:
            user_id = self.sessions[session_id]['user_id']
            del self.sessions[session_id]
            
            with self.db.transaction() as conn:
                conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
                conn.execute(
                    'INSERT INTO audit_log (user_id, action) VALUES (?, ?)',
                    (user_id, 'LOGOUT')
                )
    
    def _cleanup_sessions(self):
        """Clean up expired sessions"""
        while True:
            time.sleep(3600)  # Run every hour
            try:
                with self.db.transaction() as conn:
                    conn.execute('DELETE FROM sessions WHERE expires_at < ?', (datetime.now(),))
                
                # Clean cache
                expired = [k for k, v in self.sessions.items() 
                          if datetime.now() >= v['expires_at']]
                for k in expired:
                    del self.sessions[k]
                    
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")


# ============================================================================
# Token System
# ============================================================================

class TokenSystem:
    """Token management and transactions"""
    
    def __init__(self, db: Database):
        self.db = db
        self.tokens = {}
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from database"""
        with self.db.transaction() as conn:
            cursor = conn.execute('SELECT * FROM tokens')
            for row in cursor:
                self.tokens[row['symbol']] = {
                    'name': row['name'],
                    'decimals': row['decimals'],
                    'total_supply': Decimal(row['total_supply'])
                }
    
    def create_token(self, symbol: str, name: str, decimals: int = 18, 
                    initial_supply: Decimal = Decimal('1000000')) -> bool:
        """Create new token"""
        if symbol in self.tokens:
            return False
        
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    'INSERT INTO tokens (symbol, name, decimals, total_supply) VALUES (?, ?, ?, ?)',
                    (symbol, name, decimals, str(initial_supply))
                )
                
                self.tokens[symbol] = {
                    'name': name,
                    'decimals': decimals,
                    'total_supply': initial_supply
                }
                
                logger.info(f"Token created: {symbol} ({name})")
                return True
                
        except sqlite3.IntegrityError:
            return False
    
    def get_balance(self, user_id: int, token: str) -> Decimal:
        """Get user token balance"""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                'SELECT balance FROM balances WHERE user_id = ? AND token_symbol = ?',
                (user_id, token)
            )
            row = cursor.fetchone()
            return Decimal(row['balance']) if row else Decimal('0')
    
    def transfer(self, from_user: int, to_user: int, token: str, 
                amount: Decimal, fee: Decimal = Decimal('0')) -> Optional[str]:
        """Transfer tokens between users"""
        if amount <= 0 or token not in self.tokens:
            return None
        
        tx_id = secrets.token_hex(16)
        
        try:
            with self.db.transaction() as conn:
                # Check balance
                from_balance = self.get_balance(from_user, token)
                total_needed = amount + fee
                
                if from_balance < total_needed:
                    return None
                
                # Update sender balance
                new_from = from_balance - total_needed
                if new_from > 0:
                    conn.execute(
                        '''INSERT OR REPLACE INTO balances (user_id, token_symbol, balance)
                           VALUES (?, ?, ?)''',
                        (from_user, token, str(new_from))
                    )
                else:
                    conn.execute(
                        'DELETE FROM balances WHERE user_id = ? AND token_symbol = ?',
                        (from_user, token)
                    )
                
                # Update receiver balance
                to_balance = self.get_balance(to_user, token)
                new_to = to_balance + amount
                conn.execute(
                    '''INSERT OR REPLACE INTO balances (user_id, token_symbol, balance)
                       VALUES (?, ?, ?)''',
                    (to_user, token, str(new_to))
                )
                
                # Record transaction
                conn.execute(
                    '''INSERT INTO transactions 
                       (id, type, from_user_id, to_user_id, token_symbol, amount, fee, status, confirmed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (tx_id, 'transfer', from_user, to_user, token, str(amount), 
                     str(fee), 'confirmed', datetime.now())
                )
                
                logger.info(f"Transfer completed: {tx_id}")
                return tx_id
                
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return None
    
    def mint(self, user_id: int, token: str, amount: Decimal) -> bool:
        """Mint new tokens (admin only)"""
        if amount <= 0 or token not in self.tokens:
            return False
        
        try:
            with self.db.transaction() as conn:
                # Update balance
                balance = self.get_balance(user_id, token)
                new_balance = balance + amount
                
                conn.execute(
                    '''INSERT OR REPLACE INTO balances (user_id, token_symbol, balance)
                       VALUES (?, ?, ?)''',
                    (user_id, token, str(new_balance))
                )
                
                # Update total supply
                self.tokens[token]['total_supply'] += amount
                conn.execute(
                    'UPDATE tokens SET total_supply = ? WHERE symbol = ?',
                    (str(self.tokens[token]['total_supply']), token)
                )
                
                # Record transaction
                tx_id = secrets.token_hex(16)
                conn.execute(
                    '''INSERT INTO transactions 
                       (id, type, to_user_id, token_symbol, amount, status, confirmed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (tx_id, 'mint', user_id, token, str(amount), 'confirmed', datetime.now())
                )
                
                logger.info(f"Minted {amount} {token} to user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Mint failed: {e}")
            return False


# ============================================================================
# DeFi System  
# ============================================================================

class DeFiSystem:
    """Decentralized Finance operations"""
    
    def __init__(self, db: Database, tokens: TokenSystem):
        self.db = db
        self.tokens = tokens
        self.pools = {}
        self._load_pools()
    
    def _load_pools(self):
        """Load liquidity pools"""
        with self.db.transaction() as conn:
            cursor = conn.execute('SELECT * FROM pools')
            for row in cursor:
                pool_key = f"{row['token_a']}-{row['token_b']}"
                self.pools[pool_key] = {
                    'id': row['id'],
                    'reserve_a': Decimal(row['reserve_a']),
                    'reserve_b': Decimal(row['reserve_b']),
                    'fee_rate': Decimal(row['fee_rate']),
                    'total_shares': Decimal(row['total_shares'])
                }
    
    def create_pool(self, token_a: str, token_b: str, fee_rate: Decimal = Decimal('0.003')) -> Optional[int]:
        """Create new liquidity pool"""
        if token_a == token_b:
            return None
        
        # Canonical ordering
        if token_a > token_b:
            token_a, token_b = token_b, token_a
        
        pool_key = f"{token_a}-{token_b}"
        if pool_key in self.pools:
            return None
        
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    '''INSERT INTO pools (token_a, token_b, reserve_a, reserve_b, fee_rate)
                       VALUES (?, ?, ?, ?, ?)''',
                    (token_a, token_b, '0', '0', str(fee_rate))
                )
                
                pool_id = cursor.lastrowid
                self.pools[pool_key] = {
                    'id': pool_id,
                    'reserve_a': Decimal('0'),
                    'reserve_b': Decimal('0'),
                    'fee_rate': fee_rate,
                    'total_shares': Decimal('0')
                }
                
                logger.info(f"Pool created: {pool_key} (ID: {pool_id})")
                return pool_id
                
        except Exception as e:
            logger.error(f"Pool creation failed: {e}")
            return None
    
    def add_liquidity(self, user_id: int, token_a: str, token_b: str, 
                     amount_a: Decimal, amount_b: Decimal) -> Optional[Decimal]:
        """Add liquidity to pool"""
        # Canonical ordering
        if token_a > token_b:
            token_a, token_b = token_b, token_a
            amount_a, amount_b = amount_b, amount_a
        
        pool_key = f"{token_a}-{token_b}"
        pool = self.pools.get(pool_key)
        
        if not pool or amount_a <= 0 or amount_b <= 0:
            return None
        
        # Check balances
        balance_a = self.tokens.get_balance(user_id, token_a)
        balance_b = self.tokens.get_balance(user_id, token_b)
        
        if balance_a < amount_a or balance_b < amount_b:
            return None
        
        try:
            with self.db.transaction() as conn:
                # Calculate shares
                if pool['total_shares'] == 0:
                    # First liquidity provider
                    shares = (amount_a * amount_b) ** Decimal('0.5')
                else:
                    # Proportional shares
                    shares = min(
                        (amount_a * pool['total_shares']) / pool['reserve_a'],
                        (amount_b * pool['total_shares']) / pool['reserve_b']
                    )
                
                # Update user balances
                conn.execute(
                    'UPDATE balances SET balance = balance - ? WHERE user_id = ? AND token_symbol = ?',
                    (str(amount_a), user_id, token_a)
                )
                conn.execute(
                    'UPDATE balances SET balance = balance - ? WHERE user_id = ? AND token_symbol = ?',
                    (str(amount_b), user_id, token_b)
                )
                
                # Update pool
                pool['reserve_a'] += amount_a
                pool['reserve_b'] += amount_b
                pool['total_shares'] += shares
                
                conn.execute(
                    '''UPDATE pools SET reserve_a = ?, reserve_b = ?, total_shares = ?
                       WHERE id = ?''',
                    (str(pool['reserve_a']), str(pool['reserve_b']), 
                     str(pool['total_shares']), pool['id'])
                )
                
                # Update liquidity position
                cursor = conn.execute(
                    'SELECT shares FROM liquidity WHERE user_id = ? AND pool_id = ?',
                    (user_id, pool['id'])
                )
                existing = cursor.fetchone()
                
                if existing:
                    new_shares = Decimal(existing['shares']) + shares
                    conn.execute(
                        'UPDATE liquidity SET shares = ? WHERE user_id = ? AND pool_id = ?',
                        (str(new_shares), user_id, pool['id'])
                    )
                else:
                    conn.execute(
                        'INSERT INTO liquidity (user_id, pool_id, shares) VALUES (?, ?, ?)',
                        (user_id, pool['id'], str(shares))
                    )
                
                logger.info(f"Liquidity added: {amount_a} {token_a}, {amount_b} {token_b}")
                return shares
                
        except Exception as e:
            logger.error(f"Add liquidity failed: {e}")
            return None
    
    def swap(self, user_id: int, token_in: str, token_out: str, amount_in: Decimal) -> Optional[Decimal]:
        """Swap tokens through pool"""
        if amount_in <= 0:
            return None
        
        # Find pool
        pool_key = f"{min(token_in, token_out)}-{max(token_in, token_out)}"
        pool = self.pools.get(pool_key)
        
        if not pool:
            return None
        
        # Check balance
        balance = self.tokens.get_balance(user_id, token_in)
        if balance < amount_in:
            return None
        
        try:
            with self.db.transaction() as conn:
                # Determine reserves
                if token_in < token_out:
                    reserve_in = pool['reserve_a']
                    reserve_out = pool['reserve_b']
                else:
                    reserve_in = pool['reserve_b']
                    reserve_out = pool['reserve_a']
                
                # Calculate output (constant product formula)
                amount_in_with_fee = amount_in * (Decimal('1') - pool['fee_rate'])
                numerator = amount_in_with_fee * reserve_out
                denominator = reserve_in + amount_in_with_fee
                amount_out = numerator / denominator
                
                # Update balances
                conn.execute(
                    'UPDATE balances SET balance = balance - ? WHERE user_id = ? AND token_symbol = ?',
                    (str(amount_in), user_id, token_in)
                )
                
                current_out = self.tokens.get_balance(user_id, token_out)
                new_out = current_out + amount_out
                
                if current_out == 0:
                    conn.execute(
                        'INSERT INTO balances (user_id, token_symbol, balance) VALUES (?, ?, ?)',
                        (user_id, token_out, str(new_out))
                    )
                else:
                    conn.execute(
                        'UPDATE balances SET balance = ? WHERE user_id = ? AND token_symbol = ?',
                        (str(new_out), user_id, token_out)
                    )
                
                # Update pool reserves
                if token_in < token_out:
                    pool['reserve_a'] += amount_in
                    pool['reserve_b'] -= amount_out
                else:
                    pool['reserve_b'] += amount_in
                    pool['reserve_a'] -= amount_out
                
                conn.execute(
                    'UPDATE pools SET reserve_a = ?, reserve_b = ? WHERE id = ?',
                    (str(pool['reserve_a']), str(pool['reserve_b']), pool['id'])
                )
                
                # Record transaction
                tx_id = secrets.token_hex(16)
                conn.execute(
                    '''INSERT INTO transactions 
                       (id, type, from_user_id, token_symbol, amount, metadata, status, confirmed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (tx_id, 'swap', user_id, token_in, str(amount_in),
                     json.dumps({'token_out': token_out, 'amount_out': str(amount_out)}),
                     'confirmed', datetime.now())
                )
                
                logger.info(f"Swap completed: {amount_in} {token_in} -> {amount_out} {token_out}")
                return amount_out
                
        except Exception as e:
            logger.error(f"Swap failed: {e}")
            return None


# ============================================================================
# Monitoring System
# ============================================================================

class Monitor:
    """System monitoring and metrics"""
    
    def __init__(self, db: Database):
        self.db = db
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
    
    def record(self, name: str, value: float):
        """Record metric"""
        with self.lock:
            self.metrics[name].append((time.time(), value))
            
            # Keep only last 1000 points
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
        
        # Store in database
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    'INSERT INTO metrics (metric_name, metric_value) VALUES (?, ?)',
                    (name, value)
                )
        except Exception as e:
            logger.error(f"Metric storage error: {e}")
    
    def get_stats(self, name: str, window: int = 60) -> Dict[str, float]:
        """Get metric statistics"""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            now = time.time()
            recent = [v for t, v in self.metrics[name] if now - t <= window]
            
            if not recent:
                return {}
            
            return {
                'current': recent[-1],
                'min': min(recent),
                'max': max(recent),
                'avg': sum(recent) / len(recent),
                'count': len(recent)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        try:
            import psutil
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count(),
                'metrics': {
                    name: self.get_stats(name)
                    for name in list(self.metrics.keys())[:10]  # Top 10 metrics
                }
            }
        except ImportError:
            return {
                'timestamp': datetime.now().isoformat(),
                'active_threads': threading.active_count(),
                'metrics': {}
            }


# ============================================================================
# Main System Class
# ============================================================================

class UnifiedSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.db = Database()
        self.auth = Auth(self.db)
        self.tokens = TokenSystem(self.db)
        self.defi = DeFiSystem(self.db, self.tokens)
        self.monitor = Monitor(self.db)
        
        # Initialize default tokens
        self._init_defaults()
        
        logger.info("System initialized")
    
    def _init_defaults(self):
        """Initialize default tokens and pools"""
        # Create default tokens
        self.tokens.create_token('USDC', 'USD Coin', 6, Decimal('10000000'))
        self.tokens.create_token('ETH', 'Ethereum', 18, Decimal('10000'))
        self.tokens.create_token('QXC', 'Qenex Token', 18, Decimal('1000000'))
        
        # Create default pools
        self.defi.create_pool('ETH', 'USDC')
        self.defi.create_pool('QXC', 'USDC')
        self.defi.create_pool('ETH', 'QXC')
    
    def get_dashboard_data(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get dashboard data"""
        data = {
            'system_health': self.monitor.get_system_health(),
            'tokens': list(self.tokens.tokens.keys()),
            'pools': list(self.defi.pools.keys()),
            'total_users': 0,
            'total_transactions': 0
        }
        
        with self.db.transaction() as conn:
            # User count
            cursor = conn.execute('SELECT COUNT(*) as count FROM users')
            data['total_users'] = cursor.fetchone()['count']
            
            # Transaction count
            cursor = conn.execute('SELECT COUNT(*) as count FROM transactions')
            data['total_transactions'] = cursor.fetchone()['count']
            
            # User specific data
            if user_id:
                data['user'] = {
                    'balances': {},
                    'liquidity': []
                }
                
                # Get balances
                cursor = conn.execute(
                    'SELECT token_symbol, balance FROM balances WHERE user_id = ?',
                    (user_id,)
                )
                for row in cursor:
                    data['user']['balances'][row['token_symbol']] = row['balance']
                
                # Get liquidity positions
                cursor = conn.execute(
                    '''SELECT p.token_a, p.token_b, l.shares 
                       FROM liquidity l
                       JOIN pools p ON l.pool_id = p.id
                       WHERE l.user_id = ?''',
                    (user_id,)
                )
                for row in cursor:
                    data['user']['liquidity'].append({
                        'pool': f"{row['token_a']}-{row['token_b']}",
                        'shares': row['shares']
                    })
        
        return data
    
    def run_demo(self):
        """Run system demonstration"""
        print("\n" + "="*60)
        print("UNIFIED SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Create demo user
        print("\n1. Creating demo user...")
        user_id = self.auth.register("demo_user", "password123")
        if user_id:
            print(f"   ✓ User created (ID: {user_id})")
        
        # Login
        print("\n2. Authenticating user...")
        session = self.auth.login("demo_user", "password123")
        if session:
            print(f"   ✓ Session created: {session[:20]}...")
        
        # Mint tokens
        print("\n3. Minting tokens...")
        self.tokens.mint(user_id, 'USDC', Decimal('10000'))
        self.tokens.mint(user_id, 'ETH', Decimal('10'))
        print("   ✓ 10,000 USDC minted")
        print("   ✓ 10 ETH minted")
        
        # Add liquidity
        print("\n4. Adding liquidity to ETH/USDC pool...")
        shares = self.defi.add_liquidity(user_id, 'ETH', 'USDC', Decimal('5'), Decimal('10000'))
        if shares:
            print(f"   ✓ Liquidity added, received {shares:.4f} LP tokens")
        
        # Perform swap
        print("\n5. Swapping 1 ETH for USDC...")
        usdc_out = self.defi.swap(user_id, 'ETH', 'USDC', Decimal('1'))
        if usdc_out:
            print(f"   ✓ Received {usdc_out:.2f} USDC")
        
        # Show dashboard
        print("\n6. Dashboard Data:")
        dashboard = self.get_dashboard_data(user_id)
        print(f"   - Total Users: {dashboard['total_users']}")
        print(f"   - Total Transactions: {dashboard['total_transactions']}")
        print(f"   - Active Tokens: {', '.join(dashboard['tokens'])}")
        print(f"   - Active Pools: {', '.join(dashboard['pools'])}")
        
        if 'user' in dashboard:
            print(f"\n   User Balances:")
            for token, balance in dashboard['user']['balances'].items():
                print(f"     - {token}: {balance}")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    system = UnifiedSystem()
    
    # Record some metrics
    system.monitor.record('system.startup', 1)
    system.monitor.record('memory.usage', 100)
    
    # Run demonstration
    system.run_demo()
    
    # Keep system running
    print("\nSystem running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(10)
            # Record periodic metrics
            health = system.monitor.get_system_health()
            system.monitor.record('cpu.usage', health.get('cpu_percent', 0))
            system.monitor.record('memory.usage', health.get('memory_percent', 0))
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        logger.info("System shutdown")


if __name__ == '__main__':
    main()