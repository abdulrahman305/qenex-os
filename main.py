#!/usr/bin/env python3
"""
Main System Entry Point
"""

import os
import json
import time
import sqlite3
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal, getcontext

# Set precision for financial calculations
getcontext().prec = 28

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'system.log'),
        logging.StreamHandler()
    ]
)


# ==============================================================================
# Core Components
# ==============================================================================

class Database:
    """Database management with proper transaction handling"""
    
    def __init__(self):
        self.path = DATA_DIR / 'main.db'
        self.conn = None
        self.setup()
    
    def setup(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON')
        
        # Create tables
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS tokens (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                total_supply TEXT NOT NULL,
                decimals INTEGER DEFAULT 18
            );
            
            CREATE TABLE IF NOT EXISTS balances (
                account_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                amount TEXT NOT NULL DEFAULT '0',
                PRIMARY KEY (account_id, token),
                FOREIGN KEY (account_id) REFERENCES accounts(id),
                FOREIGN KEY (token) REFERENCES tokens(symbol)
            );
            
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash TEXT UNIQUE NOT NULL,
                from_account INTEGER,
                to_account INTEGER,
                token TEXT NOT NULL,
                amount TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_account) REFERENCES accounts(id),
                FOREIGN KEY (to_account) REFERENCES accounts(id),
                FOREIGN KEY (token) REFERENCES tokens(symbol)
            );
            
            CREATE TABLE IF NOT EXISTS pools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token0 TEXT NOT NULL,
                token1 TEXT NOT NULL,
                reserve0 TEXT NOT NULL DEFAULT '0',
                reserve1 TEXT NOT NULL DEFAULT '0',
                UNIQUE(token0, token1),
                FOREIGN KEY (token0) REFERENCES tokens(symbol),
                FOREIGN KEY (token1) REFERENCES tokens(symbol)
            );
            
            CREATE INDEX IF NOT EXISTS idx_tx_hash ON transactions(tx_hash);
            CREATE INDEX IF NOT EXISTS idx_balances ON balances(account_id, token);
        ''')
        self.conn.commit()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute query with parameters"""
        return self.conn.execute(query, params)
    
    def commit(self):
        """Commit transaction"""
        self.conn.commit()
    
    def rollback(self):
        """Rollback transaction"""
        self.conn.rollback()


@dataclass
class Account:
    """Account representation"""
    address: str
    balances: Dict[str, Decimal]
    
    @staticmethod
    def create() -> 'Account':
        """Create new account with random address"""
        address = '0x' + secrets.token_hex(20)
        return Account(address=address, balances={})


class TokenManager:
    """Token operations manager"""
    
    def __init__(self, db: Database):
        self.db = db
        self.tokens = {}
        self.load_tokens()
    
    def load_tokens(self):
        """Load tokens from database"""
        cursor = self.db.execute('SELECT * FROM tokens')
        for row in cursor:
            self.tokens[row['symbol']] = {
                'name': row['name'],
                'total_supply': Decimal(row['total_supply']),
                'decimals': row['decimals']
            }
    
    def create_token(self, symbol: str, name: str, supply: Decimal, decimals: int = 18) -> bool:
        """Create new token"""
        if symbol in self.tokens:
            return False
        
        try:
            self.db.execute(
                'INSERT INTO tokens (symbol, name, total_supply, decimals) VALUES (?, ?, ?, ?)',
                (symbol, name, str(supply), decimals)
            )
            self.db.commit()
            
            self.tokens[symbol] = {
                'name': name,
                'total_supply': supply,
                'decimals': decimals
            }
            
            logging.info(f'Token created: {symbol} ({name})')
            return True
            
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False
    
    def transfer(self, from_addr: str, to_addr: str, token: str, amount: Decimal) -> Optional[str]:
        """Transfer tokens between accounts"""
        if token not in self.tokens or amount <= 0:
            return None
        
        tx_hash = '0x' + hashlib.sha256(f'{from_addr}{to_addr}{token}{amount}{time.time()}'.encode()).hexdigest()
        
        try:
            # Get account IDs
            from_cursor = self.db.execute('SELECT id FROM accounts WHERE address = ?', (from_addr,))
            from_acc = from_cursor.fetchone()
            if not from_acc:
                return None
            
            to_cursor = self.db.execute('SELECT id FROM accounts WHERE address = ?', (to_addr,))
            to_acc = to_cursor.fetchone()
            if not to_acc:
                # Create account if doesn't exist
                self.db.execute('INSERT INTO accounts (address) VALUES (?)', (to_addr,))
                to_cursor = self.db.execute('SELECT id FROM accounts WHERE address = ?', (to_addr,))
                to_acc = to_cursor.fetchone()
            
            # Check balance
            bal_cursor = self.db.execute(
                'SELECT amount FROM balances WHERE account_id = ? AND token = ?',
                (from_acc['id'], token)
            )
            balance_row = bal_cursor.fetchone()
            
            if not balance_row or Decimal(balance_row['amount']) < amount:
                return None
            
            # Update balances
            new_from_balance = Decimal(balance_row['amount']) - amount
            self.db.execute(
                'UPDATE balances SET amount = ? WHERE account_id = ? AND token = ?',
                (str(new_from_balance), from_acc['id'], token)
            )
            
            # Update or create recipient balance
            to_bal_cursor = self.db.execute(
                'SELECT amount FROM balances WHERE account_id = ? AND token = ?',
                (to_acc['id'], token)
            )
            to_balance_row = to_bal_cursor.fetchone()
            
            if to_balance_row:
                new_to_balance = Decimal(to_balance_row['amount']) + amount
                self.db.execute(
                    'UPDATE balances SET amount = ? WHERE account_id = ? AND token = ?',
                    (str(new_to_balance), to_acc['id'], token)
                )
            else:
                self.db.execute(
                    'INSERT INTO balances (account_id, token, amount) VALUES (?, ?, ?)',
                    (to_acc['id'], token, str(amount))
                )
            
            # Record transaction
            self.db.execute(
                'INSERT INTO transactions (tx_hash, from_account, to_account, token, amount) VALUES (?, ?, ?, ?, ?)',
                (tx_hash, from_acc['id'], to_acc['id'], token, str(amount))
            )
            
            self.db.commit()
            logging.info(f'Transfer: {from_addr[:8]}... -> {to_addr[:8]}... : {amount} {token}')
            return tx_hash
            
        except Exception as e:
            self.db.rollback()
            logging.error(f'Transfer failed: {e}')
            return None
    
    def get_balance(self, address: str, token: str) -> Decimal:
        """Get account balance for token"""
        cursor = self.db.execute(
            '''SELECT b.amount 
               FROM balances b 
               JOIN accounts a ON b.account_id = a.id 
               WHERE a.address = ? AND b.token = ?''',
            (address, token)
        )
        row = cursor.fetchone()
        return Decimal(row['amount']) if row else Decimal('0')
    
    def mint(self, address: str, token: str, amount: Decimal) -> bool:
        """Mint tokens to address"""
        if token not in self.tokens or amount <= 0:
            return False
        
        try:
            # Get or create account
            cursor = self.db.execute('SELECT id FROM accounts WHERE address = ?', (address,))
            account = cursor.fetchone()
            if not account:
                self.db.execute('INSERT INTO accounts (address) VALUES (?)', (address,))
                cursor = self.db.execute('SELECT id FROM accounts WHERE address = ?', (address,))
                account = cursor.fetchone()
            
            # Update balance
            bal_cursor = self.db.execute(
                'SELECT amount FROM balances WHERE account_id = ? AND token = ?',
                (account['id'], token)
            )
            balance_row = bal_cursor.fetchone()
            
            if balance_row:
                new_balance = Decimal(balance_row['amount']) + amount
                self.db.execute(
                    'UPDATE balances SET amount = ? WHERE account_id = ? AND token = ?',
                    (str(new_balance), account['id'], token)
                )
            else:
                self.db.execute(
                    'INSERT INTO balances (account_id, token, amount) VALUES (?, ?, ?)',
                    (account['id'], token, str(amount))
                )
            
            self.db.commit()
            logging.info(f'Minted {amount} {token} to {address[:8]}...')
            return True
            
        except Exception as e:
            self.db.rollback()
            logging.error(f'Mint failed: {e}')
            return False


class LiquidityPool:
    """AMM Liquidity Pool implementation"""
    
    def __init__(self, db: Database):
        self.db = db
        self.pools = {}
        self.load_pools()
    
    def load_pools(self):
        """Load pools from database"""
        cursor = self.db.execute('SELECT * FROM pools')
        for row in cursor:
            key = f"{row['token0']}-{row['token1']}"
            self.pools[key] = {
                'id': row['id'],
                'reserve0': Decimal(row['reserve0']),
                'reserve1': Decimal(row['reserve1'])
            }
    
    def create_pool(self, token0: str, token1: str) -> bool:
        """Create new liquidity pool"""
        if token0 == token1:
            return False
        
        # Ensure consistent ordering
        if token0 > token1:
            token0, token1 = token1, token0
        
        key = f"{token0}-{token1}"
        if key in self.pools:
            return False
        
        try:
            cursor = self.db.execute(
                'INSERT INTO pools (token0, token1) VALUES (?, ?)',
                (token0, token1)
            )
            self.db.commit()
            
            self.pools[key] = {
                'id': cursor.lastrowid,
                'reserve0': Decimal('0'),
                'reserve1': Decimal('0')
            }
            
            logging.info(f'Pool created: {key}')
            return True
            
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False
    
    def add_liquidity(self, token0: str, token1: str, amount0: Decimal, amount1: Decimal) -> bool:
        """Add liquidity to pool"""
        if token0 > token1:
            token0, token1 = token1, token0
            amount0, amount1 = amount1, amount0
        
        key = f"{token0}-{token1}"
        pool = self.pools.get(key)
        
        if not pool or amount0 <= 0 or amount1 <= 0:
            return False
        
        try:
            pool['reserve0'] += amount0
            pool['reserve1'] += amount1
            
            self.db.execute(
                'UPDATE pools SET reserve0 = ?, reserve1 = ? WHERE id = ?',
                (str(pool['reserve0']), str(pool['reserve1']), pool['id'])
            )
            self.db.commit()
            
            logging.info(f'Liquidity added to {key}: {amount0} {token0}, {amount1} {token1}')
            return True
            
        except Exception as e:
            self.db.rollback()
            logging.error(f'Add liquidity failed: {e}')
            return False
    
    def swap(self, token_in: str, token_out: str, amount_in: Decimal) -> Optional[Decimal]:
        """Calculate swap output amount"""
        if amount_in <= 0:
            return None
        
        # Find pool
        key1 = f"{token_in}-{token_out}" if token_in < token_out else f"{token_out}-{token_in}"
        pool = self.pools.get(key1)
        
        if not pool:
            return None
        
        # Determine reserves
        if token_in < token_out:
            reserve_in = pool['reserve0']
            reserve_out = pool['reserve1']
        else:
            reserve_in = pool['reserve1']
            reserve_out = pool['reserve0']
        
        if reserve_in == 0 or reserve_out == 0:
            return None
        
        # Calculate output (constant product formula with 0.3% fee)
        amount_in_with_fee = amount_in * Decimal('997') / Decimal('1000')
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        amount_out = numerator / denominator
        
        # Update reserves
        try:
            if token_in < token_out:
                pool['reserve0'] = reserve_in + amount_in
                pool['reserve1'] = reserve_out - amount_out
            else:
                pool['reserve1'] = reserve_in + amount_in
                pool['reserve0'] = reserve_out - amount_out
            
            self.db.execute(
                'UPDATE pools SET reserve0 = ?, reserve1 = ? WHERE id = ?',
                (str(pool['reserve0']), str(pool['reserve1']), pool['id'])
            )
            self.db.commit()
            
            logging.info(f'Swap: {amount_in} {token_in} -> {amount_out} {token_out}')
            return amount_out
            
        except Exception as e:
            self.db.rollback()
            logging.error(f'Swap failed: {e}')
            return None
    
    def get_price(self, token0: str, token1: str) -> Optional[Decimal]:
        """Get price of token0 in terms of token1"""
        if token0 > token1:
            token0, token1 = token1, token0
            inverse = True
        else:
            inverse = False
        
        key = f"{token0}-{token1}"
        pool = self.pools.get(key)
        
        if not pool or pool['reserve0'] == 0:
            return None
        
        price = pool['reserve1'] / pool['reserve0']
        
        return Decimal('1') / price if inverse else price


class System:
    """Main system orchestrator"""
    
    def __init__(self):
        self.db = Database()
        self.tokens = TokenManager(self.db)
        self.pools = LiquidityPool(self.db)
        self.accounts = {}
        
        # Initialize default tokens
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Setup default tokens and pools"""
        # Create default tokens
        self.tokens.create_token('USDC', 'USD Coin', Decimal('1000000000'), 6)
        self.tokens.create_token('ETH', 'Ethereum', Decimal('1000000'), 18)
        self.tokens.create_token('BTC', 'Bitcoin', Decimal('21000000'), 8)
        
        # Create default pools
        self.pools.create_pool('ETH', 'USDC')
        self.pools.create_pool('BTC', 'USDC')
        self.pools.create_pool('ETH', 'BTC')
    
    def create_account(self) -> str:
        """Create new account"""
        account = Account.create()
        
        try:
            self.db.execute('INSERT INTO accounts (address) VALUES (?)', (account.address,))
            self.db.commit()
            self.accounts[account.address] = account
            logging.info(f'Account created: {account.address}')
            return account.address
        except:
            self.db.rollback()
            return None
    
    def get_account_info(self, address: str) -> Dict[str, Any]:
        """Get account information"""
        info = {
            'address': address,
            'balances': {}
        }
        
        for token in self.tokens.tokens:
            balance = self.tokens.get_balance(address, token)
            if balance > 0:
                info['balances'][token] = str(balance)
        
        return info
    
    def get_pool_info(self, token0: str, token1: str) -> Dict[str, Any]:
        """Get pool information"""
        if token0 > token1:
            token0, token1 = token1, token0
        
        key = f"{token0}-{token1}"
        pool = self.pools.pools.get(key)
        
        if not pool:
            return None
        
        return {
            'token0': token0,
            'token1': token1,
            'reserve0': str(pool['reserve0']),
            'reserve1': str(pool['reserve1']),
            'price': str(self.pools.get_price(token0, token1) or 0)
        }
    
    def run_demo(self):
        """Run system demonstration"""
        print("\n" + "="*60)
        print(" SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Create accounts
        print("\n[1] Creating Accounts...")
        acc1 = self.create_account()
        acc2 = self.create_account()
        print(f"    Account 1: {acc1[:10]}...")
        print(f"    Account 2: {acc2[:10]}...")
        
        # Mint tokens
        print("\n[2] Minting Tokens...")
        self.tokens.mint(acc1, 'ETH', Decimal('100'))
        self.tokens.mint(acc1, 'USDC', Decimal('200000'))
        self.tokens.mint(acc2, 'BTC', Decimal('10'))
        print(f"    Minted 100 ETH to Account 1")
        print(f"    Minted 200,000 USDC to Account 1")
        print(f"    Minted 10 BTC to Account 2")
        
        # Check balances
        print("\n[3] Account Balances:")
        info1 = self.get_account_info(acc1)
        for token, balance in info1['balances'].items():
            print(f"    Account 1 - {token}: {balance}")
        
        # Transfer
        print("\n[4] Transferring Tokens...")
        tx = self.tokens.transfer(acc1, acc2, 'ETH', Decimal('10'))
        if tx:
            print(f"    Transfer successful: {tx[:10]}...")
        
        # Add liquidity
        print("\n[5] Adding Liquidity...")
        self.pools.add_liquidity('ETH', 'USDC', Decimal('50'), Decimal('100000'))
        print(f"    Added 50 ETH + 100,000 USDC to pool")
        
        # Get pool info
        print("\n[6] Pool Information:")
        pool_info = self.get_pool_info('ETH', 'USDC')
        if pool_info:
            print(f"    ETH Reserve: {pool_info['reserve0']}")
            print(f"    USDC Reserve: {pool_info['reserve1']}")
            print(f"    ETH Price: {pool_info['price']} USDC")
        
        # Perform swap
        print("\n[7] Performing Swap...")
        swap_out = self.pools.swap('ETH', 'USDC', Decimal('1'))
        if swap_out:
            print(f"    Swapped 1 ETH for {swap_out:.2f} USDC")
        
        # Final balances
        print("\n[8] Final Balances:")
        info1 = self.get_account_info(acc1)
        info2 = self.get_account_info(acc2)
        print("    Account 1:")
        for token, balance in info1['balances'].items():
            print(f"      {token}: {balance}")
        print("    Account 2:")
        for token, balance in info2['balances'].items():
            print(f"      {token}: {balance}")
        
        print("\n" + "="*60)
        print(" DEMONSTRATION COMPLETE")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    system = System()
    
    # Run demonstration
    system.run_demo()
    
    # Keep running for API/service mode
    print("\nSystem ready. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()