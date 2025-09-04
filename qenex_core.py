#!/usr/bin/env python3
"""
QENEX Core Banking System
Production-ready implementation with real functionality
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import secrets
import hmac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Core Banking Engine
# ============================================================================

class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"

class AccountType(Enum):
    """Account type enumeration"""
    CHECKING = "checking"
    SAVINGS = "savings"
    LOAN = "loan"
    CREDIT = "credit"

@dataclass
class Account:
    """Bank account representation"""
    id: str
    account_number: str
    account_type: AccountType
    currency: str
    balance: Decimal
    available_balance: Decimal
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    is_frozen: bool = False
    overdraft_limit: Decimal = Decimal('0')
    interest_rate: Decimal = Decimal('0')
    metadata: Dict = field(default_factory=dict)

@dataclass
class Transaction:
    """Transaction representation"""
    id: str
    from_account: str
    to_account: str
    amount: Decimal
    currency: str
    status: TransactionStatus
    reference: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    reversed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    fee: Decimal = Decimal('0')
    exchange_rate: Decimal = Decimal('1')

class BankingCore:
    """Core banking engine with ACID compliance"""
    
    def __init__(self, db_path: str = "qenex_bank.db"):
        self.db_path = db_path
        self.init_database()
        self._transaction_locks = {}
        self._session_locks = asyncio.Lock()
        
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Accounts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                account_number TEXT UNIQUE NOT NULL,
                account_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                balance REAL NOT NULL,
                available_balance REAL NOT NULL,
                overdraft_limit REAL DEFAULT 0,
                interest_rate REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                is_frozen INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                from_account TEXT,
                to_account TEXT,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                reference TEXT,
                fee REAL DEFAULT 0,
                exchange_rate REAL DEFAULT 1,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                reversed_at TEXT,
                metadata TEXT,
                FOREIGN KEY (from_account) REFERENCES accounts(account_number),
                FOREIGN KEY (to_account) REFERENCES accounts(account_number)
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                entity_type TEXT,
                entity_id TEXT,
                old_value TEXT,
                new_value TEXT,
                metadata TEXT
            )
        ''')
        
        # Transaction ledger for double-entry bookkeeping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ledger (
                id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                account_number TEXT NOT NULL,
                debit REAL,
                credit REAL,
                balance_after REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions(id),
                FOREIGN KEY (account_number) REFERENCES accounts(account_number)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_account_number ON accounts(account_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_status ON transactions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_account ON ledger(account_number)')
        
        conn.commit()
        conn.close()
        
    async def create_account(
        self,
        account_type: AccountType,
        currency: str,
        initial_balance: Decimal = Decimal('0'),
        overdraft_limit: Decimal = Decimal('0'),
        interest_rate: Decimal = Decimal('0')
    ) -> Account:
        """Create a new bank account"""
        account_id = str(uuid.uuid4())
        account_number = self.generate_account_number()
        now = datetime.now(timezone.utc)
        
        account = Account(
            id=account_id,
            account_number=account_number,
            account_type=account_type,
            currency=currency,
            balance=initial_balance,
            available_balance=initial_balance,
            overdraft_limit=overdraft_limit,
            interest_rate=interest_rate,
            created_at=now,
            updated_at=now
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO accounts (
                id, account_number, account_type, currency, balance,
                available_balance, overdraft_limit, interest_rate,
                is_active, is_frozen, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            account.id, account.account_number, account.account_type.value,
            account.currency, float(account.balance),
            float(account.available_balance), float(account.overdraft_limit),
            float(account.interest_rate), account.is_active,
            account.is_frozen, account.created_at.isoformat(),
            account.updated_at.isoformat(), json.dumps(account.metadata)
        ))
        
        # Audit log - convert account to dict with JSON-serializable values
        account_dict = {
            k: v.value if isinstance(v, Enum) else (str(v) if isinstance(v, (datetime, Decimal)) else v)
            for k, v in asdict(account).items()
        }
        self._audit_log(conn, "ACCOUNT_CREATED", account_id, "account", account_id, None, account_dict)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Account created: {account_number}")
        return account
        
    async def process_transaction(
        self,
        from_account: str,
        to_account: str,
        amount: Decimal,
        currency: str,
        reference: str = ""
    ) -> Transaction:
        """Process a transaction with ACID guarantees"""
        if amount <= 0:
            raise ValueError("Transaction amount must be positive")
            
        transaction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        transaction = Transaction(
            id=transaction_id,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            currency=currency,
            status=TransactionStatus.PENDING,
            reference=reference,
            created_at=now
        )
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN IMMEDIATE")  # Start transaction
        
        try:
            cursor = conn.cursor()
            
            # Insert pending transaction
            cursor.execute('''
                INSERT INTO transactions (
                    id, from_account, to_account, amount, currency,
                    status, reference, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.id, transaction.from_account, transaction.to_account,
                float(transaction.amount), transaction.currency,
                transaction.status.value, transaction.reference,
                transaction.created_at.isoformat(), json.dumps(transaction.metadata)
            ))
            
            # Lock and get source account
            cursor.execute('''
                SELECT balance, available_balance, overdraft_limit, is_frozen
                FROM accounts
                WHERE account_number = ?
                FOR UPDATE
            ''', (from_account,))
            
            source = cursor.fetchone()
            if not source:
                raise ValueError(f"Source account {from_account} not found")
                
            balance, available, overdraft, is_frozen = source
            if is_frozen:
                raise ValueError("Source account is frozen")
                
            # Check sufficient funds
            total_available = Decimal(str(available)) + Decimal(str(overdraft))
            if total_available < amount:
                raise ValueError("Insufficient funds")
                
            # Lock and get destination account
            cursor.execute('''
                SELECT balance, is_frozen
                FROM accounts
                WHERE account_number = ?
                FOR UPDATE
            ''', (to_account,))
            
            dest = cursor.fetchone()
            if not dest:
                raise ValueError(f"Destination account {to_account} not found")
                
            dest_balance, dest_frozen = dest
            if dest_frozen:
                raise ValueError("Destination account is frozen")
                
            # Update balances
            new_source_balance = Decimal(str(balance)) - amount
            new_source_available = Decimal(str(available)) - amount
            new_dest_balance = Decimal(str(dest_balance)) + amount
            
            cursor.execute('''
                UPDATE accounts
                SET balance = ?, available_balance = ?, updated_at = ?
                WHERE account_number = ?
            ''', (
                float(new_source_balance), float(new_source_available),
                now.isoformat(), from_account
            ))
            
            cursor.execute('''
                UPDATE accounts
                SET balance = ?, available_balance = balance, updated_at = ?
                WHERE account_number = ?
            ''', (
                float(new_dest_balance), now.isoformat(), to_account
            ))
            
            # Record in ledger (double-entry)
            cursor.execute('''
                INSERT INTO ledger (id, transaction_id, account_number, debit, credit, balance_after, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), transaction_id, from_account,
                float(amount), None, float(new_source_balance), now.isoformat()
            ))
            
            cursor.execute('''
                INSERT INTO ledger (id, transaction_id, account_number, debit, credit, balance_after, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), transaction_id, to_account,
                None, float(amount), float(new_dest_balance), now.isoformat()
            ))
            
            # Update transaction status
            cursor.execute('''
                UPDATE transactions
                SET status = ?, completed_at = ?
                WHERE id = ?
            ''', (TransactionStatus.COMPLETED.value, now.isoformat(), transaction_id))
            
            # Audit log
            self._audit_log(conn, "TRANSACTION_COMPLETED", transaction_id, "transaction", 
                          transaction_id, None, asdict(transaction))
            
            conn.commit()
            transaction.status = TransactionStatus.COMPLETED
            transaction.completed_at = now
            
            logger.info(f"Transaction completed: {transaction_id}")
            return transaction
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            
            # Update transaction status to failed
            conn.execute('''
                UPDATE transactions
                SET status = ?
                WHERE id = ?
            ''', (TransactionStatus.FAILED.value, transaction_id))
            conn.commit()
            
            transaction.status = TransactionStatus.FAILED
            raise
            
        finally:
            conn.close()
            
    async def reverse_transaction(self, transaction_id: str) -> Transaction:
        """Reverse a completed transaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get original transaction
        cursor.execute('''
            SELECT from_account, to_account, amount, currency, reference
            FROM transactions
            WHERE id = ? AND status = ?
        ''', (transaction_id, TransactionStatus.COMPLETED.value))
        
        original = cursor.fetchone()
        if not original:
            raise ValueError("Transaction not found or not completed")
            
        conn.close()
        
        # Process reversal as new transaction
        from_acc, to_acc, amount, currency, reference = original
        return await self.process_transaction(
            to_acc, from_acc, Decimal(str(amount)), currency,
            f"Reversal of {transaction_id}: {reference}"
        )
        
    def generate_account_number(self) -> str:
        """Generate unique account number"""
        return f"ACC{int(time.time() * 1000000) % 10000000000:010d}"
        
    def _audit_log(self, conn, action, user_id, entity_type, entity_id, old_value, new_value):
        """Record audit log entry"""
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_log (
                id, timestamp, action, user_id, entity_type,
                entity_id, old_value, new_value, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), action,
            user_id, entity_type, entity_id,
            json.dumps(old_value, default=lambda x: x.value if isinstance(x, Enum) else str(x)) if old_value else None,
            json.dumps(new_value, default=lambda x: x.value if isinstance(x, Enum) else str(x)) if new_value else None,
            None
        ))
        
    async def get_balance(self, account_number: str) -> Decimal:
        """Get account balance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT balance FROM accounts WHERE account_number = ?', (account_number,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Account {account_number} not found")
            
        return Decimal(str(result[0]))
        
    async def get_account(self, account_number: str) -> Optional[Account]:
        """Get account details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM accounts WHERE account_number = ?', (account_number,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return Account(
            id=row[0],
            account_number=row[1],
            account_type=AccountType(row[2]),
            currency=row[3],
            balance=Decimal(str(row[4])),
            available_balance=Decimal(str(row[5])),
            overdraft_limit=Decimal(str(row[6])),
            interest_rate=Decimal(str(row[7])),
            is_active=bool(row[8]),
            is_frozen=bool(row[9]),
            created_at=datetime.fromisoformat(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
            metadata=json.loads(row[12]) if row[12] else {}
        )

# ============================================================================
# Security and Authentication
# ============================================================================

class AuthenticationSystem:
    """Secure authentication with hashing and sessions"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.sessions = {}
        self.users = {}
        self.failed_attempts = {}
        self.init_database()
        
    def init_database(self):
        """Initialize auth database"""
        conn = sqlite3.connect("qenex_auth.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                mfa_secret TEXT,
                is_active INTEGER DEFAULT 1,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def hash_password(self, password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for password hashing
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return key.hex(), salt
        
    async def register_user(self, username: str, email: str, password: str, is_admin: bool = False):
        """Register new user"""
        user_id = str(uuid.uuid4())
        password_hash, salt = self.hash_password(password)
        
        conn = sqlite3.connect("qenex_auth.db")
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (
                    id, username, email, password_hash, salt,
                    is_active, is_admin, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, email, password_hash, salt.hex(),
                1, int(is_admin), datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            logger.info(f"User registered: {username}")
            return user_id
            
        except sqlite3.IntegrityError:
            raise ValueError("Username or email already exists")
            
        finally:
            conn.close()
            
    async def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and create session"""
        # Check failed attempts
        if username in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[username]
            if attempts >= 5 and (datetime.now(timezone.utc) - last_attempt).seconds < 900:  # 15 min lockout
                raise ValueError("Account locked due to too many failed attempts")
                
        conn = sqlite3.connect("qenex_auth.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, password_hash, salt, is_active
            FROM users
            WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            self._record_failed_attempt(username)
            return None
            
        user_id, stored_hash, salt, is_active = user
        
        if not is_active:
            raise ValueError("Account is disabled")
            
        # Verify password
        computed_hash, _ = self.hash_password(password, bytes.fromhex(salt))
        
        if computed_hash != stored_hash:
            self._record_failed_attempt(username)
            return None
            
        # Clear failed attempts
        if username in self.failed_attempts:
            del self.failed_attempts[username]
            
        # Create session
        session_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
        
        cursor.execute('''
            INSERT INTO sessions (
                id, user_id, token, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id, user_id, token,
            datetime.now(timezone.utc).isoformat(), expires_at.isoformat()
        ))
        
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(timezone.utc).isoformat(), user_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User authenticated: {username}")
        return token
        
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username in self.failed_attempts:
            attempts, _ = self.failed_attempts[username]
            self.failed_attempts[username] = (attempts + 1, datetime.now(timezone.utc))
        else:
            self.failed_attempts[username] = (1, datetime.now(timezone.utc))
            
    async def validate_token(self, token: str) -> Optional[str]:
        """Validate session token"""
        conn = sqlite3.connect("qenex_auth.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, expires_at
            FROM sessions
            WHERE token = ?
        ''', (token,))
        
        session = cursor.fetchone()
        conn.close()
        
        if not session:
            return None
            
        user_id, expires_at = session
        
        if datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
            return None
            
        return user_id

# ============================================================================
# Main Application
# ============================================================================

async def main():
    """Main application entry point"""
    logger.info("QENEX Banking System Starting...")
    
    # Initialize systems
    banking = BankingCore()
    auth = AuthenticationSystem()
    
    # Create demo accounts
    try:
        # Create users
        admin_id = await auth.register_user("admin", "admin@qenex.ai", "AdminPass123!", is_admin=True)
        user_id = await auth.register_user("demo", "demo@qenex.ai", "DemoPass123!")
        
        # Create bank accounts
        checking = await banking.create_account(
            AccountType.CHECKING, "USD", Decimal("1000.00"), Decimal("500.00")
        )
        savings = await banking.create_account(
            AccountType.SAVINGS, "USD", Decimal("5000.00"), interest_rate=Decimal("0.02")
        )
        
        logger.info(f"Demo accounts created:")
        logger.info(f"  Checking: {checking.account_number} (Balance: ${checking.balance})")
        logger.info(f"  Savings: {savings.account_number} (Balance: ${savings.balance})")
        
        # Test transaction
        tx = await banking.process_transaction(
            checking.account_number, savings.account_number,
            Decimal("100.00"), "USD", "Test transfer"
        )
        logger.info(f"Test transaction: {tx.id} - {tx.status.value}")
        
        # Check balances
        checking_balance = await banking.get_balance(checking.account_number)
        savings_balance = await banking.get_balance(savings.account_number)
        
        logger.info(f"Updated balances:")
        logger.info(f"  Checking: ${checking_balance}")
        logger.info(f"  Savings: ${savings_balance}")
        
    except ValueError as e:
        logger.info(f"Demo setup: {e}")
        
    logger.info("QENEX Banking System Ready")
    
if __name__ == "__main__":
    asyncio.run(main())