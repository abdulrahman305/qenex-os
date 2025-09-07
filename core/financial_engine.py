#!/usr/bin/env python3
"""
QENEX Financial Engine Core
Enterprise-grade transaction processing and financial management system
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import base64

# Set maximum precision for financial calculations
getcontext().prec = 38
getcontext().traps[Decimal.InvalidOperation] = 1
getcontext().traps[Decimal.Overflow] = 1

logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    """Transaction lifecycle states"""
    PENDING = auto()
    VALIDATING = auto()
    PROCESSING = auto()
    SETTLED = auto()
    FAILED = auto()
    REVERSED = auto()
    CANCELLED = auto()

class AccountType(Enum):
    """Account classification"""
    RETAIL = auto()
    CORPORATE = auto()
    INSTITUTIONAL = auto()
    GOVERNMENT = auto()
    SETTLEMENT = auto()
    ESCROW = auto()
    RESERVE = auto()

class ComplianceLevel(Enum):
    """Regulatory compliance levels"""
    UNVERIFIED = 0
    BASIC_KYC = 1
    ENHANCED_KYC = 2
    INSTITUTIONAL = 3
    GOVERNMENT = 4
    CENTRAL_BANK = 5

@dataclass
class RiskMetrics:
    """Real-time risk assessment metrics"""
    transaction_risk: float = 0.0
    counterparty_risk: float = 0.0
    market_risk: float = 0.0
    liquidity_risk: float = 0.0
    operational_risk: float = 0.0
    compliance_risk: float = 0.0
    aggregate_score: float = 0.0
    risk_rating: str = "LOW"
    
    def calculate_aggregate(self) -> float:
        """Calculate weighted aggregate risk score"""
        weights = {
            'transaction': 0.20,
            'counterparty': 0.25,
            'market': 0.15,
            'liquidity': 0.15,
            'operational': 0.10,
            'compliance': 0.15
        }
        
        self.aggregate_score = (
            self.transaction_risk * weights['transaction'] +
            self.counterparty_risk * weights['counterparty'] +
            self.market_risk * weights['market'] +
            self.liquidity_risk * weights['liquidity'] +
            self.operational_risk * weights['operational'] +
            self.compliance_risk * weights['compliance']
        )
        
        if self.aggregate_score < 0.3:
            self.risk_rating = "LOW"
        elif self.aggregate_score < 0.6:
            self.risk_rating = "MEDIUM"
        elif self.aggregate_score < 0.8:
            self.risk_rating = "HIGH"
        else:
            self.risk_rating = "CRITICAL"
            
        return self.aggregate_score

@dataclass
class FinancialTransaction:
    """Comprehensive financial transaction record"""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_type: str = "TRANSFER"
    from_account: Optional[str] = None
    to_account: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "USD"
    exchange_rate: Decimal = Decimal("1.00")
    fee_amount: Decimal = Decimal("0.00")
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_date: Optional[datetime] = None
    reference_number: str = field(default_factory=lambda: secrets.token_hex(16))
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    compliance_checks: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail"""
        self.audit_trail.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'details': details
        })
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate transaction integrity"""
        if self.amount <= 0:
            return False, "Invalid amount"
        if not self.to_account:
            return False, "Missing recipient account"
        if self.from_account == self.to_account:
            return False, "Same account transfer not allowed"
        return True, None

@dataclass
class Account:
    """Enhanced financial account with comprehensive features"""
    account_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    account_number: str = field(default_factory=lambda: secrets.token_hex(8).upper())
    account_type: AccountType = AccountType.RETAIL
    owner_id: str = ""
    balance: Decimal = Decimal("0.00")
    available_balance: Decimal = Decimal("0.00")
    currency: str = "USD"
    status: str = "ACTIVE"
    compliance_level: ComplianceLevel = ComplianceLevel.UNVERIFIED
    daily_limit: Decimal = Decimal("50000.00")
    transaction_limit: Decimal = Decimal("10000.00")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    transaction_history: List[str] = field(default_factory=list)
    holds: List[Dict[str, Any]] = field(default_factory=list)
    
    def place_hold(self, amount: Decimal, reference: str, expires_at: datetime) -> bool:
        """Place a hold on funds"""
        if self.available_balance >= amount:
            self.holds.append({
                'amount': str(amount),
                'reference': reference,
                'placed_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': expires_at.isoformat()
            })
            self.available_balance -= amount
            return True
        return False
    
    def release_hold(self, reference: str) -> Decimal:
        """Release a specific hold"""
        for hold in self.holds[:]:
            if hold['reference'] == reference:
                amount = Decimal(hold['amount'])
                self.available_balance += amount
                self.holds.remove(hold)
                return amount
        return Decimal("0.00")

class FinancialEngine:
    """Core financial processing engine"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.db_path = self.data_dir / "financial.db"
        self._init_database()
        
        # In-memory caches
        self.accounts: Dict[str, Account] = {}
        self.transactions: Dict[str, FinancialTransaction] = {}
        self.pending_transactions: List[FinancialTransaction] = []
        
        # Processing configuration
        self.batch_size = 100
        self.settlement_interval = 60  # seconds
        self.max_retries = 3
        
        # Security
        self.master_key = secrets.token_bytes(32)
        self.session_keys: Dict[str, bytes] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.processing_lock = threading.Lock()
        self.settlement_lock = threading.Lock()
        
        # Monitoring
        self.metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'total_volume': Decimal("0.00"),
            'average_processing_time': 0.0
        }
        
        logger.info("Financial Engine initialized")
    
    def _init_database(self) -> None:
        """Initialize database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Accounts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                account_number TEXT UNIQUE NOT NULL,
                account_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                balance TEXT NOT NULL,
                available_balance TEXT NOT NULL,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                compliance_level INTEGER NOT NULL,
                daily_limit TEXT NOT NULL,
                transaction_limit TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                holds TEXT
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                transaction_type TEXT NOT NULL,
                from_account TEXT,
                to_account TEXT NOT NULL,
                amount TEXT NOT NULL,
                currency TEXT NOT NULL,
                exchange_rate TEXT NOT NULL,
                fee_amount TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                settlement_date TEXT,
                reference_number TEXT UNIQUE NOT NULL,
                metadata TEXT,
                risk_score REAL,
                audit_trail TEXT
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                actor TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_accounts_owner ON accounts(owner_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_from ON transactions(from_account)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_to ON transactions(to_account)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)')
        
        conn.commit()
        conn.close()
    
    def create_account(self, owner_id: str, account_type: AccountType = AccountType.RETAIL,
                      initial_balance: Decimal = Decimal("0.00"),
                      currency: str = "USD") -> Account:
        """Create a new financial account"""
        account = Account(
            owner_id=owner_id,
            account_type=account_type,
            balance=initial_balance,
            available_balance=initial_balance,
            currency=currency
        )
        
        # Store in cache and database
        self.accounts[account.account_id] = account
        self._persist_account(account)
        
        # Audit log
        self._audit_log("ACCOUNT_CREATED", "ACCOUNT", account.account_id, {
            'owner_id': owner_id,
            'account_type': account_type.name,
            'initial_balance': str(initial_balance),
            'currency': currency
        })
        
        logger.info(f"Account created: {account.account_id}")
        return account
    
    def _persist_account(self, account: Account) -> None:
        """Persist account to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO accounts 
            (account_id, account_number, account_type, owner_id, balance, 
             available_balance, currency, status, compliance_level, 
             daily_limit, transaction_limit, created_at, updated_at, 
             metadata, holds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            account.account_id,
            account.account_number,
            account.account_type.name,
            account.owner_id,
            str(account.balance),
            str(account.available_balance),
            account.currency,
            account.status,
            account.compliance_level.value,
            str(account.daily_limit),
            str(account.transaction_limit),
            account.created_at.isoformat(),
            account.updated_at.isoformat(),
            json.dumps(account.metadata),
            json.dumps(account.holds)
        ))
        
        conn.commit()
        conn.close()
    
    def process_transaction(self, transaction: FinancialTransaction) -> Tuple[bool, str]:
        """Process a financial transaction with full validation"""
        try:
            # Initial validation
            is_valid, error = transaction.validate()
            if not is_valid:
                transaction.status = TransactionStatus.FAILED
                return False, error
            
            transaction.status = TransactionStatus.VALIDATING
            transaction.add_audit_entry("VALIDATION_STARTED", {})
            
            # Risk assessment
            transaction.risk_metrics = self._assess_risk(transaction)
            if transaction.risk_metrics.risk_rating == "CRITICAL":
                transaction.status = TransactionStatus.FAILED
                transaction.add_audit_entry("RISK_CHECK_FAILED", {
                    'risk_score': transaction.risk_metrics.aggregate_score,
                    'risk_rating': transaction.risk_metrics.risk_rating
                })
                return False, "Transaction failed risk assessment"
            
            # Compliance checks
            compliance_result = self._compliance_check(transaction)
            if not compliance_result['passed']:
                transaction.status = TransactionStatus.FAILED
                transaction.add_audit_entry("COMPLIANCE_CHECK_FAILED", compliance_result)
                return False, f"Compliance check failed: {compliance_result.get('reason', 'Unknown')}"
            
            transaction.status = TransactionStatus.PROCESSING
            transaction.add_audit_entry("PROCESSING_STARTED", {})
            
            # Execute transaction
            with self.processing_lock:
                success, message = self._execute_transaction(transaction)
                
            if success:
                transaction.status = TransactionStatus.SETTLED
                transaction.settlement_date = datetime.now(timezone.utc)
                transaction.add_audit_entry("TRANSACTION_SETTLED", {
                    'message': message
                })
                
                # Update metrics
                self.metrics['total_transactions'] += 1
                self.metrics['successful_transactions'] += 1
                self.metrics['total_volume'] += transaction.amount
                
                # Persist transaction
                self._persist_transaction(transaction)
                
                logger.info(f"Transaction settled: {transaction.transaction_id}")
                return True, f"Transaction completed: {transaction.transaction_id}"
            else:
                transaction.status = TransactionStatus.FAILED
                transaction.add_audit_entry("TRANSACTION_FAILED", {
                    'error': message
                })
                self.metrics['failed_transactions'] += 1
                return False, message
                
        except Exception as e:
            logger.error(f"Transaction processing error: {e}")
            transaction.status = TransactionStatus.FAILED
            transaction.add_audit_entry("SYSTEM_ERROR", {
                'error': str(e)
            })
            return False, f"System error: {str(e)}"
    
    def _assess_risk(self, transaction: FinancialTransaction) -> RiskMetrics:
        """Comprehensive risk assessment"""
        risk = RiskMetrics()
        
        # Transaction risk based on amount
        if transaction.amount > Decimal("100000"):
            risk.transaction_risk = 0.8
        elif transaction.amount > Decimal("50000"):
            risk.transaction_risk = 0.6
        elif transaction.amount > Decimal("10000"):
            risk.transaction_risk = 0.4
        else:
            risk.transaction_risk = 0.2
        
        # Counterparty risk (simplified)
        if transaction.from_account and transaction.from_account in self.accounts:
            from_account = self.accounts[transaction.from_account]
            if from_account.compliance_level.value < 2:
                risk.counterparty_risk = 0.7
            else:
                risk.counterparty_risk = 0.3
        
        # Market risk for cross-currency
        if transaction.exchange_rate != Decimal("1.00"):
            risk.market_risk = 0.5
        
        # Calculate aggregate
        risk.calculate_aggregate()
        
        return risk
    
    def _compliance_check(self, transaction: FinancialTransaction) -> Dict[str, Any]:
        """Regulatory compliance verification"""
        result = {
            'passed': True,
            'checks': [],
            'reason': None
        }
        
        # AML check - amounts over threshold
        if transaction.amount > Decimal("10000"):
            result['checks'].append('AML_LARGE_TRANSACTION')
            if transaction.from_account:
                from_account = self.accounts.get(transaction.from_account)
                if from_account and from_account.compliance_level.value < 2:
                    result['passed'] = False
                    result['reason'] = "Enhanced KYC required for large transactions"
        
        # Sanctions check (simplified)
        result['checks'].append('SANCTIONS_SCREENING')
        
        # Transaction pattern analysis
        result['checks'].append('PATTERN_ANALYSIS')
        
        return result
    
    def _execute_transaction(self, transaction: FinancialTransaction) -> Tuple[bool, str]:
        """Execute the actual fund transfer"""
        try:
            # Get accounts
            to_account = self.accounts.get(transaction.to_account)
            if not to_account:
                return False, "Recipient account not found"
            
            if transaction.from_account:
                from_account = self.accounts.get(transaction.from_account)
                if not from_account:
                    return False, "Sender account not found"
                
                # Check balance
                total_amount = transaction.amount + transaction.fee_amount
                if from_account.available_balance < total_amount:
                    return False, "Insufficient funds"
                
                # Debit sender
                from_account.balance -= total_amount
                from_account.available_balance -= total_amount
                from_account.transaction_history.append(transaction.transaction_id)
                from_account.updated_at = datetime.now(timezone.utc)
                self._persist_account(from_account)
            
            # Credit recipient
            to_account.balance += transaction.amount
            to_account.available_balance += transaction.amount
            to_account.transaction_history.append(transaction.transaction_id)
            to_account.updated_at = datetime.now(timezone.utc)
            self._persist_account(to_account)
            
            return True, "Transfer completed"
            
        except Exception as e:
            logger.error(f"Transaction execution error: {e}")
            return False, str(e)
    
    def _persist_transaction(self, transaction: FinancialTransaction) -> None:
        """Persist transaction to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO transactions 
            (transaction_id, transaction_type, from_account, to_account, 
             amount, currency, exchange_rate, fee_amount, status, 
             timestamp, settlement_date, reference_number, metadata, 
             risk_score, audit_trail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction.transaction_id,
            transaction.transaction_type,
            transaction.from_account,
            transaction.to_account,
            str(transaction.amount),
            transaction.currency,
            str(transaction.exchange_rate),
            str(transaction.fee_amount),
            transaction.status.name,
            transaction.timestamp.isoformat(),
            transaction.settlement_date.isoformat() if transaction.settlement_date else None,
            transaction.reference_number,
            json.dumps(transaction.metadata),
            transaction.risk_metrics.aggregate_score,
            json.dumps(transaction.audit_trail)
        ))
        
        conn.commit()
        conn.close()
    
    def _audit_log(self, event_type: str, entity_type: str, entity_id: str, 
                   details: Dict[str, Any]) -> None:
        """Create audit log entry"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log 
            (timestamp, event_type, entity_type, entity_id, action, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            event_type,
            entity_type,
            entity_id,
            event_type,
            json.dumps(details)
        ))
        
        conn.commit()
        conn.close()
    
    def get_account_balance(self, account_id: str) -> Optional[Dict[str, str]]:
        """Get account balance information"""
        account = self.accounts.get(account_id)
        if account:
            return {
                'account_id': account.account_id,
                'balance': str(account.balance),
                'available_balance': str(account.available_balance),
                'currency': account.currency,
                'status': account.status
            }
        return None
    
    def get_transaction_history(self, account_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history for an account"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transaction_id, transaction_type, from_account, to_account, 
                   amount, currency, status, timestamp, reference_number
            FROM transactions 
            WHERE from_account = ? OR to_account = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (account_id, account_id, limit))
        
        transactions = []
        for row in cursor.fetchall():
            transactions.append({
                'transaction_id': row[0],
                'type': row[1],
                'from_account': row[2],
                'to_account': row[3],
                'amount': row[4],
                'currency': row[5],
                'status': row[6],
                'timestamp': row[7],
                'reference': row[8]
            })
        
        conn.close()
        return transactions
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            'total_transactions': self.metrics['total_transactions'],
            'successful_transactions': self.metrics['successful_transactions'],
            'failed_transactions': self.metrics['failed_transactions'],
            'success_rate': (self.metrics['successful_transactions'] / 
                           max(self.metrics['total_transactions'], 1)) * 100,
            'total_volume': str(self.metrics['total_volume']),
            'active_accounts': len(self.accounts),
            'pending_transactions': len(self.pending_transactions)
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the engine"""
        logger.info("Shutting down Financial Engine...")
        self.executor.shutdown(wait=True)
        logger.info("Financial Engine shutdown complete")

def main():
    """Main entry point for testing"""
    engine = FinancialEngine()
    
    # Create test accounts
    account1 = engine.create_account("user1", AccountType.RETAIL, Decimal("10000.00"))
    account2 = engine.create_account("user2", AccountType.RETAIL, Decimal("5000.00"))
    
    # Create and process test transaction
    transaction = FinancialTransaction(
        from_account=account1.account_id,
        to_account=account2.account_id,
        amount=Decimal("100.00"),
        transaction_type="TRANSFER"
    )
    
    success, message = engine.process_transaction(transaction)
    print(f"Transaction result: {success} - {message}")
    
    # Check balances
    balance1 = engine.get_account_balance(account1.account_id)
    balance2 = engine.get_account_balance(account2.account_id)
    print(f"Account 1 balance: {balance1}")
    print(f"Account 2 balance: {balance2}")
    
    # Get metrics
    metrics = engine.get_system_metrics()
    print(f"System metrics: {metrics}")
    
    engine.shutdown()

if __name__ == "__main__":
    main()