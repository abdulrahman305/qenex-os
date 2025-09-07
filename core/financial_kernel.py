#!/usr/bin/env python3
"""
QENEX Financial Operating System - Core Kernel
Production-ready financial transaction processing engine
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import asyncpg
import cryptography.hazmat.primitives.kdf.pbkdf2 as pbkdf2
import redis.asyncio as redis
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Set decimal precision for financial calculations
getcontext().prec = 38  # Support for 38 significant digits

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/qenex/kernel.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
transaction_counter = Counter('qenex_transactions_total', 'Total transactions processed')
transaction_latency = Histogram('qenex_transaction_duration_seconds', 'Transaction processing time')
active_connections = Gauge('qenex_active_connections', 'Number of active connections')
balance_gauge = Gauge('qenex_total_balance', 'Total system balance')


class TransactionType(Enum):
    """Financial transaction types"""
    TRANSFER = auto()
    DEPOSIT = auto()
    WITHDRAWAL = auto()
    PAYMENT = auto()
    SETTLEMENT = auto()
    FOREX = auto()
    SECURITIES = auto()
    DERIVATIVES = auto()


class TransactionStatus(Enum):
    """Transaction processing status"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REVERSED = auto()
    CANCELLED = auto()


class ComplianceLevel(Enum):
    """Compliance check levels"""
    NONE = 0
    BASIC = 1
    ENHANCED = 2
    STRICT = 3


@dataclass
class Account:
    """Financial account representation"""
    id: str
    owner_id: str
    account_type: str
    currency: str
    balance: Decimal
    created_at: datetime
    updated_at: datetime
    status: str = "ACTIVE"
    metadata: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Decimal] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert account to dictionary"""
        return {
            'id': self.id,
            'owner_id': self.owner_id,
            'account_type': self.account_type,
            'currency': self.currency,
            'balance': str(self.balance),
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata,
            'limits': {k: str(v) for k, v in self.limits.items()}
        }


@dataclass
class Transaction:
    """Financial transaction representation"""
    id: str
    type: TransactionType
    from_account: Optional[str]
    to_account: Optional[str]
    amount: Decimal
    currency: str
    status: TransactionStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    fees: Decimal = Decimal('0')
    exchange_rate: Optional[Decimal] = None
    reference: Optional[str] = None
    reversal_of: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'id': self.id,
            'type': self.type.name,
            'from_account': self.from_account,
            'to_account': self.to_account,
            'amount': str(self.amount),
            'currency': self.currency,
            'status': self.status.name,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
            'fees': str(self.fees),
            'exchange_rate': str(self.exchange_rate) if self.exchange_rate else None,
            'reference': self.reference,
            'reversal_of': self.reversal_of
        }


class CryptoEngine:
    """Cryptographic operations engine"""
    
    def __init__(self):
        self.backend = default_backend()
        self._master_key = self._generate_master_key()
        
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        # In production, this would be retrieved from HSM
        return secrets.token_bytes(32)
    
    def encrypt(self, data: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Encrypt data using AES-256-GCM"""
        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def decrypt(self, encrypted_data: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-256-GCM"""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def generate_transaction_signature(self, transaction: Transaction) -> str:
        """Generate cryptographic signature for transaction"""
        data = f"{transaction.id}:{transaction.type.name}:{transaction.amount}:{transaction.currency}"
        signature = hmac.new(
            self._master_key,
            data.encode(),
            hashlib.sha3_256
        ).hexdigest()
        return signature
    
    def verify_transaction_signature(self, transaction: Transaction, signature: str) -> bool:
        """Verify transaction signature"""
        expected = self.generate_transaction_signature(transaction)
        return hmac.compare_digest(expected, signature)


class ComplianceEngine:
    """Regulatory compliance and AML/KYC engine"""
    
    def __init__(self):
        self.sanctions_list: Set[str] = set()
        self.high_risk_countries: Set[str] = {'IR', 'KP', 'SY'}  # Example list
        self.suspicious_patterns: List[Dict[str, Any]] = []
        self._load_compliance_data()
    
    def _load_compliance_data(self):
        """Load compliance rules and sanctions lists"""
        # In production, this would load from regulatory databases
        pass
    
    async def check_transaction(
        self,
        transaction: Transaction,
        level: ComplianceLevel = ComplianceLevel.BASIC
    ) -> Tuple[bool, Optional[str]]:
        """Perform compliance checks on transaction"""
        
        # Amount threshold checks
        if transaction.amount > Decimal('10000') and level >= ComplianceLevel.BASIC:
            # CTR (Currency Transaction Report) required
            await self._file_ctr(transaction)
        
        if transaction.amount > Decimal('5000') and level >= ComplianceLevel.ENHANCED:
            # Enhanced due diligence required
            if not await self._enhanced_due_diligence(transaction):
                return False, "Enhanced due diligence failed"
        
        # Sanctions screening
        if level >= ComplianceLevel.BASIC:
            if await self._check_sanctions(transaction):
                return False, "Transaction blocked: Sanctions match"
        
        # Pattern analysis for suspicious activity
        if level >= ComplianceLevel.ENHANCED:
            suspicious = await self._detect_suspicious_activity(transaction)
            if suspicious:
                await self._file_sar(transaction)  # Suspicious Activity Report
                if level == ComplianceLevel.STRICT:
                    return False, "Transaction blocked: Suspicious activity detected"
        
        return True, None
    
    async def _check_sanctions(self, transaction: Transaction) -> bool:
        """Check transaction against sanctions lists"""
        # Simplified check - production would use full OFAC/UN lists
        metadata = transaction.metadata
        if 'sender_name' in metadata:
            if metadata['sender_name'].upper() in self.sanctions_list:
                return True
        if 'recipient_name' in metadata:
            if metadata['recipient_name'].upper() in self.sanctions_list:
                return True
        return False
    
    async def _detect_suspicious_activity(self, transaction: Transaction) -> bool:
        """Detect suspicious transaction patterns"""
        # Check for structuring (splitting large amounts)
        if transaction.amount == Decimal('9999.99'):
            return True
        
        # Check for rapid movement of funds
        if 'velocity_check' in transaction.metadata:
            if transaction.metadata['velocity_check'] > 10:  # More than 10 transactions per hour
                return True
        
        return False
    
    async def _enhanced_due_diligence(self, transaction: Transaction) -> bool:
        """Perform enhanced due diligence"""
        # In production, this would verify identity, source of funds, etc.
        return True
    
    async def _file_ctr(self, transaction: Transaction):
        """File Currency Transaction Report"""
        logger.info(f"CTR filed for transaction {transaction.id}")
        # In production, submit to FinCEN
    
    async def _file_sar(self, transaction: Transaction):
        """File Suspicious Activity Report"""
        logger.warning(f"SAR filed for transaction {transaction.id}")
        # In production, submit to FinCEN


class LedgerEngine:
    """Distributed ledger and database engine"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db_config = db_config
        
    async def initialize(self):
        """Initialize database connections"""
        # PostgreSQL connection pool
        self.db_pool = await asyncpg.create_pool(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            user=self.db_config.get('user', 'qenex'),
            password=self.db_config.get('password', ''),
            database=self.db_config.get('database', 'qenex_financial'),
            min_size=10,
            max_size=20,
            command_timeout=60
        )
        
        # Redis connection for caching
        self.redis_client = redis.Redis(
            host=self.db_config.get('redis_host', 'localhost'),
            port=self.db_config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Create tables if not exists
        await self._create_schema()
    
    async def _create_schema(self):
        """Create database schema"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id UUID PRIMARY KEY,
                    owner_id VARCHAR(255) NOT NULL,
                    account_type VARCHAR(50) NOT NULL,
                    currency VARCHAR(3) NOT NULL,
                    balance DECIMAL(38, 8) NOT NULL DEFAULT 0,
                    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    INDEX idx_owner (owner_id),
                    INDEX idx_currency (currency)
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id UUID PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    from_account UUID,
                    to_account UUID,
                    amount DECIMAL(38, 8) NOT NULL,
                    currency VARCHAR(3) NOT NULL,
                    fees DECIMAL(38, 8) DEFAULT 0,
                    exchange_rate DECIMAL(38, 8),
                    status VARCHAR(20) NOT NULL,
                    reference VARCHAR(255),
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    INDEX idx_from_account (from_account),
                    INDEX idx_to_account (to_account),
                    INDEX idx_status (status),
                    INDEX idx_created (created_at)
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id UUID NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    user_id VARCHAR(255),
                    details JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    INDEX idx_entity (entity_type, entity_id),
                    INDEX idx_created (created_at)
                )
            ''')
    
    async def create_account(self, account: Account) -> Account:
        """Create new account in ledger"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO accounts (id, owner_id, account_type, currency, balance, status, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ''', account.id, account.owner_id, account.account_type, account.currency,
                account.balance, account.status, json.dumps(account.metadata),
                account.created_at, account.updated_at)
        
        # Cache account data
        await self.redis_client.setex(
            f"account:{account.id}",
            3600,  # 1 hour TTL
            json.dumps(account.to_dict())
        )
        
        return account
    
    async def get_account(self, account_id: str) -> Optional[Account]:
        """Retrieve account from ledger"""
        # Check cache first
        cached = await self.redis_client.get(f"account:{account_id}")
        if cached:
            data = json.loads(cached)
            return Account(
                id=data['id'],
                owner_id=data['owner_id'],
                account_type=data['account_type'],
                currency=data['currency'],
                balance=Decimal(data['balance']),
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at']),
                status=data['status'],
                metadata=data['metadata']
            )
        
        # Query database
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM accounts WHERE id = $1',
                uuid.UUID(account_id)
            )
            
            if row:
                account = Account(
                    id=str(row['id']),
                    owner_id=row['owner_id'],
                    account_type=row['account_type'],
                    currency=row['currency'],
                    balance=row['balance'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    status=row['status'],
                    metadata=row['metadata'] or {}
                )
                
                # Update cache
                await self.redis_client.setex(
                    f"account:{account_id}",
                    3600,
                    json.dumps(account.to_dict())
                )
                
                return account
        
        return None
    
    async def update_balance(
        self,
        account_id: str,
        amount: Decimal,
        operation: str = 'add'
    ) -> bool:
        """Update account balance atomically"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Lock account row for update
                row = await conn.fetchrow(
                    'SELECT balance FROM accounts WHERE id = $1 FOR UPDATE',
                    uuid.UUID(account_id)
                )
                
                if not row:
                    return False
                
                current_balance = row['balance']
                
                if operation == 'add':
                    new_balance = current_balance + amount
                elif operation == 'subtract':
                    new_balance = current_balance - amount
                    if new_balance < 0:
                        raise ValueError("Insufficient funds")
                else:
                    raise ValueError(f"Invalid operation: {operation}")
                
                await conn.execute(
                    'UPDATE accounts SET balance = $1, updated_at = NOW() WHERE id = $2',
                    new_balance, uuid.UUID(account_id)
                )
                
                # Invalidate cache
                await self.redis_client.delete(f"account:{account_id}")
                
                return True
    
    async def record_transaction(self, transaction: Transaction) -> Transaction:
        """Record transaction in ledger"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO transactions (
                    id, type, from_account, to_account, amount, currency,
                    fees, exchange_rate, status, reference, metadata,
                    created_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ''', uuid.UUID(transaction.id), transaction.type.name,
                uuid.UUID(transaction.from_account) if transaction.from_account else None,
                uuid.UUID(transaction.to_account) if transaction.to_account else None,
                transaction.amount, transaction.currency, transaction.fees,
                transaction.exchange_rate, transaction.status.name,
                transaction.reference, json.dumps(transaction.metadata),
                transaction.created_at, transaction.completed_at)
        
        return transaction
    
    async def audit_log(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record audit log entry"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO audit_log (entity_type, entity_id, action, user_id, details)
                VALUES ($1, $2, $3, $4, $5)
            ''', entity_type, uuid.UUID(entity_id), action, user_id,
                json.dumps(details) if details else None)


class TransactionProcessor:
    """Core transaction processing engine"""
    
    def __init__(
        self,
        ledger: LedgerEngine,
        crypto: CryptoEngine,
        compliance: ComplianceEngine
    ):
        self.ledger = ledger
        self.crypto = crypto
        self.compliance = compliance
        self.processing_queue = asyncio.Queue()
        self.workers = []
        
    async def start_workers(self, num_workers: int = 4):
        """Start transaction processing workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._process_worker(i))
            self.workers.append(worker)
    
    async def _process_worker(self, worker_id: int):
        """Worker to process transactions from queue"""
        logger.info(f"Transaction worker {worker_id} started")
        
        while True:
            try:
                transaction = await self.processing_queue.get()
                await self._process_transaction(transaction)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def submit_transaction(self, transaction: Transaction) -> str:
        """Submit transaction for processing"""
        transaction_counter.inc()
        
        # Add to processing queue
        await self.processing_queue.put(transaction)
        
        # Record initial state
        await self.ledger.record_transaction(transaction)
        
        return transaction.id
    
    @transaction_latency.time()
    async def _process_transaction(self, transaction: Transaction):
        """Process a single transaction"""
        start_time = time.time()
        
        try:
            # Update status to processing
            transaction.status = TransactionStatus.PROCESSING
            
            # Compliance checks
            compliant, reason = await self.compliance.check_transaction(
                transaction,
                ComplianceLevel.ENHANCED
            )
            
            if not compliant:
                transaction.status = TransactionStatus.FAILED
                transaction.metadata['failure_reason'] = reason
                await self.ledger.record_transaction(transaction)
                logger.warning(f"Transaction {transaction.id} failed compliance: {reason}")
                return
            
            # Execute transaction based on type
            if transaction.type == TransactionType.TRANSFER:
                await self._execute_transfer(transaction)
            elif transaction.type == TransactionType.DEPOSIT:
                await self._execute_deposit(transaction)
            elif transaction.type == TransactionType.WITHDRAWAL:
                await self._execute_withdrawal(transaction)
            elif transaction.type == TransactionType.PAYMENT:
                await self._execute_payment(transaction)
            else:
                raise ValueError(f"Unsupported transaction type: {transaction.type}")
            
            # Mark as completed
            transaction.status = TransactionStatus.COMPLETED
            transaction.completed_at = datetime.now(timezone.utc)
            
            # Generate and store signature
            signature = self.crypto.generate_transaction_signature(transaction)
            transaction.metadata['signature'] = signature
            
            # Update ledger
            await self.ledger.record_transaction(transaction)
            
            # Audit log
            await self.ledger.audit_log(
                'transaction',
                transaction.id,
                'completed',
                details={'processing_time': time.time() - start_time}
            )
            
            logger.info(f"Transaction {transaction.id} completed in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            transaction.status = TransactionStatus.FAILED
            transaction.metadata['error'] = str(e)
            await self.ledger.record_transaction(transaction)
            logger.error(f"Transaction {transaction.id} failed: {e}")
    
    async def _execute_transfer(self, transaction: Transaction):
        """Execute transfer between accounts"""
        if not transaction.from_account or not transaction.to_account:
            raise ValueError("Transfer requires both from and to accounts")
        
        # Debit from account
        await self.ledger.update_balance(
            transaction.from_account,
            transaction.amount + transaction.fees,
            'subtract'
        )
        
        # Credit to account
        await self.ledger.update_balance(
            transaction.to_account,
            transaction.amount,
            'add'
        )
    
    async def _execute_deposit(self, transaction: Transaction):
        """Execute deposit to account"""
        if not transaction.to_account:
            raise ValueError("Deposit requires destination account")
        
        await self.ledger.update_balance(
            transaction.to_account,
            transaction.amount,
            'add'
        )
    
    async def _execute_withdrawal(self, transaction: Transaction):
        """Execute withdrawal from account"""
        if not transaction.from_account:
            raise ValueError("Withdrawal requires source account")
        
        await self.ledger.update_balance(
            transaction.from_account,
            transaction.amount + transaction.fees,
            'subtract'
        )
    
    async def _execute_payment(self, transaction: Transaction):
        """Execute payment transaction"""
        # Similar to transfer but may involve external systems
        await self._execute_transfer(transaction)


class FinancialKernel:
    """Main financial operating system kernel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crypto = CryptoEngine()
        self.compliance = ComplianceEngine()
        self.ledger = LedgerEngine(config.get('database', {}))
        self.processor = TransactionProcessor(
            self.ledger,
            self.crypto,
            self.compliance
        )
        self.running = False
        
    async def initialize(self):
        """Initialize the financial kernel"""
        logger.info("Initializing QENEX Financial Kernel...")
        
        # Initialize database
        await self.ledger.initialize()
        
        # Start transaction workers
        await self.processor.start_workers(
            self.config.get('num_workers', 4)
        )
        
        # Start metrics server
        start_http_server(self.config.get('metrics_port', 9090))
        
        self.running = True
        logger.info("QENEX Financial Kernel initialized successfully")
    
    async def create_account(
        self,
        owner_id: str,
        account_type: str,
        currency: str,
        initial_balance: Decimal = Decimal('0')
    ) -> Account:
        """Create a new financial account"""
        account = Account(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            account_type=account_type,
            currency=currency,
            balance=initial_balance,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        await self.ledger.create_account(account)
        
        # Audit log
        await self.ledger.audit_log(
            'account',
            account.id,
            'created',
            user_id=owner_id,
            details={'account_type': account_type, 'currency': currency}
        )
        
        return account
    
    async def process_transaction(
        self,
        transaction_type: TransactionType,
        from_account: Optional[str],
        to_account: Optional[str],
        amount: Decimal,
        currency: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a financial transaction"""
        transaction = Transaction(
            id=str(uuid.uuid4()),
            type=transaction_type,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            currency=currency,
            status=TransactionStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Calculate fees based on transaction type and amount
        transaction.fees = self._calculate_fees(transaction)
        
        # Submit for processing
        transaction_id = await self.processor.submit_transaction(transaction)
        
        return transaction_id
    
    def _calculate_fees(self, transaction: Transaction) -> Decimal:
        """Calculate transaction fees"""
        base_fee = Decimal('0.01')  # 1 cent base fee
        percentage_fee = transaction.amount * Decimal('0.001')  # 0.1% of amount
        
        return base_fee + percentage_fee
    
    async def get_account_balance(self, account_id: str) -> Optional[Decimal]:
        """Get current account balance"""
        account = await self.ledger.get_account(account_id)
        return account.balance if account else None
    
    async def shutdown(self):
        """Shutdown the kernel gracefully"""
        logger.info("Shutting down QENEX Financial Kernel...")
        self.running = False
        
        # Cancel workers
        for worker in self.processor.workers:
            worker.cancel()
        
        # Close database connections
        if self.ledger.db_pool:
            await self.ledger.db_pool.close()
        
        logger.info("QENEX Financial Kernel shutdown complete")


# Example usage and testing
async def main():
    """Main entry point for testing"""
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'user': 'qenex',
            'password': 'secure_password',
            'database': 'qenex_financial',
            'redis_host': 'localhost',
            'redis_port': 6379
        },
        'num_workers': 4,
        'metrics_port': 9090
    }
    
    kernel = FinancialKernel(config)
    await kernel.initialize()
    
    # Create test accounts
    account1 = await kernel.create_account(
        owner_id='user_001',
        account_type='CHECKING',
        currency='USD',
        initial_balance=Decimal('10000.00')
    )
    
    account2 = await kernel.create_account(
        owner_id='user_002',
        account_type='SAVINGS',
        currency='USD',
        initial_balance=Decimal('5000.00')
    )
    
    # Process test transaction
    transaction_id = await kernel.process_transaction(
        TransactionType.TRANSFER,
        from_account=account1.id,
        to_account=account2.id,
        amount=Decimal('100.00'),
        currency='USD',
        metadata={'description': 'Test transfer'}
    )
    
    print(f"Transaction submitted: {transaction_id}")
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check balances
    balance1 = await kernel.get_account_balance(account1.id)
    balance2 = await kernel.get_account_balance(account2.id)
    
    print(f"Account 1 balance: {balance1}")
    print(f"Account 2 balance: {balance2}")
    
    # Keep running
    try:
        while kernel.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(main())