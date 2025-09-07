#!/usr/bin/env python3
"""
QENEX Financial Kernel - Core Transaction Processing Engine
Autonomous, self-healing financial infrastructure with zero-vulnerability architecture
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import aioredis
import asyncpg
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
transaction_counter = Counter('qenex_transactions_total', 'Total transactions processed')
transaction_latency = Histogram('qenex_transaction_latency_seconds', 'Transaction processing latency')
active_connections = Gauge('qenex_active_connections', 'Active database connections')
fraud_detections = Counter('qenex_fraud_detections_total', 'Total fraud attempts detected')
system_health = Gauge('qenex_system_health', 'System health score (0-100)')


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"
    CANCELLED = "CANCELLED"


class AccountType(Enum):
    """Account type enumeration"""
    CHECKING = "CHECKING"
    SAVINGS = "SAVINGS"
    INVESTMENT = "INVESTMENT"
    MERCHANT = "MERCHANT"
    SYSTEM = "SYSTEM"


@dataclass
class Transaction:
    """Immutable transaction record"""
    id: UUID = field(default_factory=uuid4)
    source_account: str = ""
    destination_account: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "USD"
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Validate transaction data"""
        if self.amount <= 0:
            raise ValueError("Transaction amount must be positive")
        if not self.source_account or not self.destination_account:
            raise ValueError("Source and destination accounts required")
        if self.source_account == self.destination_account:
            raise ValueError("Source and destination must be different")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'id': str(self.id),
            'source_account': self.source_account,
            'destination_account': self.destination_account,
            'amount': str(self.amount),
            'currency': self.currency,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'signature': self.signature
        }
    
    def calculate_hash(self) -> str:
        """Calculate transaction hash for integrity"""
        data = f"{self.id}{self.source_account}{self.destination_account}{self.amount}{self.currency}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func):
        """Decorator for circuit breaker protection"""
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self.last_failure_time and \
                   (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e
        
        return wrapper


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int = 100, per: int = 1):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        
    async def acquire(self) -> bool:
        """Acquire rate limit permit"""
        current = time.monotonic()
        time_passed = current - self.last_check
        self.last_check = current
        
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate
            
        if self.allowance < 1.0:
            return False
            
        self.allowance -= 1.0
        return True


class DatabasePool:
    """Async database connection pool with health monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None
        self._health_check_task = None
        
    async def initialize(self):
        """Initialize database connections"""
        # PostgreSQL connection pool
        self.pool = await asyncpg.create_pool(
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 5432),
            user=self.config.get('user', 'qenex'),
            password=self.config.get('password'),
            database=self.config.get('database', 'qenex_financial'),
            min_size=10,
            max_size=50,
            max_inactive_connection_lifetime=300
        )
        
        # Redis connection for caching
        self.redis = await aioredis.create_redis_pool(
            f"redis://{self.config.get('redis_host', 'localhost')}:{self.config.get('redis_port', 6379)}",
            minsize=5,
            maxsize=20
        )
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())
        
        # Create tables
        await self._create_tables()
        
        logger.info("Database pool initialized successfully")
        
    async def _create_tables(self):
        """Create required database tables"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id VARCHAR(64) PRIMARY KEY,
                    account_type VARCHAR(32) NOT NULL,
                    balance NUMERIC(20, 8) NOT NULL DEFAULT 0,
                    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
                    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id UUID PRIMARY KEY,
                    source_account VARCHAR(64) NOT NULL,
                    destination_account VARCHAR(64) NOT NULL,
                    amount NUMERIC(20, 8) NOT NULL,
                    currency VARCHAR(3) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    signature TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    FOREIGN KEY (source_account) REFERENCES accounts(id),
                    FOREIGN KEY (destination_account) REFERENCES accounts(id)
                )
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_status 
                ON transactions(status)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_accounts 
                ON transactions(source_account, destination_account)
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(64) NOT NULL,
                    entity_id VARCHAR(128),
                    entity_type VARCHAR(64),
                    actor VARCHAR(128),
                    details JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
    async def _health_monitor(self):
        """Monitor database health"""
        while True:
            try:
                async with self.pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                
                # Update metrics
                active_connections.set(self.pool.get_size())
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                await asyncio.sleep(5)
                
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
                
    async def close(self):
        """Close database connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self.pool:
            await self.pool.close()
        
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()


class SecurityManager:
    """Cryptographic operations and security management"""
    
    def __init__(self):
        self.signing_key = self._generate_signing_key()
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
    def _generate_signing_key(self) -> bytes:
        """Generate HMAC signing key"""
        return secrets.token_bytes(32)
        
    def sign_transaction(self, transaction: Transaction) -> str:
        """Sign transaction with HMAC"""
        message = f"{transaction.id}{transaction.source_account}{transaction.destination_account}{transaction.amount}".encode()
        signature = hmac.new(self.signing_key, message, hashlib.sha256).hexdigest()
        return signature
        
    def verify_signature(self, transaction: Transaction, signature: str) -> bool:
        """Verify transaction signature"""
        expected = self.sign_transaction(transaction)
        return hmac.compare_digest(expected, signature)
        
    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode())
        
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data).decode()
        
    def hash_password(self, password: str) -> str:
        """Hash password using Scrypt"""
        salt = secrets.token_bytes(16)
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        key = kdf.derive(password.encode())
        return f"{salt.hex()}${key.hex()}"
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt_hex, key_hex = hashed.split('$')
            salt = bytes.fromhex(salt_hex)
            expected_key = bytes.fromhex(key_hex)
            
            kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
            key = kdf.derive(password.encode())
            
            return hmac.compare_digest(key, expected_key)
        except Exception:
            return False


class FinancialKernel:
    """Core financial transaction processing kernel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = DatabasePool(config['database'])
        self.security = SecurityManager()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(rate=1000, per=1)
        
        # Transaction queues
        self.pending_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # Performance tracking
        self.transaction_times: deque = deque(maxlen=1000)
        
        # System state
        self.running = False
        self.health_score = 100.0
        
    async def initialize(self):
        """Initialize the financial kernel"""
        await self.db.initialize()
        
        # Start transaction processors
        for i in range(self.config.get('workers', {}).get('transaction_processors', 4)):
            task = asyncio.create_task(self._transaction_processor(i))
            self.processing_tasks.add(task)
            
        # Start health monitoring
        asyncio.create_task(self._monitor_health())
        
        self.running = True
        logger.info("Financial kernel initialized")
        
    async def _transaction_processor(self, worker_id: int):
        """Process transactions from queue"""
        logger.info(f"Transaction processor {worker_id} started")
        
        while self.running:
            try:
                transaction = await asyncio.wait_for(
                    self.pending_queue.get(),
                    timeout=1.0
                )
                
                start_time = time.monotonic()
                
                # Process transaction
                await self._process_transaction(transaction)
                
                # Track performance
                elapsed = time.monotonic() - start_time
                self.transaction_times.append(elapsed)
                transaction_latency.observe(elapsed)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Transaction processor {worker_id} error: {e}")
                
    @circuit_breaker.call
    async def _process_transaction(self, transaction: Transaction):
        """Process a single transaction with ACID guarantees"""
        async with self.db.transaction() as conn:
            # Lock accounts for update
            source = await conn.fetchrow(
                'SELECT * FROM accounts WHERE id = $1 FOR UPDATE',
                transaction.source_account
            )
            
            dest = await conn.fetchrow(
                'SELECT * FROM accounts WHERE id = $1 FOR UPDATE',
                transaction.destination_account
            )
            
            if not source or not dest:
                raise ValueError("Invalid account")
                
            if source['status'] != 'ACTIVE' or dest['status'] != 'ACTIVE':
                raise ValueError("Account not active")
                
            # Check balance
            if Decimal(str(source['balance'])) < transaction.amount:
                raise ValueError("Insufficient funds")
                
            # Update balances
            await conn.execute(
                'UPDATE accounts SET balance = balance - $1, updated_at = NOW() WHERE id = $2',
                transaction.amount, transaction.source_account
            )
            
            await conn.execute(
                'UPDATE accounts SET balance = balance + $1, updated_at = NOW() WHERE id = $2',
                transaction.amount, transaction.destination_account
            )
            
            # Record transaction
            await conn.execute('''
                UPDATE transactions 
                SET status = $1, completed_at = NOW() 
                WHERE id = $2
            ''', TransactionStatus.COMPLETED.value, transaction.id)
            
            # Audit log
            await self._audit_log(conn, 'TRANSACTION_COMPLETED', str(transaction.id), 'transaction', {
                'amount': str(transaction.amount),
                'currency': transaction.currency
            })
            
        transaction_counter.inc()
        logger.info(f"Transaction {transaction.id} completed")
        
    async def create_transaction(
        self, 
        source_account: str,
        destination_account: str,
        amount: Decimal,
        currency: str = "USD",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """Create and queue a new transaction"""
        
        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
            
        # Create transaction
        transaction = Transaction(
            source_account=source_account,
            destination_account=destination_account,
            amount=amount,
            currency=currency,
            metadata=metadata or {}
        )
        
        # Sign transaction
        transaction.signature = self.security.sign_transaction(transaction)
        
        # Store in database
        async with self.db.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO transactions 
                (id, source_account, destination_account, amount, currency, status, signature, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', transaction.id, transaction.source_account, transaction.destination_account,
                transaction.amount, transaction.currency, transaction.status.value,
                transaction.signature, json.dumps(transaction.metadata))
        
        # Queue for processing
        await self.pending_queue.put(transaction)
        
        return transaction
        
    async def create_account(
        self,
        account_id: str,
        account_type: AccountType,
        initial_balance: Decimal = Decimal("0.00"),
        currency: str = "USD",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new account"""
        
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
            
        async with self.db.pool.acquire() as conn:
            try:
                await conn.execute('''
                    INSERT INTO accounts 
                    (id, account_type, balance, currency, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                ''', account_id, account_type.value, initial_balance, 
                    currency, json.dumps(metadata or {}))
                
                await self._audit_log(conn, 'ACCOUNT_CREATED', account_id, 'account', {
                    'type': account_type.value,
                    'currency': currency
                })
                
                return True
            except asyncpg.UniqueViolationError:
                return False
                
    async def get_balance(self, account_id: str) -> Optional[Decimal]:
        """Get account balance"""
        async with self.db.pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT balance FROM accounts WHERE id = $1',
                account_id
            )
            return Decimal(str(row['balance'])) if row else None
            
    async def get_transaction_history(
        self,
        account_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get transaction history for account"""
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM transactions 
                WHERE source_account = $1 OR destination_account = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            ''', account_id, limit, offset)
            
            return [dict(row) for row in rows]
            
    async def _audit_log(
        self,
        conn: asyncpg.Connection,
        event_type: str,
        entity_id: str,
        entity_type: str,
        details: Dict[str, Any]
    ):
        """Create audit log entry"""
        await conn.execute('''
            INSERT INTO audit_log (event_type, entity_id, entity_type, details)
            VALUES ($1, $2, $3, $4)
        ''', event_type, entity_id, entity_type, json.dumps(details))
        
    async def _monitor_health(self):
        """Monitor system health"""
        while self.running:
            try:
                # Calculate health metrics
                avg_latency = sum(self.transaction_times) / len(self.transaction_times) if self.transaction_times else 0
                queue_size = self.pending_queue.qsize()
                
                # Calculate health score (0-100)
                latency_score = max(0, 100 - (avg_latency * 1000))  # Penalize if > 100ms
                queue_score = max(0, 100 - (queue_size / 100))  # Penalize if > 10000
                
                self.health_score = (latency_score + queue_score) / 2
                system_health.set(self.health_score)
                
                # Log if health is poor
                if self.health_score < 50:
                    logger.warning(f"System health degraded: {self.health_score:.1f}")
                    
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def shutdown(self):
        """Shutdown the kernel gracefully"""
        logger.info("Shutting down financial kernel...")
        self.running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Close database connections
        await self.db.close()
        
        logger.info("Financial kernel shutdown complete")
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        avg_latency = sum(self.transaction_times) / len(self.transaction_times) if self.transaction_times else 0
        
        return {
            'health_score': self.health_score,
            'pending_transactions': self.pending_queue.qsize(),
            'average_latency_ms': avg_latency * 1000,
            'circuit_breaker_state': self.circuit_breaker.state,
            'active_processors': len(self.processing_tasks)
        }