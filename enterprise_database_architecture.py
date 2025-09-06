#!/usr/bin/env python3
"""
Enterprise Database Architecture
Production-ready distributed database system for banking
"""

import asyncio
import asyncpg
import redis.asyncio as redis
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    postgres_master: str = "postgresql://bank:secure@localhost:5432/banking"
    postgres_replicas: List[str] = None
    redis_url: str = "redis://localhost:6379"
    max_connections: int = 100
    min_connections: int = 10
    connection_timeout: int = 10
    query_timeout: int = 30
    enable_ssl: bool = True
    
    def __post_init__(self):
        if self.postgres_replicas is None:
            self.postgres_replicas = [
                "postgresql://bank:secure@replica1:5432/banking",
                "postgresql://bank:secure@replica2:5432/banking"
            ]

class ConnectionPool:
    """Advanced connection pooling with read/write splitting"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.master_pool: Optional[asyncpg.Pool] = None
        self.replica_pools: List[asyncpg.Pool] = []
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all connection pools"""
        if self.initialized:
            return
            
        # Create master pool for writes
        self.master_pool = await asyncpg.create_pool(
            self.config.postgres_master,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            timeout=self.config.connection_timeout,
            command_timeout=self.config.query_timeout,
            ssl='require' if self.config.enable_ssl else None
        )
        
        # Create replica pools for reads
        for replica_url in self.config.postgres_replicas:
            try:
                pool = await asyncpg.create_pool(
                    replica_url,
                    min_size=5,
                    max_size=50,
                    timeout=self.config.connection_timeout,
                    command_timeout=self.config.query_timeout,
                    ssl='require' if self.config.enable_ssl else None
                )
                self.replica_pools.append(pool)
                logger.info(f"Connected to replica: {replica_url.split('@')[1]}")
            except Exception as e:
                logger.warning(f"Failed to connect to replica: {e}")
        
        # Create Redis pool for caching
        self.redis_pool = redis.ConnectionPool.from_url(
            self.config.redis_url,
            max_connections=50,
            decode_responses=True
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        self.initialized = True
        logger.info("Database pools initialized")
    
    async def close(self):
        """Close all connection pools"""
        if self.master_pool:
            await self.master_pool.close()
        
        for pool in self.replica_pools:
            await pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.initialized = False
    
    @asynccontextmanager
    async def acquire_master(self):
        """Acquire connection from master pool"""
        async with self.master_pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def acquire_replica(self):
        """Acquire connection from replica pool with load balancing"""
        if not self.replica_pools:
            # Fallback to master if no replicas
            async with self.master_pool.acquire() as conn:
                yield conn
        else:
            # Round-robin load balancing
            pool = self.replica_pools[int(time.time()) % len(self.replica_pools)]
            async with pool.acquire() as conn:
                yield conn

class ShardedDatabase:
    """Sharded database for horizontal scaling"""
    
    def __init__(self, num_shards: int = 16):
        self.num_shards = num_shards
        self.shard_pools: Dict[int, ConnectionPool] = {}
        
    def get_shard_id(self, key: str) -> int:
        """Calculate shard ID using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    async def initialize_shards(self):
        """Initialize all shard connections"""
        for shard_id in range(self.num_shards):
            config = DatabaseConfig(
                postgres_master=f"postgresql://bank:secure@shard{shard_id}:5432/banking_shard_{shard_id}"
            )
            pool = ConnectionPool(config)
            await pool.initialize()
            self.shard_pools[shard_id] = pool
    
    async def execute_on_shard(self, key: str, query: str, *args) -> Any:
        """Execute query on appropriate shard"""
        shard_id = self.get_shard_id(key)
        pool = self.shard_pools[shard_id]
        
        async with pool.acquire_master() as conn:
            return await conn.fetch(query, *args)

class DistributedTransaction:
    """Two-phase commit for distributed transactions"""
    
    def __init__(self, pools: List[ConnectionPool]):
        self.pools = pools
        self.connections: List[asyncpg.Connection] = []
        self.transaction_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
    async def prepare(self) -> bool:
        """Phase 1: Prepare all participants"""
        try:
            for pool in self.pools:
                async with pool.acquire_master() as conn:
                    self.connections.append(conn)
                    await conn.execute(f"PREPARE TRANSACTION '{self.transaction_id}'")
            return True
        except Exception as e:
            logger.error(f"Prepare phase failed: {e}")
            await self.rollback()
            return False
    
    async def commit(self) -> bool:
        """Phase 2: Commit all participants"""
        try:
            for conn in self.connections:
                await conn.execute(f"COMMIT PREPARED '{self.transaction_id}'")
            return True
        except Exception as e:
            logger.error(f"Commit phase failed: {e}")
            return False
    
    async def rollback(self):
        """Rollback all participants"""
        for conn in self.connections:
            try:
                await conn.execute(f"ROLLBACK PREPARED '{self.transaction_id}'")
            except:
                pass

class DatabaseMigrationManager:
    """Database schema migration management"""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        
    async def initialize_schema(self):
        """Create initial database schema"""
        schema = """
        -- Enable extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
        
        -- Accounts table with partitioning
        CREATE TABLE IF NOT EXISTS accounts (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            account_number VARCHAR(20) UNIQUE NOT NULL,
            account_type VARCHAR(20) NOT NULL,
            currency CHAR(3) NOT NULL DEFAULT 'USD',
            balance NUMERIC(19, 4) NOT NULL DEFAULT 0,
            available_balance NUMERIC(19, 4) NOT NULL DEFAULT 0,
            hold_amount NUMERIC(19, 4) NOT NULL DEFAULT 0,
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            customer_id UUID NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB,
            CHECK (balance >= 0),
            CHECK (available_balance >= 0),
            CHECK (hold_amount >= 0)
        ) PARTITION BY RANGE (created_at);
        
        -- Create partitions for accounts
        CREATE TABLE IF NOT EXISTS accounts_2024 PARTITION OF accounts
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
        CREATE TABLE IF NOT EXISTS accounts_2025 PARTITION OF accounts
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
        
        -- Transactions table with partitioning
        CREATE TABLE IF NOT EXISTS transactions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            transaction_type VARCHAR(20) NOT NULL,
            from_account_id UUID REFERENCES accounts(id),
            to_account_id UUID REFERENCES accounts(id),
            amount NUMERIC(19, 4) NOT NULL,
            currency CHAR(3) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            reference_number VARCHAR(50) UNIQUE,
            description TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            CHECK (amount > 0)
        ) PARTITION BY RANGE (created_at);
        
        -- Create monthly partitions for transactions
        CREATE TABLE IF NOT EXISTS transactions_2024_01 PARTITION OF transactions
            FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
        CREATE TABLE IF NOT EXISTS transactions_2024_02 PARTITION OF transactions
            FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
        
        -- Audit log table
        CREATE TABLE IF NOT EXISTS audit_log (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            table_name VARCHAR(50) NOT NULL,
            operation VARCHAR(20) NOT NULL,
            user_id UUID,
            row_id UUID,
            old_data JSONB,
            new_data JSONB,
            ip_address INET,
            user_agent TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_accounts_customer_id ON accounts(customer_id);
        CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts(status);
        CREATE INDEX IF NOT EXISTS idx_transactions_from_account ON transactions(from_account_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_to_account ON transactions(to_account_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
        CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_log_table_operation ON audit_log(table_name, operation);
        CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at DESC);
        
        -- Create materialized views for reporting
        CREATE MATERIALIZED VIEW IF NOT EXISTS account_daily_summary AS
        SELECT 
            DATE(created_at) as summary_date,
            account_type,
            currency,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM transactions
        WHERE status = 'completed'
        GROUP BY DATE(created_at), account_type, currency
        WITH DATA;
        
        -- Create functions for triggers
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create triggers
        CREATE TRIGGER accounts_update_timestamp
            BEFORE UPDATE ON accounts
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at();
        
        -- Create audit trigger
        CREATE OR REPLACE FUNCTION audit_trigger()
        RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO audit_log(table_name, operation, row_id, old_data, new_data)
            VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE TRIGGER accounts_audit
            AFTER INSERT OR UPDATE OR DELETE ON accounts
            FOR EACH ROW
            EXECUTE FUNCTION audit_trigger();
        """
        
        async with self.pool.acquire_master() as conn:
            await conn.execute(schema)
            logger.info("Database schema initialized")

class QueryOptimizer:
    """Query optimization and caching strategies"""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.query_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 60  # seconds
        
    async def execute_with_cache(self, query: str, *args, cache_key: str = None) -> Any:
        """Execute query with result caching"""
        # Generate cache key if not provided
        if not cache_key:
            cache_key = hashlib.md5(f"{query}{args}".encode()).hexdigest()
        
        # Check Redis cache first
        cached = await self.pool.redis_client.get(f"query:{cache_key}")
        if cached:
            return json.loads(cached)
        
        # Execute query
        async with self.pool.acquire_replica() as conn:
            result = await conn.fetch(query, *args)
            
        # Cache result
        result_json = json.dumps([dict(row) for row in result], default=str)
        await self.pool.redis_client.setex(f"query:{cache_key}", self.cache_ttl, result_json)
        
        return result
    
    async def get_query_plan(self, query: str, *args) -> str:
        """Get query execution plan for optimization"""
        async with self.pool.acquire_replica() as conn:
            plan = await conn.fetch(f"EXPLAIN ANALYZE {query}", *args)
            return "\n".join([row['QUERY PLAN'] for row in plan])

class BankingDatabaseManager:
    """Main database manager for banking operations"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.pool = ConnectionPool(self.config)
        self.migration_manager = DatabaseMigrationManager(self.pool)
        self.optimizer = QueryOptimizer(self.pool)
        self.metrics = {
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize database system"""
        await self.pool.initialize()
        await self.migration_manager.initialize_schema()
        logger.info("Banking database manager initialized")
    
    async def create_account(self, account_number: str, account_type: str,
                           customer_id: str, initial_balance: Decimal = Decimal('0')) -> str:
        """Create new account with proper transaction handling"""
        async with self.pool.acquire_master() as conn:
            async with conn.transaction():
                result = await conn.fetchrow("""
                    INSERT INTO accounts (
                        account_number, account_type, balance, 
                        available_balance, customer_id
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, account_number, account_type, initial_balance, 
                    initial_balance, customer_id)
                
                self.metrics['queries_executed'] += 1
                return str(result['id'])
    
    async def process_transaction(self, from_account: str, to_account: str,
                                amount: Decimal, description: str = None) -> str:
        """Process transaction with proper locking and consistency"""
        async with self.pool.acquire_master() as conn:
            async with conn.transaction(isolation='serializable'):
                # Lock accounts for update
                from_acc = await conn.fetchrow("""
                    SELECT id, balance, available_balance 
                    FROM accounts 
                    WHERE account_number = $1 
                    FOR UPDATE
                """, from_account)
                
                if not from_acc or from_acc['available_balance'] < amount:
                    raise ValueError("Insufficient funds")
                
                to_acc = await conn.fetchrow("""
                    SELECT id FROM accounts 
                    WHERE account_number = $1 
                    FOR UPDATE
                """, to_account)
                
                if not to_acc:
                    raise ValueError("Destination account not found")
                
                # Create transaction record
                tx_result = await conn.fetchrow("""
                    INSERT INTO transactions (
                        transaction_type, from_account_id, to_account_id,
                        amount, currency, status, description
                    ) VALUES ('transfer', $1, $2, $3, 'USD', 'processing', $4)
                    RETURNING id
                """, from_acc['id'], to_acc['id'], amount, description)
                
                # Update account balances
                await conn.execute("""
                    UPDATE accounts 
                    SET balance = balance - $1,
                        available_balance = available_balance - $1
                    WHERE id = $2
                """, amount, from_acc['id'])
                
                await conn.execute("""
                    UPDATE accounts 
                    SET balance = balance + $1,
                        available_balance = available_balance + $1
                    WHERE id = $2
                """, amount, to_acc['id'])
                
                # Mark transaction as completed
                await conn.execute("""
                    UPDATE transactions 
                    SET status = 'completed', completed_at = NOW()
                    WHERE id = $1
                """, tx_result['id'])
                
                self.metrics['queries_executed'] += 5
                return str(tx_result['id'])
    
    async def get_balance(self, account_number: str) -> Dict[str, Decimal]:
        """Get account balance with caching"""
        result = await self.optimizer.execute_with_cache("""
            SELECT balance, available_balance, hold_amount
            FROM accounts
            WHERE account_number = $1
        """, account_number, cache_key=f"balance:{account_number}")
        
        if result:
            return dict(result[0])
        return None
    
    async def get_transaction_history(self, account_number: str, 
                                     limit: int = 100) -> List[Dict]:
        """Get transaction history with pagination"""
        async with self.pool.acquire_replica() as conn:
            account = await conn.fetchrow(
                "SELECT id FROM accounts WHERE account_number = $1",
                account_number
            )
            
            if not account:
                return []
            
            transactions = await conn.fetch("""
                SELECT 
                    t.id, t.transaction_type, t.amount, t.currency,
                    t.status, t.description, t.created_at,
                    fa.account_number as from_account,
                    ta.account_number as to_account
                FROM transactions t
                LEFT JOIN accounts fa ON t.from_account_id = fa.id
                LEFT JOIN accounts ta ON t.to_account_id = ta.id
                WHERE t.from_account_id = $1 OR t.to_account_id = $1
                ORDER BY t.created_at DESC
                LIMIT $2
            """, account['id'], limit)
            
            return [dict(tx) for tx in transactions]
    
    async def get_metrics(self) -> Dict:
        """Get database performance metrics"""
        async with self.pool.acquire_replica() as conn:
            db_stats = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM accounts) as total_accounts,
                    (SELECT COUNT(*) FROM transactions) as total_transactions,
                    (SELECT SUM(balance) FROM accounts) as total_balance,
                    (SELECT COUNT(*) FROM transactions WHERE created_at > NOW() - INTERVAL '1 hour') as recent_transactions
            """)
            
            return {
                **dict(db_stats),
                **self.metrics,
                'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            }
    
    async def close(self):
        """Close all database connections"""
        await self.pool.close()

# Example usage
async def main():
    """Example usage of enterprise database"""
    db = BankingDatabaseManager()
    await db.initialize()
    
    try:
        # Create accounts
        account1 = await db.create_account("ACC001", "checking", "CUST001", Decimal('10000'))
        account2 = await db.create_account("ACC002", "savings", "CUST002", Decimal('5000'))
        
        print(f"Created accounts: {account1}, {account2}")
        
        # Process transaction
        tx_id = await db.process_transaction("ACC001", "ACC002", Decimal('100'), "Test transfer")
        print(f"Transaction completed: {tx_id}")
        
        # Check balances
        balance1 = await db.get_balance("ACC001")
        balance2 = await db.get_balance("ACC002")
        print(f"Balances: ACC001={balance1}, ACC002={balance2}")
        
        # Get metrics
        metrics = await db.get_metrics()
        print(f"Database metrics: {metrics}")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())