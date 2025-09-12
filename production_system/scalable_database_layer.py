#!/usr/bin/env python3
"""
Production Database Layer with Connection Pooling and ORM
Solves the missing persistent storage and scalability issues
"""

import os
import json
import time
import sqlite3
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
from queue import Queue, Empty
from enum import Enum
import hashlib

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

@dataclass
class ConnectionPool:
    """Database connection pool for scalability"""
    
    def __init__(
        self,
        db_type: DatabaseType = DatabaseType.SQLITE,
        db_path: str = "production.db",
        min_connections: int = 5,
        max_connections: int = 20
    ):
        self.db_type = db_type
        self.db_path = db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        self.logger = self._setup_logging()
        
        # Initialize minimum connections
        for _ in range(min_connections):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure database logging"""
        logger = logging.getLogger('DatabasePool')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('database.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _create_connection(self):
        """Create new database connection"""
        if self.db_type == DatabaseType.SQLITE:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
            conn.execute('PRAGMA synchronous = NORMAL')
            conn.row_factory = sqlite3.Row
            return conn
        elif self.db_type == DatabaseType.POSTGRESQL:
            # PostgreSQL implementation
            import psycopg2
            from psycopg2.pool import SimpleConnectionPool
            # Implementation would go here
            pass
        elif self.db_type == DatabaseType.MYSQL:
            # MySQL implementation
            import mysql.connector
            from mysql.connector import pooling
            # Implementation would go here
            pass
    
    @contextmanager
    def get_connection(self, timeout: float = 5.0):
        """Get connection from pool with automatic return"""
        conn = None
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    conn = self.pool.get(block=False)
                    break
                except Empty:
                    with self.lock:
                        if self.active_connections < self.max_connections:
                            conn = self._create_connection()
                            self.active_connections += 1
                            break
                    time.sleep(0.1)
            
            if conn is None:
                raise TimeoutError("Could not acquire database connection")
            
            yield conn
            
        finally:
            if conn:
                # Return connection to pool
                self.pool.put(conn)

class DatabaseMigration:
    """Database schema migration system"""
    
    def __init__(self, db_pool: ConnectionPool):
        self.db_pool = db_pool
        self.migrations = []
        self._init_migration_table()
    
    def _init_migration_table(self):
        """Initialize migration tracking table"""
        with self.db_pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_migration(self, version: int, name: str, up_sql: str, down_sql: str):
        """Add migration to system"""
        self.migrations.append({
            'version': version,
            'name': name,
            'up': up_sql,
            'down': down_sql
        })
    
    def migrate(self):
        """Apply pending migrations"""
        with self.db_pool.get_connection() as conn:
            # Get current version
            cursor = conn.execute(
                'SELECT MAX(version) FROM schema_migrations'
            )
            current_version = cursor.fetchone()[0] or 0
            
            # Apply pending migrations
            for migration in sorted(self.migrations, key=lambda x: x['version']):
                if migration['version'] > current_version:
                    try:
                        conn.execute(migration['up'])
                        conn.execute('''
                            INSERT INTO schema_migrations (version, name)
                            VALUES (?, ?)
                        ''', (migration['version'], migration['name']))
                        conn.commit()
                        print(f"Applied migration {migration['version']}: {migration['name']}")
                    except Exception as e:
                        conn.rollback()
                        raise Exception(f"Migration {migration['version']} failed: {e}")

class BaseModel:
    """Base ORM model for database entities"""
    
    _table_name = None
    _primary_key = 'id'
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def create_table(cls, conn):
        """Create table for model"""
        raise NotImplementedError
    
    @classmethod
    def find(cls, db_pool: ConnectionPool, **conditions) -> Optional['BaseModel']:
        """Find single record by conditions"""
        with db_pool.get_connection() as conn:
            where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
            query = f"SELECT * FROM {cls._table_name} WHERE {where_clause}"
            
            cursor = conn.execute(query, tuple(conditions.values()))
            row = cursor.fetchone()
            
            if row:
                return cls(**dict(row))
            return None
    
    @classmethod
    def find_all(cls, db_pool: ConnectionPool, **conditions) -> List['BaseModel']:
        """Find all records matching conditions"""
        with db_pool.get_connection() as conn:
            if conditions:
                where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
                query = f"SELECT * FROM {cls._table_name} WHERE {where_clause}"
                cursor = conn.execute(query, tuple(conditions.values()))
            else:
                query = f"SELECT * FROM {cls._table_name}"
                cursor = conn.execute(query)
            
            return [cls(**dict(row)) for row in cursor.fetchall()]
    
    def save(self, db_pool: ConnectionPool):
        """Save model to database"""
        with db_pool.get_connection() as conn:
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            
            if hasattr(self, self._primary_key) and getattr(self, self._primary_key):
                # Update existing record
                set_clause = ', '.join([f"{k} = ?" for k in data.keys() if k != self._primary_key])
                query = f"UPDATE {self._table_name} SET {set_clause} WHERE {self._primary_key} = ?"
                values = [v for k, v in data.items() if k != self._primary_key]
                values.append(getattr(self, self._primary_key))
                conn.execute(query, values)
            else:
                # Insert new record
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['?' for _ in data])
                query = f"INSERT INTO {self._table_name} ({columns}) VALUES ({placeholders})"
                cursor = conn.execute(query, tuple(data.values()))
                if self._primary_key == 'id':
                    setattr(self, 'id', cursor.lastrowid)
            
            conn.commit()
    
    def delete(self, db_pool: ConnectionPool):
        """Delete model from database"""
        with db_pool.get_connection() as conn:
            query = f"DELETE FROM {self._table_name} WHERE {self._primary_key} = ?"
            conn.execute(query, (getattr(self, self._primary_key),))
            conn.commit()

class TransactionModel(BaseModel):
    """Transaction model with optimized queries"""
    
    _table_name = 'transactions'
    
    @classmethod
    def create_table(cls, conn):
        """Create optimized transaction table"""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash TEXT UNIQUE NOT NULL,
                from_address TEXT NOT NULL,
                to_address TEXT NOT NULL,
                amount REAL NOT NULL,
                fee REAL DEFAULT 0,
                status TEXT NOT NULL,
                block_number INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data TEXT,
                
                -- Indexes for query optimization
                INDEX idx_tx_hash ON transactions(tx_hash),
                INDEX idx_from_address ON transactions(from_address),
                INDEX idx_to_address ON transactions(to_address),
                INDEX idx_timestamp ON transactions(timestamp),
                INDEX idx_status ON transactions(status)
            )
        ''')
        conn.commit()
    
    @classmethod
    def get_balance(cls, db_pool: ConnectionPool, address: str) -> float:
        """Optimized balance calculation"""
        with db_pool.get_connection() as conn:
            cursor = conn.execute('''
                SELECT 
                    COALESCE(SUM(CASE WHEN to_address = ? THEN amount ELSE 0 END), 0) -
                    COALESCE(SUM(CASE WHEN from_address = ? THEN amount + fee ELSE 0 END), 0) as balance
                FROM transactions
                WHERE (to_address = ? OR from_address = ?) AND status = 'confirmed'
            ''', (address, address, address, address))
            
            return cursor.fetchone()[0] or 0.0
    
    @classmethod
    def get_transaction_history(
        cls,
        db_pool: ConnectionPool,
        address: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get paginated transaction history"""
        with db_pool.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM transactions
                WHERE from_address = ? OR to_address = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (address, address, limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]

class CacheManager:
    """In-memory cache for frequently accessed data"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        with self.lock:
            expiry = time.time() + (ttl or self.ttl)
            self.cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()

class DatabaseBackup:
    """Automated database backup system"""
    
    def __init__(self, db_pool: ConnectionPool):
        self.db_pool = db_pool
        self.backup_dir = 'backups'
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self) -> str:
        """Create database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f'backup_{timestamp}.sql')
        
        with self.db_pool.get_connection() as conn:
            with open(backup_file, 'w') as f:
                for line in conn.iterdump():
                    f.write('%s\n' % line)
        
        return backup_file
    
    def restore_backup(self, backup_file: str):
        """Restore database from backup"""
        with open(backup_file, 'r') as f:
            sql_script = f.read()
        
        with self.db_pool.get_connection() as conn:
            conn.executescript(sql_script)
            conn.commit()
    
    def cleanup_old_backups(self, days_to_keep: int = 7):
        """Remove old backup files"""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for filename in os.listdir(self.backup_dir):
            filepath = os.path.join(self.backup_dir, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)

class QueryOptimizer:
    """Query optimization and analysis"""
    
    def __init__(self, db_pool: ConnectionPool):
        self.db_pool = db_pool
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query performance"""
        with self.db_pool.get_connection() as conn:
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = cursor.fetchall()
            
            return {
                'query': query,
                'plan': [dict(row) for row in plan]
            }
    
    def optimize_indexes(self):
        """Analyze and suggest index optimizations"""
        with self.db_pool.get_connection() as conn:
            # Analyze table statistics
            conn.execute("ANALYZE")
            
            # Get table info
            cursor = conn.execute('''
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ''')
            
            suggestions = []
            for table in cursor.fetchall():
                table_name = table[0]
                
                # Check for missing indexes on foreign keys
                cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
                for fk in cursor.fetchall():
                    column = fk[3]
                    suggestions.append(f"CREATE INDEX idx_{table_name}_{column} ON {table_name}({column})")
            
            return suggestions

class ProductionDatabaseLayer:
    """Complete production database layer"""
    
    def __init__(
        self,
        db_type: DatabaseType = DatabaseType.SQLITE,
        db_path: str = "production.db"
    ):
        self.pool = ConnectionPool(db_type, db_path)
        self.cache = CacheManager()
        self.migrations = DatabaseMigration(self.pool)
        self.backup = DatabaseBackup(self.pool)
        self.optimizer = QueryOptimizer(self.pool)
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema"""
        # Add migrations
        self.migrations.add_migration(
            1,
            "create_base_tables",
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                address TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            ''',
            'DROP TABLE IF EXISTS wallets; DROP TABLE IF EXISTS users;'
        )
        
        # Apply migrations
        self.migrations.migrate()
    
    def health_check(self) -> Dict:
        """Database health check"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
            
            return {
                'status': 'healthy',
                'active_connections': self.pool.active_connections,
                'pool_size': self.pool.pool.qsize()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Initialize production database
    db = ProductionDatabaseLayer()
    
    # Health check
    print("Database health:", db.health_check())
    
    # Use ORM
    class User(BaseModel):
        _table_name = 'users'
    
    # Create user
    user = User(username='test_user', email='ceo@qenex.ai')
    user.save(db.pool)
    
    # Find user
    found_user = User.find(db.pool, username='test_user')
    print(f"Found user: {found_user.username if found_user else 'Not found'}")
    
    # Cache example
    db.cache.set('user:1', found_user, ttl=60)
    cached = db.cache.get('user:1')
    
    # Backup
    backup_file = db.backup.create_backup()
    print(f"Backup created: {backup_file}")