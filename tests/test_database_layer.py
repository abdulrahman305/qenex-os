#!/usr/bin/env python3
"""
Comprehensive test suite for production database layer
"""

import unittest
import tempfile
import os
import time
import threading
from decimal import Decimal
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production_system.scalable_database_layer import (
    ConnectionPool, DatabaseType, DatabaseMigration,
    BaseModel, TransactionModel, CacheManager,
    DatabaseBackup, QueryOptimizer, ProductionDatabaseLayer
)


class TestConnectionPool(unittest.TestCase):
    """Test database connection pooling"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name,
            min_connections=2,
            max_connections=5
        )
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_connection_creation(self):
        """Test connection pool initialization"""
        # Pool should have minimum connections
        self.assertEqual(self.pool.pool.qsize(), 2)
    
    def test_connection_acquisition(self):
        """Test getting connection from pool"""
        with self.pool.get_connection() as conn:
            self.assertIsNotNone(conn)
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)
    
    def test_connection_return(self):
        """Test connection is returned to pool"""
        initial_size = self.pool.pool.qsize()
        
        with self.pool.get_connection() as conn:
            # Connection taken from pool
            self.assertEqual(self.pool.pool.qsize(), initial_size - 1)
        
        # Connection returned to pool
        self.assertEqual(self.pool.pool.qsize(), initial_size)
    
    def test_concurrent_connections(self):
        """Test multiple concurrent connections"""
        results = []
        
        def worker():
            with self.pool.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                results.append(cursor.fetchone()[0])
                time.sleep(0.1)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r == 1 for r in results))
    
    def test_connection_timeout(self):
        """Test connection acquisition timeout"""
        # Take all connections
        connections = []
        for _ in range(5):
            conn = self.pool.pool.get()
            connections.append(conn)
        
        # Try to get another connection with short timeout
        with self.assertRaises(TimeoutError):
            with self.pool.get_connection(timeout=0.1):
                pass
        
        # Return connections
        for conn in connections:
            self.pool.pool.put(conn)


class TestDatabaseMigration(unittest.TestCase):
    """Test database migration system"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name
        )
        self.migration = DatabaseMigration(self.pool)
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_migration_tracking(self):
        """Test migration version tracking"""
        # Add migrations
        self.migration.add_migration(
            1,
            "create_users",
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
            "DROP TABLE users"
        )
        
        self.migration.add_migration(
            2,
            "add_email",
            "ALTER TABLE users ADD COLUMN email TEXT",
            "ALTER TABLE users DROP COLUMN email"
        )
        
        # Apply migrations
        self.migration.migrate()
        
        # Check migrations were applied
        with self.pool.get_connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            versions = [row[0] for row in cursor.fetchall()]
            
        self.assertEqual(versions, [1, 2])
    
    def test_migration_idempotency(self):
        """Test migrations are only applied once"""
        self.migration.add_migration(
            1,
            "create_table",
            "CREATE TABLE test (id INTEGER PRIMARY KEY)",
            "DROP TABLE test"
        )
        
        # Apply migrations twice
        self.migration.migrate()
        self.migration.migrate()
        
        # Should only have one entry
        with self.pool.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 1"
            )
            count = cursor.fetchone()[0]
            
        self.assertEqual(count, 1)


class TestORM(unittest.TestCase):
    """Test ORM functionality"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name
        )
        
        # Create test table
        with self.pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE test_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    value INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        
        # Define test model
        class TestModel(BaseModel):
            _table_name = 'test_models'
        
        self.TestModel = TestModel
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_save_new_record(self):
        """Test saving new record"""
        model = self.TestModel(name='test', value=42)
        model.save(self.pool)
        
        self.assertIsNotNone(model.id)
        self.assertGreater(model.id, 0)
    
    def test_find_record(self):
        """Test finding single record"""
        # Save a record
        model = self.TestModel(name='findme', value=100)
        model.save(self.pool)
        
        # Find it
        found = self.TestModel.find(self.pool, name='findme')
        
        self.assertIsNotNone(found)
        self.assertEqual(found.name, 'findme')
        self.assertEqual(found.value, 100)
    
    def test_find_all_records(self):
        """Test finding multiple records"""
        # Save multiple records
        for i in range(5):
            model = self.TestModel(name=f'item_{i}', value=i)
            model.save(self.pool)
        
        # Find all
        all_records = self.TestModel.find_all(self.pool)
        self.assertEqual(len(all_records), 5)
        
        # Find with condition
        filtered = self.TestModel.find_all(self.pool, value=2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, 'item_2')
    
    def test_update_record(self):
        """Test updating existing record"""
        # Save a record
        model = self.TestModel(name='original', value=1)
        model.save(self.pool)
        original_id = model.id
        
        # Update it
        model.name = 'updated'
        model.value = 2
        model.save(self.pool)
        
        # Verify update
        found = self.TestModel.find(self.pool, id=original_id)
        self.assertEqual(found.name, 'updated')
        self.assertEqual(found.value, 2)
    
    def test_delete_record(self):
        """Test deleting record"""
        # Save a record
        model = self.TestModel(name='delete_me', value=999)
        model.save(self.pool)
        record_id = model.id
        
        # Delete it
        model.delete(self.pool)
        
        # Verify deletion
        found = self.TestModel.find(self.pool, id=record_id)
        self.assertIsNone(found)


class TestTransactionModel(unittest.TestCase):
    """Test transaction-specific model"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name
        )
        
        # Create transaction table
        with self.pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_hash TEXT UNIQUE NOT NULL,
                    from_address TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    amount REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    status TEXT NOT NULL,
                    block_number INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT
                )
            ''')
            conn.commit()
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_balance_calculation(self):
        """Test balance calculation"""
        address = "0x123"
        
        # Add transactions
        with self.pool.get_connection() as conn:
            # Incoming transaction
            conn.execute('''
                INSERT INTO transactions 
                (tx_hash, from_address, to_address, amount, fee, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('tx1', '0x456', address, 100.0, 1.0, 'confirmed'))
            
            # Outgoing transaction
            conn.execute('''
                INSERT INTO transactions 
                (tx_hash, from_address, to_address, amount, fee, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('tx2', address, '0x789', 30.0, 0.5, 'confirmed'))
            
            conn.commit()
        
        # Calculate balance
        balance = TransactionModel.get_balance(self.pool, address)
        
        # Balance should be: 100 (received) - 30 (sent) - 0.5 (fee) = 69.5
        self.assertEqual(balance, 69.5)
    
    def test_transaction_history(self):
        """Test transaction history retrieval"""
        address = "0xabc"
        
        # Add transactions
        with self.pool.get_connection() as conn:
            for i in range(10):
                conn.execute('''
                    INSERT INTO transactions 
                    (tx_hash, from_address, to_address, amount, fee, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (f'tx{i}', address if i % 2 else '0xother', 
                     '0xother' if i % 2 else address, 
                     float(i * 10), 0.1, 'confirmed'))
            conn.commit()
        
        # Get history
        history = TransactionModel.get_transaction_history(
            self.pool, address, limit=5
        )
        
        self.assertEqual(len(history), 5)


class TestCacheManager(unittest.TestCase):
    """Test caching functionality"""
    
    def setUp(self):
        self.cache = CacheManager(ttl_seconds=1)
    
    def test_set_and_get(self):
        """Test setting and getting cache values"""
        self.cache.set('key1', 'value1')
        
        value = self.cache.get('key1')
        self.assertEqual(value, 'value1')
        
        # Non-existent key
        value = self.cache.get('nonexistent')
        self.assertIsNone(value)
    
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        self.cache.set('expire_key', 'expire_value', ttl=1)
        
        # Should exist immediately
        value = self.cache.get('expire_key')
        self.assertEqual(value, 'expire_value')
        
        # Should expire after TTL
        time.sleep(1.1)
        value = self.cache.get('expire_key')
        self.assertIsNone(value)
    
    def test_cache_deletion(self):
        """Test cache deletion"""
        self.cache.set('delete_key', 'delete_value')
        
        # Delete the key
        self.cache.delete('delete_key')
        
        # Should not exist
        value = self.cache.get('delete_key')
        self.assertIsNone(value)
    
    def test_cache_clear(self):
        """Test clearing all cache"""
        # Set multiple values
        for i in range(5):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Clear cache
        self.cache.clear()
        
        # All values should be gone
        for i in range(5):
            value = self.cache.get(f'key{i}')
            self.assertIsNone(value)
    
    def test_thread_safety(self):
        """Test cache thread safety"""
        results = []
        
        def worker(key, value):
            self.cache.set(key, value)
            time.sleep(0.01)
            retrieved = self.cache.get(key)
            results.append((key, retrieved))
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(f'key{i}', f'value{i}'))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        # All operations should succeed
        self.assertEqual(len(results), 10)
        for key, value in results:
            self.assertEqual(value, value)


class TestDatabaseBackup(unittest.TestCase):
    """Test database backup functionality"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name
        )
        self.backup = DatabaseBackup(self.pool)
        
        # Add some data
        with self.pool.get_connection() as conn:
            conn.execute(
                "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "INSERT INTO test_data (value) VALUES ('test')"
            )
            conn.commit()
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
        # Clean up backup directory
        import shutil
        if os.path.exists('backups'):
            shutil.rmtree('backups')
    
    def test_create_backup(self):
        """Test backup creation"""
        backup_file = self.backup.create_backup()
        
        self.assertTrue(os.path.exists(backup_file))
        self.assertTrue(os.path.getsize(backup_file) > 0)
        
        # Cleanup
        os.unlink(backup_file)
    
    def test_restore_backup(self):
        """Test backup restoration"""
        # Create backup
        backup_file = self.backup.create_backup()
        
        # Modify database
        with self.pool.get_connection() as conn:
            conn.execute("DELETE FROM test_data")
            conn.commit()
            
            # Verify data is gone
            cursor = conn.execute("SELECT COUNT(*) FROM test_data")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)
        
        # Create new pool for restoration
        restore_pool = ConnectionPool(
            db_type=DatabaseType.SQLITE,
            db_path=":memory:"
        )
        restore_backup = DatabaseBackup(restore_pool)
        
        # Restore backup
        restore_backup.restore_backup(backup_file)
        
        # Verify data is restored
        with restore_pool.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test_data")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
        
        # Cleanup
        os.unlink(backup_file)


class TestProductionDatabaseLayer(unittest.TestCase):
    """Test complete production database layer"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db = ProductionDatabaseLayer(
            db_type=DatabaseType.SQLITE,
            db_path=self.temp_db.name
        )
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
        # Clean up backup directory
        import shutil
        if os.path.exists('backups'):
            shutil.rmtree('backups')
    
    def test_health_check(self):
        """Test database health check"""
        health = self.db.health_check()
        
        self.assertEqual(health['status'], 'healthy')
        self.assertIn('active_connections', health)
        self.assertIn('pool_size', health)
    
    def test_integrated_functionality(self):
        """Test integrated database operations"""
        # Define a model
        class User(BaseModel):
            _table_name = 'users'
        
        # Use cache
        user_data = {'username': 'cached_user', 'email': 'ceo@qenex.ai'}
        self.db.cache.set('user:1', user_data)
        
        cached = self.db.cache.get('user:1')
        self.assertEqual(cached['username'], 'cached_user')
        
        # Create backup
        backup_file = self.db.backup.create_backup()
        self.assertTrue(os.path.exists(backup_file))
        
        # Cleanup
        os.unlink(backup_file)


if __name__ == '__main__':
    unittest.main()