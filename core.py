#!/usr/bin/env python3
"""
Unified Core System
"""

import os
import sys
import json
import time
import hashlib
import secrets
import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod

# Configuration
CONFIG = {
    'db_path': Path('data/core.db'),
    'log_path': Path('logs/core.log'),
    'cache_ttl': 300,
    'max_workers': 10,
    'api_timeout': 30,
}

# Ensure directories exist
CONFIG['db_path'].parent.mkdir(parents=True, exist_ok=True)
CONFIG['log_path'].parent.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_path']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Database:
    """Unified database handler"""
    
    def __init__(self, db_path: Path = CONFIG['db_path']):
        self.db_path = db_path
        self.init_schema()
    
    def init_schema(self):
        """Initialize database schema"""
        with self.connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
            ''')
    
    @contextmanager
    def connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level='DEFERRED'
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class Cache:
    """Simple cache implementation"""
    
    def __init__(self, db: Database):
        self.db = db
        self._local_cache = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        with self._lock:
            # Check local cache first
            if key in self._local_cache:
                value, expires = self._local_cache[key]
                if datetime.now() < expires:
                    return value
                del self._local_cache[key]
            
            # Check database
            with self.db.connection() as conn:
                cursor = conn.execute(
                    'SELECT value FROM cache WHERE key = ? AND expires_at > ?',
                    (key, datetime.now())
                )
                row = cursor.fetchone()
                if row:
                    value = json.loads(row['value'])
                    self._local_cache[key] = (value, datetime.now() + timedelta(seconds=60))
                    return value
        return None
    
    def set(self, key: str, value: Any, ttl: int = CONFIG['cache_ttl']):
        """Set cached value"""
        with self._lock:
            expires = datetime.now() + timedelta(seconds=ttl)
            self._local_cache[key] = (value, expires)
            
            with self.db.connection() as conn:
                conn.execute(
                    '''INSERT OR REPLACE INTO cache (key, value, expires_at)
                       VALUES (?, ?, ?)''',
                    (key, json.dumps(value), expires)
                )
    
    def clear(self):
        """Clear expired cache entries"""
        with self._lock:
            self._local_cache.clear()
            with self.db.connection() as conn:
                conn.execute('DELETE FROM cache WHERE expires_at < ?', (datetime.now(),))


class Auth:
    """Authentication handler"""
    
    def __init__(self, db: Database):
        self.db = db
        self.sessions = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = secrets.token_hex(16)
        combined = f"{password}{salt}"
        hash_value = hashlib.pbkdf2_hmac('sha256', combined.encode(), salt.encode(), 100000)
        return f"{salt}:{hash_value.hex()}"
    
    def verify_password(self, password: str, hash_str: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = hash_str.split(':')
            combined = f"{password}{salt}"
            hash_value = hashlib.pbkdf2_hmac('sha256', combined.encode(), salt.encode(), 100000)
            return hash_value.hex() == hash_hex
        except:
            return False
    
    def create_user(self, username: str, password: str) -> Optional[int]:
        """Create new user"""
        if len(password) < 8:
            return None
        
        password_hash = self.hash_password(password)
        
        try:
            with self.db.connection() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                    (username, password_hash)
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and create session"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                'SELECT id, password_hash FROM users WHERE username = ?',
                (username,)
            )
            row = cursor.fetchone()
            
            if row and self.verify_password(password, row['password_hash']):
                session_id = secrets.token_urlsafe(32)
                expires = datetime.now() + timedelta(hours=24)
                
                conn.execute(
                    'INSERT INTO sessions (id, user_id, expires_at) VALUES (?, ?, ?)',
                    (session_id, row['id'], expires)
                )
                
                self.sessions[session_id] = {
                    'user_id': row['id'],
                    'expires': expires
                }
                
                return session_id
        
        return None
    
    def verify_session(self, session_id: str) -> Optional[int]:
        """Verify session and return user_id"""
        # Check memory cache
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() < session['expires']:
                return session['user_id']
            del self.sessions[session_id]
        
        # Check database
        with self.db.connection() as conn:
            cursor = conn.execute(
                'SELECT user_id FROM sessions WHERE id = ? AND expires_at > ?',
                (session_id, datetime.now())
            )
            row = cursor.fetchone()
            if row:
                self.sessions[session_id] = {
                    'user_id': row['user_id'],
                    'expires': datetime.now() + timedelta(hours=1)
                }
                return row['user_id']
        
        return None


class Service(ABC):
    """Base service class"""
    
    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.logger = logging.getLogger(f'service.{name}')
    
    @abstractmethod
    def start(self):
        """Start service"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop service"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Service health check"""
        return {
            'name': self.name,
            'running': self.running,
            'timestamp': datetime.now().isoformat()
        }


class Worker:
    """Background worker"""
    
    def __init__(self, name: str, interval: int = 60):
        self.name = name
        self.interval = interval
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(f'worker.{name}')
    
    def start(self):
        """Start worker"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.logger.info(f'Worker {self.name} started')
    
    def stop(self):
        """Stop worker"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info(f'Worker {self.name} stopped')
    
    def _run(self):
        """Worker loop"""
        while self.running:
            try:
                self.work()
            except Exception as e:
                self.logger.error(f'Worker error: {e}')
            time.sleep(self.interval)
    
    def work(self):
        """Override this method"""
        pass


class API:
    """Simple API handler"""
    
    def __init__(self, auth: Auth, cache: Cache):
        self.auth = auth
        self.cache = cache
        self.routes = {}
    
    def route(self, path: str, method: str = 'GET'):
        """Route decorator"""
        def decorator(func):
            self.routes[f'{method} {path}'] = func
            return func
        return decorator
    
    def handle_request(self, method: str, path: str, headers: Dict, body: Any = None) -> Dict:
        """Handle API request"""
        route_key = f'{method} {path}'
        
        # Check if route exists
        if route_key not in self.routes:
            return {'status': 404, 'error': 'Not found'}
        
        # Check authentication
        session_id = headers.get('Authorization', '').replace('Bearer ', '')
        user_id = self.auth.verify_session(session_id) if session_id else None
        
        # Execute handler
        try:
            handler = self.routes[route_key]
            response = handler(user_id=user_id, headers=headers, body=body)
            return {'status': 200, 'data': response}
        except Exception as e:
            logger.error(f'API error: {e}')
            return {'status': 500, 'error': str(e)}


class Monitor:
    """System monitoring"""
    
    def __init__(self, db: Database):
        self.db = db
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record_metric(self, name: str, value: float):
        """Record metric value"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
            
            # Keep only last 100 values
            if len(self.metrics[name]) > 100:
                self.metrics[name] = self.metrics[name][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            result = {}
            for name, values in self.metrics.items():
                if values:
                    recent = [v['value'] for v in values[-10:]]
                    result[name] = {
                        'current': values[-1]['value'],
                        'avg': sum(recent) / len(recent),
                        'min': min(recent),
                        'max': max(recent)
                    }
            return result
    
    def log(self, level: str, message: str):
        """Log message to database"""
        with self.db.connection() as conn:
            conn.execute(
                'INSERT INTO logs (level, message) VALUES (?, ?)',
                (level, message)
            )


class Core:
    """Main system core"""
    
    def __init__(self):
        self.db = Database()
        self.cache = Cache(self.db)
        self.auth = Auth(self.db)
        self.monitor = Monitor(self.db)
        self.api = API(self.auth, self.cache)
        self.services = {}
        self.workers = {}
        
        # Setup API routes
        self._setup_routes()
        
        # Setup cleanup worker
        cleanup_worker = Worker('cleanup', interval=3600)
        cleanup_worker.work = self.cache.clear
        self.workers['cleanup'] = cleanup_worker
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.api.route('/health', 'GET')
        def health_check(**kwargs):
            return {
                'status': 'healthy',
                'services': {
                    name: service.health_check()
                    for name, service in self.services.items()
                },
                'metrics': self.monitor.get_metrics()
            }
        
        @self.api.route('/auth/register', 'POST')
        def register(body, **kwargs):
            username = body.get('username')
            password = body.get('password')
            
            if not username or not password:
                raise ValueError('Username and password required')
            
            user_id = self.auth.create_user(username, password)
            if user_id:
                return {'user_id': user_id}
            raise ValueError('Registration failed')
        
        @self.api.route('/auth/login', 'POST')
        def login(body, **kwargs):
            username = body.get('username')
            password = body.get('password')
            
            session_id = self.auth.authenticate(username, password)
            if session_id:
                return {'session_id': session_id}
            raise ValueError('Invalid credentials')
        
        @self.api.route('/metrics', 'GET')
        def metrics(**kwargs):
            return self.monitor.get_metrics()
    
    def start(self):
        """Start core system"""
        logger.info('Starting core system')
        
        # Start workers
        for worker in self.workers.values():
            worker.start()
        
        # Start services
        for service in self.services.values():
            service.start()
        
        logger.info('Core system started')
    
    def stop(self):
        """Stop core system"""
        logger.info('Stopping core system')
        
        # Stop services
        for service in self.services.values():
            service.stop()
        
        # Stop workers
        for worker in self.workers.values():
            worker.stop()
        
        logger.info('Core system stopped')
    
    def run(self):
        """Run core system"""
        self.start()
        
        try:
            # Keep running
            while True:
                time.sleep(1)
                
                # Record metrics
                self.monitor.record_metric('uptime', time.time())
                self.monitor.record_metric('memory', self._get_memory_usage())
                
        except KeyboardInterrupt:
            logger.info('Shutdown requested')
        finally:
            self.stop()
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0


def main():
    """Main entry point"""
    core = Core()
    
    # Example: Register and login
    user_id = core.auth.create_user('admin', 'password123')
    if user_id:
        logger.info(f'Created admin user: {user_id}')
    
    session = core.auth.authenticate('admin', 'password123')
    if session:
        logger.info(f'Admin session: {session}')
    
    # Run system
    core.run()


if __name__ == '__main__':
    main()