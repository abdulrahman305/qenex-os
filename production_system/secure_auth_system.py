#!/usr/bin/env python3
"""
Production-Grade Authentication and Authorization System
Addresses critical security vulnerabilities identified in analysis
"""

import os
import jwt
import bcrypt
import secrets
import sqlite3
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from functools import wraps
import re

class UserRole(Enum):
    """User role definitions with hierarchical permissions"""
    ADMIN = "admin"
    OPERATOR = "operator"
    USER = "user"
    READONLY = "readonly"

@dataclass
class User:
    """User entity with secure password handling"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    mfa_secret: Optional[str]
    api_keys: List[str]

class SecureAuthSystem:
    """Production authentication system with comprehensive security"""
    
    def __init__(self, db_path: str = "production.db", secret_key: str = None):
        self.db_path = db_path
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY') or secrets.token_urlsafe(32)
        self.logger = self._setup_logging()
        self.db = self._setup_database()
        self.rate_limiter = {}
        self.session_store = {}
        self.blacklisted_tokens = set()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure security logging"""
        logger = logging.getLogger('AuthSystem')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('auth_security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_database(self) -> sqlite3.Connection:
        """Initialize secure user database"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('PRAGMA foreign_keys = ON')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                mfa_secret TEXT,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                ip_address TEXT,
                success BOOLEAN,
                details TEXT
            )
        ''')
        
        conn.commit()
        return conn
    
    def _hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt"""
        # Validate password strength
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain uppercase letter")
        if not re.search(r'[a-z]', password):
            raise ValueError("Password must contain lowercase letter")
        if not re.search(r'\d', password):
            raise ValueError("Password must contain digit")
        
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                password_hash.encode('utf-8')
            )
        except Exception:
            return False
    
    def _generate_token(self, user_id: str, token_type: str = 'access') -> str:
        """Generate JWT token with expiration"""
        expiry_minutes = 15 if token_type == 'access' else 10080  # 7 days for refresh
        
        payload = {
            'user_id': user_id,
            'type': token_type,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes),
            'jti': secrets.token_urlsafe(16)  # Token ID for revocation
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Verify token hasn't expired
            if datetime.fromtimestamp(payload['exp'], timezone.utc) < datetime.now(timezone.utc):
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def _check_rate_limit(self, identifier: str, max_attempts: int = 5) -> bool:
        """Rate limiting to prevent brute force"""
        current_time = time.time()
        window = 300  # 5 minutes
        
        if identifier not in self.rate_limiter:
            self.rate_limiter[identifier] = []
        
        # Clean old attempts
        self.rate_limiter[identifier] = [
            t for t in self.rate_limiter[identifier]
            if current_time - t < window
        ]
        
        if len(self.rate_limiter[identifier]) >= max_attempts:
            return False
        
        self.rate_limiter[identifier].append(current_time)
        return True
    
    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> Tuple[bool, str, Optional[str]]:
        """Register new user with validation"""
        try:
            # Validate email
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                return False, "Invalid email format", None
            
            # Check if user exists
            cursor = self.db.execute(
                'SELECT user_id FROM users WHERE username = ? OR email = ?',
                (username, email)
            )
            if cursor.fetchone():
                return False, "User already exists", None
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            password_hash = self._hash_password(password)
            
            self.db.execute('''
                INSERT INTO users (user_id, username, email, password_hash, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, username, email, password_hash, role.value))
            
            self.db.commit()
            
            # Log registration
            self._audit_log(user_id, "USER_REGISTERED", None, None, True)
            
            # Generate initial tokens
            access_token = self._generate_token(user_id, 'access')
            
            return True, "User registered successfully", access_token
            
        except ValueError as e:
            return False, str(e), None
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False, "Registration failed", None
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str = None
    ) -> Tuple[bool, str, Optional[Dict[str, str]]]:
        """Authenticate user with comprehensive checks"""
        try:
            # Rate limiting
            if not self._check_rate_limit(f"auth:{username}"):
                return False, "Too many login attempts", None
            
            # Get user
            cursor = self.db.execute('''
                SELECT user_id, password_hash, role, is_active, 
                       failed_attempts, locked_until
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            if not user:
                self._audit_log(None, "AUTH_FAILED", username, ip_address, False)
                return False, "Invalid credentials", None
            
            user_id, password_hash, role, is_active, failed_attempts, locked_until = user
            
            # Check if account is locked
            if locked_until:
                locked_time = datetime.fromisoformat(locked_until)
                if locked_time > datetime.now():
                    return False, "Account temporarily locked", None
            
            # Check if account is active
            if not is_active:
                return False, "Account deactivated", None
            
            # Verify password
            if not self._verify_password(password, password_hash):
                # Increment failed attempts
                failed_attempts += 1
                
                # Lock account after 5 failed attempts
                if failed_attempts >= 5:
                    locked_until = datetime.now() + timedelta(minutes=30)
                    self.db.execute('''
                        UPDATE users SET failed_attempts = ?, locked_until = ?
                        WHERE user_id = ?
                    ''', (failed_attempts, locked_until.isoformat(), user_id))
                else:
                    self.db.execute('''
                        UPDATE users SET failed_attempts = ?
                        WHERE user_id = ?
                    ''', (failed_attempts, user_id))
                
                self.db.commit()
                self._audit_log(user_id, "AUTH_FAILED", username, ip_address, False)
                return False, "Invalid credentials", None
            
            # Reset failed attempts and update last login
            self.db.execute('''
                UPDATE users SET failed_attempts = 0, locked_until = NULL,
                                last_login = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user_id,))
            self.db.commit()
            
            # Generate tokens
            access_token = self._generate_token(user_id, 'access')
            refresh_token = self._generate_token(user_id, 'refresh')
            
            # Create session
            session_id = secrets.token_urlsafe(16)
            self.db.execute('''
                INSERT INTO sessions (session_id, user_id, expires_at, ip_address)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                (datetime.now() + timedelta(hours=24)).isoformat(),
                ip_address
            ))
            self.db.commit()
            
            # Audit log
            self._audit_log(user_id, "AUTH_SUCCESS", username, ip_address, True)
            
            return True, "Authentication successful", {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'session_id': session_id,
                'role': role
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, "Authentication failed", None
    
    def verify_access(self, token: str, required_role: UserRole = None) -> Tuple[bool, Optional[str]]:
        """Verify access token and check permissions"""
        payload = self._verify_token(token)
        
        if not payload:
            return False, None
        
        if payload['type'] != 'access':
            return False, None
        
        # Get user role
        cursor = self.db.execute(
            'SELECT role, is_active FROM users WHERE user_id = ?',
            (payload['user_id'],)
        )
        
        user = cursor.fetchone()
        if not user or not user[1]:  # Check if active
            return False, None
        
        user_role = UserRole(user[0])
        
        # Check role hierarchy
        if required_role:
            role_hierarchy = {
                UserRole.READONLY: 0,
                UserRole.USER: 1,
                UserRole.OPERATOR: 2,
                UserRole.ADMIN: 3
            }
            
            if role_hierarchy[user_role] < role_hierarchy[required_role]:
                return False, None
        
        return True, payload['user_id']
    
    def create_api_key(
        self,
        user_id: str,
        name: str = None,
        expires_in_days: int = 365
    ) -> Optional[str]:
        """Create API key for programmatic access"""
        try:
            key = secrets.token_urlsafe(32)
            key_id = secrets.token_urlsafe(16)
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            
            expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            self.db.execute('''
                INSERT INTO api_keys (key_id, user_id, key_hash, name, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (key_id, user_id, key_hash, name, expires_at.isoformat()))
            
            self.db.commit()
            
            return f"{key_id}.{key}"
            
        except Exception as e:
            self.logger.error(f"API key creation error: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """Verify API key and return user_id"""
        try:
            parts = api_key.split('.')
            if len(parts) != 2:
                return False, None
            
            key_id, key = parts
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            
            cursor = self.db.execute('''
                SELECT user_id, expires_at, is_active
                FROM api_keys
                WHERE key_id = ? AND key_hash = ?
            ''', (key_id, key_hash))
            
            result = cursor.fetchone()
            if not result:
                return False, None
            
            user_id, expires_at, is_active = result
            
            if not is_active:
                return False, None
            
            if datetime.fromisoformat(expires_at) < datetime.now():
                return False, None
            
            # Update last used
            self.db.execute('''
                UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                WHERE key_id = ?
            ''', (key_id,))
            self.db.commit()
            
            return True, user_id
            
        except Exception:
            return False, None
    
    def revoke_token(self, token: str):
        """Revoke a token by adding to blacklist"""
        self.blacklisted_tokens.add(token)
        self._audit_log(None, "TOKEN_REVOKED", token[:20], None, True)
    
    def _audit_log(
        self,
        user_id: Optional[str],
        action: str,
        resource: Optional[str],
        ip_address: Optional[str],
        success: bool
    ):
        """Log security events for audit"""
        try:
            self.db.execute('''
                INSERT INTO audit_log (user_id, action, resource, ip_address, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, action, resource, ip_address, success))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Audit log error: {e}")

def require_auth(required_role: UserRole = None):
    """Decorator for protecting routes"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request headers (framework-specific)
            token = kwargs.get('auth_token')
            
            if not token:
                return {'error': 'Authentication required'}, 401
            
            auth_system = SecureAuthSystem()
            valid, user_id = auth_system.verify_access(token, required_role)
            
            if not valid:
                return {'error': 'Access denied'}, 403
            
            kwargs['user_id'] = user_id
            return func(*args, **kwargs)
        
        return wrapper
    return decorator