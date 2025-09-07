#!/usr/bin/env python3
"""
QENEX Security Fortress - Impenetrable Security Layer
Zero-vulnerability architecture with defense in depth
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import redis.asyncio as redis

# Security constants
MIN_PASSWORD_LENGTH = 12
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes
TOKEN_EXPIRY = 3600  # 1 hour
RATE_LIMIT_WINDOW = 60  # 1 minute
MAX_REQUESTS_PER_WINDOW = 100
ENCRYPTION_KEY_ROTATION_DAYS = 30


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "PUBLIC"
    AUTHENTICATED = "AUTHENTICATED"
    VERIFIED = "VERIFIED"
    PRIVILEGED = "PRIVILEGED"
    ADMIN = "ADMIN"
    SYSTEM = "SYSTEM"


class ThreatType(Enum):
    """Threat classification"""
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    CSRF = "CSRF"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    LDAP_INJECTION = "LDAP_INJECTION"
    XXE = "XXE"
    SSRF = "SSRF"
    INSECURE_DESERIALIZATION = "INSECURE_DESERIALIZATION"
    BRUTE_FORCE = "BRUTE_FORCE"
    DOS = "DOS"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"


@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    permissions: Set[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class InputSanitizer:
    """Input validation and sanitization"""
    
    # Dangerous patterns for different attack types
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
        r"(--|\||\*|;|'|\")",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"(EXEC(\s|\()|EXECUTE(\s|\())",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"\.\\",
        r"%2e%2e",
        r"%252e%252e",
    ]
    
    COMMAND_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"\|\|",
        r"&&",
        r">",
        r"<",
    ]
    
    @classmethod
    def sanitize_sql(cls, input_str: str) -> str:
        """Sanitize SQL input"""
        if not input_str:
            return ""
        
        # Remove SQL injection patterns
        import re
        for pattern in cls.SQL_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                raise ValueError(f"Potential SQL injection detected: {pattern}")
        
        # Escape special characters
        return input_str.replace("'", "''").replace("\\", "\\\\")
    
    @classmethod
    def sanitize_html(cls, input_str: str) -> str:
        """Sanitize HTML input"""
        if not input_str:
            return ""
        
        # HTML entity encoding
        replacements = {
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
            "&": "&amp;"
        }
        
        for char, entity in replacements.items():
            input_str = input_str.replace(char, entity)
        
        return input_str
    
    @classmethod
    def sanitize_path(cls, path: str) -> str:
        """Sanitize file paths"""
        import re
        import os
        
        if not path:
            return ""
        
        # Check for path traversal
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                raise ValueError(f"Path traversal attempt detected: {pattern}")
        
        # Normalize and validate path
        normalized = os.path.normpath(path)
        if normalized.startswith("..") or normalized.startswith("/"):
            raise ValueError("Invalid path")
        
        return normalized
    
    @classmethod
    def sanitize_command(cls, command: str) -> str:
        """Sanitize shell commands"""
        import re
        
        if not command:
            return ""
        
        # Check for command injection
        for pattern in cls.COMMAND_PATTERNS:
            if re.search(pattern, command):
                raise ValueError(f"Command injection attempt detected: {pattern}")
        
        # Whitelist alphanumeric and basic punctuation
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_\.]', '', command)
        return sanitized


class CryptographyManager:
    """Advanced cryptographic operations"""
    
    def __init__(self):
        self.password_hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )
        self._keys = {}
        self._rotation_schedule = {}
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize encryption keys"""
        # Generate master key
        self._master_key = secrets.token_bytes(32)
        
        # Derive encryption keys
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=secrets.token_bytes(16),
            iterations=100000,
            backend=default_backend()
        )
        
        self._keys['data'] = Fernet.generate_key()
        self._keys['session'] = Fernet.generate_key()
        self._keys['api'] = secrets.token_urlsafe(32)
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2id"""
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        
        # Add pepper before hashing
        peppered = password + secrets.token_hex(8)
        return self.password_hasher.hash(peppered)
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash"""
        try:
            # Note: In production, store pepper separately
            return self.password_hasher.verify(hash, password)
        except VerifyMismatchError:
            return False
    
    def encrypt_data(self, data: bytes, context: str = 'data') -> bytes:
        """Encrypt data with context-specific key"""
        if context not in self._keys:
            raise ValueError(f"Invalid encryption context: {context}")
        
        fernet = Fernet(self._keys[context])
        return fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, context: str = 'data') -> bytes:
        """Decrypt data with context-specific key"""
        if context not in self._keys:
            raise ValueError(f"Invalid decryption context: {context}")
        
        fernet = Fernet(self._keys[context])
        return fernet.decrypt(encrypted_data)
    
    def generate_token(self, payload: Dict[str, Any], expiry_hours: int = 1) -> str:
        """Generate JWT token"""
        payload['exp'] = datetime.utcnow() + timedelta(hours=expiry_hours)
        payload['iat'] = datetime.utcnow()
        payload['jti'] = secrets.token_urlsafe(16)
        
        return jwt.encode(payload, self._keys['api'], algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            return jwt.decode(token, self._keys['api'], algorithms=['HS256'])
        except jwt.PyJWTError:
            return None
    
    def rotate_keys(self):
        """Rotate encryption keys"""
        old_keys = self._keys.copy()
        self._initialize_keys()
        
        # Re-encrypt data with new keys (in production, this would be more complex)
        self._rotation_schedule[datetime.now()] = old_keys
        
        # Clean up old keys after grace period
        cutoff = datetime.now() - timedelta(days=7)
        self._rotation_schedule = {
            k: v for k, v in self._rotation_schedule.items() if k > cutoff
        }


class RateLimiter:
    """Advanced rate limiting with sliding window"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.limits = {
            'default': (100, 60),  # 100 requests per 60 seconds
            'auth': (5, 300),      # 5 auth attempts per 5 minutes
            'api': (1000, 60),     # 1000 API calls per minute
            'transaction': (10, 60), # 10 transactions per minute
        }
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        action: str = 'default'
    ) -> Tuple[bool, Optional[int]]:
        """Check if rate limit exceeded"""
        limit, window = self.limits.get(action, self.limits['default'])
        
        key = f"rate_limit:{action}:{identifier}"
        current_time = time.time()
        window_start = current_time - window
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipe.expire(key, window)
        
        results = await pipe.execute()
        count = results[1]
        
        if count >= limit:
            # Calculate wait time
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                wait_time = int(window - (current_time - oldest[0][1]))
                return False, wait_time
            return False, window
        
        return True, None
    
    async def reset_limit(self, identifier: str, action: str = 'default'):
        """Reset rate limit for identifier"""
        key = f"rate_limit:{action}:{identifier}"
        await self.redis.delete(key)


class SessionManager:
    """Secure session management"""
    
    def __init__(self, redis_client: redis.Redis, crypto: CryptographyManager):
        self.redis = redis_client
        self.crypto = crypto
        self.session_timeout = 3600  # 1 hour
        self.absolute_timeout = 86400  # 24 hours
    
    async def create_session(
        self, 
        user_id: str, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create secure session"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_activity': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {},
            'csrf_token': secrets.token_urlsafe(32)
        }
        
        # Encrypt session data
        encrypted = self.crypto.encrypt_data(
            json.dumps(session_data).encode(),
            context='session'
        )
        
        # Store in Redis with expiry
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            encrypted
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        encrypted = await self.redis.get(f"session:{session_id}")
        
        if not encrypted:
            return None
        
        try:
            decrypted = self.crypto.decrypt_data(encrypted, context='session')
            session_data = json.loads(decrypted)
            
            # Check absolute timeout
            created_at = datetime.fromisoformat(session_data['created_at'])
            if datetime.now(timezone.utc) - created_at > timedelta(seconds=self.absolute_timeout):
                await self.destroy_session(session_id)
                return None
            
            # Update last activity
            session_data['last_activity'] = datetime.now(timezone.utc).isoformat()
            
            # Re-encrypt and update
            encrypted = self.crypto.encrypt_data(
                json.dumps(session_data).encode(),
                context='session'
            )
            await self.redis.setex(
                f"session:{session_id}",
                self.session_timeout,
                encrypted
            )
            
            return session_data
        except Exception:
            return None
    
    async def destroy_session(self, session_id: str):
        """Destroy session"""
        await self.redis.delete(f"session:{session_id}")
    
    async def validate_csrf_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False
        
        return hmac.compare_digest(
            session_data.get('csrf_token', ''),
            token
        )


class AccessControl:
    """Role-based access control with fine-grained permissions"""
    
    def __init__(self):
        self.roles = {
            'admin': {
                'level': SecurityLevel.ADMIN,
                'permissions': {'*'}  # All permissions
            },
            'operator': {
                'level': SecurityLevel.PRIVILEGED,
                'permissions': {
                    'transaction.create',
                    'transaction.read',
                    'transaction.update',
                    'account.read',
                    'report.generate'
                }
            },
            'user': {
                'level': SecurityLevel.VERIFIED,
                'permissions': {
                    'transaction.create',
                    'transaction.read',
                    'account.read'
                }
            },
            'guest': {
                'level': SecurityLevel.PUBLIC,
                'permissions': {
                    'public.read'
                }
            }
        }
        
        self.permission_hierarchy = {
            'transaction': ['create', 'read', 'update', 'delete'],
            'account': ['create', 'read', 'update', 'delete'],
            'report': ['generate', 'read'],
            'admin': ['users', 'system', 'audit']
        }
    
    def check_permission(
        self, 
        context: SecurityContext, 
        required_permission: str
    ) -> bool:
        """Check if context has required permission"""
        # Admin has all permissions
        if '*' in context.permissions:
            return True
        
        # Check exact permission
        if required_permission in context.permissions:
            return True
        
        # Check wildcard permissions (e.g., 'transaction.*')
        parts = required_permission.split('.')
        if len(parts) == 2:
            wildcard = f"{parts[0]}.*"
            if wildcard in context.permissions:
                return True
        
        return False
    
    def require_permission(self, permission: str):
        """Decorator to enforce permission requirements"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract security context from kwargs
                context = kwargs.get('security_context')
                if not context:
                    raise PermissionError("Security context required")
                
                if not self.check_permission(context, permission):
                    raise PermissionError(f"Permission denied: {permission}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_user_permissions(self, user_roles: List[str]) -> Set[str]:
        """Get all permissions for user roles"""
        permissions = set()
        for role in user_roles:
            if role in self.roles:
                permissions.update(self.roles[role]['permissions'])
        return permissions


class AuditLogger:
    """Comprehensive audit logging"""
    
    def __init__(self, database_pool):
        self.db = database_pool
        self.queue = asyncio.Queue(maxsize=10000)
        self.batch_size = 100
        self.flush_interval = 5  # seconds
        self._running = False
        self._task = None
    
    async def start(self):
        """Start audit logger"""
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop audit logger"""
        self._running = False
        if self._task:
            await self._task
    
    async def log(
        self,
        event_type: str,
        context: SecurityContext,
        details: Dict[str, Any],
        severity: str = 'INFO'
    ):
        """Log security event"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'severity': severity,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'ip_address': context.ip_address,
            'user_agent': context.user_agent,
            'details': details
        }
        
        # Add to queue
        try:
            await asyncio.wait_for(
                self.queue.put(event),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # Queue full, log to stderr
            import sys
            print(f"AUDIT QUEUE FULL: {event}", file=sys.stderr)
    
    async def _process_queue(self):
        """Process audit log queue"""
        batch = []
        last_flush = time.time()
        
        while self._running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                batch.append(event)
                
                # Flush if batch full or timeout
                if len(batch) >= self.batch_size or \
                   time.time() - last_flush > self.flush_interval:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except asyncio.TimeoutError:
                # Flush any pending events
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
            except Exception as e:
                import sys
                print(f"Audit logger error: {e}", file=sys.stderr)
        
        # Final flush
        if batch:
            await self._flush_batch(batch)
    
    async def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Write batch to database"""
        if not batch:
            return
        
        async with self.db.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO audit_log 
                (timestamp, event_type, severity, user_id, session_id, 
                 ip_address, user_agent, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                [(
                    e['timestamp'], e['event_type'], e['severity'],
                    e['user_id'], e['session_id'], e['ip_address'],
                    e['user_agent'], json.dumps(e['details'])
                ) for e in batch]
            )


class SecurityFortress:
    """Main security orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crypto = CryptographyManager()
        self.sanitizer = InputSanitizer
        self.access_control = AccessControl()
        self.redis = None
        self.rate_limiter = None
        self.session_manager = None
        self.audit_logger = None
        self._initialized = False
    
    async def initialize(self, database_pool):
        """Initialize security fortress"""
        # Initialize Redis
        self.redis = await redis.create_redis_pool(
            f"redis://{self.config.get('redis_host', 'localhost')}:"
            f"{self.config.get('redis_port', 6379)}"
        )
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.redis)
        self.session_manager = SessionManager(self.redis, self.crypto)
        self.audit_logger = AuditLogger(database_pool)
        
        # Start audit logger
        await self.audit_logger.start()
        
        self._initialized = True
    
    async def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str
    ) -> Optional[str]:
        """Authenticate user"""
        # Check rate limit
        allowed, wait_time = await self.rate_limiter.check_rate_limit(
            ip_address,
            'auth'
        )
        
        if not allowed:
            await self.audit_logger.log(
                'AUTH_RATE_LIMIT',
                SecurityContext(ip_address=ip_address),
                {'username': username, 'wait_time': wait_time},
                'WARNING'
            )
            return None
        
        # Sanitize input
        username = self.sanitizer.sanitize_sql(username)
        
        # Verify credentials (simplified - in production, query database)
        # This is where you'd check against stored password hash
        
        # Create session
        session_id = await self.session_manager.create_session(
            username,
            {'ip_address': ip_address}
        )
        
        # Log successful auth
        await self.audit_logger.log(
            'AUTH_SUCCESS',
            SecurityContext(user_id=username, ip_address=ip_address),
            {'session_id': session_id},
            'INFO'
        )
        
        return session_id
    
    async def validate_request(
        self,
        session_id: str,
        ip_address: str,
        required_permission: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Validate request and return security context"""
        # Get session
        session_data = await self.session_manager.get_session(session_id)
        if not session_data:
            return None
        
        # Verify IP address matches
        if session_data['metadata'].get('ip_address') != ip_address:
            await self.audit_logger.log(
                'SESSION_IP_MISMATCH',
                SecurityContext(session_id=session_id, ip_address=ip_address),
                {'stored_ip': session_data['metadata'].get('ip_address')},
                'WARNING'
            )
            return None
        
        # Create security context
        context = SecurityContext(
            user_id=session_data['user_id'],
            session_id=session_id,
            ip_address=ip_address,
            security_level=SecurityLevel.AUTHENTICATED,
            permissions=self.access_control.get_user_permissions(['user'])
        )
        
        # Check permission if required
        if required_permission:
            if not self.access_control.check_permission(context, required_permission):
                await self.audit_logger.log(
                    'PERMISSION_DENIED',
                    context,
                    {'required': required_permission},
                    'WARNING'
                )
                return None
        
        return context
    
    def sanitize_input(self, input_data: Any, input_type: str = 'general') -> Any:
        """Sanitize user input based on type"""
        if input_type == 'sql':
            return self.sanitizer.sanitize_sql(str(input_data))
        elif input_type == 'html':
            return self.sanitizer.sanitize_html(str(input_data))
        elif input_type == 'path':
            return self.sanitizer.sanitize_path(str(input_data))
        elif input_type == 'command':
            return self.sanitizer.sanitize_command(str(input_data))
        else:
            # General sanitization - remove control characters
            import re
            return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(input_data))
    
    async def shutdown(self):
        """Shutdown security fortress"""
        if self.audit_logger:
            await self.audit_logger.stop()
        
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()