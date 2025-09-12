#!/usr/bin/env python3
"""
Comprehensive test suite for production authentication system
"""

import unittest
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production_system.secure_auth_system import (
    SecureAuthSystem, UserRole, User, require_auth
)


class TestSecureAuthSystem(unittest.TestCase):
    """Test cases for SecureAuthSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.auth_system = SecureAuthSystem(
            db_path=self.temp_db.name,
            secret_key="test_secret_key_123"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_db.name)
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "SecurePassword123!"
        hashed = self.auth_system._hash_password(password)
        
        # Hash should be different from password
        self.assertNotEqual(password, hashed)
        
        # Should verify correctly
        self.assertTrue(self.auth_system._verify_password(password, hashed))
        
        # Wrong password should fail
        self.assertFalse(self.auth_system._verify_password("WrongPassword", hashed))
    
    def test_password_validation(self):
        """Test password strength validation"""
        weak_passwords = [
            "short",  # Too short
            "nouppercase123",  # No uppercase
            "NOLOWERCASE123",  # No lowercase
            "NoDigitsHere",  # No digits
        ]
        
        for weak_pass in weak_passwords:
            with self.assertRaises(ValueError):
                self.auth_system._hash_password(weak_pass)
    
    def test_user_registration(self):
        """Test user registration flow"""
        success, message, token = self.auth_system.register_user(
            username="testuser",
            email="ceo@qenex.ai",
            password="SecurePass123!",
            role=UserRole.USER
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(token)
        
        # Duplicate registration should fail
        success, message, token = self.auth_system.register_user(
            username="testuser",
            email="ceo@qenex.ai",
            password="SecurePass123!",
            role=UserRole.USER
        )
        
        self.assertFalse(success)
        self.assertIsNone(token)
    
    def test_email_validation(self):
        """Test email format validation"""
        invalid_emails = [
            "notanemail",
            "missing@domain",
            "@nodomain.com",
            "spaces in@email.com"
        ]
        
        for email in invalid_emails:
            success, message, token = self.auth_system.register_user(
                username="user",
                email=email,
                password="SecurePass123!",
                role=UserRole.USER
            )
            self.assertFalse(success)
    
    def test_authentication_success(self):
        """Test successful authentication"""
        # Register user
        self.auth_system.register_user(
            username="authuser",
            email="ceo@qenex.ai",
            password="AuthPass123!",
            role=UserRole.USER
        )
        
        # Authenticate
        success, message, tokens = self.auth_system.authenticate(
            username="authuser",
            password="AuthPass123!",
            ip_address="127.0.0.1"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(tokens)
        self.assertIn('access_token', tokens)
        self.assertIn('refresh_token', tokens)
        self.assertIn('session_id', tokens)
    
    def test_authentication_failure(self):
        """Test authentication with wrong credentials"""
        # Register user
        self.auth_system.register_user(
            username="authuser2",
            email="ceo@qenex.ai",
            password="AuthPass123!",
            role=UserRole.USER
        )
        
        # Wrong password
        success, message, tokens = self.auth_system.authenticate(
            username="authuser2",
            password="WrongPassword",
            ip_address="127.0.0.1"
        )
        
        self.assertFalse(success)
        self.assertIsNone(tokens)
        
        # Non-existent user
        success, message, tokens = self.auth_system.authenticate(
            username="nonexistent",
            password="AnyPassword123!",
            ip_address="127.0.0.1"
        )
        
        self.assertFalse(success)
        self.assertIsNone(tokens)
    
    def test_account_lockout(self):
        """Test account lockout after failed attempts"""
        # Register user
        self.auth_system.register_user(
            username="lockuser",
            email="ceo@qenex.ai",
            password="LockPass123!",
            role=UserRole.USER
        )
        
        # Make 5 failed attempts
        for i in range(5):
            success, message, tokens = self.auth_system.authenticate(
                username="lockuser",
                password="WrongPassword",
                ip_address="127.0.0.1"
            )
            self.assertFalse(success)
        
        # Account should be locked
        success, message, tokens = self.auth_system.authenticate(
            username="lockuser",
            password="LockPass123!",  # Even correct password
            ip_address="127.0.0.1"
        )
        
        self.assertFalse(success)
        self.assertIn("locked", message.lower())
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple requests quickly
        for i in range(5):
            result = self.auth_system._check_rate_limit("test_client")
            self.assertTrue(result)
        
        # 6th request should be blocked
        result = self.auth_system._check_rate_limit("test_client")
        self.assertFalse(result)
    
    def test_token_generation_and_verification(self):
        """Test JWT token generation and verification"""
        user_id = "test_user_123"
        
        # Generate access token
        access_token = self.auth_system._generate_token(user_id, 'access')
        self.assertIsNotNone(access_token)
        
        # Verify token
        payload = self.auth_system._verify_token(access_token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload['user_id'], user_id)
        self.assertEqual(payload['type'], 'access')
    
    def test_token_revocation(self):
        """Test token revocation"""
        user_id = "revoke_user"
        token = self.auth_system._generate_token(user_id, 'access')
        
        # Token should be valid initially
        payload = self.auth_system._verify_token(token)
        self.assertIsNotNone(payload)
        
        # Revoke token
        self.auth_system.revoke_token(token)
        
        # Token should now be invalid
        payload = self.auth_system._verify_token(token)
        self.assertIsNone(payload)
    
    def test_api_key_management(self):
        """Test API key creation and verification"""
        user_id = "api_user"
        
        # Create API key
        api_key = self.auth_system.create_api_key(
            user_id,
            name="Test API Key",
            expires_in_days=30
        )
        
        self.assertIsNotNone(api_key)
        
        # Verify API key
        valid, verified_user_id = self.auth_system.verify_api_key(api_key)
        self.assertTrue(valid)
        self.assertEqual(verified_user_id, user_id)
        
        # Invalid API key
        valid, verified_user_id = self.auth_system.verify_api_key("invalid_key")
        self.assertFalse(valid)
        self.assertIsNone(verified_user_id)
    
    def test_role_based_access_control(self):
        """Test RBAC functionality"""
        # Register users with different roles
        roles = [UserRole.ADMIN, UserRole.OPERATOR, UserRole.USER, UserRole.READONLY]
        tokens = {}
        
        for i, role in enumerate(roles):
            success, message, token = self.auth_system.register_user(
                username=f"user_{role.value}",
                email=f"{role.value}@example.com",
                password="SecurePass123!",
                role=role
            )
            tokens[role] = token
        
        # Test access control
        # Admin should access everything
        valid, user_id = self.auth_system.verify_access(
            tokens[UserRole.ADMIN],
            UserRole.ADMIN
        )
        self.assertTrue(valid)
        
        # User should not access admin resources
        valid, user_id = self.auth_system.verify_access(
            tokens[UserRole.USER],
            UserRole.ADMIN
        )
        self.assertFalse(valid)
        
        # Admin can access user resources
        valid, user_id = self.auth_system.verify_access(
            tokens[UserRole.ADMIN],
            UserRole.USER
        )
        self.assertTrue(valid)


class TestAuthDecorator(unittest.TestCase):
    """Test authentication decorator"""
    
    def test_require_auth_decorator(self):
        """Test require_auth decorator"""
        
        @require_auth(UserRole.USER)
        def protected_endpoint(**kwargs):
            return {'message': 'Success', 'user_id': kwargs.get('user_id')}
        
        # Without token
        result = protected_endpoint()
        self.assertEqual(result[1], 401)
        
        # With invalid token
        result = protected_endpoint(auth_token='invalid_token')
        self.assertEqual(result[1], 403)
        
        # With valid token (mocked)
        with patch('production_system.secure_auth_system.SecureAuthSystem.verify_access') as mock_verify:
            mock_verify.return_value = (True, 'test_user_id')
            result = protected_endpoint(auth_token='valid_token')
            self.assertEqual(result['message'], 'Success')
            self.assertEqual(result['user_id'], 'test_user_id')


class TestSecurityFeatures(unittest.TestCase):
    """Test security-specific features"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.auth_system = SecureAuthSystem(
            db_path=self.temp_db.name,
            secret_key="test_secret_key_456"
        )
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        # Attempt SQL injection in username
        malicious_username = "admin' OR '1'='1"
        
        success, message, tokens = self.auth_system.authenticate(
            username=malicious_username,
            password="AnyPassword",
            ip_address="127.0.0.1"
        )
        
        # Should fail safely without SQL error
        self.assertFalse(success)
        self.assertIsNone(tokens)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks"""
        # Register a user
        self.auth_system.register_user(
            username="timing_user",
            email="ceo@qenex.ai",
            password="TimingPass123!",
            role=UserRole.USER
        )
        
        # Time authentication with correct username
        start = time.time()
        self.auth_system.authenticate(
            username="timing_user",
            password="WrongPassword",
            ip_address="127.0.0.1"
        )
        time_existing = time.time() - start
        
        # Time authentication with non-existent username
        start = time.time()
        self.auth_system.authenticate(
            username="nonexistent_user",
            password="WrongPassword",
            ip_address="127.0.0.1"
        )
        time_nonexistent = time.time() - start
        
        # Times should be similar (within reasonable variance)
        # This prevents username enumeration
        time_diff = abs(time_existing - time_nonexistent)
        self.assertLess(time_diff, 0.1)  # Less than 100ms difference
    
    def test_audit_logging(self):
        """Test audit log functionality"""
        # Perform actions that should be logged
        self.auth_system.register_user(
            username="audit_user",
            email="ceo@qenex.ai",
            password="AuditPass123!",
            role=UserRole.USER
        )
        
        self.auth_system.authenticate(
            username="audit_user",
            password="AuditPass123!",
            ip_address="192.168.1.1"
        )
        
        self.auth_system.authenticate(
            username="audit_user",
            password="WrongPassword",
            ip_address="192.168.1.1"
        )
        
        # Check audit log entries exist
        cursor = self.auth_system.db.execute(
            'SELECT action FROM audit_log ORDER BY timestamp'
        )
        actions = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('USER_REGISTERED', actions)
        self.assertIn('AUTH_SUCCESS', actions)
        self.assertIn('AUTH_FAILED', actions)


if __name__ == '__main__':
    unittest.main()