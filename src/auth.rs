//! Secure Authentication and Authorization System
//! Production-ready with proper password hashing and JWT tokens

use argon2::{
    password_hash::{
        rand_core::OsRng,
        PasswordHash, PasswordHasher, PasswordVerifier, SaltString
    },
    Argon2
};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// Authentication system with secure password handling
pub struct AuthenticationSystem {
    users: Arc<RwLock<HashMap<String, User>>>,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    jwt_secret: String,
    password_policy: PasswordPolicy,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

/// User account structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub roles: HashSet<String>,
    pub permissions: HashSet<String>,
    pub mfa_enabled: bool,
    pub mfa_secret: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
    pub failed_attempts: u32,
    pub locked_until: Option<DateTime<Utc>>,
    pub password_history: Vec<String>,
    pub must_change_password: bool,
}

/// User session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_id: Uuid,
    pub token: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub ip_address: String,
    pub user_agent: String,
}

/// JWT Claims
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub session_id: String,
}

/// Password policy configuration
#[derive(Debug, Clone)]
pub struct PasswordPolicy {
    pub min_length: usize,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_numbers: bool,
    pub require_special: bool,
    pub password_history_size: usize,
    pub max_age_days: u32,
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 12,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special: true,
            password_history_size: 5,
            max_age_days: 90,
        }
    }
}

/// Rate limiter for authentication attempts
pub struct RateLimiter {
    attempts: HashMap<String, Vec<DateTime<Utc>>>,
    max_attempts: u32,
    window_minutes: i64,
}

impl AuthenticationSystem {
    pub fn new(jwt_secret: String) -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            jwt_secret,
            password_policy: PasswordPolicy::default(),
            rate_limiter: Arc::new(RwLock::new(RateLimiter {
                attempts: HashMap::new(),
                max_attempts: 5,
                window_minutes: 15,
            })),
        }
    }
    
    /// Register a new user
    pub fn register_user(
        &self,
        username: String,
        email: String,
        password: String,
        roles: HashSet<String>,
    ) -> Result<Uuid, String> {
        // Validate password against policy
        self.validate_password(&password)?;
        
        // Check if username or email already exists
        {
            let users = self.users.read().unwrap();
            if users.values().any(|u| u.username == username || u.email == email) {
                return Err("Username or email already exists".to_string());
            }
        }
        
        // Hash password using Argon2
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| format!("Failed to hash password: {}", e))?
            .to_string();
        
        // Create user
        let user = User {
            id: Uuid::new_v4(),
            username: username.clone(),
            email,
            password_hash: password_hash.clone(),
            roles,
            permissions: HashSet::new(),
            mfa_enabled: false,
            mfa_secret: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_login: None,
            failed_attempts: 0,
            locked_until: None,
            password_history: vec![password_hash],
            must_change_password: false,
        };
        
        let user_id = user.id;
        
        // Store user
        {
            let mut users = self.users.write().unwrap();
            users.insert(username, user);
        }
        
        Ok(user_id)
    }
    
    /// Authenticate user and create session
    pub fn authenticate(
        &self,
        username: &str,
        password: &str,
        ip_address: String,
        user_agent: String,
    ) -> Result<String, String> {
        // Check rate limiting
        if !self.check_rate_limit(username) {
            return Err("Too many failed attempts. Please try again later.".to_string());
        }
        
        // Get user
        let mut users = self.users.write().unwrap();
        let user = users.get_mut(username)
            .ok_or("Invalid username or password")?;
        
        // Check if account is locked
        if let Some(locked_until) = user.locked_until {
            if Utc::now() < locked_until {
                return Err("Account is locked. Please try again later.".to_string());
            } else {
                user.locked_until = None;
                user.failed_attempts = 0;
            }
        }
        
        // Verify password
        let parsed_hash = PasswordHash::new(&user.password_hash)
            .map_err(|_| "Invalid password hash")?;
        
        let argon2 = Argon2::default();
        if argon2.verify_password(password.as_bytes(), &parsed_hash).is_err() {
            // Increment failed attempts
            user.failed_attempts += 1;
            
            // Lock account after 5 failed attempts
            if user.failed_attempts >= 5 {
                user.locked_until = Some(Utc::now() + Duration::minutes(30));
            }
            
            // Record failed attempt for rate limiting
            self.record_failed_attempt(username);
            
            return Err("Invalid username or password".to_string());
        }
        
        // Reset failed attempts on successful login
        user.failed_attempts = 0;
        user.last_login = Some(Utc::now());
        
        // Create session
        let session_id = Uuid::new_v4().to_string();
        let expires_at = Utc::now() + Duration::hours(8);
        
        // Create JWT token
        let claims = Claims {
            sub: user.id.to_string(),
            exp: expires_at.timestamp(),
            iat: Utc::now().timestamp(),
            roles: user.roles.iter().cloned().collect(),
            permissions: user.permissions.iter().cloned().collect(),
            session_id: session_id.clone(),
        };
        
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_ref()),
        ).map_err(|e| format!("Failed to create token: {}", e))?;
        
        // Store session
        let session = Session {
            id: session_id.clone(),
            user_id: user.id,
            token: token.clone(),
            created_at: Utc::now(),
            expires_at,
            ip_address,
            user_agent,
        };
        
        {
            let mut sessions = self.sessions.write().unwrap();
            sessions.insert(session_id, session);
        }
        
        Ok(token)
    }
    
    /// Validate JWT token
    pub fn validate_token(&self, token: &str) -> Result<Claims, String> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_ref()),
            &Validation::default(),
        ).map_err(|e| format!("Invalid token: {}", e))?;
        
        // Check if session exists and is valid
        {
            let sessions = self.sessions.read().unwrap();
            let session = sessions.get(&token_data.claims.session_id)
                .ok_or("Session not found")?;
            
            if Utc::now() > session.expires_at {
                return Err("Session expired".to_string());
            }
        }
        
        Ok(token_data.claims)
    }
    
    /// Logout and invalidate session
    pub fn logout(&self, token: &str) -> Result<(), String> {
        let claims = self.validate_token(token)?;
        
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(&claims.session_id);
        
        Ok(())
    }
    
    /// Change user password
    pub fn change_password(
        &self,
        username: &str,
        old_password: &str,
        new_password: &str,
    ) -> Result<(), String> {
        // Validate new password
        self.validate_password(new_password)?;
        
        let mut users = self.users.write().unwrap();
        let user = users.get_mut(username)
            .ok_or("User not found")?;
        
        // Verify old password
        let parsed_hash = PasswordHash::new(&user.password_hash)
            .map_err(|_| "Invalid password hash")?;
        
        let argon2 = Argon2::default();
        if argon2.verify_password(old_password.as_bytes(), &parsed_hash).is_err() {
            return Err("Invalid old password".to_string());
        }
        
        // Check password history
        for old_hash in &user.password_history {
            let old_parsed = PasswordHash::new(old_hash)
                .map_err(|_| "Invalid password hash")?;
            if argon2.verify_password(new_password.as_bytes(), &old_parsed).is_ok() {
                return Err("Password has been used recently".to_string());
            }
        }
        
        // Hash new password
        let salt = SaltString::generate(&mut OsRng);
        let new_hash = argon2
            .hash_password(new_password.as_bytes(), &salt)
            .map_err(|e| format!("Failed to hash password: {}", e))?
            .to_string();
        
        // Update password and history
        user.password_hash = new_hash.clone();
        user.password_history.push(new_hash);
        if user.password_history.len() > self.password_policy.password_history_size {
            user.password_history.remove(0);
        }
        user.updated_at = Utc::now();
        user.must_change_password = false;
        
        Ok(())
    }
    
    /// Enable MFA for user
    pub fn enable_mfa(&self, username: &str) -> Result<String, String> {
        use totp_lite::{totp, Sha1};
        
        let mut users = self.users.write().unwrap();
        let user = users.get_mut(username)
            .ok_or("User not found")?;
        
        // Generate secret
        let secret: String = (0..32)
            .map(|_| {
                let n = rand::random::<u8>() % 62;
                if n < 10 { (b'0' + n) as char }
                else if n < 36 { (b'A' + n - 10) as char }
                else { (b'a' + n - 36) as char }
            })
            .collect();
        
        user.mfa_enabled = true;
        user.mfa_secret = Some(secret.clone());
        user.updated_at = Utc::now();
        
        Ok(secret)
    }
    
    /// Verify MFA token
    pub fn verify_mfa(&self, username: &str, token: &str) -> Result<bool, String> {
        use totp_lite::{totp, Sha1};
        
        let users = self.users.read().unwrap();
        let user = users.get(username)
            .ok_or("User not found")?;
        
        if !user.mfa_enabled {
            return Ok(true); // MFA not enabled, always pass
        }
        
        let secret = user.mfa_secret.as_ref()
            .ok_or("MFA secret not found")?;
        
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() / 30;
        
        let expected = totp::<Sha1>(secret.as_bytes(), time);
        
        Ok(token == expected.to_string())
    }
    
    /// Validate password against policy
    fn validate_password(&self, password: &str) -> Result<(), String> {
        if password.len() < self.password_policy.min_length {
            return Err(format!("Password must be at least {} characters long", self.password_policy.min_length));
        }
        
        if self.password_policy.require_uppercase && !password.chars().any(|c| c.is_uppercase()) {
            return Err("Password must contain at least one uppercase letter".to_string());
        }
        
        if self.password_policy.require_lowercase && !password.chars().any(|c| c.is_lowercase()) {
            return Err("Password must contain at least one lowercase letter".to_string());
        }
        
        if self.password_policy.require_numbers && !password.chars().any(|c| c.is_numeric()) {
            return Err("Password must contain at least one number".to_string());
        }
        
        if self.password_policy.require_special && !password.chars().any(|c| !c.is_alphanumeric()) {
            return Err("Password must contain at least one special character".to_string());
        }
        
        Ok(())
    }
    
    /// Check rate limiting
    fn check_rate_limit(&self, identifier: &str) -> bool {
        let mut limiter = self.rate_limiter.write().unwrap();
        let now = Utc::now();
        let window_start = now - Duration::minutes(limiter.window_minutes);
        
        // Clean old attempts
        let attempts = limiter.attempts.entry(identifier.to_string()).or_insert_with(Vec::new);
        attempts.retain(|&t| t > window_start);
        
        attempts.len() < limiter.max_attempts as usize
    }
    
    /// Record failed authentication attempt
    fn record_failed_attempt(&self, identifier: &str) {
        let mut limiter = self.rate_limiter.write().unwrap();
        let attempts = limiter.attempts.entry(identifier.to_string()).or_insert_with(Vec::new);
        attempts.push(Utc::now());
    }
    
    /// Check if user has permission
    pub fn has_permission(&self, username: &str, permission: &str) -> bool {
        let users = self.users.read().unwrap();
        users.get(username)
            .map(|user| user.permissions.contains(permission))
            .unwrap_or(false)
    }
    
    /// Check if user has role
    pub fn has_role(&self, username: &str, role: &str) -> bool {
        let users = self.users.read().unwrap();
        users.get(username)
            .map(|user| user.roles.contains(role))
            .unwrap_or(false)
    }
    
    /// Grant role to user
    pub fn grant_role(&self, username: &str, role: String) -> Result<(), String> {
        let mut users = self.users.write().unwrap();
        let user = users.get_mut(username)
            .ok_or("User not found")?;
        
        user.roles.insert(role);
        user.updated_at = Utc::now();
        
        Ok(())
    }
    
    /// Revoke role from user
    pub fn revoke_role(&self, username: &str, role: &str) -> Result<(), String> {
        let mut users = self.users.write().unwrap();
        let user = users.get_mut(username)
            .ok_or("User not found")?;
        
        user.roles.remove(role);
        user.updated_at = Utc::now();
        
        Ok(())
    }
    
    /// Clean up expired sessions
    pub fn cleanup_sessions(&self) {
        let mut sessions = self.sessions.write().unwrap();
        let now = Utc::now();
        sessions.retain(|_, session| session.expires_at > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_registration() {
        let auth = AuthenticationSystem::new("test_secret".to_string());
        
        let mut roles = HashSet::new();
        roles.insert("user".to_string());
        
        let user_id = auth.register_user(
            "testuser".to_string(),
            "test@example.com".to_string(),
            "SecureP@ssw0rd123!".to_string(),
            roles,
        ).unwrap();
        
        assert!(!user_id.is_nil());
    }
    
    #[test]
    fn test_authentication() {
        let auth = AuthenticationSystem::new("test_secret".to_string());
        
        // Register user
        let mut roles = HashSet::new();
        roles.insert("user".to_string());
        
        auth.register_user(
            "testuser".to_string(),
            "test@example.com".to_string(),
            "SecureP@ssw0rd123!".to_string(),
            roles,
        ).unwrap();
        
        // Authenticate
        let token = auth.authenticate(
            "testuser",
            "SecureP@ssw0rd123!",
            "127.0.0.1".to_string(),
            "TestAgent".to_string(),
        ).unwrap();
        
        assert!(!token.is_empty());
        
        // Validate token
        let claims = auth.validate_token(&token).unwrap();
        assert!(claims.roles.contains(&"user".to_string()));
    }
    
    #[test]
    fn test_password_validation() {
        let auth = AuthenticationSystem::new("test_secret".to_string());
        
        // Test weak passwords
        assert!(auth.validate_password("short").is_err());
        assert!(auth.validate_password("nouppercase123!").is_err());
        assert!(auth.validate_password("NOLOWERCASE123!").is_err());
        assert!(auth.validate_password("NoNumbers!").is_err());
        assert!(auth.validate_password("NoSpecialChar123").is_err());
        
        // Test strong password
        assert!(auth.validate_password("SecureP@ssw0rd123!").is_ok());
    }
    
    #[test]
    fn test_rate_limiting() {
        let auth = AuthenticationSystem::new("test_secret".to_string());
        
        // Register user
        let mut roles = HashSet::new();
        roles.insert("user".to_string());
        
        auth.register_user(
            "testuser".to_string(),
            "test@example.com".to_string(),
            "SecureP@ssw0rd123!".to_string(),
            roles,
        ).unwrap();
        
        // Attempt authentication with wrong password multiple times
        for _ in 0..5 {
            let _ = auth.authenticate(
                "testuser",
                "WrongPassword",
                "127.0.0.1".to_string(),
                "TestAgent".to_string(),
            );
        }
        
        // Next attempt should be rate limited
        let result = auth.authenticate(
            "testuser",
            "SecureP@ssw0rd123!",
            "127.0.0.1".to_string(),
            "TestAgent".to_string(),
        );
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Too many failed attempts"));
    }
    
    #[test]
    fn test_password_change() {
        let auth = AuthenticationSystem::new("test_secret".to_string());
        
        // Register user
        let mut roles = HashSet::new();
        roles.insert("user".to_string());
        
        auth.register_user(
            "testuser".to_string(),
            "test@example.com".to_string(),
            "OldP@ssw0rd123!".to_string(),
            roles,
        ).unwrap();
        
        // Change password
        auth.change_password(
            "testuser",
            "OldP@ssw0rd123!",
            "NewP@ssw0rd456!",
        ).unwrap();
        
        // Old password should fail
        let result = auth.authenticate(
            "testuser",
            "OldP@ssw0rd123!",
            "127.0.0.1".to_string(),
            "TestAgent".to_string(),
        );
        assert!(result.is_err());
        
        // New password should work
        let token = auth.authenticate(
            "testuser",
            "NewP@ssw0rd456!",
            "127.0.0.1".to_string(),
            "TestAgent".to_string(),
        ).unwrap();
        assert!(!token.is_empty());
    }
}