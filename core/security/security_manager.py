#!/usr/bin/env python3
"""
QENEX OS Security Manager - System security and threat detection
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

@dataclass
class Threat:
    """Represents a security threat"""
    threat_id: str
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    detected_at: float
    source: str
    mitigated: bool = False

class SecurityManager:
    """Manages system security"""
    
    def __init__(self):
        self.encryption_key = None
        self.threats: List[Threat] = []
        self.blocked_ips: Set[str] = set()
        self.firewall_rules: List[Dict] = []
        self.audit_log: List[Dict] = []
        self.initialized = False
        self.threat_level = "low"
        
    async def initialize(self):
        """Initialize security manager"""
        print("🔒 Initializing Security Manager...")
        
        # Generate encryption key
        self.encryption_key = Fernet.generate_key()
        
        # Load firewall rules
        self._load_firewall_rules()
        
        # Start security monitors
        asyncio.create_task(self.threat_monitor())
        asyncio.create_task(self.intrusion_detection())
        
        self.initialized = True
        print("✅ Security Manager initialized")
    
    async def shutdown(self):
        """Shutdown security manager"""
        self.initialized = False
        print("🔒 Security Manager shutdown")
    
    def _load_firewall_rules(self):
        """Load default firewall rules"""
        self.firewall_rules = [
            {"action": "allow", "protocol": "tcp", "port": 22, "source": "any"},  # SSH
            {"action": "allow", "protocol": "tcp", "port": 80, "source": "any"},  # HTTP
            {"action": "allow", "protocol": "tcp", "port": 443, "source": "any"}, # HTTPS
            {"action": "deny", "protocol": "any", "port": "any", "source": "any"}  # Default deny
        ]
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using Fernet"""
        if not self.encryption_key:
            raise Exception("Encryption key not initialized")
        
        f = Fernet(self.encryption_key)
        return f.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet"""
        if not self.encryption_key:
            raise Exception("Encryption key not initialized")
        
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data)
    
    def hash_password(self, password: str, salt: bytes = None):
        """Hash password using PBKDF2"""
        if salt is None:
            salt = os.urandom(32)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode())
        return key, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify password hash"""
        new_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_hex(length)
    
    async def scan_system(self) -> List[Threat]:
        """Scan system for threats"""
        detected_threats = []
        
        # Check for suspicious processes
        import psutil
        suspicious_processes = ["keylogger", "backdoor", "trojan", "malware"]
        
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name'].lower()
                for suspicious in suspicious_processes:
                    if suspicious in name:
                        threat = Threat(
                            threat_id=self.generate_token(8),
                            threat_type="suspicious_process",
                            severity="high",
                            description=f"Suspicious process detected: {proc.info['name']}",
                            detected_at=time.time(),
                            source=f"PID: {proc.pid}"
                        )
                        detected_threats.append(threat)
                        self.threats.append(threat)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Check for open ports
        for conn in psutil.net_connections():
            if conn.status == 'LISTEN' and conn.laddr.port not in [22, 80, 443, 8080]:
                threat = Threat(
                    threat_id=self.generate_token(8),
                    threat_type="suspicious_port",
                    severity="medium",
                    description=f"Unusual port open: {conn.laddr.port}",
                    detected_at=time.time(),
                    source=f"Port: {conn.laddr.port}"
                )
                detected_threats.append(threat)
                self.threats.append(threat)
        
        # Update threat level
        self._update_threat_level()
        
        return detected_threats
    
    def check_firewall(self, protocol: str, port: int, source_ip: str) -> bool:
        """Check if connection is allowed by firewall"""
        for rule in self.firewall_rules:
            if (rule["protocol"] in [protocol, "any"] and
                (rule["port"] == port or rule["port"] == "any") and
                (rule["source"] == source_ip or rule["source"] == "any")):
                
                allowed = rule["action"] == "allow"
                
                # Log access attempt
                self.audit_log.append({
                    "timestamp": time.time(),
                    "protocol": protocol,
                    "port": port,
                    "source_ip": source_ip,
                    "action": "allowed" if allowed else "blocked"
                })
                
                return allowed
        
        return False
    
    def block_ip(self, ip: str, reason: str = ""):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        self.audit_log.append({
            "timestamp": time.time(),
            "action": "block_ip",
            "ip": ip,
            "reason": reason
        })
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            self.audit_log.append({
                "timestamp": time.time(),
                "action": "unblock_ip",
                "ip": ip
            })
    
    async def threat_monitor(self):
        """Monitor for threats continuously"""
        while self.initialized:
            await asyncio.sleep(30)
            
            # Run system scan
            threats = await self.scan_system()
            
            if threats:
                print(f"⚠️ {len(threats)} threats detected!")
                
                # Auto-mitigate high severity threats
                for threat in threats:
                    if threat.severity in ["high", "critical"] and not threat.mitigated:
                        await self.mitigate_threat(threat)
    
    async def intrusion_detection(self):
        """Detect intrusion attempts"""
        failed_attempts: Dict[str, int] = {}
        
        while self.initialized:
            await asyncio.sleep(5)
            
            # Check recent audit logs for failed attempts
            recent_logs = [log for log in self.audit_log 
                          if time.time() - log["timestamp"] < 60]
            
            for log in recent_logs:
                if log.get("action") == "blocked":
                    ip = log.get("source_ip", "unknown")
                    failed_attempts[ip] = failed_attempts.get(ip, 0) + 1
                    
                    # Block IP after 5 failed attempts
                    if failed_attempts[ip] >= 5:
                        self.block_ip(ip, "Multiple failed access attempts")
                        failed_attempts[ip] = 0
    
    async def mitigate_threat(self, threat: Threat):
        """Mitigate a detected threat"""
        print(f"🛡️ Mitigating threat: {threat.description}")
        
        if threat.threat_type == "suspicious_process":
            # Kill suspicious process
            try:
                pid = int(threat.source.split(": ")[1])
                os.kill(pid, 9)
                threat.mitigated = True
                print(f"✅ Killed suspicious process: PID {pid}")
            except Exception as e:
                print(f"❌ Failed to kill process: {e}")
        
        elif threat.threat_type == "suspicious_port":
            # Add firewall rule to block port
            port = int(threat.source.split(": ")[1])
            self.firewall_rules.insert(0, {
                "action": "deny",
                "protocol": "any",
                "port": port,
                "source": "any"
            })
            threat.mitigated = True
            print(f"✅ Blocked port: {port}")
    
    def _update_threat_level(self):
        """Update overall threat level based on recent threats"""
        recent_threats = [t for t in self.threats 
                         if time.time() - t.detected_at < 3600]  # Last hour
        
        critical_count = sum(1 for t in recent_threats if t.severity == "critical")
        high_count = sum(1 for t in recent_threats if t.severity == "high")
        
        if critical_count > 0:
            self.threat_level = "critical"
        elif high_count > 2:
            self.threat_level = "high"
        elif high_count > 0 or len(recent_threats) > 5:
            self.threat_level = "medium"
        else:
            self.threat_level = "low"
    
    async def update_definitions(self):
        """Update security definitions"""
        print("📥 Updating security definitions...")
        
        # Simulate downloading new threat definitions
        await asyncio.sleep(2)
        
        # Add new threat patterns
        self.audit_log.append({
            "timestamp": time.time(),
            "action": "update_definitions",
            "status": "success"
        })
        
        print("✅ Security definitions updated")
    
    def get_status(self) -> Dict:
        """Get security status"""
        return {
            "status": "active" if self.initialized else "inactive",
            "threat_level": self.threat_level,
            "threats_detected": len(self.threats),
            "active_threats": sum(1 for t in self.threats if not t.mitigated),
            "blocked_ips": len(self.blocked_ips),
            "firewall_rules": len(self.firewall_rules),
            "audit_entries": len(self.audit_log)
        }

# Singleton instance
security_manager = SecurityManager()

async def main():
    """Main function for testing"""
    await security_manager.initialize()
    
    # Test encryption
    data = b"Secret data"
    encrypted = security_manager.encrypt_data(data)
    decrypted = security_manager.decrypt_data(encrypted)
    print(f"Encryption test: {decrypted == data}")
    
    # Test password hashing
    password = "SecurePassword123!"
    hashed, salt = security_manager.hash_password(password)
    verified = security_manager.verify_password(password, hashed, salt)
    print(f"Password test: {verified}")
    
    # Test firewall
    allowed = security_manager.check_firewall("tcp", 80, "192.168.1.1")
    print(f"Firewall test (port 80): {allowed}")
    
    blocked = security_manager.check_firewall("tcp", 1337, "192.168.1.1")
    print(f"Firewall test (port 1337): {blocked}")
    
    # Run system scan
    threats = await security_manager.scan_system()
    print(f"Threats detected: {len(threats)}")
    
    # Get status
    status = security_manager.get_status()
    print(f"Security status: {json.dumps(status, indent=2)}")
    
    await security_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())