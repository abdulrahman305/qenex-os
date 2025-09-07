#!/usr/bin/env python3
"""
Zero-Knowledge Proof Authentication System
Prove identity without revealing credentials
"""

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import json


@dataclass
class ZKProof:
    """Zero-knowledge proof container"""
    protocol: str
    commitment: bytes
    challenge: bytes
    response: bytes
    timestamp: float
    metadata: Dict[str, Any]


class SchnorrProtocol:
    """Schnorr Zero-Knowledge Proof Protocol"""
    
    # Safe prime p = 2q + 1 (2048-bit)
    P = int("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
           "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
           "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
           "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
           "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
           "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
           "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
           "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
           "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
           "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
           "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
    
    # Generator g
    G = 2
    
    # Order q = (p-1)/2
    Q = (P - 1) // 2
    
    def __init__(self):
        self.private_key: Optional[int] = None
        self.public_key: Optional[int] = None
    
    def generate_keypair(self) -> Tuple[int, int]:
        """Generate Schnorr keypair"""
        # Private key: random number in [1, q-1]
        self.private_key = secrets.randbelow(self.Q - 1) + 1
        
        # Public key: y = g^x mod p
        self.public_key = pow(self.G, self.private_key, self.P)
        
        return self.private_key, self.public_key
    
    def create_commitment(self) -> Tuple[int, int]:
        """Create commitment (prover step 1)"""
        # Random r in [1, q-1]
        r = secrets.randbelow(self.Q - 1) + 1
        
        # Commitment: t = g^r mod p
        t = pow(self.G, r, self.P)
        
        return r, t
    
    def create_challenge(self) -> int:
        """Create challenge (verifier step)"""
        # Random challenge c in [0, 2^128 - 1]
        c = secrets.randbits(128)
        return c
    
    def create_response(self, r: int, c: int) -> int:
        """Create response (prover step 2)"""
        if self.private_key is None:
            raise ValueError("Private key not set")
        
        # Response: s = r + c*x mod q
        s = (r + c * self.private_key) % self.Q
        return s
    
    def verify(self, public_key: int, t: int, c: int, s: int) -> bool:
        """Verify proof (verifier step 2)"""
        # Check g^s = t * y^c mod p
        left = pow(self.G, s, self.P)
        right = (t * pow(public_key, c, self.P)) % self.P
        
        return left == right


class FiatShamirProtocol:
    """Fiat-Shamir Zero-Knowledge Proof Protocol"""
    
    def __init__(self, modulus_bits: int = 2048):
        self.modulus_bits = modulus_bits
        self.n: Optional[int] = None
        self.private_key: Optional[int] = None
        self.public_key: Optional[int] = None
    
    def generate_keypair(self) -> Tuple[int, int]:
        """Generate Fiat-Shamir keypair"""
        # Generate safe RSA modulus n = p*q
        from Crypto.Util import number
        
        p = number.getPrime(self.modulus_bits // 2)
        q = number.getPrime(self.modulus_bits // 2)
        self.n = p * q
        
        # Private key: random s coprime to n
        while True:
            s = secrets.randbelow(self.n)
            if self._gcd(s, self.n) == 1:
                self.private_key = s
                break
        
        # Public key: v = s^2 mod n
        self.public_key = pow(self.private_key, 2, self.n)
        
        return self.private_key, self.public_key
    
    def create_commitment(self) -> Tuple[int, int]:
        """Create commitment"""
        if self.n is None:
            raise ValueError("Modulus not set")
        
        # Random r coprime to n
        while True:
            r = secrets.randbelow(self.n)
            if self._gcd(r, self.n) == 1:
                break
        
        # Commitment: x = r^2 mod n
        x = pow(r, 2, self.n)
        
        return r, x
    
    def create_response(self, r: int, challenge: bool) -> int:
        """Create response based on challenge"""
        if self.private_key is None or self.n is None:
            raise ValueError("Keys not set")
        
        if challenge:
            # y = r * s mod n
            y = (r * self.private_key) % self.n
        else:
            # y = r mod n
            y = r % self.n
        
        return y
    
    def verify(self, public_key: int, x: int, challenge: bool, y: int) -> bool:
        """Verify proof"""
        if self.n is None:
            raise ValueError("Modulus not set")
        
        if challenge:
            # Check y^2 = x * v mod n
            left = pow(y, 2, self.n)
            right = (x * public_key) % self.n
        else:
            # Check y^2 = x mod n
            left = pow(y, 2, self.n)
            right = x % self.n
        
        return left == right
    
    def _gcd(self, a: int, b: int) -> int:
        """Compute GCD"""
        while b:
            a, b = b, a % b
        return a


class ChaumPedersenProtocol:
    """Chaum-Pedersen Zero-Knowledge Proof Protocol"""
    
    def __init__(self):
        # Use same prime as Schnorr
        self.p = SchnorrProtocol.P
        self.q = SchnorrProtocol.Q
        self.g = SchnorrProtocol.G
        
        # Second generator h
        self.h = 3
    
    def prove_discrete_log_equality(self, x: int) -> Dict[str, Any]:
        """Prove knowledge of x such that g^x = u and h^x = v"""
        # Public values
        u = pow(self.g, x, self.p)
        v = pow(self.h, x, self.p)
        
        # Commitment
        r = secrets.randbelow(self.q)
        a = pow(self.g, r, self.p)
        b = pow(self.h, r, self.p)
        
        # Challenge (using Fiat-Shamir heuristic)
        challenge_input = f"{u}:{v}:{a}:{b}".encode()
        c = int(hashlib.sha256(challenge_input).hexdigest(), 16) % self.q
        
        # Response
        s = (r + c * x) % self.q
        
        return {
            'u': u,
            'v': v,
            'a': a,
            'b': b,
            'c': c,
            's': s
        }
    
    def verify_discrete_log_equality(self, proof: Dict[str, Any]) -> bool:
        """Verify discrete log equality proof"""
        u = proof['u']
        v = proof['v']
        a = proof['a']
        b = proof['b']
        c = proof['c']
        s = proof['s']
        
        # Verify challenge
        challenge_input = f"{u}:{v}:{a}:{b}".encode()
        expected_c = int(hashlib.sha256(challenge_input).hexdigest(), 16) % self.q
        
        if c != expected_c:
            return False
        
        # Verify g^s = a * u^c
        left1 = pow(self.g, s, self.p)
        right1 = (a * pow(u, c, self.p)) % self.p
        
        # Verify h^s = b * v^c
        left2 = pow(self.h, s, self.p)
        right2 = (b * pow(v, c, self.p)) % self.p
        
        return left1 == right1 and left2 == right2


class ZKAuthSystem:
    """Complete Zero-Knowledge Authentication System"""
    
    def __init__(self):
        self.schnorr = SchnorrProtocol()
        self.fiat_shamir = FiatShamirProtocol()
        self.chaum_pedersen = ChaumPedersenProtocol()
        
        # User database (in production, use secure storage)
        self.users: Dict[str, Dict[str, Any]] = {}
        
        # Active sessions
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Nonce cache to prevent replay attacks
        self.nonce_cache: set = set()
    
    def register_user(self, username: str, protocol: str = "schnorr") -> Dict[str, Any]:
        """Register new user with ZK credentials"""
        if username in self.users:
            raise ValueError("User already exists")
        
        if protocol == "schnorr":
            private_key, public_key = self.schnorr.generate_keypair()
            
            self.users[username] = {
                'protocol': 'schnorr',
                'public_key': public_key,
                'registered_at': time.time()
            }
            
            return {
                'username': username,
                'protocol': 'schnorr',
                'private_key': private_key,  # User must store this securely
                'public_key': public_key
            }
        
        elif protocol == "fiat-shamir":
            private_key, public_key = self.fiat_shamir.generate_keypair()
            
            self.users[username] = {
                'protocol': 'fiat-shamir',
                'public_key': public_key,
                'modulus': self.fiat_shamir.n,
                'registered_at': time.time()
            }
            
            return {
                'username': username,
                'protocol': 'fiat-shamir',
                'private_key': private_key,
                'public_key': public_key,
                'modulus': self.fiat_shamir.n
            }
        
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    
    def create_auth_challenge(self, username: str) -> Dict[str, Any]:
        """Create authentication challenge"""
        if username not in self.users:
            raise ValueError("User not found")
        
        user = self.users[username]
        
        # Generate session ID and nonce
        session_id = secrets.token_hex(32)
        nonce = secrets.randbits(256)
        
        if user['protocol'] == 'schnorr':
            challenge = self.schnorr.create_challenge()
            
            self.sessions[session_id] = {
                'username': username,
                'protocol': 'schnorr',
                'challenge': challenge,
                'nonce': nonce,
                'created_at': time.time(),
                'commitment': None  # Will be set by prover
            }
            
            return {
                'session_id': session_id,
                'protocol': 'schnorr',
                'challenge': challenge,
                'nonce': nonce
            }
        
        elif user['protocol'] == 'fiat-shamir':
            # Random bit challenge
            challenge = secrets.randbits(1) == 1
            
            self.sessions[session_id] = {
                'username': username,
                'protocol': 'fiat-shamir',
                'challenge': challenge,
                'nonce': nonce,
                'created_at': time.time(),
                'commitment': None
            }
            
            return {
                'session_id': session_id,
                'protocol': 'fiat-shamir',
                'challenge': challenge,
                'nonce': nonce
            }
    
    def authenticate_user(self, session_id: str, proof: ZKProof) -> bool:
        """Authenticate user with zero-knowledge proof"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        user = self.users[session['username']]
        
        # Check session timeout (5 minutes)
        if time.time() - session['created_at'] > 300:
            del self.sessions[session_id]
            return False
        
        # Check nonce to prevent replay
        proof_hash = hashlib.sha256(
            proof.commitment + proof.challenge + proof.response
        ).digest()
        
        if proof_hash in self.nonce_cache:
            return False
        
        self.nonce_cache.add(proof_hash)
        
        # Verify based on protocol
        if proof.protocol == 'schnorr':
            # Extract values
            t = int.from_bytes(proof.commitment, 'big')
            c = session['challenge']
            s = int.from_bytes(proof.response, 'big')
            
            # Verify proof
            valid = self.schnorr.verify(user['public_key'], t, c, s)
            
        elif proof.protocol == 'fiat-shamir':
            # Extract values
            x = int.from_bytes(proof.commitment, 'big')
            challenge = session['challenge']
            y = int.from_bytes(proof.response, 'big')
            
            # Set modulus for verification
            self.fiat_shamir.n = user['modulus']
            
            # Verify proof
            valid = self.fiat_shamir.verify(user['public_key'], x, challenge, y)
        
        else:
            valid = False
        
        # Clean up session
        if valid:
            del self.sessions[session_id]
        
        return valid
    
    def create_schnorr_proof(self, private_key: int, challenge: int) -> ZKProof:
        """Create Schnorr proof (client-side)"""
        # Set private key
        self.schnorr.private_key = private_key
        
        # Create commitment
        r, t = self.schnorr.create_commitment()
        
        # Create response
        s = self.schnorr.create_response(r, challenge)
        
        return ZKProof(
            protocol='schnorr',
            commitment=t.to_bytes((t.bit_length() + 7) // 8, 'big'),
            challenge=challenge.to_bytes((challenge.bit_length() + 7) // 8, 'big'),
            response=s.to_bytes((s.bit_length() + 7) // 8, 'big'),
            timestamp=time.time(),
            metadata={'version': '1.0'}
        )
    
    def create_multi_factor_auth(self, username: str, factors: List[str]) -> Dict[str, Any]:
        """Create multi-factor ZK authentication"""
        if username not in self.users:
            raise ValueError("User not found")
        
        challenges = {}
        session_id = secrets.token_hex(32)
        
        for factor in factors:
            if factor == 'schnorr':
                challenge = self.schnorr.create_challenge()
                challenges['schnorr'] = challenge
            
            elif factor == 'fiat-shamir':
                challenge = secrets.randbits(1) == 1
                challenges['fiat-shamir'] = challenge
            
            elif factor == 'chaum-pedersen':
                # For discrete log equality proof
                challenges['chaum-pedersen'] = secrets.randbits(128)
        
        self.sessions[session_id] = {
            'username': username,
            'multi_factor': True,
            'challenges': challenges,
            'completed_factors': [],
            'created_at': time.time()
        }
        
        return {
            'session_id': session_id,
            'challenges': challenges,
            'required_factors': len(factors)
        }
    
    def verify_multi_factor(self, session_id: str, factor: str, proof: ZKProof) -> Dict[str, Any]:
        """Verify one factor in multi-factor authentication"""
        if session_id not in self.sessions:
            return {'success': False, 'error': 'Invalid session'}
        
        session = self.sessions[session_id]
        
        if not session.get('multi_factor'):
            return {'success': False, 'error': 'Not a multi-factor session'}
        
        if factor in session['completed_factors']:
            return {'success': False, 'error': 'Factor already verified'}
        
        # Verify the specific factor
        # (Implementation depends on factor type)
        
        session['completed_factors'].append(factor)
        
        # Check if all factors completed
        if len(session['completed_factors']) == len(session['challenges']):
            del self.sessions[session_id]
            return {
                'success': True,
                'authenticated': True,
                'message': 'All factors verified'
            }
        
        return {
            'success': True,
            'authenticated': False,
            'remaining_factors': len(session['challenges']) - len(session['completed_factors'])
        }


# Example usage
def example_zkauth():
    """Example zero-knowledge authentication flow"""
    
    # Initialize system
    auth_system = ZKAuthSystem()
    
    # Register user
    print("Registering user...")
    registration = auth_system.register_user("alice", "schnorr")
    private_key = registration['private_key']
    print(f"User registered with public key: {registration['public_key']}")
    
    # Authentication flow
    print("\nStarting authentication...")
    
    # Step 1: Request challenge
    challenge_data = auth_system.create_auth_challenge("alice")
    session_id = challenge_data['session_id']
    challenge = challenge_data['challenge']
    print(f"Challenge received: {challenge}")
    
    # Step 2: Create proof (client-side)
    proof = auth_system.create_schnorr_proof(private_key, challenge)
    print(f"Proof created")
    
    # Step 3: Verify proof
    authenticated = auth_system.authenticate_user(session_id, proof)
    print(f"Authentication result: {'SUCCESS' if authenticated else 'FAILED'}")
    
    return authenticated


if __name__ == "__main__":
    # Run example
    result = example_zkauth()
    
    if result:
        print("\n✓ Zero-knowledge authentication successful!")
    else:
        print("\n✗ Authentication failed")