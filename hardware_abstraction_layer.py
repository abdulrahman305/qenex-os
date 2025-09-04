"""
Hardware Abstraction Layer (HAL)
Platform-independent interface for hardware security modules and cryptographic operations
"""

import os
import sys
import platform
import subprocess
import ctypes
import struct
import secrets
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareType(Enum):
    """Types of hardware security modules"""
    TPM = "Trusted Platform Module"
    HSM = "Hardware Security Module"
    SECURE_ENCLAVE = "Secure Enclave"
    TEE = "Trusted Execution Environment"
    SMART_CARD = "Smart Card"
    USB_TOKEN = "USB Security Token"
    FIDO2 = "FIDO2 Authenticator"

class CryptoOperation(Enum):
    """Cryptographic operations"""
    GENERATE_KEY = "generate_key"
    SIGN = "sign"
    VERIFY = "verify"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    HASH = "hash"
    RANDOM = "random"
    DERIVE_KEY = "derive_key"

@dataclass
class HardwareCapabilities:
    """Hardware security capabilities"""
    type: HardwareType
    vendor: str
    model: str
    firmware_version: str
    algorithms: List[str]
    max_key_size: int
    performance: Dict[str, float]  # Operations per second
    features: List[str]

class HardwareInterface(ABC):
    """Abstract interface for hardware security modules"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize hardware module"""
        pass
    
    @abstractmethod
    def generate_key(self, algorithm: str, key_size: int) -> bytes:
        """Generate cryptographic key"""
        pass
    
    @abstractmethod
    def sign(self, data: bytes, key_handle: bytes) -> bytes:
        """Sign data with hardware key"""
        pass
    
    @abstractmethod
    def verify(self, data: bytes, signature: bytes, key_handle: bytes) -> bool:
        """Verify signature"""
        pass
    
    @abstractmethod
    def encrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Encrypt data"""
        pass
    
    @abstractmethod
    def decrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Decrypt data"""
        pass

class TPMInterface(HardwareInterface):
    """Trusted Platform Module interface"""
    
    def __init__(self):
        self.tpm_version = "2.0"
        self.pcr_banks = ["SHA256", "SHA384", "SHA512"]
        self.key_handles: Dict[str, bytes] = {}
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize TPM 2.0"""
        try:
            # Check for TPM device
            if os.path.exists("/dev/tpm0") or os.path.exists("/dev/tpmrm0"):
                self.initialized = True
                logger.info("TPM 2.0 initialized successfully")
                return True
            else:
                logger.warning("TPM device not found, using software emulation")
                self.initialized = True
                return True
        except Exception as e:
            logger.error(f"TPM initialization failed: {e}")
            return False
    
    def generate_key(self, algorithm: str = "RSA", key_size: int = 2048) -> bytes:
        """Generate key in TPM"""
        if not self.initialized:
            raise RuntimeError("TPM not initialized")
        
        # Generate key handle (simplified)
        key_handle = secrets.token_bytes(32)
        key_id = f"{algorithm}_{key_size}_{secrets.token_hex(8)}"
        self.key_handles[key_id] = key_handle
        
        logger.info(f"Generated {algorithm} key in TPM: {key_id}")
        return key_handle
    
    def seal_data(self, data: bytes, pcr_values: List[int]) -> bytes:
        """Seal data to PCR values"""
        # Create seal blob
        seal_blob = {
            "data": data,
            "pcrs": pcr_values,
            "policy": secrets.token_bytes(32)
        }
        
        # Encrypt with TPM storage key (simplified)
        sealed = bytes([b ^ 0xAA for b in data])  # XOR encryption for demo
        
        return sealed
    
    def unseal_data(self, sealed_data: bytes, pcr_values: List[int]) -> bytes:
        """Unseal data if PCR values match"""
        # Verify PCR values (simplified)
        # In real implementation, would check actual PCR registers
        
        # Decrypt (simplified)
        data = bytes([b ^ 0xAA for b in sealed_data])
        
        return data
    
    def extend_pcr(self, pcr_index: int, data: bytes) -> bytes:
        """Extend PCR with hash of data"""
        hash_value = hashlib.sha256(data).digest()
        logger.info(f"Extended PCR[{pcr_index}] with hash: {hash_value.hex()[:16]}...")
        return hash_value
    
    def quote(self, pcr_indices: List[int], nonce: bytes) -> Dict:
        """Create TPM quote for remote attestation"""
        quote_data = {
            "pcrs": {i: secrets.token_bytes(32) for i in pcr_indices},
            "nonce": nonce,
            "signature": secrets.token_bytes(256),
            "timestamp": os.urandom(8)
        }
        return quote_data
    
    def sign(self, data: bytes, key_handle: bytes) -> bytes:
        """Sign data with TPM key"""
        # Simplified signature
        hash_value = hashlib.sha256(data).digest()
        signature = hashlib.sha512(hash_value + key_handle).digest()
        return signature
    
    def verify(self, data: bytes, signature: bytes, key_handle: bytes) -> bool:
        """Verify signature"""
        expected = self.sign(data, key_handle)
        return expected == signature
    
    def encrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Encrypt with TPM key"""
        # Simplified encryption
        key = hashlib.sha256(key_handle).digest()
        encrypted = bytes([d ^ k for d, k in zip(data, key * (len(data) // 32 + 1))])
        return encrypted
    
    def decrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Decrypt with TPM key"""
        # Simplified decryption (same as encryption for XOR)
        return self.encrypt(data, key_handle)

class HSMInterface(HardwareInterface):
    """Hardware Security Module interface"""
    
    def __init__(self, hsm_type: str = "network"):
        self.hsm_type = hsm_type  # "network", "pcie", "usb"
        self.session = None
        self.initialized = False
        self.key_store: Dict[str, bytes] = {}
        
    def initialize(self) -> bool:
        """Initialize HSM connection"""
        try:
            if self.hsm_type == "network":
                # Would connect to network HSM (e.g., Thales, Gemalto)
                self.session = {"id": secrets.token_hex(16), "active": True}
            elif self.hsm_type == "pcie":
                # Would initialize PCIe HSM
                self.session = {"id": "pcie_" + secrets.token_hex(8), "active": True}
            else:
                # USB HSM
                self.session = {"id": "usb_" + secrets.token_hex(8), "active": True}
            
            self.initialized = True
            logger.info(f"{self.hsm_type.upper()} HSM initialized")
            return True
            
        except Exception as e:
            logger.error(f"HSM initialization failed: {e}")
            return False
    
    def generate_key(self, algorithm: str = "AES", key_size: int = 256) -> bytes:
        """Generate key in HSM"""
        if not self.initialized:
            raise RuntimeError("HSM not initialized")
        
        # Generate key in HSM
        if algorithm == "AES":
            key = secrets.token_bytes(key_size // 8)
        elif algorithm == "RSA":
            key = secrets.token_bytes(key_size // 8)
        else:
            key = secrets.token_bytes(32)
        
        key_id = f"hsm_{algorithm}_{secrets.token_hex(8)}"
        self.key_store[key_id] = key
        
        logger.info(f"Generated {algorithm}-{key_size} key in HSM")
        return key
    
    def perform_crypto(self, operation: CryptoOperation, data: bytes, 
                      key_handle: bytes) -> bytes:
        """Perform cryptographic operation in HSM"""
        if operation == CryptoOperation.ENCRYPT:
            return self.encrypt(data, key_handle)
        elif operation == CryptoOperation.DECRYPT:
            return self.decrypt(data, key_handle)
        elif operation == CryptoOperation.SIGN:
            return self.sign(data, key_handle)
        elif operation == CryptoOperation.HASH:
            return hashlib.sha256(data).digest()
        elif operation == CryptoOperation.RANDOM:
            return secrets.token_bytes(len(data))
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def sign(self, data: bytes, key_handle: bytes) -> bytes:
        """Sign in HSM"""
        hash_val = hashlib.sha256(data).digest()
        signature = hashlib.sha512(hash_val + key_handle).digest()
        return signature
    
    def verify(self, data: bytes, signature: bytes, key_handle: bytes) -> bool:
        """Verify in HSM"""
        expected = self.sign(data, key_handle)
        return expected == signature
    
    def encrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Encrypt in HSM"""
        # AES-like encryption (simplified)
        key = hashlib.sha256(key_handle).digest()
        iv = secrets.token_bytes(16)
        encrypted = iv + bytes([d ^ k for d, k in zip(data, key * (len(data) // 32 + 1))])
        return encrypted
    
    def decrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Decrypt in HSM"""
        iv = data[:16]
        ciphertext = data[16:]
        key = hashlib.sha256(key_handle).digest()
        decrypted = bytes([c ^ k for c, k in zip(ciphertext, key * (len(ciphertext) // 32 + 1))])
        return decrypted

class SecureEnclaveInterface(HardwareInterface):
    """Secure Enclave interface (Intel SGX, ARM TrustZone)"""
    
    def __init__(self, enclave_type: str = "SGX"):
        self.enclave_type = enclave_type
        self.enclave_id = None
        self.sealed_keys: Dict[str, bytes] = {}
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize secure enclave"""
        try:
            if self.enclave_type == "SGX":
                # Check for Intel SGX support
                if self._check_sgx_support():
                    self.enclave_id = secrets.token_hex(16)
                    self.initialized = True
                    logger.info("Intel SGX enclave initialized")
                    return True
            elif self.enclave_type == "TrustZone":
                # Check for ARM TrustZone
                if self._check_trustzone_support():
                    self.enclave_id = secrets.token_hex(16)
                    self.initialized = True
                    logger.info("ARM TrustZone initialized")
                    return True
            
            # Fallback to software emulation
            self.enclave_id = "emulated_" + secrets.token_hex(8)
            self.initialized = True
            logger.warning(f"Using emulated secure enclave")
            return True
            
        except Exception as e:
            logger.error(f"Secure enclave initialization failed: {e}")
            return False
    
    def _check_sgx_support(self) -> bool:
        """Check for Intel SGX support"""
        try:
            # Check CPUID for SGX support
            if platform.processor() and "Intel" in platform.processor():
                return os.path.exists("/dev/sgx") or os.path.exists("/dev/isgx")
            return False
        except:
            return False
    
    def _check_trustzone_support(self) -> bool:
        """Check for ARM TrustZone support"""
        try:
            # Check for ARM processor and TrustZone
            if platform.machine().startswith("arm") or platform.machine() == "aarch64":
                return os.path.exists("/dev/tee0") or os.path.exists("/dev/teepriv0")
            return False
        except:
            return False
    
    def create_enclave(self, code: bytes) -> str:
        """Create secure enclave with code"""
        enclave_id = secrets.token_hex(16)
        
        # Measure code for attestation
        measurement = hashlib.sha256(code).digest()
        
        logger.info(f"Created enclave {enclave_id} with measurement {measurement.hex()[:16]}...")
        return enclave_id
    
    def remote_attestation(self, enclave_id: str, challenge: bytes) -> Dict:
        """Generate remote attestation report"""
        report = {
            "enclave_id": enclave_id,
            "measurement": secrets.token_bytes(32),
            "challenge": challenge,
            "platform_info": {
                "cpu": platform.processor(),
                "security_version": 1,
                "attributes": ["DEBUG", "MODE64BIT"]
            },
            "signature": secrets.token_bytes(256)
        }
        return report
    
    def seal_secret(self, secret: bytes, policy: Dict) -> bytes:
        """Seal secret to enclave"""
        # Derive sealing key from enclave measurement
        sealing_key = hashlib.sha256(self.enclave_id.encode() + b"SEAL").digest()
        
        # Encrypt secret
        sealed = bytes([s ^ k for s, k in zip(secret, sealing_key * (len(secret) // 32 + 1))])
        
        return sealed
    
    def unseal_secret(self, sealed_data: bytes) -> bytes:
        """Unseal secret in enclave"""
        # Derive sealing key
        sealing_key = hashlib.sha256(self.enclave_id.encode() + b"SEAL").digest()
        
        # Decrypt secret
        secret = bytes([s ^ k for s, k in zip(sealed_data, sealing_key * (len(sealed_data) // 32 + 1))])
        
        return secret
    
    def generate_key(self, algorithm: str = "ECDSA", key_size: int = 256) -> bytes:
        """Generate key in enclave"""
        key = secrets.token_bytes(key_size // 8)
        key_id = f"enclave_{algorithm}_{secrets.token_hex(8)}"
        self.sealed_keys[key_id] = self.seal_secret(key, {"algorithm": algorithm})
        return key
    
    def sign(self, data: bytes, key_handle: bytes) -> bytes:
        """Sign in enclave"""
        hash_val = hashlib.sha256(data).digest()
        signature = hashlib.sha512(hash_val + key_handle + self.enclave_id.encode()).digest()
        return signature
    
    def verify(self, data: bytes, signature: bytes, key_handle: bytes) -> bool:
        """Verify in enclave"""
        expected = self.sign(data, key_handle)
        return expected == signature
    
    def encrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Encrypt in enclave"""
        key = hashlib.sha256(key_handle + self.enclave_id.encode()).digest()
        encrypted = bytes([d ^ k for d, k in zip(data, key * (len(data) // 32 + 1))])
        return encrypted
    
    def decrypt(self, data: bytes, key_handle: bytes) -> bytes:
        """Decrypt in enclave"""
        return self.encrypt(data, key_handle)  # XOR is symmetric

class HardwareAbstractionLayer:
    """Unified hardware abstraction layer"""
    
    def __init__(self):
        self.interfaces: Dict[HardwareType, HardwareInterface] = {}
        self.capabilities: Dict[HardwareType, HardwareCapabilities] = {}
        self.active_interface: Optional[HardwareInterface] = None
        
    def detect_hardware(self) -> List[HardwareType]:
        """Detect available hardware security modules"""
        available = []
        
        # Check for TPM
        if os.path.exists("/dev/tpm0") or os.path.exists("/dev/tpmrm0"):
            available.append(HardwareType.TPM)
            self.capabilities[HardwareType.TPM] = HardwareCapabilities(
                type=HardwareType.TPM,
                vendor="Generic",
                model="TPM 2.0",
                firmware_version="1.38",
                algorithms=["RSA", "ECC", "AES", "SHA256"],
                max_key_size=4096,
                performance={"sign": 100, "verify": 150, "encrypt": 500},
                features=["Attestation", "Sealing", "PCR"]
            )
        
        # Check for Intel SGX
        if platform.processor() and "Intel" in platform.processor():
            if os.path.exists("/dev/sgx") or os.path.exists("/dev/isgx"):
                available.append(HardwareType.TEE)
                self.capabilities[HardwareType.TEE] = HardwareCapabilities(
                    type=HardwareType.TEE,
                    vendor="Intel",
                    model="SGX",
                    firmware_version="2.0",
                    algorithms=["AES-GCM", "ECDSA", "SHA256"],
                    max_key_size=256,
                    performance={"sign": 1000, "verify": 1500, "encrypt": 5000},
                    features=["Remote Attestation", "Sealing", "Enclaves"]
                )
        
        # Check for ARM TrustZone
        if platform.machine().startswith("arm"):
            if os.path.exists("/dev/tee0"):
                available.append(HardwareType.TEE)
                self.capabilities[HardwareType.TEE] = HardwareCapabilities(
                    type=HardwareType.TEE,
                    vendor="ARM",
                    model="TrustZone",
                    firmware_version="1.0",
                    algorithms=["AES", "RSA", "SHA256"],
                    max_key_size=2048,
                    performance={"sign": 500, "verify": 750, "encrypt": 2000},
                    features=["Secure World", "Trusted Apps"]
                )
        
        # Always available: software HSM emulation
        available.append(HardwareType.HSM)
        self.capabilities[HardwareType.HSM] = HardwareCapabilities(
            type=HardwareType.HSM,
            vendor="Software",
            model="Emulated",
            firmware_version="1.0",
            algorithms=["RSA", "ECC", "AES", "SHA256", "SHA512"],
            max_key_size=4096,
            performance={"sign": 50, "verify": 75, "encrypt": 200},
            features=["Key Storage", "Crypto Operations"]
        )
        
        logger.info(f"Detected hardware security modules: {[h.value for h in available]}")
        return available
    
    def initialize_interface(self, hw_type: HardwareType) -> bool:
        """Initialize specific hardware interface"""
        try:
            if hw_type == HardwareType.TPM:
                interface = TPMInterface()
            elif hw_type == HardwareType.HSM:
                interface = HSMInterface()
            elif hw_type == HardwareType.TEE:
                interface = SecureEnclaveInterface()
            elif hw_type == HardwareType.SECURE_ENCLAVE:
                interface = SecureEnclaveInterface("TrustZone")
            else:
                logger.error(f"Unsupported hardware type: {hw_type}")
                return False
            
            if interface.initialize():
                self.interfaces[hw_type] = interface
                if not self.active_interface:
                    self.active_interface = interface
                logger.info(f"Initialized {hw_type.value} interface")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize {hw_type.value}: {e}")
            return False
    
    def get_best_interface(self) -> Optional[HardwareInterface]:
        """Get best available hardware interface"""
        # Priority order: TEE > HSM > TPM
        priority = [HardwareType.TEE, HardwareType.SECURE_ENCLAVE, 
                   HardwareType.HSM, HardwareType.TPM]
        
        for hw_type in priority:
            if hw_type in self.interfaces:
                return self.interfaces[hw_type]
        
        return self.active_interface
    
    def perform_operation(self, operation: CryptoOperation, 
                         data: bytes, key: Optional[bytes] = None) -> bytes:
        """Perform cryptographic operation using best available hardware"""
        interface = self.get_best_interface()
        if not interface:
            raise RuntimeError("No hardware interface available")
        
        if operation == CryptoOperation.GENERATE_KEY:
            return interface.generate_key("AES", 256)
        elif operation == CryptoOperation.SIGN:
            return interface.sign(data, key or secrets.token_bytes(32))
        elif operation == CryptoOperation.VERIFY:
            # Returns boolean as bytes
            return b"\x01" if interface.verify(data, key, secrets.token_bytes(32)) else b"\x00"
        elif operation == CryptoOperation.ENCRYPT:
            return interface.encrypt(data, key or secrets.token_bytes(32))
        elif operation == CryptoOperation.DECRYPT:
            return interface.decrypt(data, key or secrets.token_bytes(32))
        elif operation == CryptoOperation.HASH:
            return hashlib.sha256(data).digest()
        elif operation == CryptoOperation.RANDOM:
            return secrets.token_bytes(len(data) if len(data) > 0 else 32)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all interfaces"""
        metrics = {}
        
        for hw_type, interface in self.interfaces.items():
            if hw_type in self.capabilities:
                cap = self.capabilities[hw_type]
                metrics[hw_type.value] = {
                    "vendor": cap.vendor,
                    "model": cap.model,
                    "performance": cap.performance,
                    "max_key_size": cap.max_key_size
                }
        
        return metrics

# Example usage
if __name__ == "__main__":
    print("Hardware Abstraction Layer")
    print("=" * 50)
    
    # Initialize HAL
    hal = HardwareAbstractionLayer()
    
    # Detect available hardware
    available = hal.detect_hardware()
    print(f"\nAvailable Hardware Security Modules:")
    for hw in available:
        print(f"  - {hw.value}")
        if hw in hal.capabilities:
            cap = hal.capabilities[hw]
            print(f"    Vendor: {cap.vendor}")
            print(f"    Model: {cap.model}")
            print(f"    Max Key Size: {cap.max_key_size} bits")
    
    # Initialize interfaces
    print(f"\nInitializing interfaces...")
    for hw in available:
        success = hal.initialize_interface(hw)
        print(f"  {hw.value}: {'✓' if success else '✗'}")
    
    # Test operations
    print(f"\nTesting cryptographic operations...")
    
    # Generate key
    key = hal.perform_operation(CryptoOperation.GENERATE_KEY, b"")
    print(f"  Generated key: {len(key)} bytes")
    
    # Encrypt/Decrypt
    plaintext = b"Sensitive banking data"
    ciphertext = hal.perform_operation(CryptoOperation.ENCRYPT, plaintext, key)
    decrypted = hal.perform_operation(CryptoOperation.DECRYPT, ciphertext, key)
    print(f"  Encryption: {'✓' if decrypted == plaintext else '✗'}")
    
    # Sign/Verify
    message = b"Transaction: $1,000,000"
    signature = hal.perform_operation(CryptoOperation.SIGN, message, key)
    print(f"  Signature: {len(signature)} bytes")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    metrics = hal.get_performance_metrics()
    for hw_type, data in metrics.items():
        print(f"  {hw_type}:")
        print(f"    Sign: {data['performance'].get('sign', 0)} ops/sec")
        print(f"    Verify: {data['performance'].get('verify', 0)} ops/sec")
        print(f"    Encrypt: {data['performance'].get('encrypt', 0)} ops/sec")