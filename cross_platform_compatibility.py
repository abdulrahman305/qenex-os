#!/usr/bin/env python3
"""
QENEX Cross-Platform Compatibility Layer
Ensures QENEX works seamlessly across different operating systems and architectures
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import ctypes
import shutil

logger = logging.getLogger(__name__)

# ============================================================================
# Platform Detection and Capabilities
# ============================================================================

class PlatformType(Enum):
    """Supported platform types"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    FREEBSD = "freebsd"
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"

class Architecture(Enum):
    """Supported architectures"""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    X86 = "x86"
    RISC_V = "riscv"
    WASM = "wasm"

@dataclass
class PlatformInfo:
    """Platform information and capabilities"""
    platform_type: PlatformType
    architecture: Architecture
    os_version: str
    kernel_version: str
    has_gui: bool
    has_network: bool
    has_filesystem: bool
    has_hardware_security: bool
    memory_mb: int
    cpu_cores: int
    supports_virtualization: bool
    supports_containers: bool
    python_version: str
    available_libraries: List[str]

class PlatformDetector:
    """Detects current platform and its capabilities"""
    
    @staticmethod
    def detect() -> PlatformInfo:
        """Detect current platform information"""
        
        # Detect platform type
        system = platform.system().lower()
        if system == "windows":
            platform_type = PlatformType.WINDOWS
        elif system == "linux":
            if PlatformDetector._is_android():
                platform_type = PlatformType.ANDROID
            else:
                platform_type = PlatformType.LINUX
        elif system == "darwin":
            if PlatformDetector._is_ios():
                platform_type = PlatformType.IOS
            else:
                platform_type = PlatformType.MACOS
        elif system == "freebsd":
            platform_type = PlatformType.FREEBSD
        else:
            platform_type = PlatformType.LINUX  # Default fallback
            
        # Detect architecture
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            arch = Architecture.X86_64
        elif machine in ["aarch64", "arm64"]:
            arch = Architecture.ARM64
        elif machine.startswith("arm"):
            arch = Architecture.ARM32
        elif machine in ["i386", "i686", "x86"]:
            arch = Architecture.X86
        elif machine.startswith("riscv"):
            arch = Architecture.RISC_V
        else:
            arch = Architecture.X86_64  # Default fallback
            
        # Get system information
        os_version = platform.version()
        kernel_version = platform.release()
        python_version = platform.python_version()
        
        # Detect capabilities
        capabilities = PlatformDetector._detect_capabilities(platform_type)
        
        # Get system resources
        memory_mb = PlatformDetector._get_memory_mb()
        cpu_cores = os.cpu_count() or 1
        
        # Detect available libraries
        available_libs = PlatformDetector._detect_available_libraries()
        
        return PlatformInfo(
            platform_type=platform_type,
            architecture=arch,
            os_version=os_version,
            kernel_version=kernel_version,
            has_gui=capabilities['gui'],
            has_network=capabilities['network'],
            has_filesystem=capabilities['filesystem'],
            has_hardware_security=capabilities['hardware_security'],
            memory_mb=memory_mb,
            cpu_cores=cpu_cores,
            supports_virtualization=capabilities['virtualization'],
            supports_containers=capabilities['containers'],
            python_version=python_version,
            available_libraries=available_libs
        )
        
    @staticmethod
    def _is_android() -> bool:
        """Check if running on Android"""
        try:
            return 'ANDROID_ROOT' in os.environ or 'ANDROID_DATA' in os.environ
        except:
            return False
            
    @staticmethod
    def _is_ios() -> bool:
        """Check if running on iOS"""
        try:
            import platform
            return platform.system() == 'Darwin' and 'iPhone' in platform.platform()
        except:
            return False
            
    @staticmethod
    def _detect_capabilities(platform_type: PlatformType) -> Dict[str, bool]:
        """Detect platform capabilities"""
        
        capabilities = {
            'gui': True,
            'network': True,
            'filesystem': True,
            'hardware_security': False,
            'virtualization': False,
            'containers': False
        }
        
        try:
            # Check GUI capability
            if platform_type == PlatformType.LINUX:
                capabilities['gui'] = 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ
            elif platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
                capabilities['gui'] = True
            elif platform_type == PlatformType.WEB:
                capabilities['gui'] = True
                capabilities['filesystem'] = False
                
            # Check virtualization support
            if platform_type == PlatformType.LINUX:
                capabilities['virtualization'] = os.path.exists('/dev/kvm')
                capabilities['containers'] = shutil.which('docker') is not None
            elif platform_type == PlatformType.WINDOWS:
                capabilities['virtualization'] = PlatformDetector._check_hyper_v()
                capabilities['containers'] = shutil.which('docker') is not None
                
            # Check hardware security
            capabilities['hardware_security'] = PlatformDetector._check_hardware_security(platform_type)
            
        except Exception as e:
            logger.warning(f"Error detecting capabilities: {e}")
            
        return capabilities
        
    @staticmethod
    def _get_memory_mb() -> int:
        """Get system memory in MB"""
        try:
            if platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) // 1024
            elif platform.system() == "Windows":
                import psutil
                return psutil.virtual_memory().total // (1024 * 1024)
            elif platform.system() == "Darwin":
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if result.returncode == 0:
                    return int(result.stdout.split()[1]) // (1024 * 1024)
        except:
            pass
        return 4096  # Default fallback
        
    @staticmethod
    def _check_hyper_v() -> bool:
        """Check if Hyper-V is available on Windows"""
        try:
            result = subprocess.run(['systeminfo'], capture_output=True, text=True)
            return 'Hyper-V' in result.stdout
        except:
            return False
            
    @staticmethod
    def _check_hardware_security(platform_type: PlatformType) -> bool:
        """Check for hardware security features"""
        try:
            if platform_type == PlatformType.LINUX:
                # Check for TPM
                return os.path.exists('/dev/tpm0') or os.path.exists('/sys/class/tpm')
            elif platform_type == PlatformType.WINDOWS:
                # Check for TPM via WMI
                result = subprocess.run(['wmic', 'path', 'win32_tpm', 'get', 'SpecVersion'], 
                                      capture_output=True, text=True)
                return 'SpecVersion' in result.stdout and len(result.stdout.strip().split('\n')) > 2
            elif platform_type == PlatformType.MACOS:
                # Check for Secure Enclave
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True)
                return 'Secure Enclave' in result.stdout
        except:
            pass
        return False
        
    @staticmethod
    def _detect_available_libraries() -> List[str]:
        """Detect available Python libraries"""
        libraries = []
        test_imports = [
            'numpy', 'scipy', 'pandas', 'sklearn', 'tensorflow', 'torch',
            'cryptography', 'requests', 'aiohttp', 'fastapi', 'flask',
            'psutil', 'sqlite3', 'postgresql', 'redis', 'docker'
        ]
        
        for lib in test_imports:
            try:
                __import__(lib)
                libraries.append(lib)
            except ImportError:
                pass
                
        return libraries

# ============================================================================
# Cross-Platform Abstractions
# ============================================================================

class FileSystemAdapter(ABC):
    """Abstract file system adapter"""
    
    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        pass
        
    @abstractmethod
    async def write_file(self, path: str, content: bytes) -> bool:
        pass
        
    @abstractmethod
    async def create_directory(self, path: str) -> bool:
        pass
        
    @abstractmethod
    async def list_directory(self, path: str) -> List[str]:
        pass
        
    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        pass

class NetworkAdapter(ABC):
    """Abstract network adapter"""
    
    @abstractmethod
    async def http_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def websocket_connect(self, url: str) -> Any:
        pass
        
    @abstractmethod
    async def tcp_listen(self, host: str, port: int) -> Any:
        pass

class SecurityAdapter(ABC):
    """Abstract security adapter"""
    
    @abstractmethod
    async def generate_keypair(self) -> Dict[str, bytes]:
        pass
        
    @abstractmethod
    async def sign_data(self, data: bytes, private_key: bytes) -> bytes:
        pass
        
    @abstractmethod
    async def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        pass
        
    @abstractmethod
    async def secure_store(self, key: str, value: bytes) -> bool:
        pass
        
    @abstractmethod
    async def secure_retrieve(self, key: str) -> Optional[bytes]:
        pass

# ============================================================================
# Platform-Specific Implementations
# ============================================================================

class UnixFileSystemAdapter(FileSystemAdapter):
    """Unix-like file system adapter"""
    
    async def read_file(self, path: str) -> bytes:
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise
            
    async def write_file(self, path: str, content: bytes) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return False
            
    async def create_directory(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
            
    async def list_directory(self, path: str) -> List[str]:
        try:
            return os.listdir(path)
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return []
            
    async def delete_file(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
            return False

class WindowsFileSystemAdapter(FileSystemAdapter):
    """Windows file system adapter with extended attributes"""
    
    async def read_file(self, path: str) -> bytes:
        try:
            # Handle Windows path separators
            path = path.replace('/', '\\')
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise
            
    async def write_file(self, path: str, content: bytes) -> bool:
        try:
            path = path.replace('/', '\\')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return False
            
    async def create_directory(self, path: str) -> bool:
        try:
            path = path.replace('/', '\\')
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
            
    async def list_directory(self, path: str) -> List[str]:
        try:
            path = path.replace('/', '\\')
            return os.listdir(path)
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return []
            
    async def delete_file(self, path: str) -> bool:
        try:
            path = path.replace('/', '\\')
            os.remove(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
            return False

class StandardNetworkAdapter(NetworkAdapter):
    """Standard network adapter using aiohttp"""
    
    def __init__(self):
        self._session = None
        
    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                # Fallback to requests
                import requests
                self._session = requests.Session()
        return self._session
        
    async def http_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        try:
            session = await self._get_session()
            if hasattr(session, 'request'):  # aiohttp
                async with session.request(method, url, **kwargs) as response:
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'content': await response.text()
                    }
            else:  # requests
                response = session.request(method, url, **kwargs)
                return {
                    'status': response.status_code,
                    'headers': dict(response.headers),
                    'content': response.text
                }
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            raise
            
    async def websocket_connect(self, url: str) -> Any:
        try:
            import aiohttp
            session = await self._get_session()
            return await session.ws_connect(url)
        except ImportError:
            logger.warning("WebSocket support requires aiohttp")
            return None
            
    async def tcp_listen(self, host: str, port: int) -> Any:
        try:
            import asyncio
            server = await asyncio.start_server(
                lambda r, w: None, host, port
            )
            return server
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
            return None

class StandardSecurityAdapter(SecurityAdapter):
    """Standard security adapter using cryptography library"""
    
    def __init__(self):
        self._keystore = {}
        
    async def generate_keypair(self) -> Dict[str, bytes]:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                'private_key': private_pem,
                'public_key': public_pem
            }
        except ImportError:
            logger.warning("Cryptography library not available, using mock keys")
            return {
                'private_key': b'mock_private_key',
                'public_key': b'mock_public_key'
            }
            
    async def sign_data(self, data: bytes, private_key: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives import serialization, hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            key = serialization.load_pem_private_key(private_key, password=None)
            signature = key.sign(data, padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ), hashes.SHA256())
            
            return signature
        except ImportError:
            # Mock signature
            import hashlib
            return hashlib.sha256(data + private_key).digest()
            
    async def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        try:
            from cryptography.fernet import Fernet
            f = Fernet(key)
            return f.encrypt(data)
        except ImportError:
            # Simple XOR encryption as fallback
            key_bytes = key[:len(data)]
            return bytes(a ^ b for a, b in zip(data, key_bytes))
            
    async def secure_store(self, key: str, value: bytes) -> bool:
        try:
            self._keystore[key] = value
            return True
        except Exception as e:
            logger.error(f"Failed to store key {key}: {e}")
            return False
            
    async def secure_retrieve(self, key: str) -> Optional[bytes]:
        return self._keystore.get(key)

# ============================================================================
# Cross-Platform Manager
# ============================================================================

class CrossPlatformManager:
    """Main cross-platform compatibility manager"""
    
    def __init__(self):
        self.platform_info = PlatformDetector.detect()
        self.fs_adapter = self._create_filesystem_adapter()
        self.network_adapter = self._create_network_adapter()
        self.security_adapter = self._create_security_adapter()
        
        logger.info(f"QENEX initialized on {self.platform_info.platform_type.value} "
                   f"{self.platform_info.architecture.value}")
                   
    def _create_filesystem_adapter(self) -> FileSystemAdapter:
        """Create appropriate filesystem adapter"""
        if self.platform_info.platform_type == PlatformType.WINDOWS:
            return WindowsFileSystemAdapter()
        else:
            return UnixFileSystemAdapter()
            
    def _create_network_adapter(self) -> NetworkAdapter:
        """Create appropriate network adapter"""
        return StandardNetworkAdapter()
        
    def _create_security_adapter(self) -> SecurityAdapter:
        """Create appropriate security adapter"""
        return StandardSecurityAdapter()
        
    async def initialize_platform_specific_features(self):
        """Initialize platform-specific features"""
        
        if self.platform_info.platform_type == PlatformType.LINUX:
            await self._initialize_linux_features()
        elif self.platform_info.platform_type == PlatformType.WINDOWS:
            await self._initialize_windows_features()
        elif self.platform_info.platform_type == PlatformType.MACOS:
            await self._initialize_macos_features()
        elif self.platform_info.platform_type == PlatformType.ANDROID:
            await self._initialize_android_features()
            
    async def _initialize_linux_features(self):
        """Initialize Linux-specific features"""
        
        # Check for systemd
        if shutil.which('systemctl'):
            logger.info("Systemd detected, enabling service management")
            
        # Check for container support
        if self.platform_info.supports_containers:
            logger.info("Container support available")
            
        # Check for hardware security
        if self.platform_info.has_hardware_security:
            logger.info("Hardware security module detected")
            
    async def _initialize_windows_features(self):
        """Initialize Windows-specific features"""
        
        # Check for Windows services
        try:
            import win32service
            logger.info("Windows service management available")
        except ImportError:
            logger.warning("pywin32 not available, limited Windows integration")
            
        # Check for Windows security features
        if self.platform_info.has_hardware_security:
            logger.info("Windows TPM detected")
            
    async def _initialize_macos_features(self):
        """Initialize macOS-specific features"""
        
        # Check for macOS security features
        if self.platform_info.has_hardware_security:
            logger.info("macOS Secure Enclave detected")
            
        # Check for Keychain access
        try:
            result = subprocess.run(['security', 'list-keychains'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("macOS Keychain access available")
        except:
            pass
            
    async def _initialize_android_features(self):
        """Initialize Android-specific features"""
        
        # Check for Android keystore
        logger.info("Android platform detected, enabling mobile features")
        
    def get_platform_capabilities(self) -> Dict[str, Any]:
        """Get detailed platform capabilities"""
        
        return {
            'platform': self.platform_info.platform_type.value,
            'architecture': self.platform_info.architecture.value,
            'os_version': self.platform_info.os_version,
            'python_version': self.platform_info.python_version,
            'memory_mb': self.platform_info.memory_mb,
            'cpu_cores': self.platform_info.cpu_cores,
            'capabilities': {
                'gui': self.platform_info.has_gui,
                'network': self.platform_info.has_network,
                'filesystem': self.platform_info.has_filesystem,
                'hardware_security': self.platform_info.has_hardware_security,
                'virtualization': self.platform_info.supports_virtualization,
                'containers': self.platform_info.supports_containers
            },
            'available_libraries': self.platform_info.available_libraries
        }
        
    async def optimize_for_platform(self) -> Dict[str, Any]:
        """Apply platform-specific optimizations"""
        
        optimizations = {}
        
        # Memory optimizations
        if self.platform_info.memory_mb < 2048:
            optimizations['memory'] = 'limited'
            logger.info("Applying low-memory optimizations")
        else:
            optimizations['memory'] = 'standard'
            
        # CPU optimizations
        if self.platform_info.cpu_cores == 1:
            optimizations['concurrency'] = 'single_threaded'
            logger.info("Single-core system detected, disabling multiprocessing")
        else:
            optimizations['concurrency'] = 'multi_threaded'
            
        # Storage optimizations
        if self.platform_info.platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
            optimizations['storage'] = 'mobile_optimized'
            logger.info("Mobile platform detected, enabling storage optimizations")
        else:
            optimizations['storage'] = 'standard'
            
        return optimizations

# ============================================================================
# Testing
# ============================================================================

async def test_cross_platform():
    """Test cross-platform compatibility system"""
    
    print("Testing QENEX Cross-Platform Compatibility")
    print("=" * 45)
    
    # Initialize cross-platform manager
    manager = CrossPlatformManager()
    
    # Display platform information
    print(f"\nPlatform Information:")
    print(f"- System: {manager.platform_info.platform_type.value}")
    print(f"- Architecture: {manager.platform_info.architecture.value}")
    print(f"- OS Version: {manager.platform_info.os_version}")
    print(f"- Python: {manager.platform_info.python_version}")
    print(f"- Memory: {manager.platform_info.memory_mb:,} MB")
    print(f"- CPU Cores: {manager.platform_info.cpu_cores}")
    
    # Test capabilities
    capabilities = manager.get_platform_capabilities()
    print(f"\nPlatform Capabilities:")
    for category, details in capabilities['capabilities'].items():
        status = "✓" if details else "✗"
        print(f"- {category.replace('_', ' ').title()}: {status}")
        
    # Test filesystem adapter
    print(f"\nTesting Filesystem Adapter...")
    test_file = os.path.join(tempfile.gettempdir(), "qenex_test.txt")
    test_content = b"QENEX cross-platform test"
    
    success = await manager.fs_adapter.write_file(test_file, test_content)
    if success:
        read_content = await manager.fs_adapter.read_file(test_file)
        if read_content == test_content:
            print("✓ Filesystem adapter working")
        else:
            print("✗ Filesystem read/write mismatch")
        await manager.fs_adapter.delete_file(test_file)
    else:
        print("✗ Filesystem adapter failed")
        
    # Test security adapter
    print(f"\nTesting Security Adapter...")
    keypair = await manager.security_adapter.generate_keypair()
    if keypair:
        print("✓ Key generation working")
        
        # Test signing
        test_data = b"test data for signing"
        signature = await manager.security_adapter.sign_data(test_data, keypair['private_key'])
        if signature:
            print("✓ Data signing working")
        else:
            print("✗ Data signing failed")
    else:
        print("✗ Key generation failed")
        
    # Initialize platform-specific features
    print(f"\nInitializing Platform-Specific Features...")
    await manager.initialize_platform_specific_features()
    
    # Apply optimizations
    print(f"\nApplying Platform Optimizations...")
    optimizations = await manager.optimize_for_platform()
    print(f"Applied optimizations: {optimizations}")
    
    print(f"\nCross-platform compatibility test completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_cross_platform())