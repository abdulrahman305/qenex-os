#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import json
import time
import threading
import socket
import struct
import hashlib
import base64
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import secrets

@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    architecture: str
    processor: str
    python_version: str
    hostname: str
    memory_gb: float
    cpu_count: int
    kernel_version: str = ""
    distribution: str = ""

class PlatformDetector:
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.version = platform.version()
        self.processor = platform.processor()
        
    def get_system_info(self) -> SystemInfo:
        info = SystemInfo(
            os_name=self.system,
            os_version=self.version,
            architecture=self.machine,
            processor=self.processor,
            python_version=platform.python_version(),
            hostname=socket.gethostname(),
            memory_gb=self.get_memory_size(),
            cpu_count=os.cpu_count() or 1
        )
        
        if self.system == "Linux":
            info.kernel_version = platform.release()
            try:
                with open('/etc/os-release') as f:
                    for line in f:
                        if line.startswith('PRETTY_NAME='):
                            info.distribution = line.split('=')[1].strip().strip('"')
                            break
            except:
                info.distribution = "Unknown Linux"
        elif self.system == "Darwin":
            info.distribution = f"macOS {platform.mac_ver()[0]}"
        elif self.system == "Windows":
            info.distribution = f"Windows {platform.win32_ver()[0]}"
        
        return info
    
    def get_memory_size(self) -> float:
        try:
            if self.system == "Linux":
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal'):
                            return int(line.split()[1]) / (1024 * 1024)
            elif self.system == "Darwin":
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True)
                return int(result.stdout.strip()) / (1024 * 1024 * 1024)
            elif self.system == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]
                
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024 * 1024 * 1024)
        except:
            pass
        return 0.0

class FileSystemManager:
    def __init__(self):
        self.platform = platform.system()
        self.home_dir = Path.home()
        self.config_dir = self.get_config_directory()
        self.data_dir = self.get_data_directory()
        
    def get_config_directory(self) -> Path:
        if self.platform == "Windows":
            base = os.environ.get('APPDATA', self.home_dir / 'AppData' / 'Roaming')
        elif self.platform == "Darwin":
            base = self.home_dir / 'Library' / 'Application Support'
        else:
            base = Path(os.environ.get('XDG_CONFIG_HOME', self.home_dir / '.config'))
        
        config_dir = Path(base) / 'qenex'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def get_data_directory(self) -> Path:
        if self.platform == "Windows":
            base = os.environ.get('LOCALAPPDATA', self.home_dir / 'AppData' / 'Local')
        elif self.platform == "Darwin":
            base = self.home_dir / 'Library' / 'Application Support'
        else:
            base = Path(os.environ.get('XDG_DATA_HOME', self.home_dir / '.local' / 'share'))
        
        data_dir = Path(base) / 'qenex'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_temp_directory(self) -> Path:
        if self.platform == "Windows":
            return Path(os.environ.get('TEMP', '/tmp'))
        else:
            return Path('/tmp')
    
    def normalize_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        path = os.path.abspath(path)
        
        if self.platform == "Windows":
            path = path.replace('/', '\\')
        else:
            path = path.replace('\\', '/')
        
        return path
    
    def read_file(self, path: str) -> Optional[str]:
        try:
            normalized_path = self.normalize_path(path)
            with open(normalized_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def write_file(self, path: str, content: str) -> bool:
        try:
            normalized_path = self.normalize_path(path)
            os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
            
            with open(normalized_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False

class ProcessManager:
    def __init__(self):
        self.platform = platform.system()
        self.processes = {}
        
    def run_command(self, command: List[str], shell: bool = False) -> Tuple[int, str, str]:
        try:
            if self.platform == "Windows" and not shell:
                command = [cmd.replace('/', '\\') for cmd in command]
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=shell,
                text=True
            )
            
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr
        except Exception as e:
            return -1, "", str(e)
    
    def start_background_process(self, name: str, command: List[str]) -> bool:
        try:
            if self.platform == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=creation_flags
                )
            else:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            self.processes[name] = process
            return True
        except Exception as e:
            print(f"Error starting process: {e}")
            return False
    
    def stop_process(self, name: str) -> bool:
        if name not in self.processes:
            return False
        
        try:
            process = self.processes[name]
            
            if self.platform == "Windows":
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), 15)
            
            process.wait(timeout=5)
            del self.processes[name]
            return True
        except Exception as e:
            print(f"Error stopping process: {e}")
            return False
    
    def is_process_running(self, name: str) -> bool:
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        return process.poll() is None

class NetworkManager:
    def __init__(self):
        self.platform = platform.system()
        self.interfaces = self.get_network_interfaces()
        
    def get_network_interfaces(self) -> List[Dict]:
        interfaces = []
        
        try:
            if self.platform in ["Linux", "Darwin"]:
                result, stdout, _ = ProcessManager().run_command(['ifconfig', '-a'])
                if result == 0:
                    current_interface = None
                    for line in stdout.split('\n'):
                        if line and not line.startswith(' '):
                            interface_name = line.split(':')[0].split()[0]
                            current_interface = {'name': interface_name, 'addresses': []}
                            interfaces.append(current_interface)
                        elif current_interface and 'inet ' in line:
                            parts = line.split()
                            inet_idx = parts.index('inet')
                            if inet_idx + 1 < len(parts):
                                current_interface['addresses'].append(parts[inet_idx + 1])
            
            elif self.platform == "Windows":
                result, stdout, _ = ProcessManager().run_command(['ipconfig', '/all'])
                if result == 0:
                    current_interface = None
                    for line in stdout.split('\n'):
                        if 'adapter' in line.lower():
                            interface_name = line.split('adapter')[1].strip(':').strip()
                            current_interface = {'name': interface_name, 'addresses': []}
                            interfaces.append(current_interface)
                        elif current_interface and 'ipv4' in line.lower():
                            parts = line.split(':')
                            if len(parts) > 1:
                                addr = parts[1].strip().split('(')[0].strip()
                                if addr:
                                    current_interface['addresses'].append(addr)
        except:
            pass
        
        if not interfaces:
            interfaces.append({
                'name': 'default',
                'addresses': ['127.0.0.1']
            })
        
        return interfaces
    
    def get_primary_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return '127.0.0.1'
    
    def is_port_available(self, port: int, host: str = '0.0.0.0') -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except:
            return False

class SecurityManager:
    def __init__(self):
        self.platform = platform.system()
        self.key_store = {}
        
    def generate_secure_key(self, length: int = 32) -> bytes:
        return secrets.token_bytes(length)
    
    def hash_data(self, data: bytes, algorithm: str = 'sha256') -> str:
        if algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data).hexdigest()
        elif algorithm == 'blake2b':
            return hashlib.blake2b(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key[i % len(key)])
        return bytes(encrypted)
    
    def decrypt_data(self, encrypted: bytes, key: bytes) -> bytes:
        return self.encrypt_data(encrypted, key)
    
    def secure_random(self, min_val: int = 0, max_val: int = 100) -> int:
        return secrets.randbelow(max_val - min_val) + min_val
    
    def generate_token(self, length: int = 32) -> str:
        return secrets.token_urlsafe(length)

class CrossPlatformBridge:
    def __init__(self):
        self.detector = PlatformDetector()
        self.filesystem = FileSystemManager()
        self.process = ProcessManager()
        self.network = NetworkManager()
        self.security = SecurityManager()
        self.system_info = self.detector.get_system_info()
        
    def get_executable_extension(self) -> str:
        if self.system_info.os_name == "Windows":
            return ".exe"
        return ""
    
    def get_library_extension(self) -> str:
        if self.system_info.os_name == "Windows":
            return ".dll"
        elif self.system_info.os_name == "Darwin":
            return ".dylib"
        else:
            return ".so"
    
    def get_path_separator(self) -> str:
        if self.system_info.os_name == "Windows":
            return ";"
        return ":"
    
    def get_line_ending(self) -> str:
        if self.system_info.os_name == "Windows":
            return "\r\n"
        return "\n"
    
    def ensure_compatibility(self) -> Dict[str, bool]:
        checks = {
            'python_version': sys.version_info >= (3, 6),
            'memory_available': self.system_info.memory_gb >= 0.5,
            'network_connectivity': self.check_network_connectivity(),
            'filesystem_access': self.check_filesystem_access(),
            'security_features': self.check_security_features()
        }
        return checks
    
    def check_network_connectivity(self) -> bool:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def check_filesystem_access(self) -> bool:
        try:
            test_file = self.filesystem.get_temp_directory() / 'qenex_test.txt'
            test_file.write_text('test')
            content = test_file.read_text()
            test_file.unlink()
            return content == 'test'
        except:
            return False
    
    def check_security_features(self) -> bool:
        try:
            key = self.security.generate_secure_key()
            data = b"test data"
            encrypted = self.security.encrypt_data(data, key)
            decrypted = self.security.decrypt_data(encrypted, key)
            return decrypted == data
        except:
            return False

class UniversalAdapter:
    def __init__(self):
        self.bridge = CrossPlatformBridge()
        self.adapters = {
            'Windows': self.windows_adapter,
            'Darwin': self.macos_adapter,
            'Linux': self.linux_adapter
        }
        
    def execute(self, function: Callable, *args, **kwargs) -> Any:
        os_name = self.bridge.system_info.os_name
        
        if os_name in self.adapters:
            return self.adapters[os_name](function, *args, **kwargs)
        else:
            return self.generic_adapter(function, *args, **kwargs)
    
    def windows_adapter(self, function: Callable, *args, **kwargs) -> Any:
        kwargs['platform'] = 'windows'
        kwargs['path_sep'] = '\\'
        return function(*args, **kwargs)
    
    def macos_adapter(self, function: Callable, *args, **kwargs) -> Any:
        kwargs['platform'] = 'macos'
        kwargs['path_sep'] = '/'
        return function(*args, **kwargs)
    
    def linux_adapter(self, function: Callable, *args, **kwargs) -> Any:
        kwargs['platform'] = 'linux'
        kwargs['path_sep'] = '/'
        return function(*args, **kwargs)
    
    def generic_adapter(self, function: Callable, *args, **kwargs) -> Any:
        kwargs['platform'] = 'generic'
        kwargs['path_sep'] = os.sep
        return function(*args, **kwargs)

def main():
    print("QENEX Cross-Platform Compatibility Layer")
    print("=" * 50)
    
    bridge = CrossPlatformBridge()
    adapter = UniversalAdapter()
    
    print("\n1. System Information:")
    info = bridge.system_info
    print(f"  OS: {info.os_name} {info.os_version}")
    print(f"  Architecture: {info.architecture}")
    print(f"  Processor: {info.processor}")
    print(f"  Python: {info.python_version}")
    print(f"  Memory: {info.memory_gb:.2f} GB")
    print(f"  CPUs: {info.cpu_count}")
    print(f"  Distribution: {info.distribution}")
    
    print("\n2. Platform-Specific Paths:")
    print(f"  Config Directory: {bridge.filesystem.config_dir}")
    print(f"  Data Directory: {bridge.filesystem.data_dir}")
    print(f"  Temp Directory: {bridge.filesystem.get_temp_directory()}")
    print(f"  Executable Extension: {bridge.get_executable_extension()}")
    print(f"  Library Extension: {bridge.get_library_extension()}")
    
    print("\n3. Network Configuration:")
    print(f"  Primary IP: {bridge.network.get_primary_ip()}")
    interfaces = bridge.network.interfaces[:3]
    for interface in interfaces:
        addrs = ', '.join(interface['addresses'][:2])
        print(f"  Interface {interface['name']}: {addrs}")
    
    print("\n4. Compatibility Checks:")
    checks = bridge.ensure_compatibility()
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    print("\n5. Security Features:")
    token = bridge.security.generate_token(16)
    print(f"  Generated Token: {token}")
    
    key = bridge.security.generate_secure_key(32)
    test_data = b"Sensitive financial data"
    encrypted = bridge.security.encrypt_data(test_data, key)
    decrypted = bridge.security.decrypt_data(encrypted, key)
    print(f"  Encryption Test: {'✓' if decrypted == test_data else '✗'}")
    
    print("\n6. Process Management:")
    code, stdout, stderr = bridge.process.run_command(['echo', 'QENEX OS Test'])
    if code == 0:
        print(f"  Command Execution: ✓")
    else:
        print(f"  Command Execution: ✗")
    
    print("\n7. File System Operations:")
    test_file = bridge.filesystem.get_temp_directory() / 'qenex_compatibility_test.json'
    test_data = {
        'platform': info.os_name,
        'timestamp': time.time(),
        'status': 'operational'
    }
    
    if bridge.filesystem.write_file(str(test_file), json.dumps(test_data)):
        print(f"  File Write: ✓")
        content = bridge.filesystem.read_file(str(test_file))
        if content:
            print(f"  File Read: ✓")
            try:
                test_file.unlink()
            except:
                pass
        else:
            print(f"  File Read: ✗")
    else:
        print(f"  File Write: ✗")
    
    print("\n8. Universal Adapter:")
    def test_function(**kwargs):
        return f"Platform: {kwargs.get('platform', 'unknown')}, Sep: {kwargs.get('path_sep', '?')}"
    
    result = adapter.execute(test_function)
    print(f"  {result}")
    
    print("\n✅ Cross-platform compatibility layer operational")

if __name__ == "__main__":
    main()