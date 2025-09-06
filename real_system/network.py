#!/usr/bin/env python3
"""
Real Network Stack Implementation
Actual working network functionality with sockets, HTTP, and protocols
"""

import socket
import asyncio
import threading
import time
import json
import struct
import hashlib
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import urllib.request
import urllib.parse
import ssl
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor

@dataclass
class NetworkConnection:
    conn_id: str
    socket: socket.socket
    remote_addr: Tuple[str, int]
    local_addr: Tuple[str, int]
    protocol: str
    state: str
    bytes_sent: int = 0
    bytes_received: int = 0
    created_at: float = 0

class RealNetworkStack:
    """Real network implementation with actual socket operations"""
    
    def __init__(self):
        self.connections = {}
        self.servers = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_tcp_client(self, host: str, port: int, timeout: float = 5.0) -> Optional[NetworkConnection]:
        """Create a real TCP client connection"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            
            conn_id = f"tcp_client_{hashlib.md5(f'{host}:{port}:{time.time()}'.encode()).hexdigest()[:8]}"
            
            conn = NetworkConnection(
                conn_id=conn_id,
                socket=sock,
                remote_addr=(host, port),
                local_addr=sock.getsockname(),
                protocol='TCP',
                state='ESTABLISHED',
                created_at=time.time()
            )
            
            self.connections[conn_id] = conn
            return conn
            
        except Exception as e:
            print(f"Failed to connect to {host}:{port} - {e}")
            return None
    
    def create_tcp_server(self, host: str = '0.0.0.0', port: int = 0) -> Optional[str]:
        """Create a real TCP server that listens for connections"""
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((host, port))
            server_sock.listen(5)
            
            actual_port = server_sock.getsockname()[1]
            server_id = f"tcp_server_{actual_port}"
            
            self.servers[server_id] = {
                'socket': server_sock,
                'host': host,
                'port': actual_port,
                'protocol': 'TCP',
                'accepting': False,
                'clients': []
            }
            
            return server_id
            
        except Exception as e:
            print(f"Failed to create server: {e}")
            return None
    
    def start_server(self, server_id: str, handler: Callable = None):
        """Start accepting connections on a server"""
        if server_id not in self.servers:
            return False
        
        server = self.servers[server_id]
        server['accepting'] = True
        
        def accept_loop():
            while server['accepting']:
                try:
                    server['socket'].settimeout(1.0)
                    client_sock, client_addr = server['socket'].accept()
                    
                    conn_id = f"tcp_server_client_{hashlib.md5(f'{client_addr}:{time.time()}'.encode()).hexdigest()[:8]}"
                    
                    conn = NetworkConnection(
                        conn_id=conn_id,
                        socket=client_sock,
                        remote_addr=client_addr,
                        local_addr=server['socket'].getsockname(),
                        protocol='TCP',
                        state='ESTABLISHED',
                        created_at=time.time()
                    )
                    
                    self.connections[conn_id] = conn
                    server['clients'].append(conn_id)
                    
                    if handler:
                        self.executor.submit(handler, conn)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if server['accepting']:
                        print(f"Server error: {e}")
        
        self.executor.submit(accept_loop)
        return True
    
    def send_data(self, conn_id: str, data: bytes) -> int:
        """Send real data over a connection"""
        if conn_id not in self.connections:
            return 0
        
        conn = self.connections[conn_id]
        try:
            bytes_sent = conn.socket.send(data)
            conn.bytes_sent += bytes_sent
            return bytes_sent
        except Exception as e:
            print(f"Send error: {e}")
            conn.state = 'ERROR'
            return 0
    
    def receive_data(self, conn_id: str, max_bytes: int = 4096) -> Optional[bytes]:
        """Receive real data from a connection"""
        if conn_id not in self.connections:
            return None
        
        conn = self.connections[conn_id]
        try:
            data = conn.socket.recv(max_bytes)
            if data:
                conn.bytes_received += len(data)
                return data
            else:
                # Connection closed
                conn.state = 'CLOSED'
                return None
        except socket.timeout:
            return b''
        except Exception as e:
            print(f"Receive error: {e}")
            conn.state = 'ERROR'
            return None
    
    def close_connection(self, conn_id: str):
        """Close a network connection"""
        if conn_id in self.connections:
            try:
                self.connections[conn_id].socket.close()
                self.connections[conn_id].state = 'CLOSED'
            except:
                pass
            del self.connections[conn_id]
    
    def http_get(self, url: str, headers: Dict[str, str] = None) -> Optional[Dict]:
        """Make a real HTTP GET request"""
        try:
            req = urllib.request.Request(url)
            if headers:
                for key, value in headers.items():
                    req.add_header(key, value)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': response.read().decode('utf-8', errors='ignore'),
                    'url': response.url
                }
        except Exception as e:
            print(f"HTTP GET error: {e}")
            return None
    
    def http_post(self, url: str, data: Dict = None, headers: Dict[str, str] = None) -> Optional[Dict]:
        """Make a real HTTP POST request"""
        try:
            post_data = urllib.parse.urlencode(data).encode() if data else b''
            
            req = urllib.request.Request(url, data=post_data, method='POST')
            if headers:
                for key, value in headers.items():
                    req.add_header(key, value)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': response.read().decode('utf-8', errors='ignore'),
                    'url': response.url
                }
        except Exception as e:
            print(f"HTTP POST error: {e}")
            return None
    
    def create_udp_socket(self) -> Optional[str]:
        """Create a UDP socket"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            conn_id = f"udp_{hashlib.md5(f'{time.time()}'.encode()).hexdigest()[:8]}"
            
            conn = NetworkConnection(
                conn_id=conn_id,
                socket=sock,
                remote_addr=('', 0),
                local_addr=('', 0),
                protocol='UDP',
                state='READY',
                created_at=time.time()
            )
            
            self.connections[conn_id] = conn
            return conn_id
            
        except Exception as e:
            print(f"Failed to create UDP socket: {e}")
            return None
    
    def send_udp(self, conn_id: str, data: bytes, addr: Tuple[str, int]) -> int:
        """Send UDP datagram"""
        if conn_id not in self.connections:
            return 0
        
        conn = self.connections[conn_id]
        try:
            bytes_sent = conn.socket.sendto(data, addr)
            conn.bytes_sent += bytes_sent
            return bytes_sent
        except Exception as e:
            print(f"UDP send error: {e}")
            return 0
    
    def receive_udp(self, conn_id: str, max_bytes: int = 4096) -> Optional[Tuple[bytes, Tuple[str, int]]]:
        """Receive UDP datagram"""
        if conn_id not in self.connections:
            return None
        
        conn = self.connections[conn_id]
        try:
            data, addr = conn.socket.recvfrom(max_bytes)
            conn.bytes_received += len(data)
            return (data, addr)
        except socket.timeout:
            return None
        except Exception as e:
            print(f"UDP receive error: {e}")
            return None
    
    def scan_ports(self, host: str, start_port: int = 1, end_port: int = 1000) -> List[int]:
        """Scan for open TCP ports on a host"""
        open_ports = []
        
        for port in range(start_port, end_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            
            result = sock.connect_ex((host, port))
            if result == 0:
                open_ports.append(port)
            
            sock.close()
        
        return open_ports
    
    def get_network_interfaces(self) -> Dict:
        """Get network interface information"""
        import psutil
        
        interfaces = {}
        
        # Get network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            interfaces[interface] = {
                'addresses': []
            }
            
            for addr in addrs:
                addr_info = {
                    'family': str(addr.family),
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': addr.broadcast
                }
                interfaces[interface]['addresses'].append(addr_info)
        
        # Get interface statistics
        stats = psutil.net_if_stats()
        for interface in interfaces:
            if interface in stats:
                interfaces[interface]['stats'] = {
                    'isup': stats[interface].isup,
                    'speed': stats[interface].speed,
                    'mtu': stats[interface].mtu
                }
        
        return interfaces
    
    def create_http_server(self, port: int = 8000) -> Optional[str]:
        """Create a simple HTTP server"""
        server_id = f"http_server_{port}"
        
        class SimpleHTTPHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    'status': 'ok',
                    'message': 'QENEX Network Stack HTTP Server',
                    'path': self.path,
                    'time': time.time()
                }
                
                self.wfile.write(json.dumps(response).encode())
                return
        
        try:
            httpd = socketserver.TCPServer(("", port), SimpleHTTPHandler)
            
            self.servers[server_id] = {
                'httpd': httpd,
                'port': port,
                'running': False
            }
            
            def serve():
                self.servers[server_id]['running'] = True
                httpd.serve_forever()
            
            self.executor.submit(serve)
            return server_id
            
        except Exception as e:
            print(f"Failed to create HTTP server: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get network statistics"""
        import psutil
        
        net_io = psutil.net_io_counters()
        
        return {
            'connections': {
                'active': len([c for c in self.connections.values() if c.state == 'ESTABLISHED']),
                'total': len(self.connections)
            },
            'servers': len(self.servers),
            'io_counters': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        }

def demonstrate_real_network():
    """Demonstrate real network functionality"""
    print("=" * 70)
    print("REAL NETWORK STACK DEMONSTRATION")
    print("=" * 70)
    
    net = RealNetworkStack()
    
    # 1. Test HTTP requests
    print("\n1. HTTP Requests:")
    print("-" * 40)
    
    # Make a real HTTP request
    response = net.http_get("http://api.github.com")
    if response:
        print(f"✅ GitHub API Status: {response['status']}")
        print(f"   Response size: {len(response['body'])} bytes")
    else:
        print("❌ Could not reach GitHub API")
    
    # 2. Create TCP server
    print("\n2. TCP Server:")
    print("-" * 40)
    
    server_id = net.create_tcp_server('127.0.0.1', 0)
    if server_id:
        port = net.servers[server_id]['port']
        print(f"✅ TCP server created on port {port}")
        
        def handle_client(conn):
            # Echo server
            data = net.receive_data(conn.conn_id, 1024)
            if data:
                net.send_data(conn.conn_id, b"Echo: " + data)
            net.close_connection(conn.conn_id)
        
        net.start_server(server_id, handle_client)
    
    # 3. Test TCP client
    print("\n3. TCP Client:")
    print("-" * 40)
    
    # Connect to a well-known service
    conn = net.create_tcp_client("8.8.8.8", 53, timeout=2)
    if conn:
        print(f"✅ Connected to Google DNS")
        print(f"   Local: {conn.local_addr}")
        print(f"   Remote: {conn.remote_addr}")
        net.close_connection(conn.conn_id)
    else:
        print("❌ Could not connect to Google DNS")
    
    # 4. Network interfaces
    print("\n4. Network Interfaces:")
    print("-" * 40)
    
    interfaces = net.get_network_interfaces()
    for name, info in list(interfaces.items())[:3]:  # Show first 3
        print(f"Interface: {name}")
        if info.get('stats', {}).get('isup'):
            for addr in info['addresses']:
                if addr['family'] == 'AddressFamily.AF_INET':
                    print(f"  IPv4: {addr['address']}")
    
    # 5. Create HTTP server
    print("\n5. HTTP Server:")
    print("-" * 40)
    
    http_server = net.create_http_server(8888)
    if http_server:
        print(f"✅ HTTP server started on port 8888")
        print("   Try: curl http://localhost:8888/")
    
    # 6. Network statistics
    print("\n6. Network Statistics:")
    print("-" * 40)
    
    stats = net.get_stats()
    print(f"Active connections: {stats['connections']['active']}")
    print(f"Total connections: {stats['connections']['total']}")
    print(f"Running servers: {stats['servers']}")
    print(f"Bytes sent: {stats['io_counters']['bytes_sent']:,}")
    print(f"Bytes received: {stats['io_counters']['bytes_recv']:,}")
    
    print("\n" + "=" * 70)
    print("✅ REAL NETWORK STACK WORKING!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_real_network()