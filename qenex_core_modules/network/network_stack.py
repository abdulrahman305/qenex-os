#!/usr/bin/env python3
"""
QENEX OS Network Stack - Network connectivity and blockchain integration
"""

import asyncio
import hashlib
import json
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import aiohttp

@dataclass
class Connection:
    """Represents a network connection"""
    conn_id: str
    remote_addr: str
    remote_port: int
    protocol: str
    status: str
    created_at: float
    bytes_sent: int = 0
    bytes_received: int = 0

class NetworkStack:
    """Network connectivity manager"""
    
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self.blockchain_nodes: List[str] = [
            "https://eth-mainnet.g.alchemy.com/v2/demo",
            "https://eth-sepolia.g.alchemy.com/v2/demo"
        ]
        self.p2p_peers: Set[str] = set()
        self.running = False
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Start network stack"""
        print("🌐 Starting Network Stack...")
        self.running = True
        self.session = aiohttp.ClientSession()
        
        # Start network services
        asyncio.create_task(self.connection_monitor())
        asyncio.create_task(self.blockchain_sync())
        asyncio.create_task(self.p2p_discovery())
        
        print("✅ Network Stack started")
    
    async def stop(self):
        """Stop network stack"""
        self.running = False
        
        # Close all connections
        for conn_id in list(self.connections.keys()):
            await self.disconnect(conn_id)
        
        if self.session:
            await self.session.close()
        
        print("🌐 Network Stack stopped")
    
    async def connect(self, address: str, port: int = 80, protocol: str = "tcp") -> Optional[str]:
        """Establish a network connection"""
        try:
            conn_id = hashlib.sha256(f"{address}:{port}:{time.time()}".encode()).hexdigest()[:16]
            
            connection = Connection(
                conn_id=conn_id,
                remote_addr=address,
                remote_port=port,
                protocol=protocol,
                status="connecting",
                created_at=time.time()
            )
            
            self.connections[conn_id] = connection
            
            # Simulate connection establishment
            await asyncio.sleep(0.5)
            
            connection.status = "connected"
            print(f"✅ Connected to {address}:{port}")
            
            return conn_id
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return None
    
    async def disconnect(self, conn_id: str) -> bool:
        """Close a network connection"""
        if conn_id in self.connections:
            self.connections[conn_id].status = "closed"
            del self.connections[conn_id]
            return True
        return False
    
    async def send_data(self, conn_id: str, data: bytes) -> bool:
        """Send data through a connection"""
        if conn_id not in self.connections:
            return False
        
        connection = self.connections[conn_id]
        if connection.status != "connected":
            return False
        
        # Simulate sending data
        connection.bytes_sent += len(data)
        await asyncio.sleep(0.01)  # Simulate network latency
        
        return True
    
    async def receive_data(self, conn_id: str, max_bytes: int = 4096) -> Optional[bytes]:
        """Receive data from a connection"""
        if conn_id not in self.connections:
            return None
        
        connection = self.connections[conn_id]
        if connection.status != "connected":
            return None
        
        # Simulate receiving data
        data = b"received_data" * (max_bytes // 13)
        connection.bytes_received += len(data)
        
        return data[:max_bytes]
    
    async def blockchain_request(self, method: str, params: List = None) -> Optional[Dict]:
        """Make a blockchain RPC request"""
        if not self.session:
            return None
        
        for node in self.blockchain_nodes:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or [],
                    "id": 1
                }
                
                async with self.session.post(
                    node,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("result")
            except Exception:
                continue
        
        return None
    
    async def get_block_number(self) -> Optional[int]:
        """Get current blockchain block number"""
        result = await self.blockchain_request("eth_blockNumber")
        if result:
            return int(result, 16)
        return None
    
    async def get_gas_price(self) -> Optional[int]:
        """Get current gas price"""
        result = await self.blockchain_request("eth_gasPrice")
        if result:
            return int(result, 16)
        return None
    
    def add_p2p_peer(self, peer_address: str):
        """Add a P2P peer"""
        self.p2p_peers.add(peer_address)
    
    def remove_p2p_peer(self, peer_address: str):
        """Remove a P2P peer"""
        self.p2p_peers.discard(peer_address)
    
    async def broadcast_to_peers(self, message: Dict):
        """Broadcast message to all P2P peers"""
        tasks = []
        for peer in self.p2p_peers:
            tasks.append(self.send_to_peer(peer, message))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_peer(self, peer_address: str, message: Dict):
        """Send message to a specific peer"""
        try:
            # Simulate P2P message sending
            conn_id = await self.connect(peer_address, 8545, "p2p")
            if conn_id:
                data = json.dumps(message).encode()
                await self.send_data(conn_id, data)
                await self.disconnect(conn_id)
        except Exception as e:
            print(f"Failed to send to peer {peer_address}: {e}")
    
    async def connection_monitor(self):
        """Monitor network connections"""
        while self.running:
            await asyncio.sleep(10)
            
            # Clean up stale connections
            now = time.time()
            stale_connections = [
                conn_id for conn_id, conn in self.connections.items()
                if conn.status == "connected" and now - conn.created_at > 300  # 5 minutes
            ]
            
            for conn_id in stale_connections:
                await self.disconnect(conn_id)
    
    async def blockchain_sync(self):
        """Sync with blockchain network"""
        while self.running:
            await asyncio.sleep(30)
            
            block_number = await self.get_block_number()
            if block_number:
                print(f"📦 Blockchain height: {block_number}")
            
            gas_price = await self.get_gas_price()
            if gas_price:
                print(f"⛽ Gas price: {gas_price / 1e9:.2f} Gwei")
    
    async def p2p_discovery(self):
        """Discover P2P peers"""
        while self.running:
            await asyncio.sleep(60)
            
            # Simulate peer discovery
            if len(self.p2p_peers) < 5:
                new_peer = f"peer{len(self.p2p_peers)}.qenex.network"
                self.add_p2p_peer(new_peer)
                print(f"🤝 Discovered new peer: {new_peer}")
    
    def get_bandwidth_usage(self) -> Dict:
        """Get bandwidth usage statistics"""
        total_sent = sum(conn.bytes_sent for conn in self.connections.values())
        total_received = sum(conn.bytes_received for conn in self.connections.values())
        
        return {
            "bytes_sent": total_sent,
            "bytes_received": total_received,
            "total": total_sent + total_received
        }
    
    def get_status(self) -> Dict:
        """Get network stack status"""
        return {
            "status": "running" if self.running else "stopped",
            "connections": len(self.connections),
            "active_connections": sum(1 for c in self.connections.values() if c.status == "connected"),
            "blockchain_nodes": len(self.blockchain_nodes),
            "p2p_peers": len(self.p2p_peers),
            "bandwidth": self.get_bandwidth_usage()
        }

# Singleton instance
network_stack = NetworkStack()

async def main():
    """Main function for testing"""
    await network_stack.start()
    
    # Test connection
    conn_id = await network_stack.connect("example.com", 443, "tcp")
    if conn_id:
        print(f"Connection established: {conn_id}")
        
        # Send data
        await network_stack.send_data(conn_id, b"Hello, World!")
        
        # Receive data
        data = await network_stack.receive_data(conn_id)
        print(f"Received: {data[:50]}...")
        
        # Disconnect
        await network_stack.disconnect(conn_id)
    
    # Test blockchain
    block = await network_stack.get_block_number()
    print(f"Current block: {block}")
    
    gas = await network_stack.get_gas_price()
    if gas:
        print(f"Gas price: {gas / 1e9:.2f} Gwei")
    
    # Test P2P
    network_stack.add_p2p_peer("peer1.qenex.network")
    await network_stack.broadcast_to_peers({"type": "ping", "timestamp": time.time()})
    
    # Get status
    status = network_stack.get_status()
    print(f"Network status: {json.dumps(status, indent=2)}")
    
    await network_stack.stop()

if __name__ == "__main__":
    asyncio.run(main())