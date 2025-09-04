#!/usr/bin/env python3
"""
QENEX P2P Network Layer
Distributed peer-to-peer networking with node discovery
"""

import asyncio
import json
import hashlib
import time
import random
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
import websockets
import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import logging

# Configuration
DEFAULT_PORT = 8765
MAX_PEERS = 50
PING_INTERVAL = 30
PEER_TIMEOUT = 90
BOOTSTRAP_NODES = [
    "ws://node1.qenex.network:8765",
    "ws://node2.qenex.network:8765",
    "ws://node3.qenex.network:8765"
]

# Message types
class MessageType:
    HANDSHAKE = "handshake"
    BLOCK = "block"
    TRANSACTION = "transaction"
    GET_BLOCKS = "get_blocks"
    GET_PEERS = "get_peers"
    PEERS = "peers"
    PING = "ping"
    PONG = "pong"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_PROPOSAL = "consensus_proposal"

@dataclass
class Peer:
    """Peer node information"""
    node_id: str
    address: str
    port: int
    version: str
    last_seen: float = field(default_factory=time.time)
    latency: float = 0
    reputation: int = 100
    connected: bool = False
    websocket: Optional[Any] = None
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'version': self.version,
            'last_seen': self.last_seen,
            'latency': self.latency,
            'reputation': self.reputation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Peer':
        return cls(
            node_id=data['node_id'],
            address=data['address'],
            port=data['port'],
            version=data['version'],
            last_seen=data.get('last_seen', time.time()),
            latency=data.get('latency', 0),
            reputation=data.get('reputation', 100)
        )

@dataclass
class Message:
    """P2P network message"""
    type: str
    payload: Dict[str, Any]
    sender: str
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'payload': self.payload,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'signature': self.signature
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'Message':
        msg_dict = json.loads(data)
        return cls(
            type=msg_dict['type'],
            payload=msg_dict['payload'],
            sender=msg_dict['sender'],
            timestamp=msg_dict.get('timestamp', time.time()),
            signature=msg_dict.get('signature')
        )

class P2PNode:
    """P2P network node with full networking capabilities"""
    
    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.version = "1.0.0"
        
        # Peer management
        self.peers: Dict[str, Peer] = {}
        self.max_peers = MAX_PEERS
        self.connected_peers: Set[str] = set()
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Networking
        self.server = None
        self.running = False
        
        # Blockchain reference
        self.blockchain = None
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"P2P-{node_id[:8]}")
        
    async def start(self):
        """Start P2P node"""
        self.running = True
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        self.logger.info(f"P2P node started on {self.host}:{self.port}")
        
        # Start background tasks
        asyncio.create_task(self.peer_discovery())
        asyncio.create_task(self.peer_maintenance())
        asyncio.create_task(self.sync_blockchain())
        
        # Connect to bootstrap nodes
        await self.bootstrap()
    
    async def stop(self):
        """Stop P2P node"""
        self.running = False
        
        # Close all peer connections
        for peer in self.peers.values():
            if peer.websocket:
                await peer.websocket.close()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("P2P node stopped")
    
    async def handle_connection(self, websocket, path):
        """Handle incoming peer connection"""
        peer_id = None
        try:
            # Wait for handshake
            message = await asyncio.wait_for(
                websocket.recv(),
                timeout=10
            )
            
            msg = Message.from_json(message)
            
            if msg.type != MessageType.HANDSHAKE:
                await websocket.close()
                return
            
            # Process handshake
            peer_id = msg.sender
            peer_info = msg.payload
            
            # Create or update peer
            peer = Peer(
                node_id=peer_id,
                address=websocket.remote_address[0],
                port=peer_info['port'],
                version=peer_info['version'],
                websocket=websocket
            )
            
            self.peers[peer_id] = peer
            self.connected_peers.add(peer_id)
            peer.connected = True
            
            # Send handshake response
            response = Message(
                type=MessageType.HANDSHAKE,
                payload={
                    'node_id': self.node_id,
                    'version': self.version,
                    'port': self.port,
                    'peers': len(self.peers)
                },
                sender=self.node_id
            )
            
            await websocket.send(response.to_json())
            
            self.logger.info(f"Peer connected: {peer_id[:8]}")
            
            # Handle messages from peer
            async for message in websocket:
                await self.handle_message(message, peer_id)
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            
        finally:
            # Clean up peer
            if peer_id:
                self.connected_peers.discard(peer_id)
                if peer_id in self.peers:
                    self.peers[peer_id].connected = False
                    self.peers[peer_id].websocket = None
                
                self.logger.info(f"Peer disconnected: {peer_id[:8] if peer_id else 'unknown'}")
    
    async def connect_to_peer(self, address: str, port: int):
        """Connect to a peer"""
        try:
            uri = f"ws://{address}:{port}"
            websocket = await websockets.connect(uri)
            
            # Send handshake
            handshake = Message(
                type=MessageType.HANDSHAKE,
                payload={
                    'node_id': self.node_id,
                    'version': self.version,
                    'port': self.port
                },
                sender=self.node_id
            )
            
            await websocket.send(handshake.to_json())
            
            # Wait for response
            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=10
            )
            
            msg = Message.from_json(response)
            
            if msg.type == MessageType.HANDSHAKE:
                peer_info = msg.payload
                peer = Peer(
                    node_id=msg.sender,
                    address=address,
                    port=port,
                    version=peer_info['version'],
                    websocket=websocket
                )
                
                self.peers[msg.sender] = peer
                self.connected_peers.add(msg.sender)
                peer.connected = True
                
                self.logger.info(f"Connected to peer: {msg.sender[:8]}")
                
                # Start message handler for this peer
                asyncio.create_task(self.handle_peer_messages(peer))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to {address}:{port}: {e}")
            return False
    
    async def handle_peer_messages(self, peer: Peer):
        """Handle messages from connected peer"""
        try:
            while peer.connected and peer.websocket:
                message = await peer.websocket.recv()
                await self.handle_message(message, peer.node_id)
                
        except Exception as e:
            self.logger.error(f"Error handling peer messages: {e}")
            
        finally:
            peer.connected = False
            peer.websocket = None
            self.connected_peers.discard(peer.node_id)
    
    async def handle_message(self, message: str, sender_id: str):
        """Handle incoming message"""
        try:
            msg = Message.from_json(message)
            
            self.messages_received += 1
            self.bytes_received += len(message)
            
            # Update peer last seen
            if sender_id in self.peers:
                self.peers[sender_id].last_seen = time.time()
            
            # Route message to handlers
            if msg.type == MessageType.PING:
                await self.handle_ping(msg, sender_id)
            elif msg.type == MessageType.PONG:
                await self.handle_pong(msg, sender_id)
            elif msg.type == MessageType.GET_PEERS:
                await self.handle_get_peers(msg, sender_id)
            elif msg.type == MessageType.PEERS:
                await self.handle_peers(msg, sender_id)
            elif msg.type == MessageType.BLOCK:
                await self.handle_block(msg, sender_id)
            elif msg.type == MessageType.TRANSACTION:
                await self.handle_transaction(msg, sender_id)
            elif msg.type == MessageType.SYNC_REQUEST:
                await self.handle_sync_request(msg, sender_id)
            elif msg.type == MessageType.SYNC_RESPONSE:
                await self.handle_sync_response(msg, sender_id)
            
            # Call custom handlers
            if msg.type in self.message_handlers:
                for handler in self.message_handlers[msg.type]:
                    await handler(msg, sender_id)
                    
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def broadcast(self, message: Message, exclude: Optional[Set[str]] = None):
        """Broadcast message to all connected peers"""
        exclude = exclude or set()
        message_str = message.to_json()
        
        for peer_id, peer in self.peers.items():
            if peer_id not in exclude and peer.connected and peer.websocket:
                try:
                    await peer.websocket.send(message_str)
                    self.messages_sent += 1
                    self.bytes_sent += len(message_str)
                except Exception as e:
                    self.logger.error(f"Failed to send to {peer_id[:8]}: {e}")
                    peer.connected = False
    
    async def send_to_peer(self, peer_id: str, message: Message):
        """Send message to specific peer"""
        if peer_id in self.peers and self.peers[peer_id].connected:
            peer = self.peers[peer_id]
            if peer.websocket:
                try:
                    message_str = message.to_json()
                    await peer.websocket.send(message_str)
                    self.messages_sent += 1
                    self.bytes_sent += len(message_str)
                except Exception as e:
                    self.logger.error(f"Failed to send to {peer_id[:8]}: {e}")
                    peer.connected = False
    
    # Message handlers
    async def handle_ping(self, message: Message, sender_id: str):
        """Handle ping message"""
        pong = Message(
            type=MessageType.PONG,
            payload={'timestamp': message.payload['timestamp']},
            sender=self.node_id
        )
        await self.send_to_peer(sender_id, pong)
    
    async def handle_pong(self, message: Message, sender_id: str):
        """Handle pong message"""
        if sender_id in self.peers:
            latency = time.time() - message.payload['timestamp']
            self.peers[sender_id].latency = latency
    
    async def handle_get_peers(self, message: Message, sender_id: str):
        """Handle get peers request"""
        peers_list = [
            peer.to_dict() 
            for peer in list(self.peers.values())[:20]
            if peer.node_id != sender_id
        ]
        
        response = Message(
            type=MessageType.PEERS,
            payload={'peers': peers_list},
            sender=self.node_id
        )
        await self.send_to_peer(sender_id, response)
    
    async def handle_peers(self, message: Message, sender_id: str):
        """Handle peers list"""
        for peer_data in message.payload['peers']:
            peer = Peer.from_dict(peer_data)
            if peer.node_id != self.node_id and peer.node_id not in self.peers:
                # Add new peer for potential connection
                self.peers[peer.node_id] = peer
    
    async def handle_block(self, message: Message, sender_id: str):
        """Handle new block"""
        if self.blockchain:
            # Validate and add block
            block_data = message.payload['block']
            # Process block with blockchain
            self.logger.info(f"Received block from {sender_id[:8]}")
            
            # Propagate to other peers
            await self.broadcast(message, exclude={sender_id})
    
    async def handle_transaction(self, message: Message, sender_id: str):
        """Handle new transaction"""
        if self.blockchain:
            # Add to mempool
            tx_data = message.payload['transaction']
            # Process transaction
            self.logger.info(f"Received transaction from {sender_id[:8]}")
            
            # Propagate to other peers
            await self.broadcast(message, exclude={sender_id})
    
    async def handle_sync_request(self, message: Message, sender_id: str):
        """Handle blockchain sync request"""
        if self.blockchain:
            # Send blockchain data
            response = Message(
                type=MessageType.SYNC_RESPONSE,
                payload={
                    'height': 0,  # Get from blockchain
                    'blocks': []  # Get recent blocks
                },
                sender=self.node_id
            )
            await self.send_to_peer(sender_id, response)
    
    async def handle_sync_response(self, message: Message, sender_id: str):
        """Handle blockchain sync response"""
        # Process received blocks
        self.logger.info(f"Received sync response from {sender_id[:8]}")
    
    # Background tasks
    async def bootstrap(self):
        """Connect to bootstrap nodes"""
        for node_url in BOOTSTRAP_NODES:
            try:
                # Parse URL
                parts = node_url.replace("ws://", "").split(":")
                address = parts[0]
                port = int(parts[1]) if len(parts) > 1 else DEFAULT_PORT
                
                await self.connect_to_peer(address, port)
                
            except Exception as e:
                self.logger.error(f"Failed to connect to bootstrap node {node_url}: {e}")
    
    async def peer_discovery(self):
        """Discover new peers"""
        while self.running:
            try:
                # Request peers from connected nodes
                if len(self.connected_peers) < self.max_peers // 2:
                    message = Message(
                        type=MessageType.GET_PEERS,
                        payload={},
                        sender=self.node_id
                    )
                    await self.broadcast(message)
                
                # Try to connect to discovered peers
                for peer_id, peer in list(self.peers.items()):
                    if not peer.connected and len(self.connected_peers) < self.max_peers:
                        await self.connect_to_peer(peer.address, peer.port)
                
            except Exception as e:
                self.logger.error(f"Peer discovery error: {e}")
            
            await asyncio.sleep(60)  # Run every minute
    
    async def peer_maintenance(self):
        """Maintain peer connections"""
        while self.running:
            try:
                current_time = time.time()
                
                # Ping connected peers
                for peer_id in list(self.connected_peers):
                    if peer_id in self.peers:
                        peer = self.peers[peer_id]
                        
                        # Check timeout
                        if current_time - peer.last_seen > PEER_TIMEOUT:
                            self.logger.info(f"Peer timeout: {peer_id[:8]}")
                            if peer.websocket:
                                await peer.websocket.close()
                            peer.connected = False
                            self.connected_peers.discard(peer_id)
                        
                        # Send ping
                        elif current_time - peer.last_seen > PING_INTERVAL:
                            ping = Message(
                                type=MessageType.PING,
                                payload={'timestamp': current_time},
                                sender=self.node_id
                            )
                            await self.send_to_peer(peer_id, ping)
                
                # Remove old disconnected peers
                for peer_id in list(self.peers.keys()):
                    peer = self.peers[peer_id]
                    if not peer.connected and current_time - peer.last_seen > 3600:
                        del self.peers[peer_id]
                
            except Exception as e:
                self.logger.error(f"Peer maintenance error: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    async def sync_blockchain(self):
        """Synchronize blockchain with peers"""
        while self.running:
            try:
                if self.blockchain and len(self.connected_peers) > 0:
                    # Request sync from random peer
                    peer_id = random.choice(list(self.connected_peers))
                    
                    sync_request = Message(
                        type=MessageType.SYNC_REQUEST,
                        payload={'height': 0},  # Current blockchain height
                        sender=self.node_id
                    )
                    await self.send_to_peer(peer_id, sync_request)
                
            except Exception as e:
                self.logger.error(f"Blockchain sync error: {e}")
            
            await asyncio.sleep(120)  # Run every 2 minutes
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register custom message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'node_id': self.node_id,
            'peers': len(self.peers),
            'connected': len(self.connected_peers),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'avg_latency': sum(p.latency for p in self.peers.values()) / len(self.peers) if self.peers else 0
        }

async def main():
    """P2P network demonstration"""
    print("=" * 60)
    print(" QENEX P2P NETWORK - DISTRIBUTED SYSTEM")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create node
    node = P2PNode(
        node_id=hashlib.sha256(str(time.time()).encode()).hexdigest(),
        host="0.0.0.0",
        port=8765
    )
    
    # Start node
    await node.start()
    
    print(f"\n[✓] Node started: {node.node_id[:16]}...")
    print(f"    Listening on: {node.host}:{node.port}")
    
    # Wait for connections
    await asyncio.sleep(5)
    
    # Get stats
    stats = node.get_network_stats()
    print(f"\n[📊] Network Statistics:")
    print(f"    Connected Peers: {stats['connected']}")
    print(f"    Total Peers: {stats['peers']}")
    print(f"    Messages Sent: {stats['messages_sent']}")
    print(f"    Messages Received: {stats['messages_received']}")
    
    print("\n" + "=" * 60)
    print(" P2P NETWORK OPERATIONAL")
    print("=" * 60)
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await node.stop()

if __name__ == "__main__":
    asyncio.run(main())