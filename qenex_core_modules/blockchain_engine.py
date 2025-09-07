#!/usr/bin/env python3
"""
QENEX Blockchain Engine
High-performance distributed ledger with Byzantine fault tolerance
"""

import hashlib
import json
import time
import asyncio
import secrets
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
from decimal import Decimal
import logging
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

logger = logging.getLogger(__name__)

@dataclass
class Block:
    """Immutable block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    merkle_root: str = ""
    validator: str = ""
    signature: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
        
        # Create leaf nodes
        hashes = []
        for tx in self.transactions:
            tx_string = json.dumps(tx, sort_keys=True)
            tx_hash = hashlib.sha256(tx_string.encode()).hexdigest()
            hashes.append(tx_hash)
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])  # Duplicate last hash if odd number
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_hashes.append(new_hash)
            hashes = new_hashes
        
        return hashes[0]

@dataclass
class Node:
    """Blockchain network node"""
    node_id: str
    public_key: str
    stake: Decimal = Decimal("0")
    reputation: float = 1.0
    is_validator: bool = False
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated_blocks: int = 0
    failed_validations: int = 0

class ConsensusProtocol:
    """Byzantine Fault Tolerant consensus mechanism"""
    
    def __init__(self, min_validators: int = 4):
        self.min_validators = min_validators
        self.voting_threshold = 0.67  # 2/3 majority
        self.timeout = 30  # seconds
        self.view_number = 0
        self.prepared_blocks: Dict[str, Set[str]] = {}
        self.committed_blocks: Dict[str, Set[str]] = {}
        
    def propose_block(self, block: Block, validators: List[Node]) -> bool:
        """Propose a block for validation"""
        if len(validators) < self.min_validators:
            logger.warning(f"Insufficient validators: {len(validators)}/{self.min_validators}")
            return False
        
        block_hash = block.calculate_hash()
        self.prepared_blocks[block_hash] = set()
        
        # Simulate voting (in production, this would be network communication)
        required_votes = int(len(validators) * self.voting_threshold)
        
        for validator in validators:
            if validator.is_validator:
                # Validator votes based on reputation and stake
                vote_probability = validator.reputation * min(float(validator.stake) / 1000, 1.0)
                if secrets.random() < vote_probability:
                    self.prepared_blocks[block_hash].add(validator.node_id)
        
        if len(self.prepared_blocks[block_hash]) >= required_votes:
            self.committed_blocks[block_hash] = self.prepared_blocks[block_hash]
            return True
        
        return False

class BlockchainEngine:
    """High-performance blockchain engine with advanced features"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("./blockchain_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Blockchain storage
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.transaction_pool: Dict[str, Dict[str, Any]] = {}
        
        # Network nodes
        self.nodes: Dict[str, Node] = {}
        self.validators: List[Node] = []
        
        # Consensus
        self.consensus = ConsensusProtocol()
        self.block_time = 10  # seconds
        self.max_block_size = 1_000_000  # bytes
        self.difficulty = 4  # Number of leading zeros in hash
        
        # Database
        self.db_path = self.data_dir / "blockchain.db"
        self._init_database()
        
        # Threading
        self.mining_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Cryptography
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # State management
        self.state_db: Dict[str, Any] = {}
        self.state_root = ""
        
        # Initialize genesis block
        self._create_genesis_block()
        
        logger.info("Blockchain Engine initialized")
    
    def _init_database(self) -> None:
        """Initialize blockchain database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                block_index INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                hash TEXT UNIQUE NOT NULL,
                previous_hash TEXT NOT NULL,
                merkle_root TEXT NOT NULL,
                nonce INTEGER NOT NULL,
                validator TEXT,
                signature TEXT,
                transactions TEXT NOT NULL
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                tx_hash TEXT PRIMARY KEY,
                block_index INTEGER,
                sender TEXT,
                recipient TEXT,
                amount TEXT,
                fee TEXT,
                timestamp REAL NOT NULL,
                signature TEXT,
                status TEXT,
                data TEXT,
                FOREIGN KEY (block_index) REFERENCES blocks(block_index)
            )
        ''')
        
        # Nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                stake TEXT,
                reputation REAL,
                is_validator INTEGER,
                last_seen TEXT,
                validated_blocks INTEGER,
                failed_validations INTEGER
            )
        ''')
        
        # State table for smart contract storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                block_height INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_block ON transactions(block_index)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_recipient ON transactions(recipient)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_height ON state(block_height)')
        
        conn.commit()
        conn.close()
    
    def _create_genesis_block(self) -> None:
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            nonce=0
        )
        genesis_block.merkle_root = genesis_block.calculate_merkle_root()
        genesis_block.hash = genesis_block.calculate_hash()
        
        self.chain.append(genesis_block)
        self._persist_block(genesis_block)
        logger.info("Genesis block created")
    
    def _persist_block(self, block: Block) -> None:
        """Persist block to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO blocks 
            (block_index, timestamp, hash, previous_hash, merkle_root, 
             nonce, validator, signature, transactions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            block.index,
            block.timestamp,
            block.hash,
            block.previous_hash,
            block.merkle_root,
            block.nonce,
            block.validator,
            block.signature,
            json.dumps(block.transactions)
        ))
        
        # Persist individual transactions
        for tx in block.transactions:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (tx_hash, block_index, sender, recipient, amount, fee, 
                 timestamp, signature, status, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx.get('hash'),
                block.index,
                tx.get('sender'),
                tx.get('recipient'),
                str(tx.get('amount', '0')),
                str(tx.get('fee', '0')),
                tx.get('timestamp'),
                tx.get('signature'),
                'CONFIRMED',
                json.dumps(tx.get('data', {}))
            ))
        
        conn.commit()
        conn.close()
    
    def add_transaction(self, transaction: Dict[str, Any]) -> str:
        """Add a transaction to the pending pool"""
        # Generate transaction hash
        tx_string = json.dumps(transaction, sort_keys=True)
        tx_hash = hashlib.sha256(tx_string.encode()).hexdigest()
        transaction['hash'] = tx_hash
        transaction['timestamp'] = time.time()
        
        # Validate transaction
        if not self._validate_transaction(transaction):
            raise ValueError("Invalid transaction")
        
        # Add to pool
        self.transaction_pool[tx_hash] = transaction
        self.pending_transactions.append(transaction)
        
        logger.info(f"Transaction added: {tx_hash}")
        return tx_hash
    
    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction structure and signatures"""
        required_fields = ['sender', 'recipient', 'amount']
        for field in required_fields:
            if field not in transaction:
                return False
        
        # Validate amount
        try:
            amount = Decimal(str(transaction['amount']))
            if amount <= 0:
                return False
        except:
            return False
        
        # Additional validation would include signature verification
        # and balance checks in production
        
        return True
    
    def mine_block(self) -> Optional[Block]:
        """Mine a new block with proof of work"""
        if not self.pending_transactions:
            return None
        
        with self.mining_lock:
            # Select transactions for block
            block_transactions = []
            block_size = 0
            
            for tx in self.pending_transactions[:]:
                tx_size = len(json.dumps(tx))
                if block_size + tx_size > self.max_block_size:
                    break
                block_transactions.append(tx)
                block_size += tx_size
                self.pending_transactions.remove(tx)
            
            if not block_transactions:
                return None
            
            # Create new block
            previous_block = self.chain[-1]
            new_block = Block(
                index=len(self.chain),
                timestamp=time.time(),
                transactions=block_transactions,
                previous_hash=previous_block.hash
            )
            
            # Calculate Merkle root
            new_block.merkle_root = new_block.calculate_merkle_root()
            
            # Proof of Work
            new_block = self._proof_of_work(new_block)
            
            # Validate with consensus
            if self.validators and not self.consensus.propose_block(new_block, self.validators):
                logger.warning("Block rejected by consensus")
                # Return transactions to pool
                self.pending_transactions.extend(block_transactions)
                return None
            
            # Add block to chain
            self.chain.append(new_block)
            self._persist_block(new_block)
            
            # Update state
            self._update_state(new_block)
            
            logger.info(f"Block mined: {new_block.hash}")
            return new_block
    
    def _proof_of_work(self, block: Block) -> Block:
        """Implement proof of work algorithm"""
        target = "0" * self.difficulty
        
        while True:
            block.hash = block.calculate_hash()
            if block.hash.startswith(target):
                return block
            block.nonce += 1
    
    def _update_state(self, block: Block) -> None:
        """Update blockchain state after new block"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for tx in block.transactions:
            # Update balances in state
            sender = tx.get('sender')
            recipient = tx.get('recipient')
            amount = Decimal(str(tx.get('amount', '0')))
            
            if sender:
                sender_balance = self._get_balance(sender) - amount
                self._set_balance(sender, sender_balance)
            
            if recipient:
                recipient_balance = self._get_balance(recipient) + amount
                self._set_balance(recipient, recipient_balance)
        
        # Calculate new state root
        state_string = json.dumps(self.state_db, sort_keys=True)
        self.state_root = hashlib.sha256(state_string.encode()).hexdigest()
        
        conn.close()
    
    def _get_balance(self, address: str) -> Decimal:
        """Get balance from state"""
        return Decimal(str(self.state_db.get(f"balance:{address}", "0")))
    
    def _set_balance(self, address: str, balance: Decimal) -> None:
        """Set balance in state"""
        self.state_db[f"balance:{address}"] = str(balance)
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check hash calculation
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Invalid hash at block {i}")
                return False
            
            # Check previous hash link
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid chain link at block {i}")
                return False
            
            # Check Merkle root
            calculated_merkle = current_block.calculate_merkle_root()
            if current_block.merkle_root != calculated_merkle:
                logger.error(f"Invalid Merkle root at block {i}")
                return False
            
            # Check proof of work
            if not current_block.hash.startswith("0" * self.difficulty):
                logger.error(f"Invalid proof of work at block {i}")
                return False
        
        return True
    
    def register_node(self, node_id: str, public_key: str, stake: Decimal = Decimal("0")) -> Node:
        """Register a new node in the network"""
        node = Node(
            node_id=node_id,
            public_key=public_key,
            stake=stake,
            is_validator=stake >= Decimal("1000")  # Minimum stake for validator
        )
        
        self.nodes[node_id] = node
        if node.is_validator:
            self.validators.append(node)
        
        # Persist node
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (node_id, public_key, stake, reputation, is_validator, 
             last_seen, validated_blocks, failed_validations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.public_key,
            str(node.stake),
            node.reputation,
            int(node.is_validator),
            node.last_seen.isoformat(),
            node.validated_blocks,
            node.failed_validations
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Node registered: {node_id}")
        return node
    
    def get_block(self, index: int) -> Optional[Block]:
        """Get a specific block by index"""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get a transaction by hash"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT tx_hash, block_index, sender, recipient, amount, 
                   fee, timestamp, signature, status, data
            FROM transactions
            WHERE tx_hash = ?
        ''', (tx_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'hash': row[0],
                'block_index': row[1],
                'sender': row[2],
                'recipient': row[3],
                'amount': row[4],
                'fee': row[5],
                'timestamp': row[6],
                'signature': row[7],
                'status': row[8],
                'data': json.loads(row[9]) if row[9] else {}
            }
        return None
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            'height': len(self.chain),
            'latest_block_hash': self.chain[-1].hash if self.chain else None,
            'latest_block_time': self.chain[-1].timestamp if self.chain else None,
            'pending_transactions': len(self.pending_transactions),
            'total_transactions': sum(len(block.transactions) for block in self.chain),
            'difficulty': self.difficulty,
            'validators': len(self.validators),
            'nodes': len(self.nodes),
            'state_root': self.state_root
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the blockchain engine"""
        logger.info("Shutting down Blockchain Engine...")
        self.executor.shutdown(wait=True)
        logger.info("Blockchain Engine shutdown complete")

def main():
    """Main entry point for testing"""
    engine = BlockchainEngine()
    
    # Register some nodes
    node1 = engine.register_node("node1", "pubkey1", Decimal("5000"))
    node2 = engine.register_node("node2", "pubkey2", Decimal("3000"))
    
    # Add transactions
    tx1 = engine.add_transaction({
        'sender': 'alice',
        'recipient': 'bob',
        'amount': '100.50',
        'fee': '0.50'
    })
    
    tx2 = engine.add_transaction({
        'sender': 'bob',
        'recipient': 'charlie',
        'amount': '50.25',
        'fee': '0.25'
    })
    
    # Mine a block
    block = engine.mine_block()
    if block:
        print(f"Block mined: {block.hash}")
    
    # Validate chain
    is_valid = engine.validate_chain()
    print(f"Chain valid: {is_valid}")
    
    # Get chain info
    info = engine.get_chain_info()
    print(f"Chain info: {info}")
    
    engine.shutdown()

if __name__ == "__main__":
    main()