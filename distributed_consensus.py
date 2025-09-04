"""
Distributed Consensus Mechanism
Byzantine Fault Tolerant consensus for distributed banking
"""

import asyncio
import hashlib
import time
import json
import secrets
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Consensus message types"""
    PREPARE = "prepare"
    PROMISE = "promise"
    PROPOSE = "propose"
    ACCEPT = "accept"
    COMMIT = "commit"
    HEARTBEAT = "heartbeat"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"

class NodeState(Enum):
    """Node states in consensus"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    BYZANTINE = "byzantine"

@dataclass
class ConsensusMessage:
    """Message in consensus protocol"""
    type: MessageType
    sender: str
    view: int
    sequence: int
    value: Any
    signature: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Transaction:
    """Banking transaction"""
    id: str
    from_account: str
    to_account: str
    amount: float
    timestamp: float
    signature: str

class RaftConsensus:
    """Raft consensus implementation for leader election"""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[Dict] = []
        self.commit_index = 0
        self.last_applied = 0
        self.next_index = {peer: 0 for peer in peers}
        self.match_index = {peer: 0 for peer in peers}
        self.leader_id: Optional[str] = None
        self.election_timeout = None
        self.heartbeat_interval = 0.5
        
    async def start_election(self):
        """Start leader election"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        votes = 1
        
        # Request votes from peers
        tasks = []
        for peer in self.peers:
            tasks.append(self._request_vote(peer))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result and not isinstance(result, Exception):
                votes += 1
        
        # Check if won election
        if votes > len(self.peers) // 2:
            self.state = NodeState.LEADER
            self.leader_id = self.node_id
            logger.info(f"Node {self.node_id} elected as leader for term {self.current_term}")
            await self._send_heartbeats()
        else:
            self.state = NodeState.FOLLOWER
    
    async def _request_vote(self, peer: str) -> bool:
        """Request vote from peer"""
        # Simulate network request
        await asyncio.sleep(0.01)
        # Simplified - would send actual network request
        return secrets.choice([True, False])
    
    async def _send_heartbeats(self):
        """Send heartbeats to maintain leadership"""
        while self.state == NodeState.LEADER:
            tasks = []
            for peer in self.peers:
                tasks.append(self._send_heartbeat(peer))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self, peer: str) -> bool:
        """Send heartbeat to peer"""
        # Simulate network request
        await asyncio.sleep(0.01)
        return True
    
    def append_entry(self, entry: Dict) -> bool:
        """Append entry to log"""
        if self.state != NodeState.LEADER:
            return False
        
        self.log.append({
            "term": self.current_term,
            "index": len(self.log),
            "entry": entry,
            "timestamp": time.time()
        })
        return True

class PBFTConsensus:
    """Practical Byzantine Fault Tolerance implementation"""
    
    def __init__(self, node_id: str, nodes: List[str], f: int = 1):
        """
        Initialize PBFT consensus
        f: number of byzantine failures to tolerate
        """
        self.node_id = node_id
        self.nodes = nodes
        self.f = f
        self.view = 0
        self.sequence = 0
        self.is_primary = False
        self.prepared: Dict[int, Set[str]] = defaultdict(set)
        self.committed: Dict[int, Set[str]] = defaultdict(set)
        self.log: List[Transaction] = []
        self.pending_transactions: List[Transaction] = []
        self.checkpoints: Dict[int, bytes] = {}
        
    def is_primary_node(self) -> bool:
        """Check if this node is primary"""
        primary_index = self.view % len(self.nodes)
        return self.nodes[primary_index] == self.node_id
    
    async def propose_transaction(self, transaction: Transaction) -> bool:
        """Propose transaction to network"""
        if not self.is_primary_node():
            # Forward to primary
            return await self._forward_to_primary(transaction)
        
        # Primary processes transaction
        self.sequence += 1
        
        # Send pre-prepare to all replicas
        message = ConsensusMessage(
            type=MessageType.PREPARE,
            sender=self.node_id,
            view=self.view,
            sequence=self.sequence,
            value=transaction.__dict__,
            signature=self._sign_message(transaction)
        )
        
        await self._broadcast(message)
        return True
    
    async def handle_prepare(self, message: ConsensusMessage):
        """Handle prepare message"""
        # Verify message
        if not self._verify_message(message):
            return
        
        # Add to prepared set
        self.prepared[message.sequence].add(message.sender)
        
        # If received 2f prepares, send commit
        if len(self.prepared[message.sequence]) >= 2 * self.f:
            commit_msg = ConsensusMessage(
                type=MessageType.COMMIT,
                sender=self.node_id,
                view=self.view,
                sequence=message.sequence,
                value=message.value,
                signature=self._sign_message(message.value)
            )
            await self._broadcast(commit_msg)
    
    async def handle_commit(self, message: ConsensusMessage):
        """Handle commit message"""
        # Verify message
        if not self._verify_message(message):
            return
        
        # Add to committed set
        self.committed[message.sequence].add(message.sender)
        
        # If received 2f+1 commits, execute transaction
        if len(self.committed[message.sequence]) >= 2 * self.f + 1:
            transaction = Transaction(**message.value)
            await self._execute_transaction(transaction)
    
    async def _execute_transaction(self, transaction: Transaction):
        """Execute committed transaction"""
        self.log.append(transaction)
        logger.info(f"Executed transaction {transaction.id} at sequence {self.sequence}")
        
        # Create checkpoint every 100 transactions
        if len(self.log) % 100 == 0:
            await self._create_checkpoint()
    
    async def _create_checkpoint(self):
        """Create state checkpoint"""
        state_hash = hashlib.sha256(
            json.dumps([t.__dict__ for t in self.log]).encode()
        ).digest()
        
        self.checkpoints[len(self.log)] = state_hash
        logger.info(f"Created checkpoint at sequence {len(self.log)}")
    
    def _sign_message(self, data: Any) -> str:
        """Sign message (simplified)"""
        data_str = json.dumps(data) if not isinstance(data, str) else data
        return hashlib.sha256(f"{self.node_id}:{data_str}".encode()).hexdigest()
    
    def _verify_message(self, message: ConsensusMessage) -> bool:
        """Verify message signature (simplified)"""
        expected_sig = hashlib.sha256(
            f"{message.sender}:{json.dumps(message.value)}".encode()
        ).hexdigest()
        return message.signature == expected_sig
    
    async def _broadcast(self, message: ConsensusMessage):
        """Broadcast message to all nodes"""
        tasks = []
        for node in self.nodes:
            if node != self.node_id:
                tasks.append(self._send_message(node, message))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_message(self, node: str, message: ConsensusMessage):
        """Send message to specific node"""
        # Simulate network delay
        await asyncio.sleep(0.01)
        # In production, would use actual network transport
        logger.debug(f"Sent {message.type.value} to {node}")
    
    async def _forward_to_primary(self, transaction: Transaction) -> bool:
        """Forward transaction to primary node"""
        primary_index = self.view % len(self.nodes)
        primary = self.nodes[primary_index]
        
        # Simulate forwarding
        await asyncio.sleep(0.01)
        logger.info(f"Forwarded transaction {transaction.id} to primary {primary}")
        return True

class HotStuffConsensus:
    """HotStuff consensus protocol (used by Diem/Libra)"""
    
    def __init__(self, node_id: str, nodes: List[str]):
        self.node_id = node_id
        self.nodes = nodes
        self.view = 0
        self.highest_qc_view = 0
        self.locked_view = 0
        self.executed_view = 0
        self.pending_votes: Dict[int, Set[str]] = defaultdict(set)
        self.qc_high: Optional[Dict] = None
        
    async def propose_block(self, transactions: List[Transaction]) -> Dict:
        """Propose new block"""
        block = {
            "view": self.view,
            "parent": self.qc_high,
            "transactions": [t.__dict__ for t in transactions],
            "timestamp": time.time()
        }
        
        # Create block hash
        block["hash"] = hashlib.sha256(
            json.dumps(block).encode()
        ).hexdigest()
        
        # Send to all replicas
        await self._broadcast_proposal(block)
        
        return block
    
    async def handle_vote(self, vote: Dict):
        """Handle vote from replica"""
        view = vote["view"]
        self.pending_votes[view].add(vote["sender"])
        
        # Check if have quorum certificate
        if len(self.pending_votes[view]) >= 2 * len(self.nodes) // 3 + 1:
            qc = {
                "view": view,
                "votes": list(self.pending_votes[view]),
                "block_hash": vote["block_hash"]
            }
            
            await self._process_qc(qc)
    
    async def _process_qc(self, qc: Dict):
        """Process quorum certificate"""
        if qc["view"] > self.highest_qc_view:
            self.highest_qc_view = qc["view"]
            self.qc_high = qc
            
            # Check for commit
            await self._try_commit()
    
    async def _try_commit(self):
        """Try to commit blocks with 3-chain rule"""
        # HotStuff commits when there's a 3-chain
        if self.qc_high and "parent" in self.qc_high:
            parent = self.qc_high["parent"]
            if parent and "parent" in parent:
                grandparent = parent["parent"]
                if grandparent:
                    # Commit grandparent
                    await self._commit_block(grandparent)
    
    async def _commit_block(self, block: Dict):
        """Commit block to ledger"""
        logger.info(f"Committed block at view {block['view']}")
        self.executed_view = block["view"]
    
    async def _broadcast_proposal(self, block: Dict):
        """Broadcast block proposal"""
        tasks = []
        for node in self.nodes:
            if node != self.node_id:
                tasks.append(self._send_proposal(node, block))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_proposal(self, node: str, block: Dict):
        """Send proposal to node"""
        await asyncio.sleep(0.01)
        logger.debug(f"Sent proposal to {node}")

class TendermintConsensus:
    """Tendermint consensus protocol"""
    
    def __init__(self, node_id: str, validators: List[str]):
        self.node_id = node_id
        self.validators = validators
        self.height = 0
        self.round = 0
        self.step = "propose"  # propose, prevote, precommit
        self.locked_value = None
        self.locked_round = -1
        self.valid_value = None
        self.valid_round = -1
        
    async def run_round(self, value: Any) -> bool:
        """Run consensus round"""
        self.round += 1
        
        # Propose phase
        if self._is_proposer():
            proposal = await self._create_proposal(value)
            await self._broadcast_proposal(proposal)
        
        # Prevote phase
        await self._prevote()
        
        # Precommit phase
        decision = await self._precommit()
        
        if decision:
            await self._commit(value)
            return True
        
        return False
    
    def _is_proposer(self) -> bool:
        """Check if this node is proposer for current round"""
        proposer_index = (self.height + self.round) % len(self.validators)
        return self.validators[proposer_index] == self.node_id
    
    async def _create_proposal(self, value: Any) -> Dict:
        """Create block proposal"""
        return {
            "height": self.height,
            "round": self.round,
            "value": value,
            "proposer": self.node_id,
            "timestamp": time.time()
        }
    
    async def _prevote(self):
        """Prevote phase"""
        # Simulate prevoting
        await asyncio.sleep(0.01)
        self.step = "prevote"
    
    async def _precommit(self) -> bool:
        """Precommit phase"""
        # Simulate precommit
        await asyncio.sleep(0.01)
        self.step = "precommit"
        
        # Simplified - would check actual votes
        return secrets.choice([True, False])
    
    async def _commit(self, value: Any):
        """Commit value"""
        self.height += 1
        self.round = 0
        logger.info(f"Committed value at height {self.height}")
    
    async def _broadcast_proposal(self, proposal: Dict):
        """Broadcast proposal to validators"""
        tasks = []
        for validator in self.validators:
            if validator != self.node_id:
                tasks.append(self._send_to_validator(validator, proposal))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_validator(self, validator: str, data: Dict):
        """Send data to validator"""
        await asyncio.sleep(0.01)

class ConsensusOrchestrator:
    """Orchestrate multiple consensus mechanisms"""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        
        # Initialize consensus mechanisms
        self.raft = RaftConsensus(node_id, peers)
        self.pbft = PBFTConsensus(node_id, peers)
        self.hotstuff = HotStuffConsensus(node_id, peers)
        self.tendermint = TendermintConsensus(node_id, peers)
        
        # Metrics
        self.consensus_times: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, int] = defaultdict(int)
        self.total_attempts: Dict[str, int] = defaultdict(int)
    
    async def achieve_consensus(self, transaction: Transaction, 
                               algorithm: str = "pbft") -> bool:
        """Achieve consensus using specified algorithm"""
        start_time = time.time()
        success = False
        
        self.total_attempts[algorithm] += 1
        
        try:
            if algorithm == "raft":
                # Use Raft for leader election
                if self.raft.state != NodeState.LEADER:
                    await self.raft.start_election()
                success = self.raft.append_entry(transaction.__dict__)
                
            elif algorithm == "pbft":
                # Use PBFT for Byzantine tolerance
                success = await self.pbft.propose_transaction(transaction)
                
            elif algorithm == "hotstuff":
                # Use HotStuff for high performance
                block = await self.hotstuff.propose_block([transaction])
                success = block is not None
                
            elif algorithm == "tendermint":
                # Use Tendermint for blockchain consensus
                success = await self.tendermint.run_round(transaction)
            
            if success:
                self.success_rates[algorithm] += 1
                
            # Record consensus time
            consensus_time = time.time() - start_time
            self.consensus_times[algorithm].append(consensus_time)
            
            logger.info(f"Consensus achieved using {algorithm} in {consensus_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Consensus failed using {algorithm}: {e}")
            success = False
        
        return success
    
    def get_metrics(self) -> Dict:
        """Get consensus metrics"""
        metrics = {}
        
        for algo in ["raft", "pbft", "hotstuff", "tendermint"]:
            if algo in self.consensus_times and self.consensus_times[algo]:
                avg_time = sum(self.consensus_times[algo]) / len(self.consensus_times[algo])
                success_rate = (self.success_rates[algo] / self.total_attempts[algo] * 100 
                               if self.total_attempts[algo] > 0 else 0)
                
                metrics[algo] = {
                    "average_time": avg_time,
                    "success_rate": success_rate,
                    "total_attempts": self.total_attempts[algo]
                }
        
        return metrics
    
    async def run_comparison_test(self, num_transactions: int = 100):
        """Run comparison test of consensus algorithms"""
        print("\nConsensus Algorithm Comparison Test")
        print("=" * 50)
        
        algorithms = ["raft", "pbft", "hotstuff", "tendermint"]
        
        for algo in algorithms:
            print(f"\nTesting {algo.upper()}...")
            
            for i in range(num_transactions):
                tx = Transaction(
                    id=f"tx_{algo}_{i}",
                    from_account=f"acc_{secrets.randbelow(100)}",
                    to_account=f"acc_{secrets.randbelow(100)}",
                    amount=secrets.randbelow(10000),
                    timestamp=time.time(),
                    signature=secrets.token_hex(32)
                )
                
                await self.achieve_consensus(tx, algo)
        
        # Display results
        print("\nResults:")
        print("-" * 50)
        
        metrics = self.get_metrics()
        for algo, data in metrics.items():
            print(f"\n{algo.upper()}:")
            print(f"  Average Time: {data['average_time']:.4f}s")
            print(f"  Success Rate: {data['success_rate']:.1f}%")
            print(f"  Transactions: {data['total_attempts']}")

# Example usage
async def main():
    # Initialize nodes
    nodes = [f"node_{i}" for i in range(7)]  # 7 nodes for 2 Byzantine failures
    
    # Create orchestrator for first node
    orchestrator = ConsensusOrchestrator("node_0", nodes[1:])
    
    print("Distributed Consensus System")
    print("=" * 50)
    
    # Test single transaction
    tx = Transaction(
        id="tx_001",
        from_account="alice",
        to_account="bob",
        amount=1000.0,
        timestamp=time.time(),
        signature=secrets.token_hex(32)
    )
    
    # Test different consensus algorithms
    for algo in ["raft", "pbft", "hotstuff", "tendermint"]:
        print(f"\nTesting {algo.upper()} consensus...")
        success = await orchestrator.achieve_consensus(tx, algo)
        print(f"  Result: {'✓ Success' if success else '✗ Failed'}")
    
    # Run comparison test
    await orchestrator.run_comparison_test(20)

if __name__ == "__main__":
    asyncio.run(main())