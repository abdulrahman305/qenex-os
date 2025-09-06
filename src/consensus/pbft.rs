//! Practical Byzantine Fault Tolerant Consensus Implementation

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};

/// PBFT consensus state machine
pub struct PBFTConsensus {
    node_id: String,
    keypair: Keypair,
    nodes: HashMap<String, PublicKey>,
    view: u64,
    sequence: u64,
    state: ConsensusState,
    message_log: Arc<RwLock<MessageLog>>,
    prepared: HashSet<String>,
    committed: HashSet<String>,
    checkpoint_interval: u64,
    f: usize, // Maximum number of faulty nodes
}

#[derive(Debug, Clone, PartialEq)]
enum ConsensusState {
    Normal,
    ViewChange,
    Checkpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    Request {
        client_id: String,
        timestamp: u64,
        operation: Vec<u8>,
    },
    PrePrepare {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        request: Box<ConsensusMessage>,
    },
    Prepare {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    },
    Commit {
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    },
    Reply {
        view: u64,
        timestamp: u64,
        node_id: String,
        result: Vec<u8>,
    },
    ViewChange {
        new_view: u64,
        node_id: String,
        prepared_proof: Vec<PreparedCertificate>,
    },
    NewView {
        view: u64,
        view_changes: Vec<ConsensusMessage>,
        pre_prepares: Vec<ConsensusMessage>,
    },
    Checkpoint {
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreparedCertificate {
    sequence: u64,
    digest: Vec<u8>,
    pre_prepare: ConsensusMessage,
    prepares: Vec<ConsensusMessage>,
}

struct MessageLog {
    requests: HashMap<String, ConsensusMessage>,
    pre_prepares: HashMap<(u64, u64), ConsensusMessage>,
    prepares: HashMap<(u64, u64), Vec<ConsensusMessage>>,
    commits: HashMap<(u64, u64), Vec<ConsensusMessage>>,
    checkpoints: HashMap<u64, Vec<ConsensusMessage>>,
}

impl PBFTConsensus {
    pub fn new(
        node_id: String,
        keypair: Keypair,
        nodes: HashMap<String, PublicKey>,
        f: usize,
    ) -> Self {
        Self {
            node_id,
            keypair,
            nodes,
            view: 0,
            sequence: 0,
            state: ConsensusState::Normal,
            message_log: Arc::new(RwLock::new(MessageLog {
                requests: HashMap::new(),
                pre_prepares: HashMap::new(),
                prepares: HashMap::new(),
                commits: HashMap::new(),
                checkpoints: HashMap::new(),
            })),
            prepared: HashSet::new(),
            committed: HashSet::new(),
            checkpoint_interval: 100,
            f,
        }
    }

    /// Process incoming consensus message
    pub async fn process_message(&mut self, message: ConsensusMessage) -> Result<(), Box<dyn std::error::Error>> {
        match message {
            ConsensusMessage::Request { .. } => {
                if self.is_primary() {
                    self.handle_request(message).await?;
                }
            }
            ConsensusMessage::PrePrepare { view, sequence, digest, request } => {
                self.handle_pre_prepare(view, sequence, digest, *request).await?;
            }
            ConsensusMessage::Prepare { view, sequence, digest, node_id } => {
                self.handle_prepare(view, sequence, digest, node_id).await?;
            }
            ConsensusMessage::Commit { view, sequence, digest, node_id } => {
                self.handle_commit(view, sequence, digest, node_id).await?;
            }
            ConsensusMessage::ViewChange { new_view, node_id, prepared_proof } => {
                self.handle_view_change(new_view, node_id, prepared_proof).await?;
            }
            ConsensusMessage::Checkpoint { sequence, digest, node_id } => {
                self.handle_checkpoint(sequence, digest, node_id).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Check if this node is the primary for current view
    fn is_primary(&self) -> bool {
        let primary_index = (self.view as usize) % self.nodes.len();
        let mut node_ids: Vec<_> = self.nodes.keys().cloned().collect();
        node_ids.sort();
        node_ids[primary_index] == self.node_id
    }

    /// Handle client request (primary only)
    async fn handle_request(&mut self, request: ConsensusMessage) -> Result<(), Box<dyn std::error::Error>> {
        self.sequence += 1;
        let digest = self.compute_digest(&request);
        
        let pre_prepare = ConsensusMessage::PrePrepare {
            view: self.view,
            sequence: self.sequence,
            digest: digest.clone(),
            request: Box::new(request.clone()),
        };

        // Log pre-prepare
        {
            let mut log = self.message_log.write().unwrap();
            log.pre_prepares.insert((self.view, self.sequence), pre_prepare.clone());
        }

        // Broadcast pre-prepare to all nodes
        self.broadcast(pre_prepare).await?;
        
        // Send prepare as well
        self.send_prepare(self.view, self.sequence, digest).await?;
        
        Ok(())
    }

    /// Handle pre-prepare message
    async fn handle_pre_prepare(
        &mut self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        request: ConsensusMessage,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Verify conditions
        if view != self.view {
            return Ok(());
        }

        // Verify digest
        let computed_digest = self.compute_digest(&request);
        if computed_digest != digest {
            return Err("Invalid digest".into());
        }

        // Log pre-prepare
        {
            let mut log = self.message_log.write().unwrap();
            log.pre_prepares.insert((view, sequence), ConsensusMessage::PrePrepare {
                view,
                sequence,
                digest: digest.clone(),
                request: Box::new(request),
            });
        }

        // Send prepare
        self.send_prepare(view, sequence, digest).await?;
        
        Ok(())
    }

    /// Send prepare message
    async fn send_prepare(&mut self, view: u64, sequence: u64, digest: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let prepare = ConsensusMessage::Prepare {
            view,
            sequence,
            digest,
            node_id: self.node_id.clone(),
        };

        // Log our prepare
        {
            let mut log = self.message_log.write().unwrap();
            log.prepares.entry((view, sequence)).or_insert_with(Vec::new).push(prepare.clone());
        }

        // Broadcast prepare
        self.broadcast(prepare).await?;
        
        Ok(())
    }

    /// Handle prepare message
    async fn handle_prepare(
        &mut self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if view != self.view {
            return Ok(());
        }

        // Log prepare
        {
            let mut log = self.message_log.write().unwrap();
            log.prepares.entry((view, sequence)).or_insert_with(Vec::new).push(
                ConsensusMessage::Prepare { view, sequence, digest: digest.clone(), node_id }
            );
        }

        // Check if prepared
        if self.check_prepared(view, sequence) && !self.prepared.contains(&format!("{}-{}", view, sequence)) {
            self.prepared.insert(format!("{}-{}", view, sequence));
            self.send_commit(view, sequence, digest).await?;
        }

        Ok(())
    }

    /// Check if request is prepared
    fn check_prepared(&self, view: u64, sequence: u64) -> bool {
        let log = self.message_log.read().unwrap();
        
        // Need pre-prepare
        if !log.pre_prepares.contains_key(&(view, sequence)) {
            return false;
        }

        // Need 2f prepares
        if let Some(prepares) = log.prepares.get(&(view, sequence)) {
            prepares.len() >= 2 * self.f
        } else {
            false
        }
    }

    /// Send commit message
    async fn send_commit(&mut self, view: u64, sequence: u64, digest: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let commit = ConsensusMessage::Commit {
            view,
            sequence,
            digest,
            node_id: self.node_id.clone(),
        };

        // Log our commit
        {
            let mut log = self.message_log.write().unwrap();
            log.commits.entry((view, sequence)).or_insert_with(Vec::new).push(commit.clone());
        }

        // Broadcast commit
        self.broadcast(commit).await?;
        
        Ok(())
    }

    /// Handle commit message
    async fn handle_commit(
        &mut self,
        view: u64,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if view != self.view {
            return Ok(());
        }

        // Log commit
        {
            let mut log = self.message_log.write().unwrap();
            log.commits.entry((view, sequence)).or_insert_with(Vec::new).push(
                ConsensusMessage::Commit { view, sequence, digest, node_id }
            );
        }

        // Check if committed
        if self.check_committed(view, sequence) && !self.committed.contains(&format!("{}-{}", view, sequence)) {
            self.committed.insert(format!("{}-{}", view, sequence));
            self.execute_request(view, sequence).await?;
        }

        Ok(())
    }

    /// Check if request is committed
    fn check_committed(&self, view: u64, sequence: u64) -> bool {
        let log = self.message_log.read().unwrap();
        
        // Need to be prepared first
        if !self.prepared.contains(&format!("{}-{}", view, sequence)) {
            return false;
        }

        // Need 2f+1 commits
        if let Some(commits) = log.commits.get(&(view, sequence)) {
            commits.len() >= 2 * self.f + 1
        } else {
            false
        }
    }

    /// Execute committed request
    async fn execute_request(&mut self, view: u64, sequence: u64) -> Result<(), Box<dyn std::error::Error>> {
        let log = self.message_log.read().unwrap();
        
        if let Some(pre_prepare) = log.pre_prepares.get(&(view, sequence)) {
            if let ConsensusMessage::PrePrepare { request, .. } = pre_prepare {
                // Execute the request
                println!("Executing request at sequence {}", sequence);
                
                // Check if checkpoint needed
                if sequence % self.checkpoint_interval == 0 {
                    self.create_checkpoint(sequence).await?;
                }
            }
        }
        
        Ok(())
    }

    /// Create checkpoint
    async fn create_checkpoint(&mut self, sequence: u64) -> Result<(), Box<dyn std::error::Error>> {
        let digest = self.compute_state_digest();
        
        let checkpoint = ConsensusMessage::Checkpoint {
            sequence,
            digest,
            node_id: self.node_id.clone(),
        };

        // Log checkpoint
        {
            let mut log = self.message_log.write().unwrap();
            log.checkpoints.entry(sequence).or_insert_with(Vec::new).push(checkpoint.clone());
        }

        // Broadcast checkpoint
        self.broadcast(checkpoint).await?;
        
        Ok(())
    }

    /// Handle checkpoint message
    async fn handle_checkpoint(
        &mut self,
        sequence: u64,
        digest: Vec<u8>,
        node_id: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Log checkpoint
        {
            let mut log = self.message_log.write().unwrap();
            log.checkpoints.entry(sequence).or_insert_with(Vec::new).push(
                ConsensusMessage::Checkpoint { sequence, digest, node_id }
            );
        }

        // Check if stable checkpoint
        if self.check_stable_checkpoint(sequence) {
            self.garbage_collect(sequence)?;
        }

        Ok(())
    }

    /// Check if checkpoint is stable
    fn check_stable_checkpoint(&self, sequence: u64) -> bool {
        let log = self.message_log.read().unwrap();
        
        if let Some(checkpoints) = log.checkpoints.get(&sequence) {
            checkpoints.len() >= 2 * self.f + 1
        } else {
            false
        }
    }

    /// Garbage collect old messages
    fn garbage_collect(&mut self, stable_sequence: u64) -> Result<(), Box<dyn std::error::Error>> {
        let mut log = self.message_log.write().unwrap();
        
        // Remove old pre-prepares
        log.pre_prepares.retain(|(_, seq), _| *seq > stable_sequence);
        
        // Remove old prepares
        log.prepares.retain(|(_, seq), _| *seq > stable_sequence);
        
        // Remove old commits
        log.commits.retain(|(_, seq), _| *seq > stable_sequence);
        
        Ok(())
    }

    /// Handle view change
    async fn handle_view_change(
        &mut self,
        new_view: u64,
        node_id: String,
        prepared_proof: Vec<PreparedCertificate>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implement view change protocol
        if new_view > self.view {
            self.state = ConsensusState::ViewChange;
            // Collect view-change messages
            // When have 2f+1, construct new-view message
        }
        Ok(())
    }

    /// Compute message digest
    fn compute_digest(&self, message: &ConsensusMessage) -> Vec<u8> {
        let serialized = bincode::serialize(message).unwrap();
        let mut hasher = Sha3_256::new();
        hasher.update(&serialized);
        hasher.finalize().to_vec()
    }

    /// Compute state digest
    fn compute_state_digest(&self) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(format!("state-{}", self.sequence).as_bytes());
        hasher.finalize().to_vec()
    }

    /// Broadcast message to all nodes
    async fn broadcast(&self, message: ConsensusMessage) -> Result<(), Box<dyn std::error::Error>> {
        // In production, this would send over network
        println!("Broadcasting {:?}", message);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_basic() {
        let keypair = Keypair::generate(&mut rand::thread_rng());
        let mut nodes = HashMap::new();
        nodes.insert("node1".to_string(), keypair.public);
        
        let mut consensus = PBFTConsensus::new(
            "node1".to_string(),
            keypair,
            nodes,
            0,
        );

        let request = ConsensusMessage::Request {
            client_id: "client1".to_string(),
            timestamp: 1234567890,
            operation: b"transfer".to_vec(),
        };

        consensus.process_message(request).await.unwrap();
    }
}