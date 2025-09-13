//! QENEX Root-Level Consensus Module
//! 
//! Public interface for consensus operations

pub use crate::consensus::pbft::*;

pub mod pbft;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Public consensus interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusInterface {
    pub node_id: Uuid,
}

impl ConsensusInterface {
    pub fn new() -> Self {
        Self {
            node_id: Uuid::new_v4(),
        }
    }
}