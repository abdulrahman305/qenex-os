pub mod quantum;

pub use quantum::{
    QuantumResistantEngine, 
    SignatureAlgorithm, 
    KEMAlgorithm, 
    SymmetricAlgorithm,
    HashAlgorithm,
    SecurityLevel,
    KeyPurpose,
    CryptoError,
    QuantumSignature,
    QuantumCiphertext,
};

use std::sync::Arc;

/// Main cryptographic provider for QENEX banking system
pub struct CryptoProvider {
    quantum_engine: Arc<QuantumResistantEngine>,
}

impl CryptoProvider {
    pub fn new() -> Self {
        Self {
            quantum_engine: Arc::new(QuantumResistantEngine::new()),
        }
    }
    
    pub fn with_hsm(hsm_config: quantum::HSMConfig) -> Result<Self, CryptoError> {
        Ok(Self {
            quantum_engine: Arc::new(QuantumResistantEngine::with_hsm(hsm_config)?),
        })
    }
    
    pub fn engine(&self) -> &QuantumResistantEngine {
        &self.quantum_engine
    }
}

/// Legacy signature type for compatibility
pub struct Signature {
    pub algorithm: String,
    pub signature_data: Vec<u8>,
}

impl From<QuantumSignature> for Signature {
    fn from(quantum_sig: QuantumSignature) -> Self {
        Self {
            algorithm: format!("{:?}", quantum_sig.algorithm),
            signature_data: quantum_sig.signature_data,
        }
    }
}

/// Legacy key type enum for compatibility
#[derive(Debug, Clone, PartialEq)]
pub enum KeyType {
    Dilithium,
    Falcon,
    SphincsPlus,
    Kyber,
    AES256,
    ChaCha20,
}

impl From<quantum::SignatureAlgorithm> for KeyType {
    fn from(alg: quantum::SignatureAlgorithm) -> Self {
        match alg {
            quantum::SignatureAlgorithm::Dilithium => KeyType::Dilithium,
            quantum::SignatureAlgorithm::Falcon => KeyType::Falcon,
            quantum::SignatureAlgorithm::SphincsPlus => KeyType::SphincsPlus,
        }
    }
}

impl From<quantum::KEMAlgorithm> for KeyType {
    fn from(alg: quantum::KEMAlgorithm) -> Self {
        match alg {
            quantum::KEMAlgorithm::Kyber => KeyType::Kyber,
            quantum::KEMAlgorithm::McEliece => KeyType::Kyber, // Map to Kyber for now
        }
    }
}

impl From<quantum::SymmetricAlgorithm> for KeyType {
    fn from(alg: quantum::SymmetricAlgorithm) -> Self {
        match alg {
            quantum::SymmetricAlgorithm::AES256GCM => KeyType::AES256,
            quantum::SymmetricAlgorithm::ChaCha20Poly1305 => KeyType::ChaCha20,
        }
    }
}