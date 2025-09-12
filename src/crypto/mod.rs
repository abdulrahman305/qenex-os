pub mod quantum;
pub mod post_quantum;
pub mod hsm;

#[cfg(test)]
pub mod pq_tests;

// Export the real post-quantum implementation as primary
pub use post_quantum::{
    PostQuantumEngine,
    PostQuantumSignature,
    PostQuantumCiphertext,
    BankingCryptoContext,
    PQCryptoError,
    DilithiumSigner,
    KyberKEM,
    SphincsPlus,
};

// Export HSM functionality for banking hardware integration
pub use hsm::{
    BankingHSM,
    HSMConfig,
    HSMVendor,
    FIPS140Level,
    CommonCriteriaLevel,
    HSMKeyHandle,
    HSMSignature,
    HSMHealthReport,
    HSMStatus,
    BankingComplianceStatus,
};

// Keep legacy quantum module for compatibility
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

#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(not(feature = "std"))]
use spin::Mutex as Arc;

/// Production cryptographic provider for QENEX banking system
/// Uses real post-quantum algorithms for maximum security
pub struct CryptoProvider {
    pq_engine: Arc<PostQuantumEngine>,
    legacy_engine: Arc<QuantumResistantEngine>,
}

impl CryptoProvider {
    /// Create new crypto provider with post-quantum algorithms
    pub fn new() -> Result<Self, PQCryptoError> {
        Ok(Self {
            pq_engine: Arc::new(PostQuantumEngine::new()?),
            legacy_engine: Arc::new(QuantumResistantEngine::new()),
        })
    }
    
    /// Create crypto provider with Hardware Security Module integration
    #[cfg(feature = "pkcs11")]
    pub fn with_hsm(hsm_config: quantum::HSMConfig) -> Result<Self, PQCryptoError> {
        let pq_engine = PostQuantumEngine::new()?;
        let legacy_engine = QuantumResistantEngine::with_hsm(hsm_config)
            .map_err(|_| PQCryptoError::HSMError)?;
            
        Ok(Self {
            pq_engine: Arc::new(pq_engine),
            legacy_engine: Arc::new(legacy_engine),
        })
    }
    
    /// Get post-quantum cryptographic engine (primary)
    pub fn pq_engine(&self) -> &PostQuantumEngine {
        &self.pq_engine
    }
    
    /// Get legacy quantum engine (for compatibility)
    pub fn legacy_engine(&self) -> &QuantumResistantEngine {
        &self.legacy_engine
    }
    
    /// Create banking cryptographic context
    pub fn create_banking_context(&mut self, bank_id: &str) -> Result<BankingCryptoContext, PQCryptoError> {
        // Use the real post-quantum implementation
        let engine = Arc::get_mut(&mut self.pq_engine)
            .ok_or(PQCryptoError::KeyGenerationFailed)?;
        engine.create_banking_context(bank_id)
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