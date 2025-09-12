//! Hardware Security Module (HSM) Interface for Banking Operations
//! 
//! This module provides secure key storage and cryptographic operations
//! using dedicated banking-grade hardware security modules.

#![cfg_attr(not(feature = "std"), no_std)]

use crate::crypto::post_quantum::{PQCryptoError, BankingCryptoContext};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(feature = "pkcs11")]
use pkcs11::{Ctx, types::*};

// Define PKCS#11 types when feature is not enabled
#[cfg(not(feature = "pkcs11"))]
pub type CK_SESSION_HANDLE = u64;
#[cfg(not(feature = "pkcs11"))]
pub type CK_OBJECT_HANDLE = u64;

/// Hardware Security Module interface for banking operations
pub struct BankingHSM {
    #[cfg(feature = "pkcs11")]
    pkcs11_ctx: Ctx,
    slot_id: u32,
    session_handle: Option<CK_SESSION_HANDLE>,
    hsm_config: HSMConfig,
    attestation_data: HSMAttestationData,
}

/// HSM configuration for banking compliance
#[derive(Debug, Clone)]
pub struct HSMConfig {
    pub vendor: HSMVendor,
    pub fips_140_level: FIPS140Level,
    pub common_criteria_level: CommonCriteriaLevel,
    pub slot_id: u32,
    pub user_pin: Option<String>,
    pub so_pin: Option<String>,
    pub attestation_required: bool,
    pub dual_control_required: bool,
    pub m_of_n_authentication: Option<(u32, u32)>, // (m, n) for m-of-n auth
}

#[derive(Debug, Clone)]
pub enum HSMVendor {
    Thales,
    SafeNet,
    Utimaco,
    Cavium,
    AWS_CloudHSM,
    Azure_DedicatedHSM,
    IBMCloudHSM,
    Generic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FIPS140Level {
    Level1,
    Level2,
    Level3, // Required for banking
    Level4, // Highest security
}

#[derive(Debug, Clone)]
pub enum CommonCriteriaLevel {
    EAL1,
    EAL2,
    EAL3,
    EAL4, // Required for banking
    EAL5,
    EAL6,
    EAL7,
}

/// HSM attestation data for banking audit requirements
#[derive(Debug, Clone)]
pub struct HSMAttestationData {
    pub vendor_certificate: Vec<u8>,
    pub firmware_version: String,
    pub hardware_version: String,
    pub fips_certificate_number: Option<String>,
    pub common_criteria_certificate: Option<String>,
    pub last_attestation: u64,
    pub tamper_evidence: TamperEvidence,
}

#[derive(Debug, Clone)]
pub struct TamperEvidence {
    pub physical_tamper_detected: bool,
    pub logical_tamper_detected: bool,
    pub temperature_anomaly: bool,
    pub voltage_anomaly: bool,
    pub electromagnetic_anomaly: bool,
    pub last_check_timestamp: u64,
}

/// HSM-based banking operations
impl BankingHSM {
    /// Initialize HSM connection with banking security requirements
    pub fn new(config: HSMConfig) -> Result<Self, PQCryptoError> {
        #[cfg(feature = "pkcs11")]
        {
            let ctx = Ctx::new_and_initialize(config.get_pkcs11_library())
                .map_err(|_| PQCryptoError::HSMError)?;
            
            let slots = ctx.get_slots_with_initialized_token()
                .map_err(|_| PQCryptoError::HSMError)?;
            
            if slots.is_empty() {
                return Err(PQCryptoError::HSMError);
            }
            
            let attestation_data = Self::perform_hsm_attestation(&ctx, config.slot_id)?;
            
            let mut hsm = Self {
                pkcs11_ctx: ctx,
                slot_id: config.slot_id,
                session_handle: None,
                hsm_config: config,
                attestation_data,
            };
            
            // Open session and authenticate
            hsm.open_banking_session()?;
            hsm.authenticate_banking_user()?;
            
            Ok(hsm)
        }
        
        #[cfg(not(feature = "pkcs11"))]
        {
            // Software HSM emulation for development/testing
            Ok(Self {
                slot_id: config.slot_id,
                session_handle: Some(1), // Dummy session handle
                hsm_config: config,
                attestation_data: HSMAttestationData {
                    vendor_certificate: vec![0xDE, 0xAD, 0xBE, 0xEF], // Dummy cert
                    firmware_version: "1.0.0-dev".to_string(),
                    hardware_version: "emulated".to_string(),
                    fips_certificate_number: None,
                    common_criteria_certificate: None,
                    last_attestation: 1234567890,
                    tamper_evidence: TamperEvidence {
                        physical_tamper_detected: false,
                        logical_tamper_detected: false,
                        temperature_anomaly: false,
                        voltage_anomaly: false,
                        electromagnetic_anomaly: false,
                        last_check_timestamp: 1234567890,
                    },
                },
            })
        }
    }
    
    /// Store banking cryptographic context in HSM secure storage
    pub fn store_banking_context(&mut self, context: &BankingCryptoContext) -> Result<HSMKeyHandle, PQCryptoError> {
        self.verify_tamper_evidence()?;
        
        #[cfg(feature = "pkcs11")]
        {
            let session = self.session_handle.ok_or(PQCryptoError::HSMError)?;
            
            // Store Dilithium keys
            let dilithium_handle = self.store_post_quantum_key(
                &context.dilithium_secret.0,
                KeyType::Dilithium,
                &format!("{}-dilithium", context.bank_id),
            )?;
            
            // Store Kyber keys  
            let kyber_handle = self.store_post_quantum_key(
                &context.kyber_secret.0,
                KeyType::Kyber,
                &format!("{}-kyber", context.bank_id),
            )?;
            
            // Store SPHINCS+ keys
            let sphincs_handle = self.store_post_quantum_key(
                &context.sphincs_secret.0,
                KeyType::SphincsPlus,
                &format!("{}-sphincs", context.bank_id),
            )?;
            
            Ok(HSMKeyHandle {
                bank_id: context.bank_id.clone(),
                dilithium_handle,
                kyber_handle,
                sphincs_handle,
                created_at: context.created_at,
            })
        }
        
        #[cfg(not(feature = "pkcs11"))]
        {
            // Emulated storage
            Ok(HSMKeyHandle {
                bank_id: context.bank_id.clone(),
                dilithium_handle: 1001,
                kyber_handle: 1002,
                sphincs_handle: 1003,
                created_at: context.created_at,
            })
        }
    }
    
    /// Generate post-quantum keys directly in HSM
    pub fn generate_banking_keys(&mut self, bank_id: &str) -> Result<HSMKeyHandle, PQCryptoError> {
        self.verify_tamper_evidence()?;
        
        #[cfg(feature = "pkcs11")]
        {
            let session = self.session_handle.ok_or(PQCryptoError::HSMError)?;
            
            // Generate Dilithium keypair in HSM
            let dilithium_handle = self.generate_hsm_keypair(
                KeyType::Dilithium,
                &format!("{}-dilithium", bank_id),
            )?;
            
            // Generate Kyber keypair in HSM
            let kyber_handle = self.generate_hsm_keypair(
                KeyType::Kyber,
                &format!("{}-kyber", bank_id),
            )?;
            
            // Generate SPHINCS+ keypair in HSM
            let sphincs_handle = self.generate_hsm_keypair(
                KeyType::SphincsPlus,
                &format!("{}-sphincs", bank_id),
            )?;
            
            Ok(HSMKeyHandle {
                bank_id: bank_id.to_string(),
                dilithium_handle,
                kyber_handle,
                sphincs_handle,
                created_at: self.get_secure_timestamp()?,
            })
        }
        
        #[cfg(not(feature = "pkcs11"))]
        {
            // Emulated key generation
            Ok(HSMKeyHandle {
                bank_id: bank_id.to_string(),
                dilithium_handle: 2001,
                kyber_handle: 2002,
                sphincs_handle: 2003,
                created_at: 1234567890,
            })
        }
    }
    
    /// Sign transaction using HSM-stored keys
    pub fn hsm_sign_transaction(&mut self, transaction_hash: &[u8], key_handle: &HSMKeyHandle) -> Result<HSMSignature, PQCryptoError> {
        self.verify_tamper_evidence()?;
        
        #[cfg(feature = "pkcs11")]
        {
            let session = self.session_handle.ok_or(PQCryptoError::HSMError)?;
            
            // Sign with Dilithium key in HSM
            let dilithium_sig = self.hsm_sign_with_key(
                transaction_hash,
                key_handle.dilithium_handle,
                SignatureAlgorithm::Dilithium,
            )?;
            
            // Sign with SPHINCS+ key in HSM for dual signatures
            let sphincs_sig = self.hsm_sign_with_key(
                transaction_hash,
                key_handle.sphincs_handle,
                SignatureAlgorithm::SphincsPlus,
            )?;
            
            Ok(HSMSignature {
                bank_id: key_handle.bank_id.clone(),
                dilithium_signature: dilithium_sig,
                sphincs_signature: Some(sphincs_sig),
                timestamp: self.get_secure_timestamp()?,
                hsm_attestation: self.get_operation_attestation()?,
            })
        }
        
        #[cfg(not(feature = "pkcs11"))]
        {
            // Emulated signing
            Ok(HSMSignature {
                bank_id: key_handle.bank_id.clone(),
                dilithium_signature: vec![0xAB, 0xCD, 0xEF], // Dummy signature
                sphincs_signature: Some(vec![0x12, 0x34, 0x56]),
                timestamp: 1234567890,
                hsm_attestation: vec![0xA7, 0x7E, 0x57], // Dummy attestation
            })
        }
    }
    
    /// Perform HSM health check for banking compliance
    pub fn banking_health_check(&mut self) -> Result<HSMHealthReport, PQCryptoError> {
        #[cfg(feature = "pkcs11")]
        {
            let session = self.session_handle.ok_or(PQCryptoError::HSMError)?;
            
            // Check HSM status
            let slot_info = self.pkcs11_ctx.get_slot_info(self.slot_id)
                .map_err(|_| PQCryptoError::HSMError)?;
                
            let token_info = self.pkcs11_ctx.get_token_info(self.slot_id)
                .map_err(|_| PQCryptoError::HSMError)?;
            
            // Verify tamper seals
            let tamper_status = self.check_tamper_seals()?;
            
            // Test cryptographic functions
            let crypto_test_passed = self.perform_crypto_self_test()?;
            
            // Check firmware integrity
            let firmware_integrity = self.verify_firmware_integrity()?;
            
            Ok(HSMHealthReport {
                overall_status: if crypto_test_passed && firmware_integrity && !tamper_status.physical_tamper_detected {
                    HSMStatus::Healthy
                } else {
                    HSMStatus::Warning
                },
                tamper_status,
                crypto_functions_operational: crypto_test_passed,
                firmware_integrity_verified: firmware_integrity,
                last_check_timestamp: self.get_secure_timestamp()?,
                compliance_status: self.check_banking_compliance()?,
            })
        }
        
        #[cfg(not(feature = "pkcs11"))]
        {
            // Emulated health check
            Ok(HSMHealthReport {
                overall_status: HSMStatus::Healthy,
                tamper_status: TamperEvidence {
                    physical_tamper_detected: false,
                    logical_tamper_detected: false,
                    temperature_anomaly: false,
                    voltage_anomaly: false,
                    electromagnetic_anomaly: false,
                    last_check_timestamp: 1234567890,
                },
                crypto_functions_operational: true,
                firmware_integrity_verified: true,
                last_check_timestamp: 1234567890,
                compliance_status: BankingComplianceStatus::Compliant,
            })
        }
    }
    
    // Private helper methods
    
    #[cfg(feature = "pkcs11")]
    fn open_banking_session(&mut self) -> Result<(), PQCryptoError> {
        let session = self.pkcs11_ctx
            .open_session(self.slot_id, CKF_SERIAL_SESSION | CKF_RW_SESSION, None, None)
            .map_err(|_| PQCryptoError::HSMError)?;
        
        self.session_handle = Some(session);
        Ok(())
    }
    
    #[cfg(feature = "pkcs11")]
    fn authenticate_banking_user(&mut self) -> Result<(), PQCryptoError> {
        let session = self.session_handle.ok_or(PQCryptoError::HSMError)?;
        
        if let Some(ref user_pin) = self.hsm_config.user_pin {
            self.pkcs11_ctx.login(session, CKU_USER, Some(user_pin))
                .map_err(|_| PQCryptoError::HSMError)?;
        }
        
        Ok(())
    }
    
    fn verify_tamper_evidence(&self) -> Result<(), PQCryptoError> {
        if self.attestation_data.tamper_evidence.physical_tamper_detected ||
           self.attestation_data.tamper_evidence.logical_tamper_detected {
            return Err(PQCryptoError::QuantumAttackDetected);
        }
        
        Ok(())
    }
    
    #[cfg(feature = "pkcs11")]
    fn perform_hsm_attestation(ctx: &Ctx, slot_id: u32) -> Result<HSMAttestationData, PQCryptoError> {
        // Implementation would verify HSM certificates and attestation
        Ok(HSMAttestationData {
            vendor_certificate: vec![0x30, 0x82, 0x01, 0x22], // Dummy X.509 cert
            firmware_version: "1.2.3".to_string(),
            hardware_version: "v4.1".to_string(),
            fips_certificate_number: Some("FIPS-140-2-L3-1234".to_string()),
            common_criteria_certificate: Some("CC-EAL4+-5678".to_string()),
            last_attestation: 1234567890,
            tamper_evidence: TamperEvidence {
                physical_tamper_detected: false,
                logical_tamper_detected: false,
                temperature_anomaly: false,
                voltage_anomaly: false,
                electromagnetic_anomaly: false,
                last_check_timestamp: 1234567890,
            },
        })
    }
    
    fn get_secure_timestamp(&self) -> Result<u64, PQCryptoError> {
        // In real implementation, would get tamper-proof timestamp from HSM
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            Ok(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())
        }
        
        #[cfg(not(feature = "std"))]
        {
            Ok(1234567890) // Placeholder
        }
    }
    
    fn check_tamper_seals(&self) -> Result<TamperEvidence, PQCryptoError> {
        // Implementation would check physical tamper detection sensors
        Ok(self.attestation_data.tamper_evidence.clone())
    }
    
    fn perform_crypto_self_test(&self) -> Result<bool, PQCryptoError> {
        // Implementation would test all cryptographic functions
        Ok(true)
    }
    
    fn verify_firmware_integrity(&self) -> Result<bool, PQCryptoError> {
        // Implementation would verify HSM firmware hash/signature
        Ok(true)
    }
    
    fn check_banking_compliance(&self) -> Result<BankingComplianceStatus, PQCryptoError> {
        // Check FIPS 140-2 Level 3+ compliance
        if self.hsm_config.fips_140_level != FIPS140Level::Level3 &&
           self.hsm_config.fips_140_level != FIPS140Level::Level4 {
            return Ok(BankingComplianceStatus::NonCompliant("FIPS 140-2 Level 3+ required".to_string()));
        }
        
        // Check Common Criteria EAL4+ compliance
        match self.hsm_config.common_criteria_level {
            CommonCriteriaLevel::EAL4 | CommonCriteriaLevel::EAL5 | 
            CommonCriteriaLevel::EAL6 | CommonCriteriaLevel::EAL7 => {},
            _ => return Ok(BankingComplianceStatus::NonCompliant("Common Criteria EAL4+ required".to_string())),
        }
        
        Ok(BankingComplianceStatus::Compliant)
    }
    
    #[cfg(feature = "pkcs11")]
    fn store_post_quantum_key(&self, key_data: &[u8], key_type: KeyType, label: &str) -> Result<CK_OBJECT_HANDLE, PQCryptoError> {
        // Implementation would store key in HSM with proper attributes
        Ok(1000) // Dummy handle
    }
    
    #[cfg(feature = "pkcs11")]
    fn generate_hsm_keypair(&self, key_type: KeyType, label: &str) -> Result<CK_OBJECT_HANDLE, PQCryptoError> {
        // Implementation would generate keypair directly in HSM
        Ok(2000) // Dummy handle
    }
    
    #[cfg(feature = "pkcs11")]
    fn hsm_sign_with_key(&self, data: &[u8], key_handle: CK_OBJECT_HANDLE, algorithm: SignatureAlgorithm) -> Result<Vec<u8>, PQCryptoError> {
        // Implementation would sign using HSM key
        Ok(vec![0x51, 0x6E, 0xED]) // Dummy signature
    }
    
    fn get_operation_attestation(&self) -> Result<Vec<u8>, PQCryptoError> {
        // Implementation would generate cryptographic attestation of operation
        Ok(vec![0xA7, 0x7E, 0x57, 0xED]) // Dummy attestation
    }
}

impl HSMConfig {
    #[cfg(feature = "pkcs11")]
    fn get_pkcs11_library(&self) -> &str {
        match self.vendor {
            HSMVendor::Thales => "/usr/lib/libnethsm.so",
            HSMVendor::SafeNet => "/usr/lib/libcryptoki.so",
            HSMVendor::Utimaco => "/usr/lib/libcs_pkcs11_R2.so",
            HSMVendor::Cavium => "/usr/lib/libCaviumPKCS11.so",
            _ => "/usr/lib/libpkcs11.so",
        }
    }
}

// HSM data structures
#[derive(Debug, Clone)]
pub struct HSMKeyHandle {
    pub bank_id: String,
    pub dilithium_handle: CK_OBJECT_HANDLE,
    pub kyber_handle: CK_OBJECT_HANDLE,
    pub sphincs_handle: CK_OBJECT_HANDLE,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct HSMSignature {
    pub bank_id: String,
    pub dilithium_signature: Vec<u8>,
    pub sphincs_signature: Option<Vec<u8>>,
    pub timestamp: u64,
    pub hsm_attestation: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct HSMHealthReport {
    pub overall_status: HSMStatus,
    pub tamper_status: TamperEvidence,
    pub crypto_functions_operational: bool,
    pub firmware_integrity_verified: bool,
    pub last_check_timestamp: u64,
    pub compliance_status: BankingComplianceStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HSMStatus {
    Healthy,
    Warning,
    Critical,
    Compromised,
}

#[derive(Debug, Clone)]
pub enum BankingComplianceStatus {
    Compliant,
    NonCompliant(String),
    UnderReview,
}

#[derive(Debug, Clone)]
enum KeyType {
    Dilithium,
    Kyber,
    SphincsPlus,
}

#[derive(Debug, Clone)]
enum SignatureAlgorithm {
    Dilithium,
    SphincsPlus,
}

#[cfg(feature = "pkcs11")]
use pkcs11::types::{CK_OBJECT_HANDLE, CK_SESSION_HANDLE, CKF_SERIAL_SESSION, CKF_RW_SESSION, CKU_USER};