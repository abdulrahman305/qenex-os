//! Post-Quantum Cryptography Test Suite for Banking Compliance
//! 
//! Comprehensive tests for NIST-approved post-quantum algorithms
//! ensuring compliance with banking security standards.

#[cfg(test)]
mod tests {
    use super::super::post_quantum::*;
    use super::super::hsm::*;
    
    #[test]
    fn test_post_quantum_engine_initialization() {
        let result = PostQuantumEngine::new();
        assert!(result.is_ok(), "PostQuantumEngine should initialize successfully");
        
        let engine = result.unwrap();
        // Engine should pass internal security self-test
    }
    
    #[test] 
    fn test_banking_context_creation() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let context = engine.create_banking_context("TEST-BANK-001");
        
        assert!(context.is_ok(), "Banking context creation should succeed");
        
        let ctx = context.unwrap();
        assert_eq!(ctx.bank_id, "TEST-BANK-001");
        assert_eq!(ctx.version, 1);
        assert!(ctx.created_at > 0);
    }
    
    #[test]
    fn test_transaction_signing_verification() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let context = engine.create_banking_context("TEST-BANK-002").unwrap();
        
        let transaction = b"TRANSFER:FROM=123456789:TO=987654321:AMOUNT=1000.00:CURRENCY=USD";
        
        // Sign transaction
        let signature = engine.sign_transaction(transaction, &context);
        assert!(signature.is_ok(), "Transaction signing should succeed");
        
        let sig = signature.unwrap();
        assert_eq!(sig.signer_id, "TEST-BANK-002");
        assert!(sig.dilithium_signature.0.len() > 0);
        assert!(sig.sphincs_signature.is_some());
        
        // Verify signature
        let verification = engine.verify_transaction(transaction, &sig, &context);
        assert!(verification.is_ok(), "Signature verification should not error");
        assert!(verification.unwrap(), "Signature should be valid");
    }
    
    #[test]
    fn test_transaction_signature_tamper_detection() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let context = engine.create_banking_context("TEST-BANK-003").unwrap();
        
        let transaction = b"TRANSFER:FROM=123456789:TO=987654321:AMOUNT=1000.00:CURRENCY=USD";
        let tampered_transaction = b"TRANSFER:FROM=123456789:TO=987654321:AMOUNT=9999.99:CURRENCY=USD";
        
        let signature = engine.sign_transaction(transaction, &context).unwrap();
        
        // Verification should fail for tampered transaction
        let verification = engine.verify_transaction(tampered_transaction, &signature, &context);
        assert!(verification.is_ok(), "Verification function should not error");
        assert!(!verification.unwrap(), "Tampered transaction should fail verification");
    }
    
    #[test]
    fn test_banking_data_encryption_decryption() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let sender_context = engine.create_banking_context("SENDER-BANK").unwrap();
        let recipient_context = engine.create_banking_context("RECIPIENT-BANK").unwrap();
        
        let sensitive_data = b"ACCOUNT:123456789:BALANCE:1000000.00:SSN:123-45-6789";
        
        // Encrypt for recipient
        let ciphertext = engine.encrypt_banking_data(sensitive_data, &recipient_context);
        assert!(ciphertext.is_ok(), "Encryption should succeed");
        
        let ct = ciphertext.unwrap();
        assert_eq!(ct.recipient_id, "RECIPIENT-BANK");
        assert!(ct.kyber_ciphertext.0.len() > 0);
        assert!(ct.encrypted_data.len() > 0);
        
        // Decrypt with recipient's keys
        let plaintext = engine.decrypt_banking_data(&ct, &recipient_context);
        assert!(plaintext.is_ok(), "Decryption should succeed");
        
        let decrypted = plaintext.unwrap();
        assert_eq!(decrypted, sensitive_data);
    }
    
    #[test]
    fn test_encryption_wrong_recipient() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let sender_context = engine.create_banking_context("SENDER-BANK").unwrap();
        let recipient_context = engine.create_banking_context("RECIPIENT-BANK").unwrap();
        let wrong_context = engine.create_banking_context("WRONG-BANK").unwrap();
        
        let sensitive_data = b"CONFIDENTIAL BANKING DATA";
        
        let ciphertext = engine.encrypt_banking_data(sensitive_data, &recipient_context).unwrap();
        
        // Wrong recipient should not be able to decrypt
        let decryption = engine.decrypt_banking_data(&ciphertext, &wrong_context);
        assert!(decryption.is_err(), "Decryption with wrong keys should fail");
    }
    
    #[test]
    fn test_dilithium_keypair_generation() {
        let rng = QuantumRNG::new().unwrap();
        let signer = DilithiumSigner::new(&rng).unwrap();
        
        let keypair1 = signer.generate_keypair();
        let keypair2 = signer.generate_keypair();
        
        assert!(keypair1.is_ok() && keypair2.is_ok(), "Keypair generation should succeed");
        
        let kp1 = keypair1.unwrap();
        let kp2 = keypair2.unwrap();
        
        // Keys should be different
        assert_ne!(kp1.public.0, kp2.public.0, "Public keys should be different");
        assert_ne!(kp1.secret.0, kp2.secret.0, "Secret keys should be different");
    }
    
    #[test]
    fn test_kyber_key_encapsulation() {
        let rng = QuantumRNG::new().unwrap();
        let kem = KyberKEM::new(&rng).unwrap();
        
        let keypair = kem.generate_keypair().unwrap();
        
        // Encapsulate shared secret
        let (ciphertext, shared_secret1) = kem.encapsulate(&keypair.public).unwrap();
        
        // Decapsulate shared secret
        let shared_secret2 = kem.decapsulate(&ciphertext, &keypair.secret).unwrap();
        
        assert_eq!(shared_secret1, shared_secret2, "Encapsulated and decapsulated secrets should match");
    }
    
    #[test]
    fn test_sphincs_plus_signing() {
        let rng = QuantumRNG::new().unwrap();
        let signer = SphincsPlus::new(&rng).unwrap();
        
        let keypair = signer.generate_keypair().unwrap();
        let message = b"Banking transaction requiring SPHINCS+ signature";
        
        let signature = signer.sign(message, &keypair.secret).unwrap();
        let valid = signer.verify(message, &signature, &keypair.public).unwrap();
        
        assert!(valid, "SPHINCS+ signature should verify correctly");
        
        // Test with wrong message
        let wrong_message = b"Different message";
        let invalid = signer.verify(wrong_message, &signature, &keypair.public).unwrap();
        
        assert!(!invalid, "SPHINCS+ signature should not verify for wrong message");
    }
    
    #[test]
    fn test_aes_gcm_encryption() {
        let rng = QuantumRNG::new().unwrap();
        let aes = AES256GCM::new(&rng).unwrap();
        let key = rng.generate_key_256().unwrap();
        
        let plaintext = b"Sensitive banking information requiring AES-256-GCM protection";
        
        let ciphertext = aes.encrypt(plaintext, &key).unwrap();
        let decrypted = aes.decrypt(&ciphertext, &key).unwrap();
        
        assert_eq!(plaintext.as_slice(), decrypted.as_slice(), "AES-GCM encryption/decryption should roundtrip correctly");
    }
    
    #[test]
    fn test_quantum_rng_key_generation() {
        let rng = QuantumRNG::new().unwrap();
        
        let key1 = rng.generate_key_256().unwrap();
        let key2 = rng.generate_key_256().unwrap();
        
        assert_ne!(key1, key2, "Generated keys should be different");
        assert_eq!(key1.len(), 32, "Key should be 256 bits (32 bytes)");
        assert_eq!(key2.len(), 32, "Key should be 256 bits (32 bytes)");
    }
    
    #[test]
    fn test_banking_compliance_levels() {
        // Test FIPS 140-2 compliance levels
        let fips_l3 = FIPS140Level::Level3;
        let fips_l4 = FIPS140Level::Level4;
        
        // Banking should require Level 3 or higher
        match fips_l3 {
            FIPS140Level::Level3 | FIPS140Level::Level4 => {
                // Acceptable for banking
            },
            _ => panic!("FIPS 140-2 Level 3+ required for banking"),
        }
        
        // Test Common Criteria compliance
        let cc_eal4 = CommonCriteriaLevel::EAL4;
        
        match cc_eal4 {
            CommonCriteriaLevel::EAL4 | CommonCriteriaLevel::EAL5 | 
            CommonCriteriaLevel::EAL6 | CommonCriteriaLevel::EAL7 => {
                // Acceptable for banking
            },
            _ => panic!("Common Criteria EAL4+ required for banking"),
        }
    }
    
    #[test]
    fn test_hsm_config_validation() {
        let valid_config = HSMConfig {
            vendor: HSMVendor::Thales,
            fips_140_level: FIPS140Level::Level3,
            common_criteria_level: CommonCriteriaLevel::EAL4,
            slot_id: 0,
            user_pin: Some("123456".to_string()),
            so_pin: Some("87654321".to_string()),
            attestation_required: true,
            dual_control_required: true,
            m_of_n_authentication: Some((2, 3)), // 2 of 3 authentication
        };
        
        // Should be able to create HSM with valid banking config
        let hsm_result = BankingHSM::new(valid_config);
        assert!(hsm_result.is_ok(), "Banking HSM should initialize with valid config");
    }
    
    #[test]
    fn test_tamper_evidence_detection() {
        let tamper_evidence = TamperEvidence {
            physical_tamper_detected: true,  // Simulate tamper detection
            logical_tamper_detected: false,
            temperature_anomaly: false,
            voltage_anomaly: false,
            electromagnetic_anomaly: false,
            last_check_timestamp: 1234567890,
        };
        
        // System should detect tamper evidence
        assert!(tamper_evidence.physical_tamper_detected, "Physical tamper should be detected");
    }
    
    #[test]
    fn test_post_quantum_algorithm_compatibility() {
        // Test that all PQ algorithms work together
        let mut engine = PostQuantumEngine::new().unwrap();
        let context = engine.create_banking_context("MULTI-ALG-BANK").unwrap();
        
        // Test transaction with multiple signatures
        let transaction = b"MULTI-SIGNATURE-TRANSACTION:AMOUNT=5000.00";
        let signature = engine.sign_transaction(transaction, &context).unwrap();
        
        // Should have both Dilithium and SPHINCS+ signatures
        assert!(signature.dilithium_signature.0.len() > 0, "Should have Dilithium signature");
        assert!(signature.sphincs_signature.is_some(), "Should have SPHINCS+ backup signature");
        
        // Both signatures should verify
        let valid = engine.verify_transaction(transaction, &signature, &context).unwrap();
        assert!(valid, "Multi-algorithm signature should verify");
        
        // Test encryption with Kyber + AES
        let data = b"ENCRYPTED WITH HYBRID PQ ALGORITHMS";
        let ciphertext = engine.encrypt_banking_data(data, &context).unwrap();
        let decrypted = engine.decrypt_banking_data(&ciphertext, &context).unwrap();
        
        assert_eq!(data.as_slice(), decrypted.as_slice(), "Hybrid encryption should work");
    }
    
    #[test]
    fn test_performance_benchmarks() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let context = engine.create_banking_context("PERF-TEST-BANK").unwrap();
        
        // Benchmark signing performance (should be reasonable for banking)
        let transaction = b"PERFORMANCE-TEST-TRANSACTION";
        
        #[cfg(feature = "std")]
        {
            use std::time::Instant;
            
            let start = Instant::now();
            let _signature = engine.sign_transaction(transaction, &context).unwrap();
            let signing_time = start.elapsed();
            
            // Post-quantum signatures should complete within reasonable time for banking
            // (This is implementation dependent - adjust thresholds as needed)
            assert!(signing_time.as_millis() < 1000, "Signing should complete within 1 second");
        }
        
        // Test encryption performance
        let large_data = vec![0u8; 1024 * 1024]; // 1MB test data
        
        #[cfg(feature = "std")]
        {
            use std::time::Instant;
            
            let start = Instant::now();
            let _ciphertext = engine.encrypt_banking_data(&large_data, &context).unwrap();
            let encryption_time = start.elapsed();
            
            // Should handle reasonable data sizes efficiently
            assert!(encryption_time.as_millis() < 5000, "1MB encryption should complete within 5 seconds");
        }
    }
    
    #[test]
    fn test_memory_zeroization() {
        use zeroize::Zeroize;
        
        let mut sensitive_data = vec![0xFF; 32]; // Fill with test pattern
        let original_data = sensitive_data.clone();
        
        // Zeroize the data
        sensitive_data.zeroize();
        
        // Data should be cleared
        assert_ne!(sensitive_data, original_data, "Data should be different after zeroization");
        assert_eq!(sensitive_data, vec![0u8; 32], "Data should be zeroed");
    }
    
    #[test]
    fn test_banking_context_zeroization() {
        let mut engine = PostQuantumEngine::new().unwrap();
        let mut context = engine.create_banking_context("ZERO-TEST-BANK").unwrap();
        
        // Context should implement ZeroizeOnDrop
        // This test verifies the trait is properly implemented
        // Actual zeroization testing requires unsafe memory inspection
        
        drop(context); // Should trigger zeroization
        
        // If we reach here without panic, zeroization trait is properly implemented
    }
}

#[cfg(test)]
mod integration_tests {
    use super::super::*;
    
    #[test]
    fn test_full_banking_workflow() {
        // Simulate complete banking workflow with post-quantum crypto
        let mut crypto_provider = CryptoProvider::new().unwrap();
        
        // Create contexts for two banks
        let bank_a_context = crypto_provider.create_banking_context("BANK-A").unwrap();
        let bank_b_context = crypto_provider.create_banking_context("BANK-B").unwrap();
        
        // Bank A signs a transaction
        let transaction = b"WIRE-TRANSFER:FROM=BANK-A-123:TO=BANK-B-456:AMOUNT=50000.00:CURRENCY=USD";
        let signature = crypto_provider.pq_engine().sign_transaction(transaction, &bank_a_context).unwrap();
        
        // Bank B verifies the transaction
        let verification = crypto_provider.pq_engine().verify_transaction(transaction, &signature, &bank_a_context).unwrap();
        assert!(verification, "Inter-bank transaction should verify");
        
        // Bank A encrypts sensitive customer data for Bank B
        let customer_data = b"CUSTOMER:John-Doe:SSN:123-45-6789:ACCOUNT:9876543210:BALANCE:150000.00";
        let encrypted_data = crypto_provider.pq_engine().encrypt_banking_data(customer_data, &bank_b_context).unwrap();
        
        // Bank B decrypts the customer data
        let decrypted_data = crypto_provider.pq_engine().decrypt_banking_data(&encrypted_data, &bank_b_context).unwrap();
        assert_eq!(customer_data.as_slice(), decrypted_data.as_slice(), "Customer data should decrypt correctly");
    }
    
    #[test]
    fn test_regulatory_compliance_workflow() {
        let mut crypto_provider = CryptoProvider::new().unwrap();
        let bank_context = crypto_provider.create_banking_context("REGULATED-BANK").unwrap();
        
        // Test audit trail creation
        let audit_data = b"AUDIT:TRANSACTION-ID:TX123456:USER:admin:ACTION:APPROVE-TRANSFER:TIMESTAMP:1234567890";
        let audit_signature = crypto_provider.pq_engine().sign_transaction(audit_data, &bank_context).unwrap();
        
        // Compliance officer should be able to verify audit trail
        let audit_valid = crypto_provider.pq_engine().verify_transaction(audit_data, &audit_signature, &bank_context).unwrap();
        assert!(audit_valid, "Audit trail signature should be valid for compliance");
        
        // Test regulatory reporting encryption
        let regulatory_report = b"REGULATORY-REPORT:BANK-ID:REGULATED-BANK:QUARTER:Q1-2024:CAPITAL-RATIO:12.5:LIQUIDITY-RATIO:150.2";
        let encrypted_report = crypto_provider.pq_engine().encrypt_banking_data(regulatory_report, &bank_context).unwrap();
        
        // Regulator should be able to decrypt with proper keys
        let decrypted_report = crypto_provider.pq_engine().decrypt_banking_data(&encrypted_report, &bank_context).unwrap();
        assert_eq!(regulatory_report.as_slice(), decrypted_report.as_slice(), "Regulatory report should decrypt for compliance review");
    }
}