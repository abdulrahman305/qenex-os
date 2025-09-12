use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::RwLock;
use sqlx::PgPool;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftMessage {
    pub id: Uuid,
    pub message_type: SwiftMessageType,
    pub sender_bic: String,
    pub receiver_bic: String,
    pub transaction_reference: String,
    pub amount: rust_decimal::Decimal,
    pub currency: String,
    pub value_date: DateTime<Utc>,
    pub ordering_customer: String,
    pub beneficiary: String,
    pub remittance_info: Option<String>,
    pub charges: SwiftCharges,
    pub created_at: DateTime<Utc>,
    pub sent_at: Option<DateTime<Utc>>,
    pub status: MessageStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwiftMessageType {
    MT103, // Single Customer Credit Transfer
    MT202, // General Financial Institution Transfer
    MT950, // Statement Message
    MT940, // Customer Statement Message
    MT910, // Confirmation of Credit
    MT900, // Confirmation of Debit
    MT199, // Free Format Message
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwiftCharges {
    OUR,    // All charges borne by ordering customer
    BEN,    // All charges borne by beneficiary
    SHA,    // Charges shared
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageStatus {
    Draft,
    Queued,
    Sent,
    Acknowledged,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISO20022Message {
    pub id: Uuid,
    pub message_type: ISO20022MessageType,
    pub message_id: String,
    pub creation_date_time: DateTime<Utc>,
    pub initiating_party: String,
    pub payment_info: PaymentInfo,
    pub credit_transfer_info: Vec<CreditTransferInfo>,
    pub status: MessageStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ISO20022MessageType {
    Pacs008, // CustomerCreditTransferInitiation
    Pacs002, // PaymentStatusReport
    Pain001, // CustomerCreditTransferInitiation
    Camt053, // BankToCustomerStatement
    Camt054, // BankToCustomerDebitCreditNotification
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentInfo {
    pub payment_info_id: String,
    pub payment_method: String,
    pub requested_execution_date: DateTime<Utc>,
    pub debtor: PartyIdentification,
    pub debtor_account: AccountIdentification,
    pub debtor_agent: BankIdentification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditTransferInfo {
    pub payment_id: String,
    pub amount: rust_decimal::Decimal,
    pub currency: String,
    pub creditor: PartyIdentification,
    pub creditor_account: AccountIdentification,
    pub creditor_agent: BankIdentification,
    pub remittance_info: Option<String>,
    pub purpose_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartyIdentification {
    pub name: String,
    pub postal_address: Option<PostalAddress>,
    pub identification: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostalAddress {
    pub street_name: String,
    pub building_number: Option<String>,
    pub postal_code: String,
    pub town_name: String,
    pub country: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountIdentification {
    pub iban: Option<String>,
    pub other: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankIdentification {
    pub bic: String,
    pub name: Option<String>,
    pub postal_address: Option<PostalAddress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEPAPayment {
    pub id: Uuid,
    pub sepa_type: SEPAType,
    pub message_id: String,
    pub payment_info_id: String,
    pub debtor_iban: String,
    pub debtor_name: String,
    pub creditor_iban: String,
    pub creditor_name: String,
    pub amount: rust_decimal::Decimal,
    pub currency: String,
    pub end_to_end_id: String,
    pub remittance_info: Option<String>,
    pub execution_date: DateTime<Utc>,
    pub batch_booking: bool,
    pub created_at: DateTime<Utc>,
    pub status: MessageStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SEPAType {
    SCT,    // SEPA Credit Transfer
    SDD,    // SEPA Direct Debit
    INST,   // SEPA Instant Credit Transfer
}

pub struct BankingProtocolEngine {
    db: Arc<PgPool>,
    swift_config: SwiftConfig,
    sepa_config: SEPAConfig,
    message_queue: Arc<RwLock<Vec<ProtocolMessage>>>,
}

#[derive(Debug, Clone)]
pub struct SwiftConfig {
    pub bic: String,
    pub member_id: String,
    pub network_service: String,
    pub security_config: SwiftSecurityConfig,
}

#[derive(Debug, Clone)]
pub struct SwiftSecurityConfig {
    pub certificate_path: String,
    pub private_key_path: String,
    pub hsm_module: Option<String>,
    pub authentication_key: String,
}

#[derive(Debug, Clone)]
pub struct SEPAConfig {
    pub creditor_id: String,
    pub bic: String,
    pub country_code: String,
    pub scheme_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolMessage {
    Swift(SwiftMessage),
    ISO20022(ISO20022Message),
    SEPA(SEPAPayment),
}

impl BankingProtocolEngine {
    pub async fn new(
        db: Arc<PgPool>, 
        swift_config: SwiftConfig, 
        sepa_config: SEPAConfig
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            db,
            swift_config,
            sepa_config,
            message_queue: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn create_swift_mt103(&self, 
        sender_bic: String,
        receiver_bic: String,
        amount: rust_decimal::Decimal,
        currency: String,
        ordering_customer: String,
        beneficiary: String,
        remittance_info: Option<String>
    ) -> Result<SwiftMessage, Box<dyn std::error::Error + Send + Sync>> {
        
        let message = SwiftMessage {
            id: Uuid::new_v4(),
            message_type: SwiftMessageType::MT103,
            sender_bic,
            receiver_bic,
            transaction_reference: self.generate_transaction_reference().await?,
            amount,
            currency,
            value_date: Utc::now(),
            ordering_customer,
            beneficiary,
            remittance_info,
            charges: SwiftCharges::SHA,
            created_at: Utc::now(),
            sent_at: None,
            status: MessageStatus::Draft,
        };

        // Store in database
        self.store_swift_message(&message).await?;
        
        Ok(message)
    }

    pub async fn create_iso20022_pain001(&self,
        initiating_party: String,
        payment_info: PaymentInfo,
        credit_transfers: Vec<CreditTransferInfo>
    ) -> Result<ISO20022Message, Box<dyn std::error::Error + Send + Sync>> {
        
        let message = ISO20022Message {
            id: Uuid::new_v4(),
            message_type: ISO20022MessageType::Pain001,
            message_id: self.generate_message_id().await?,
            creation_date_time: Utc::now(),
            initiating_party,
            payment_info,
            credit_transfer_info: credit_transfers,
            status: MessageStatus::Draft,
            created_at: Utc::now(),
        };

        // Store in database
        self.store_iso20022_message(&message).await?;
        
        Ok(message)
    }

    pub async fn create_sepa_sct(&self,
        debtor_iban: String,
        debtor_name: String,
        creditor_iban: String,
        creditor_name: String,
        amount: rust_decimal::Decimal,
        remittance_info: Option<String>
    ) -> Result<SEPAPayment, Box<dyn std::error::Error + Send + Sync>> {
        
        // Validate IBAN
        self.validate_iban(&debtor_iban)?;
        self.validate_iban(&creditor_iban)?;
        
        let payment = SEPAPayment {
            id: Uuid::new_v4(),
            sepa_type: SEPAType::SCT,
            message_id: self.generate_message_id().await?,
            payment_info_id: self.generate_payment_info_id().await?,
            debtor_iban,
            debtor_name,
            creditor_iban,
            creditor_name,
            amount,
            currency: "EUR".to_string(),
            end_to_end_id: self.generate_end_to_end_id().await?,
            remittance_info,
            execution_date: Utc::now(),
            batch_booking: false,
            created_at: Utc::now(),
            status: MessageStatus::Draft,
        };

        // Store in database
        self.store_sepa_payment(&payment).await?;
        
        Ok(payment)
    }

    pub async fn send_swift_message(&self, message_id: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut message = self.get_swift_message(message_id).await?;
        
        // Convert to SWIFT format
        let swift_format = self.format_swift_message(&message).await?;
        
        // Sign message
        let signed_message = self.sign_swift_message(&swift_format).await?;
        
        // Send via SWIFT network
        self.transmit_swift_message(&signed_message).await?;
        
        // Update status
        message.status = MessageStatus::Sent;
        message.sent_at = Some(Utc::now());
        self.update_swift_message(&message).await?;
        
        Ok(())
    }

    pub async fn send_iso20022_message(&self, message_id: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut message = self.get_iso20022_message(message_id).await?;
        
        // Convert to XML format
        let xml_message = self.format_iso20022_xml(&message).await?;
        
        // Validate against XSD schema
        self.validate_iso20022_schema(&xml_message).await?;
        
        // Sign message
        let signed_xml = self.sign_iso20022_message(&xml_message).await?;
        
        // Send via appropriate channel
        self.transmit_iso20022_message(&signed_xml).await?;
        
        // Update status
        message.status = MessageStatus::Sent;
        self.update_iso20022_message(&message).await?;
        
        Ok(())
    }

    pub async fn process_sepa_batch(&self, payment_ids: Vec<Uuid>) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut payments = Vec::new();
        
        for payment_id in payment_ids {
            let payment = self.get_sepa_payment(payment_id).await?;
            payments.push(payment);
        }
        
        // Create SEPA XML batch
        let batch_xml = self.create_sepa_batch_xml(&payments).await?;
        
        // Validate batch
        self.validate_sepa_batch(&batch_xml).await?;
        
        // Sign batch
        let signed_batch = self.sign_sepa_batch(&batch_xml).await?;
        
        // Submit to clearing system
        let batch_id = self.submit_sepa_batch(&signed_batch).await?;
        
        // Update payment statuses
        for mut payment in payments {
            payment.status = MessageStatus::Sent;
            self.update_sepa_payment(&payment).await?;
        }
        
        Ok(batch_id)
    }

    async fn format_swift_message(&self, message: &SwiftMessage) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match message.message_type {
            SwiftMessageType::MT103 => {
                Ok(format!(
                    ":20:{}\n:23B:CRED\n:32A:{}{}{}\n:50K:{}\n:59:{}\n:70:{}",
                    message.transaction_reference,
                    message.value_date.format("%y%m%d"),
                    message.currency,
                    message.amount,
                    message.ordering_customer,
                    message.beneficiary,
                    message.remittance_info.as_deref().unwrap_or("")
                ))
            },
            _ => Err("Unsupported SWIFT message type".into()),
        }
    }

    async fn format_iso20022_xml(&self, message: &ISO20022Message) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Generate ISO 20022 XML
        let xml = format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">
    <CstmrCdtTrfInitn>
        <GrpHdr>
            <MsgId>{}</MsgId>
            <CreDtTm>{}</CreDtTm>
            <NbOfTxs>{}</NbOfTxs>
            <InitgPty>
                <Nm>{}</Nm>
            </InitgPty>
        </GrpHdr>
        <PmtInf>
            <PmtInfId>{}</PmtInfId>
            <PmtMtd>TRF</PmtMtd>
            <ReqdExctnDt>{}</ReqdExctnDt>
            <Dbtr>
                <Nm>{}</Nm>
            </Dbtr>
            <DbtrAcct>
                <Id>
                    <IBAN>{}</IBAN>
                </Id>
            </DbtrAcct>
            <DbtrAgt>
                <FinInstnId>
                    <BIC>{}</BIC>
                </FinInstnId>
            </DbtrAgt>
            {}
        </PmtInf>
    </CstmrCdtTrfInitn>
</Document>"#,
            message.message_id,
            message.creation_date_time.format("%Y-%m-%dT%H:%M:%S"),
            message.credit_transfer_info.len(),
            message.initiating_party,
            message.payment_info.payment_info_id,
            message.payment_info.requested_execution_date.format("%Y-%m-%d"),
            message.payment_info.debtor.name,
            message.payment_info.debtor_account.iban.as_deref().unwrap_or(""),
            message.payment_info.debtor_agent.bic,
            self.format_credit_transfers(&message.credit_transfer_info).await?
        );
        
        Ok(xml)
    }

    async fn format_credit_transfers(&self, transfers: &[CreditTransferInfo]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut xml = String::new();
        
        for transfer in transfers {
            xml.push_str(&format!(r#"
            <CdtTrfTxInf>
                <PmtId>
                    <EndToEndId>{}</EndToEndId>
                </PmtId>
                <Amt>
                    <InstdAmt Ccy="{}">{}</InstdAmt>
                </Amt>
                <Cdtr>
                    <Nm>{}</Nm>
                </Cdtr>
                <CdtrAcct>
                    <Id>
                        <IBAN>{}</IBAN>
                    </Id>
                </CdtrAcct>
                <CdtrAgt>
                    <FinInstnId>
                        <BIC>{}</BIC>
                    </FinInstnId>
                </CdtrAgt>
                {}
            </CdtTrfTxInf>"#,
                transfer.payment_id,
                transfer.currency,
                transfer.amount,
                transfer.creditor.name,
                transfer.creditor_account.iban.as_deref().unwrap_or(""),
                transfer.creditor_agent.bic,
                if let Some(ref info) = transfer.remittance_info {
                    format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", info)
                } else {
                    String::new()
                }
            ));
        }
        
        Ok(xml)
    }

    async fn create_sepa_batch_xml(&self, payments: &[SEPAPayment]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let total_amount: rust_decimal::Decimal = payments.iter()
            .map(|p| p.amount)
            .sum();
        
        let xml = format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">
    <CstmrCdtTrfInitn>
        <GrpHdr>
            <MsgId>{}</MsgId>
            <CreDtTm>{}</CreDtTm>
            <NbOfTxs>{}</NbOfTxs>
            <CtrlSum>{}</CtrlSum>
            <InitgPty>
                <Id>
                    <OrgId>
                        <Othr>
                            <Id>{}</Id>
                        </Othr>
                    </OrgId>
                </InitgPty>
        </GrpHdr>
        {}
    </CstmrCdtTrfInitn>
</Document>"#,
            self.generate_message_id().await?,
            Utc::now().format("%Y-%m-%dT%H:%M:%S"),
            payments.len(),
            total_amount,
            self.sepa_config.creditor_id,
            self.format_sepa_payments(payments).await?
        );
        
        Ok(xml)
    }

    async fn format_sepa_payments(&self, payments: &[SEPAPayment]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut xml = String::new();
        
        for payment in payments {
            xml.push_str(&format!(r#"
        <PmtInf>
            <PmtInfId>{}</PmtInfId>
            <PmtMtd>TRF</PmtMtd>
            <BtchBookg>{}</BtchBookg>
            <ReqdExctnDt>{}</ReqdExctnDt>
            <Dbtr>
                <Nm>{}</Nm>
            </Dbtr>
            <DbtrAcct>
                <Id>
                    <IBAN>{}</IBAN>
                </Id>
            </DbtrAcct>
            <DbtrAgt>
                <FinInstnId>
                    <BIC>{}</BIC>
                </FinInstnId>
            </DbtrAgt>
            <CdtTrfTxInf>
                <PmtId>
                    <EndToEndId>{}</EndToEndId>
                </PmtId>
                <Amt>
                    <InstdAmt Ccy="{}">{}</InstdAmt>
                </Amt>
                <Cdtr>
                    <Nm>{}</Nm>
                </Cdtr>
                <CdtrAcct>
                    <Id>
                        <IBAN>{}</IBAN>
                    </Id>
                </CdtrAcct>
                {}
            </CdtTrfTxInf>
        </PmtInf>"#,
                payment.payment_info_id,
                payment.batch_booking,
                payment.execution_date.format("%Y-%m-%d"),
                payment.debtor_name,
                payment.debtor_iban,
                self.sepa_config.bic,
                payment.end_to_end_id,
                payment.currency,
                payment.amount,
                payment.creditor_name,
                payment.creditor_iban,
                if let Some(ref info) = payment.remittance_info {
                    format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", info)
                } else {
                    String::new()
                }
            ));
        }
        
        Ok(xml)
    }

    fn validate_iban(&self, iban: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if iban.len() < 15 || iban.len() > 34 {
            return Err("Invalid IBAN length".into());
        }
        
        // Basic IBAN validation (simplified)
        if !iban.chars().all(|c| c.is_alphanumeric()) {
            return Err("IBAN contains invalid characters".into());
        }
        
        Ok(())
    }

    async fn sign_swift_message(&self, message: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // SWIFT message signing with LAU/PAC
        // This is a simplified implementation
        let signature = format!("LAU:{}", self.swift_config.security_config.authentication_key);
        Ok(format!("{}\n{}", message, signature))
    }

    async fn sign_iso20022_message(&self, xml: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // XML Digital Signature
        // This is a simplified implementation
        let signature = format!("<!-- Digital Signature: {} -->", "signature_placeholder");
        Ok(format!("{}\n{}", xml, signature))
    }

    async fn sign_sepa_batch(&self, xml: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // SEPA batch signing
        self.sign_iso20022_message(xml).await
    }

    async fn transmit_swift_message(&self, _message: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Simulate SWIFT network transmission
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(())
    }

    async fn transmit_iso20022_message(&self, _xml: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Simulate ISO 20022 message transmission
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(())
    }

    async fn submit_sepa_batch(&self, _xml: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate SEPA batch submission
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        Ok(format!("BATCH_{}", Uuid::new_v4().simple()))
    }

    async fn validate_iso20022_schema(&self, _xml: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate against ISO 20022 XSD schema
        // This is a placeholder for actual schema validation
        Ok(())
    }

    async fn validate_sepa_batch(&self, _xml: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate SEPA batch format
        Ok(())
    }

    async fn generate_transaction_reference(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(format!("TXN{}{}", Utc::now().timestamp(), rand::random::<u16>()))
    }

    async fn generate_message_id(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(format!("MSG{}", Uuid::new_v4().simple()))
    }

    async fn generate_payment_info_id(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(format!("PMT{}", Uuid::new_v4().simple()))
    }

    async fn generate_end_to_end_id(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(format!("E2E{}", Uuid::new_v4().simple()))
    }

    // Database operations
    async fn store_swift_message(&self, message: &SwiftMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO swift_messages 
             (id, message_type, sender_bic, receiver_bic, transaction_reference, amount, currency, 
              value_date, ordering_customer, beneficiary, remittance_info, charges, created_at, status)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)"
        )
        .bind(message.id)
        .bind(serde_json::to_string(&message.message_type)?)
        .bind(&message.sender_bic)
        .bind(&message.receiver_bic)
        .bind(&message.transaction_reference)
        .bind(message.amount)
        .bind(&message.currency)
        .bind(message.value_date)
        .bind(&message.ordering_customer)
        .bind(&message.beneficiary)
        .bind(&message.remittance_info)
        .bind(serde_json::to_string(&message.charges)?)
        .bind(message.created_at)
        .bind(serde_json::to_string(&message.status)?)
        .execute(self.db.as_ref())
        .await?;

        Ok(())
    }

    async fn store_iso20022_message(&self, message: &ISO20022Message) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO iso20022_messages 
             (id, message_type, message_id, creation_date_time, initiating_party, 
              payment_info, credit_transfer_info, status, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
        )
        .bind(message.id)
        .bind(serde_json::to_string(&message.message_type)?)
        .bind(&message.message_id)
        .bind(message.creation_date_time)
        .bind(&message.initiating_party)
        .bind(serde_json::to_string(&message.payment_info)?)
        .bind(serde_json::to_string(&message.credit_transfer_info)?)
        .bind(serde_json::to_string(&message.status)?)
        .bind(message.created_at)
        .execute(self.db.as_ref())
        .await?;

        Ok(())
    }

    async fn store_sepa_payment(&self, payment: &SEPAPayment) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO sepa_payments 
             (id, sepa_type, message_id, payment_info_id, debtor_iban, debtor_name, 
              creditor_iban, creditor_name, amount, currency, end_to_end_id, 
              remittance_info, execution_date, batch_booking, created_at, status)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)"
        )
        .bind(payment.id)
        .bind(serde_json::to_string(&payment.sepa_type)?)
        .bind(&payment.message_id)
        .bind(&payment.payment_info_id)
        .bind(&payment.debtor_iban)
        .bind(&payment.debtor_name)
        .bind(&payment.creditor_iban)
        .bind(&payment.creditor_name)
        .bind(payment.amount)
        .bind(&payment.currency)
        .bind(&payment.end_to_end_id)
        .bind(&payment.remittance_info)
        .bind(payment.execution_date)
        .bind(payment.batch_booking)
        .bind(payment.created_at)
        .bind(serde_json::to_string(&payment.status)?)
        .execute(self.db.as_ref())
        .await?;

        Ok(())
    }

    async fn get_swift_message(&self, id: Uuid) -> Result<SwiftMessage, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT * FROM swift_messages WHERE id = $1")
            .bind(id)
            .fetch_one(self.db.as_ref())
            .await?;

        Ok(SwiftMessage {
            id: row.get::<uuid::Uuid, _>("id"),
            message_type: serde_json::from_str(&row.get::<String, _>("message_type"))?,
            sender_bic: row.get::<String, _>("sender_bic"),
            receiver_bic: row.get::<String, _>("receiver_bic"),
            transaction_reference: row.get::<String, _>("transaction_reference"),
            amount: row.get::<rust_decimal::Decimal, _>("amount"),
            currency: row.get::<String, _>("currency"),
            value_date: row.get::<chrono::DateTime<Utc>, _>("value_date"),
            ordering_customer: row.get::<String, _>("ordering_customer"),
            beneficiary: row.get::<String, _>("beneficiary"),
            remittance_info: row.get::<Option<String>, _>("remittance_info"),
            charges: serde_json::from_str(&row.get::<String, _>("charges"))?,
            created_at: row.get::<chrono::DateTime<Utc>, _>("created_at"),
            sent_at: row.get::<Option<chrono::DateTime<Utc>>, _>("sent_at"),
            status: serde_json::from_str(&row.get::<String, _>("status"))?,
        })
    }

    async fn get_iso20022_message(&self, id: Uuid) -> Result<ISO20022Message, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT * FROM iso20022_messages WHERE id = $1")
            .bind(id)
            .fetch_one(self.db.as_ref())
            .await?;

        Ok(ISO20022Message {
            id: row.get::<uuid::Uuid, _>("id"),
            message_type: serde_json::from_str(&row.get::<String, _>("message_type"))?,
            message_id: row.get::<String, _>("message_id"),
            creation_date_time: row.get::<chrono::DateTime<Utc>, _>("creation_date_time"),
            initiating_party: row.get::<String, _>("initiating_party"),
            payment_info: serde_json::from_str(&row.get::<String, _>("payment_info"))?,
            credit_transfer_info: serde_json::from_str(&row.get::<String, _>("credit_transfer_info"))?,
            status: serde_json::from_str(&row.get::<String, _>("status"))?,
            created_at: row.get::<chrono::DateTime<Utc>, _>("created_at"),
        })
    }

    async fn get_sepa_payment(&self, id: Uuid) -> Result<SEPAPayment, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT * FROM sepa_payments WHERE id = $1")
            .bind(id)
            .fetch_one(self.db.as_ref())
            .await?;

        Ok(SEPAPayment {
            id: row.get::<uuid::Uuid, _>("id"),
            sepa_type: serde_json::from_str(&row.get::<String, _>("sepa_type"))?,
            message_id: row.get::<String, _>("message_id"),
            payment_info_id: row.get::<String, _>("payment_info_id"),
            debtor_iban: row.get::<String, _>("debtor_iban"),
            debtor_name: row.get::<String, _>("debtor_name"),
            creditor_iban: row.get::<String, _>("creditor_iban"),
            creditor_name: row.get::<String, _>("creditor_name"),
            amount: row.get::<rust_decimal::Decimal, _>("amount"),
            currency: row.get::<String, _>("currency"),
            end_to_end_id: row.get::<String, _>("end_to_end_id"),
            remittance_info: row.get::<Option<String>, _>("remittance_info"),
            execution_date: row.get::<chrono::DateTime<Utc>, _>("execution_date"),
            batch_booking: row.get::<bool, _>("batch_booking"),
            created_at: row.get::<chrono::DateTime<Utc>, _>("created_at"),
            status: serde_json::from_str(&row.get::<String, _>("status"))?,
        })
    }

    async fn update_swift_message(&self, message: &SwiftMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE swift_messages SET status = $1, sent_at = $2 WHERE id = $3")
            .bind(serde_json::to_string(&message.status)?)
            .bind(message.sent_at)
            .bind(message.id)
            .execute(self.db.as_ref())
            .await?;

        Ok(())
    }

    async fn update_iso20022_message(&self, message: &ISO20022Message) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE iso20022_messages SET status = $1 WHERE id = $2")
            .bind(serde_json::to_string(&message.status)?)
            .bind(message.id)
            .execute(self.db.as_ref())
            .await?;

        Ok(())
    }

    async fn update_sepa_payment(&self, payment: &SEPAPayment) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE sepa_payments SET status = $1 WHERE id = $2")
            .bind(serde_json::to_string(&payment.status)?)
            .bind(payment.id)
            .execute(self.db.as_ref())
            .await?;

        Ok(())
    }
}