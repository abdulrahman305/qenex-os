pub mod live_feeds;
pub mod real_time_compliance;

#[cfg(test)]
pub mod compliance_tests;

// Export both legacy and real-time compliance systems
pub use real_time_compliance::{
    RealTimeComplianceEngine,
    ComplianceEngineConfig,
    OfficialAPIConfig,
    RegulatoryJurisdiction,
    CustomerData,
    TransactionData,
    ComplianceScreeningResult,
    TransactionComplianceResult,
    ComplianceError,
    OFACScreener,
    EUScreener,
    AlertDeliveryMethod,
};

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningRequest {
    pub entity_name: String,
    pub entity_type: EntityType,
    pub transaction_id: Option<Uuid>,
}
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::RwLock;
use sqlx::PgPool;
use std::sync::Arc;

pub use live_feeds::{
    LiveComplianceSystem,
    LiveComplianceConfig,
    ScreeningResult,
    ComplianceMatch,
    ScreeningRecommendation,
};

// Use RiskLevel from real_time_compliance to avoid conflicts
pub use real_time_compliance::RiskLevel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub parameters: HashMap<String, String>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    AML,        // Anti-Money Laundering
    KYC,        // Know Your Customer
    CTF,        // Counter-Terrorism Financing
    PEP,        // Politically Exposed Persons
    Sanctions,  // Economic Sanctions
    Reporting,  // Regulatory Reporting
    RiskLimit,  // Risk Management
    DataProtection, // GDPR/Privacy
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub id: Uuid,
    pub transaction_id: Uuid,
    pub rule_id: Uuid,
    pub check_type: RuleType,
    pub status: ComplianceStatus,
    pub risk_score: f64,
    pub details: String,
    pub reviewed_by: Option<String>,
    pub reviewed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Approved,
    Rejected,
    Pending,
    UnderReview,
    Escalated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryReport {
    pub id: Uuid,
    pub report_type: ReportType,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub data: serde_json::Value,
    pub status: ReportStatus,
    pub submitted_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    SAR,    // Suspicious Activity Report
    CTR,    // Currency Transaction Report  
    FBAR,   // Foreign Bank Account Report
    FATCA,  // Foreign Account Tax Compliance Act
    CRS,    // Common Reporting Standard
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportStatus {
    Draft,
    Ready,
    Submitted,
    Acknowledged,
    Rejected,
}

pub struct ComplianceEngine {
    db: Arc<PgPool>,
    rules: Arc<RwLock<HashMap<Uuid, ComplianceRule>>>,
    watchlists: Arc<RwLock<HashMap<String, WatchlistEntry>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchlistEntry {
    pub id: String,
    pub entity_type: EntityType,
    pub names: Vec<String>,
    pub aliases: Vec<String>,
    pub risk_level: RiskLevel,
    pub source: String,
    pub added_at: DateTime<Utc>,
}

// Define EntityType locally for this module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Individual,
    Organization,
    Country,
    Account,
}

impl ComplianceEngine {
    pub async fn new(db: Arc<PgPool>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let engine = Self {
            db: db.clone(),
            rules: Arc::new(RwLock::new(HashMap::new())),
            watchlists: Arc::new(RwLock::new(HashMap::new())),
        };
        
        engine.load_rules().await?;
        engine.load_watchlists().await?;
        
        Ok(engine)
    }

    pub async fn check_transaction(&self, transaction_id: Uuid) -> Result<Vec<ComplianceCheck>, Box<dyn std::error::Error + Send + Sync>> {
        let mut checks = Vec::new();
        let rules = self.rules.read().await;
        
        for rule in rules.values() {
            if rule.active {
                let check = self.execute_rule(transaction_id, rule).await?;
                checks.push(check);
            }
        }
        
        // Store checks in database
        for check in &checks {
            self.store_compliance_check(check).await?;
        }
        
        Ok(checks)
    }

    async fn execute_rule(&self, transaction_id: Uuid, rule: &ComplianceRule) -> Result<ComplianceCheck, Box<dyn std::error::Error + Send + Sync>> {
        let mut risk_score: f64 = 0.0;
        let mut status = ComplianceStatus::Approved;
        let mut details = String::new();

        match rule.rule_type {
            RuleType::AML => {
                risk_score = self.check_aml_risk(transaction_id).await?;
                if risk_score > 0.7 {
                    status = ComplianceStatus::UnderReview;
                    details = "High AML risk detected".to_string();
                } else if risk_score > 0.5 {
                    status = ComplianceStatus::Pending;
                    details = "Medium AML risk detected".to_string();
                }
            },
            RuleType::KYC => {
                risk_score = self.check_kyc_compliance(transaction_id).await?;
                if risk_score > 0.8 {
                    status = ComplianceStatus::Rejected;
                    details = "KYC verification failed".to_string();
                }
            },
            RuleType::Sanctions => {
                risk_score = self.check_sanctions(transaction_id).await?;
                if risk_score > 0.9 {
                    status = ComplianceStatus::Rejected;
                    details = "Sanctions violation detected".to_string();
                } else if risk_score > 0.5 {
                    status = ComplianceStatus::Escalated;
                    details = "Potential sanctions risk".to_string();
                }
            },
            _ => {
                // Implement other rule types
            }
        }

        Ok(ComplianceCheck {
            id: Uuid::new_v4(),
            transaction_id,
            rule_id: rule.id,
            check_type: rule.rule_type.clone(),
            status,
            risk_score,
            details,
            reviewed_by: None,
            reviewed_at: None,
            created_at: Utc::now(),
        })
    }

    async fn check_aml_risk(&self, transaction_id: Uuid) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Get transaction details
        let transaction = sqlx::query("SELECT from_account, to_account, amount, currency FROM transactions WHERE id = $1")
            .bind(transaction_id)
            .fetch_one(self.db.as_ref())
            .await?;

        let mut risk_score: f64 = 0.0;

        // Check transaction amount thresholds
        let amount = transaction.get::<rust_decimal::Decimal, _>("amount");
        if amount > rust_decimal::Decimal::from(10000) {
            risk_score += 0.3;
        }

        // Check velocity (multiple transactions in short time)
        let from_account = transaction.get::<String, _>("from_account");
        let recent_count = sqlx::query_scalar(
            "SELECT COUNT(*) FROM transactions 
             WHERE (from_account = $1 OR to_account = $1) 
             AND created_at > NOW() - INTERVAL '1 hour'"
        )
        .bind(&from_account)
        .fetch_one(self.db.as_ref())
        .await?
        .get::<i64, _>(0);

        if recent_count > 10 {
            risk_score += 0.4;
        }

        // Check watchlist
        let watchlists = self.watchlists.read().await;
        let to_account = transaction.get::<String, _>("to_account");
        if watchlists.contains_key(&from_account) || 
           watchlists.contains_key(&to_account) {
            risk_score += 0.6;
        }

        // Structuring detection (amounts just below reporting thresholds)
        if amount > rust_decimal::Decimal::from(9500) && 
           amount < rust_decimal::Decimal::from(10000) {
            risk_score += 0.5;
        }

        Ok(risk_score.min(1.0))
    }

    async fn check_kyc_compliance(&self, transaction_id: Uuid) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let transaction = sqlx::query("SELECT from_account, to_account FROM transactions WHERE id = $1")
            .bind(transaction_id)
            .fetch_one(self.db.as_ref())
            .await?;

        // Check if accounts have completed KYC
        let from_account = transaction.get::<String, _>("from_account");
        let from_kyc = sqlx::query_scalar("SELECT kyc_verified FROM accounts WHERE id = $1")
            .bind(&from_account)
            .fetch_one(self.db.as_ref())
            .await?
            .get::<bool, _>(0);

        let to_account = transaction.get::<String, _>("to_account");
        let to_kyc = sqlx::query_scalar("SELECT kyc_verified FROM accounts WHERE id = $1")
            .bind(&to_account)
            .fetch_one(self.db.as_ref())
            .await?
            .get::<bool, _>(0);

        let mut risk_score: f64 = 0.0;

        if !from_kyc {
            risk_score += 0.5;
        }

        if !to_kyc {
            risk_score += 0.5;
        }

        Ok(risk_score)
    }

    async fn check_sanctions(&self, transaction_id: Uuid) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let transaction = sqlx::query("SELECT from_account, to_account FROM transactions WHERE id = $1")
            .bind(transaction_id)
            .fetch_one(self.db.as_ref())
            .await?;

        let watchlists = self.watchlists.read().await;
        let mut risk_score: f64 = 0.0;

        // Check for sanctioned entities
        let from_account = transaction.get::<String, _>("from_account");
        let to_account = transaction.get::<String, _>("to_account");
        for entry in watchlists.values() {
            if entry.source.contains("OFAC") || entry.source.contains("EU_SANCTIONS") {
                if entry.names.iter().any(|name| 
                    from_account.contains(name) || 
                    to_account.contains(name)) {
                    match entry.risk_level {
                        RiskLevel::High => risk_score = 1.0, // Map High to Critical behavior
                        RiskLevel::Medium => risk_score = 0.6,
                        RiskLevel::Low => risk_score = 0.3,
                    }
                    break;
                }
            }
        }

        Ok(risk_score)
    }

    pub async fn generate_sar_report(&self, transaction_id: Uuid, reason: String) -> Result<RegulatoryReport, Box<dyn std::error::Error + Send + Sync>> {
        let transaction = sqlx::query("SELECT * FROM transactions WHERE id = $1")
            .bind(transaction_id)
            .fetch_one(self.db.as_ref())
            .await?;

        let report_data = serde_json::json!({
            "transaction_id": transaction_id,
            "from_account": transaction.get::<String, _>("from_account"),
            "to_account": transaction.get::<String, _>("to_account"),
            "amount": transaction.get::<rust_decimal::Decimal, _>("amount"),
            "currency": transaction.get::<String, _>("currency"),
            "reason": reason,
            "transaction_date": transaction.get::<chrono::DateTime<Utc>, _>("created_at"),
            "report_date": Utc::now()
        });

        let report = RegulatoryReport {
            id: Uuid::new_v4(),
            report_type: ReportType::SAR,
            period_start: transaction.get::<chrono::DateTime<Utc>, _>("created_at"),
            period_end: Utc::now(),
            data: report_data,
            status: ReportStatus::Draft,
            submitted_at: None,
            created_at: Utc::now(),
        };

        // Store report
        sqlx::query(
            "INSERT INTO regulatory_reports (id, report_type, period_start, period_end, data, status, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7)"
        )
        .bind(report.id)
        .bind(serde_json::to_string(&report.report_type)?)
        .bind(report.period_start)
        .bind(report.period_end)
        .bind(&report.data)
        .bind(serde_json::to_string(&report.status)?)
        .bind(report.created_at)
        .execute(self.db.as_ref())
        .await?;

        Ok(report)
    }

    async fn load_rules(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rules_data = sqlx::query(
            "SELECT id, name, description, rule_type, parameters, active, created_at, updated_at 
             FROM compliance_rules"
        )
        .fetch_all(self.db.as_ref())
        .await?;

        let mut rules = self.rules.write().await;
        
        for row in rules_data {
            let rule = ComplianceRule {
                id: row.get::<uuid::Uuid, _>("id"),
                name: row.get::<String, _>("name"),
                description: row.get::<String, _>("description"),
                rule_type: serde_json::from_str(&row.get::<String, _>("rule_type"))?,
                parameters: serde_json::from_str(&row.get::<String, _>("parameters"))?,
                active: row.get::<bool, _>("active"),
                created_at: row.get::<chrono::DateTime<Utc>, _>("created_at"),
                updated_at: row.get::<chrono::DateTime<Utc>, _>("updated_at"),
            };
            rules.insert(rule.id, rule);
        }

        Ok(())
    }

    async fn load_watchlists(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let watchlist_data = sqlx::query(
            "SELECT id, entity_type, names, aliases, risk_level, source, added_at 
             FROM watchlist_entries"
        )
        .fetch_all(self.db.as_ref())
        .await?;

        let mut watchlists = self.watchlists.write().await;
        
        for row in watchlist_data {
            let entry = WatchlistEntry {
                id: row.get::<String, _>("id"),
                entity_type: serde_json::from_str(&row.get::<String, _>("entity_type"))?,
                names: serde_json::from_str(&row.get::<String, _>("names"))?,
                aliases: serde_json::from_str(&row.get::<String, _>("aliases"))?,
                risk_level: serde_json::from_str(&row.get::<String, _>("risk_level"))?,
                source: row.get::<String, _>("source"),
                added_at: row.get::<chrono::DateTime<Utc>, _>("added_at"),
            };
            watchlists.insert(entry.id.clone(), entry);
        }

        Ok(())
    }

    async fn store_compliance_check(&self, check: &ComplianceCheck) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO compliance_checks 
             (id, transaction_id, rule_id, check_type, status, risk_score, details, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
        )
        .bind(check.id)
        .bind(check.transaction_id)
        .bind(check.rule_id)
        .bind(serde_json::to_string(&check.check_type)?)
        .bind(serde_json::to_string(&check.status)?)
        .bind(check.risk_score)
        .bind(&check.details)
        .bind(check.created_at)
        .execute(self.db.as_ref())
        .await?;

        Ok(())
    }

    pub async fn update_watchlist(&self, entries: Vec<WatchlistEntry>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut watchlists = self.watchlists.write().await;
        
        for entry in entries {
            // Store in database
            sqlx::query(
                "INSERT INTO watchlist_entries 
                 (id, entity_type, names, aliases, risk_level, source, added_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (id) DO UPDATE SET
                 entity_type = EXCLUDED.entity_type,
                 names = EXCLUDED.names,
                 aliases = EXCLUDED.aliases,
                 risk_level = EXCLUDED.risk_level,
                 source = EXCLUDED.source"
            )
            .bind(&entry.id)
            .bind(serde_json::to_string(&entry.entity_type)?)
            .bind(serde_json::to_string(&entry.names)?)
            .bind(serde_json::to_string(&entry.aliases)?)
            .bind(serde_json::to_string(&entry.risk_level)?)
            .bind(&entry.source)
            .bind(entry.added_at)
            .execute(self.db.as_ref())
            .await?;

            watchlists.insert(entry.id.clone(), entry);
        }

        Ok(())
    }
}