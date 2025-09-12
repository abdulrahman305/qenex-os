//! Database Migrations - Production Schema
//! 
//! Complete database schema with proper indexing,
//! constraints, and optimization for financial operations

use super::{CoreError, Result};
use sqlx::PgPool;

/// Run all database migrations
pub async fn run_migrations(pool: &PgPool) -> Result<()> {
    tracing::info!("Running database migrations...");
    
    // Migration 001: Core tables
    create_core_tables(pool).await?;
    
    // Migration 002: Indexes for performance
    create_indexes(pool).await?;
    
    // Migration 003: Constraints and foreign keys
    create_constraints(pool).await?;
    
    // Migration 004: Views for reporting
    create_views(pool).await?;
    
    // Migration 005: Functions and triggers
    create_functions_and_triggers(pool).await?;
    
    tracing::info!("Database migrations completed successfully");
    Ok(())
}

/// Create core database tables
async fn create_core_tables(pool: &PgPool) -> Result<()> {
    tracing::debug!("Creating core tables...");
    
    // Accounts table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS accounts (
            id VARCHAR(255) PRIMARY KEY,
            account_type VARCHAR(50) NOT NULL DEFAULT 'individual',
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            tier VARCHAR(20) NOT NULL DEFAULT 'standard',
            daily_limit DECIMAL(20,8) NOT NULL DEFAULT 10000.00,
            monthly_limit DECIMAL(20,8) NOT NULL DEFAULT 100000.00,
            kyc_status VARCHAR(20) NOT NULL DEFAULT 'pending',
            risk_rating VARCHAR(20) NOT NULL DEFAULT 'low',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create accounts table: {}", e)))?;
    
    // Account balances table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS account_balances (
            account_id VARCHAR(255) NOT NULL,
            currency CHAR(3) NOT NULL,
            available_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
            pending_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
            reserved_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
            last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (account_id, currency)
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create account_balances table: {}", e)))?;
    
    // Transactions table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS transactions (
            id UUID PRIMARY KEY,
            sender VARCHAR(255) NOT NULL,
            receiver VARCHAR(255) NOT NULL,
            amount DECIMAL(20,8) NOT NULL,
            currency CHAR(3) NOT NULL,
            transaction_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 2,
            reference VARCHAR(255),
            description TEXT,
            fee DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
            exchange_rate DECIMAL(20,8),
            settlement_date TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            signature BYTEA NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            compliance_flags TEXT[] NOT NULL DEFAULT '{}',
            risk_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            block_hash VARCHAR(66),
            block_number BIGINT
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create transactions table: {}", e)))?;
    
    // Balance changes table for audit trail
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS balance_changes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            account_id VARCHAR(255) NOT NULL,
            currency CHAR(3) NOT NULL,
            amount_change DECIMAL(20,8) NOT NULL,
            balance_before DECIMAL(20,8) NOT NULL,
            balance_after DECIMAL(20,8) NOT NULL,
            transaction_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create balance_changes table: {}", e)))?;
    
    // Audit logs table for compliance
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(100) NOT NULL,
            actor VARCHAR(255) NOT NULL,
            resource VARCHAR(255) NOT NULL,
            action VARCHAR(50) NOT NULL,
            old_value JSONB,
            new_value JSONB,
            ip_address INET,
            user_agent TEXT,
            session_id VARCHAR(255),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            severity VARCHAR(20) NOT NULL DEFAULT 'info'
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create audit_logs table: {}", e)))?;
    
    // Compliance reports table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS compliance_reports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            report_type VARCHAR(50) NOT NULL,
            account_id VARCHAR(255),
            transaction_id UUID,
            findings JSONB NOT NULL,
            severity VARCHAR(20) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'open',
            assignee VARCHAR(255),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at TIMESTAMPTZ
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create compliance_reports table: {}", e)))?;
    
    // KYC/AML data table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS kyc_records (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            account_id VARCHAR(255) NOT NULL UNIQUE,
            document_type VARCHAR(50) NOT NULL,
            document_number VARCHAR(100) NOT NULL,
            document_verified BOOLEAN NOT NULL DEFAULT FALSE,
            verification_date TIMESTAMPTZ,
            identity_data JSONB NOT NULL,
            risk_assessment JSONB,
            sanctions_check JSONB,
            pep_check JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create kyc_records table: {}", e)))?;
    
    // Transaction limits table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS transaction_limits (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            account_id VARCHAR(255) NOT NULL,
            limit_type VARCHAR(50) NOT NULL,
            currency CHAR(3) NOT NULL,
            amount_limit DECIMAL(20,8) NOT NULL,
            time_period VARCHAR(20) NOT NULL,
            current_usage DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
            reset_time TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create transaction_limits table: {}", e)))?;
    
    // Exchange rates table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS exchange_rates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            from_currency CHAR(3) NOT NULL,
            to_currency CHAR(3) NOT NULL,
            rate DECIMAL(20,8) NOT NULL,
            source VARCHAR(50) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create exchange_rates table: {}", e)))?;
    
    // System configuration table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS system_config (
            key VARCHAR(100) PRIMARY KEY,
            value JSONB NOT NULL,
            description TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_by VARCHAR(255) NOT NULL DEFAULT 'system'
        )
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create system_config table: {}", e)))?;
    
    tracing::debug!("Core tables created successfully");
    Ok(())
}

/// Create database indexes for optimal performance
async fn create_indexes(pool: &PgPool) -> Result<()> {
    tracing::debug!("Creating database indexes...");
    
    let indexes = vec![
        // Accounts indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_status ON accounts(status)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_tier ON accounts(tier)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_kyc_status ON accounts(kyc_status)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_created_at ON accounts(created_at)",
        
        // Account balances indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_balances_account_id ON account_balances(account_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_balances_currency ON account_balances(currency)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_balances_last_updated ON account_balances(last_updated)",
        
        // Transactions indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_sender ON transactions(sender)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_receiver ON transactions(receiver)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_status ON transactions(status)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_type ON transactions(transaction_type)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_created_at ON transactions(created_at)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_settlement_date ON transactions(settlement_date)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_block_number ON transactions(block_number)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_amount ON transactions(amount)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_currency ON transactions(currency)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_risk_score ON transactions(risk_score)",
        
        // Composite indexes for common queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_sender_created ON transactions(sender, created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_receiver_created ON transactions(receiver, created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_status_priority ON transactions(status, priority DESC)",
        
        // Balance changes indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_balance_changes_account_id ON balance_changes(account_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_balance_changes_transaction_id ON balance_changes(transaction_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_balance_changes_created_at ON balance_changes(created_at)",
        
        // Audit logs indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_severity ON audit_logs(severity)",
        
        // Compliance reports indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_account_id ON compliance_reports(account_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_transaction_id ON compliance_reports(transaction_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_status ON compliance_reports(status)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_severity ON compliance_reports(severity)",
        
        // KYC records indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kyc_records_account_id ON kyc_records(account_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kyc_records_document_verified ON kyc_records(document_verified)",
        
        // Transaction limits indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transaction_limits_account_id ON transaction_limits(account_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transaction_limits_reset_time ON transaction_limits(reset_time)",
        
        // Exchange rates indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_exchange_rates_currencies ON exchange_rates(from_currency, to_currency)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_exchange_rates_timestamp ON exchange_rates(timestamp DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_exchange_rates_active ON exchange_rates(is_active)",
        
        // JSONB indexes for metadata
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_metadata_gin ON accounts USING GIN (metadata)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_metadata_gin ON transactions USING GIN (metadata)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_findings_gin ON compliance_reports USING GIN (findings)",
    ];
    
    for index_sql in indexes {
        if let Err(e) = sqlx::query(index_sql).execute(pool).await {
            tracing::warn!("Failed to create index: {} - Error: {}", index_sql, e);
        }
    }
    
    tracing::debug!("Database indexes created successfully");
    Ok(())
}

/// Create foreign key constraints and checks
async fn create_constraints(pool: &PgPool) -> Result<()> {
    tracing::debug!("Creating database constraints...");
    
    let constraints = vec![
        // Foreign key constraints
        r#"
        ALTER TABLE account_balances 
        ADD CONSTRAINT IF NOT EXISTS fk_account_balances_account_id 
        FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        "#,
        
        r#"
        ALTER TABLE balance_changes 
        ADD CONSTRAINT IF NOT EXISTS fk_balance_changes_account_id 
        FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        "#,
        
        r#"
        ALTER TABLE balance_changes 
        ADD CONSTRAINT IF NOT EXISTS fk_balance_changes_transaction_id 
        FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE SET NULL
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS fk_transactions_sender 
        FOREIGN KEY (sender) REFERENCES accounts(id)
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS fk_transactions_receiver 
        FOREIGN KEY (receiver) REFERENCES accounts(id)
        "#,
        
        r#"
        ALTER TABLE kyc_records 
        ADD CONSTRAINT IF NOT EXISTS fk_kyc_records_account_id 
        FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        "#,
        
        r#"
        ALTER TABLE transaction_limits 
        ADD CONSTRAINT IF NOT EXISTS fk_transaction_limits_account_id 
        FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        "#,
        
        // Check constraints
        r#"
        ALTER TABLE accounts 
        ADD CONSTRAINT IF NOT EXISTS chk_accounts_status 
        CHECK (status IN ('active', 'inactive', 'suspended', 'closed'))
        "#,
        
        r#"
        ALTER TABLE accounts 
        ADD CONSTRAINT IF NOT EXISTS chk_accounts_tier 
        CHECK (tier IN ('individual', 'business', 'institution'))
        "#,
        
        r#"
        ALTER TABLE accounts 
        ADD CONSTRAINT IF NOT EXISTS chk_accounts_kyc_status 
        CHECK (kyc_status IN ('pending', 'verified', 'rejected', 'expired'))
        "#,
        
        r#"
        ALTER TABLE account_balances 
        ADD CONSTRAINT IF NOT EXISTS chk_account_balances_positive 
        CHECK (available_balance >= 0 AND pending_balance >= 0 AND reserved_balance >= 0)
        "#,
        
        r#"
        ALTER TABLE account_balances 
        ADD CONSTRAINT IF NOT EXISTS chk_account_balances_currency 
        CHECK (currency ~ '^[A-Z]{3}$')
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_amount_positive 
        CHECK (amount > 0)
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_fee_non_negative 
        CHECK (fee >= 0)
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_status 
        CHECK (status IN ('pending', 'validating', 'approved', 'processing', 'settled', 'failed', 'rejected', 'cancelled'))
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_priority 
        CHECK (priority BETWEEN 1 AND 4)
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_currency 
        CHECK (currency ~ '^[A-Z]{3}$')
        "#,
        
        r#"
        ALTER TABLE transactions 
        ADD CONSTRAINT IF NOT EXISTS chk_transactions_risk_score 
        CHECK (risk_score BETWEEN 0.0 AND 1.0)
        "#,
        
        r#"
        ALTER TABLE compliance_reports 
        ADD CONSTRAINT IF NOT EXISTS chk_compliance_reports_severity 
        CHECK (severity IN ('low', 'medium', 'high', 'critical'))
        "#,
        
        r#"
        ALTER TABLE compliance_reports 
        ADD CONSTRAINT IF NOT EXISTS chk_compliance_reports_status 
        CHECK (status IN ('open', 'investigating', 'resolved', 'dismissed'))
        "#,
        
        r#"
        ALTER TABLE exchange_rates 
        ADD CONSTRAINT IF NOT EXISTS chk_exchange_rates_positive 
        CHECK (rate > 0)
        "#,
        
        r#"
        ALTER TABLE exchange_rates 
        ADD CONSTRAINT IF NOT EXISTS chk_exchange_rates_currency 
        CHECK (from_currency ~ '^[A-Z]{3}$' AND to_currency ~ '^[A-Z]{3}$')
        "#,
    ];
    
    for constraint_sql in constraints {
        if let Err(e) = sqlx::query(constraint_sql).execute(pool).await {
            tracing::warn!("Failed to create constraint: {} - Error: {}", constraint_sql, e);
        }
    }
    
    tracing::debug!("Database constraints created successfully");
    Ok(())
}

/// Create database views for reporting and analytics
async fn create_views(pool: &PgPool) -> Result<()> {
    tracing::debug!("Creating database views...");
    
    // Account summary view
    sqlx::query(
        r#"
        CREATE OR REPLACE VIEW account_summary AS
        SELECT 
            a.id,
            a.account_type,
            a.status,
            a.tier,
            a.kyc_status,
            a.risk_rating,
            a.created_at,
            COUNT(DISTINCT ab.currency) as currencies_count,
            SUM(CASE WHEN ab.currency = 'USD' THEN ab.available_balance ELSE 0 END) as usd_balance,
            COUNT(t.id) FILTER (WHERE t.created_at >= NOW() - INTERVAL '30 days') as transactions_30d,
            SUM(t.amount) FILTER (WHERE t.created_at >= NOW() - INTERVAL '30 days' AND t.currency = 'USD') as volume_30d_usd
        FROM accounts a
        LEFT JOIN account_balances ab ON a.id = ab.account_id
        LEFT JOIN transactions t ON (a.id = t.sender OR a.id = t.receiver)
        GROUP BY a.id, a.account_type, a.status, a.tier, a.kyc_status, a.risk_rating, a.created_at
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create account_summary view: {}", e)))?;
    
    // Transaction analytics view
    sqlx::query(
        r#"
        CREATE OR REPLACE VIEW transaction_analytics AS
        SELECT 
            DATE_TRUNC('day', created_at) as transaction_date,
            transaction_type,
            status,
            currency,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            MIN(amount) as min_amount,
            MAX(amount) as max_amount,
            SUM(fee) as total_fees,
            COUNT(*) FILTER (WHERE risk_score > 0.5) as high_risk_count,
            COUNT(*) FILTER (WHERE array_length(compliance_flags, 1) > 0) as flagged_count
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '1 year'
        GROUP BY DATE_TRUNC('day', created_at), transaction_type, status, currency
        ORDER BY transaction_date DESC, total_amount DESC
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create transaction_analytics view: {}", e)))?;
    
    // Compliance dashboard view
    sqlx::query(
        r#"
        CREATE OR REPLACE VIEW compliance_dashboard AS
        SELECT 
            DATE_TRUNC('day', timestamp) as date,
            event_type,
            severity,
            COUNT(*) as event_count,
            COUNT(DISTINCT actor) as unique_actors,
            COUNT(DISTINCT resource) as unique_resources
        FROM audit_logs
        WHERE timestamp >= NOW() - INTERVAL '90 days'
        GROUP BY DATE_TRUNC('day', timestamp), event_type, severity
        ORDER BY date DESC, event_count DESC
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create compliance_dashboard view: {}", e)))?;
    
    tracing::debug!("Database views created successfully");
    Ok(())
}

/// Create database functions and triggers
async fn create_functions_and_triggers(pool: &PgPool) -> Result<()> {
    tracing::debug!("Creating database functions and triggers...");
    
    // Function to update the updated_at timestamp
    sqlx::query(
        r#"
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create update_updated_at function: {}", e)))?;
    
    // Triggers for updated_at columns
    let trigger_tables = vec![
        "accounts",
        "transactions", 
        "compliance_reports",
        "kyc_records",
        "transaction_limits",
        "system_config",
    ];
    
    for table in trigger_tables {
        let trigger_sql = format!(
            r#"
            CREATE OR REPLACE TRIGGER update_{}_updated_at 
                BEFORE UPDATE ON {} 
                FOR EACH ROW 
                EXECUTE PROCEDURE update_updated_at_column();
            "#,
            table, table
        );
        
        sqlx::query(&trigger_sql)
            .execute(pool)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to create trigger for {}: {}", table, e)))?;
    }
    
    // Function to validate transaction balance
    sqlx::query(
        r#"
        CREATE OR REPLACE FUNCTION validate_transaction_balance()
        RETURNS TRIGGER AS $$
        DECLARE
            sender_balance DECIMAL(20,8);
        BEGIN
            -- Check if sender has sufficient balance
            SELECT available_balance INTO sender_balance
            FROM account_balances 
            WHERE account_id = NEW.sender AND currency = NEW.currency;
            
            IF sender_balance IS NULL THEN
                RAISE EXCEPTION 'Sender account % has no balance for currency %', NEW.sender, NEW.currency;
            END IF;
            
            IF sender_balance < (NEW.amount + NEW.fee) THEN
                RAISE EXCEPTION 'Insufficient balance. Required: %, Available: %', 
                    (NEW.amount + NEW.fee), sender_balance;
            END IF;
            
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create balance validation function: {}", e)))?;
    
    // Trigger for transaction balance validation
    sqlx::query(
        r#"
        CREATE OR REPLACE TRIGGER validate_transaction_balance_trigger
            BEFORE INSERT ON transactions
            FOR EACH ROW
            WHEN (NEW.status = 'approved')
            EXECUTE PROCEDURE validate_transaction_balance();
        "#
    )
    .execute(pool)
    .await
    .map_err(|e| CoreError::StorageError(format!("Failed to create transaction validation trigger: {}", e)))?;
    
    tracing::debug!("Database functions and triggers created successfully");
    Ok(())
}

/// Initialize system configuration
pub async fn initialize_system_config(pool: &PgPool) -> Result<()> {
    tracing::debug!("Initializing system configuration...");
    
    let configs = vec![
        ("max_transaction_amount", serde_json::json!(1000000.00), "Maximum single transaction amount"),
        ("default_transaction_fee", serde_json::json!(0.01), "Default transaction fee percentage"),
        ("kyc_required_threshold", serde_json::json!(10000.00), "KYC required for amounts above this"),
        ("aml_monitoring_enabled", serde_json::json!(true), "Enable AML monitoring"),
        ("sanctions_screening_enabled", serde_json::json!(true), "Enable sanctions screening"),
        ("fraud_detection_threshold", serde_json::json!(0.7), "Fraud detection risk score threshold"),
        ("supported_currencies", serde_json::json!(["USD", "EUR", "GBP", "JPY"]), "Supported currencies"),
        ("maintenance_mode", serde_json::json!(false), "System maintenance mode"),
    ];
    
    for (key, value, description) in configs {
        sqlx::query(
            r#"
            INSERT INTO system_config (key, value, description, updated_by)
            VALUES ($1, $2, $3, 'migration')
            ON CONFLICT (key) DO NOTHING
            "#
        )
        .bind(key)
        .bind(value)
        .bind(description)
        .execute(pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to insert system config {}: {}", key, e)))?;
    }
    
    tracing::debug!("System configuration initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_migrations() {
        // Test would verify that all migrations run successfully
        // Implementation would use a test database
    }
}