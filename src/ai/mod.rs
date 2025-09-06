use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::RwLock;
use sqlx::PgPool;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModel {
    pub id: Uuid,
    pub name: String,
    pub model_type: ModelType,
    pub version: String,
    pub parameters: serde_json::Value,
    pub training_data: Option<String>,
    pub accuracy: f64,
    pub last_trained: DateTime<Utc>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    FraudDetection,
    RiskAssessment,
    PriceForecasting,
    CustomerSegmentation,
    ComplianceScoring,
    AnomalyDetection,
    ReinforcementLearning,
    NaturalLanguageProcessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub id: Uuid,
    pub model_id: Uuid,
    pub input_data: serde_json::Value,
    pub prediction: serde_json::Value,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
    pub actual_outcome: Option<serde_json::Value>,
    pub feedback_received: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSession {
    pub id: Uuid,
    pub model_id: Uuid,
    pub dataset_size: usize,
    pub epochs: u32,
    pub learning_rate: f64,
    pub validation_accuracy: f64,
    pub loss: f64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: TrainingStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

pub struct AIEngine {
    db: Arc<PgPool>,
    models: Arc<RwLock<HashMap<Uuid, AIModel>>>,
    predictions: Arc<RwLock<HashMap<Uuid, Prediction>>>,
    training_queue: Arc<RwLock<Vec<TrainingSession>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FraudDetectionInput {
    pub transaction_amount: f64,
    pub account_balance: f64,
    pub transaction_time: DateTime<Utc>,
    pub merchant_category: String,
    pub location: Option<String>,
    pub previous_transaction_amount: f64,
    pub time_since_last_transaction: i64,
    pub account_age_days: i64,
    pub average_transaction_amount: f64,
    pub transaction_count_last_hour: i32,
    pub is_weekend: bool,
    pub is_night_time: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskAssessmentInput {
    pub customer_id: String,
    pub transaction_history: Vec<f64>,
    pub account_balance: f64,
    pub credit_score: Option<i32>,
    pub employment_status: String,
    pub annual_income: Option<f64>,
    pub debt_to_income_ratio: Option<f64>,
    pub number_of_accounts: i32,
    pub kyc_status: bool,
    pub country_risk_score: f64,
}

impl AIEngine {
    pub async fn new(db: Arc<PgPool>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let engine = Self {
            db: db.clone(),
            models: Arc::new(RwLock::new(HashMap::new())),
            predictions: Arc::new(RwLock::new(HashMap::new())),
            training_queue: Arc::new(RwLock::new(Vec::new())),
        };
        
        engine.initialize_models().await?;
        
        Ok(engine)
    }

    pub async fn predict_fraud(&self, input: FraudDetectionInput) -> Result<Prediction, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.models.read().await;
        let fraud_model = models.values()
            .find(|m| matches!(m.model_type, ModelType::FraudDetection) && m.active)
            .ok_or("No active fraud detection model found")?;

        // Feature engineering
        let features = self.extract_fraud_features(&input).await?;
        
        // Neural network inference simulation
        let prediction_score = self.neural_network_inference(&features, &fraud_model.parameters).await?;
        
        let prediction = Prediction {
            id: Uuid::new_v4(),
            model_id: fraud_model.id,
            input_data: serde_json::to_value(&input)?,
            prediction: serde_json::json!({
                "is_fraud": prediction_score > 0.5,
                "fraud_probability": prediction_score,
                "risk_level": if prediction_score > 0.8 { "HIGH" } 
                             else if prediction_score > 0.5 { "MEDIUM" } 
                             else { "LOW" }
            }),
            confidence: prediction_score,
            created_at: Utc::now(),
            actual_outcome: None,
            feedback_received: false,
        };

        // Store prediction
        self.store_prediction(&prediction).await?;
        
        Ok(prediction)
    }

    pub async fn assess_risk(&self, input: RiskAssessmentInput) -> Result<Prediction, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.models.read().await;
        let risk_model = models.values()
            .find(|m| matches!(m.model_type, ModelType::RiskAssessment) && m.active)
            .ok_or("No active risk assessment model found")?;

        let features = self.extract_risk_features(&input).await?;
        let risk_score = self.neural_network_inference(&features, &risk_model.parameters).await?;
        
        let prediction = Prediction {
            id: Uuid::new_v4(),
            model_id: risk_model.id,
            input_data: serde_json::to_value(&input)?,
            prediction: serde_json::json!({
                "risk_score": risk_score,
                "risk_category": self.categorize_risk(risk_score),
                "recommended_action": self.recommend_action(risk_score),
                "confidence_interval": [risk_score - 0.1, risk_score + 0.1]
            }),
            confidence: 0.85, // Model confidence
            created_at: Utc::now(),
            actual_outcome: None,
            feedback_received: false,
        };

        self.store_prediction(&prediction).await?;
        
        Ok(prediction)
    }

    pub async fn detect_anomalies(&self, transaction_data: Vec<f64>) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let models = self.models.read().await;
        let anomaly_model = models.values()
            .find(|m| matches!(m.model_type, ModelType::AnomalyDetection) && m.active)
            .ok_or("No active anomaly detection model found")?;

        // Implement isolation forest algorithm simulation
        let anomaly_scores = self.isolation_forest(&transaction_data, &anomaly_model.parameters).await?;
        
        Ok(anomaly_scores)
    }

    pub async fn reinforcement_learning_update(&self, action_taken: String, reward: f64, state: serde_json::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let models = self.models.read().await;
        let rl_model = models.values()
            .find(|m| matches!(m.model_type, ModelType::ReinforcementLearning) && m.active);

        if let Some(model) = rl_model {
            // Q-learning update simulation
            self.update_q_table(model.id, action_taken, reward, state).await?;
        }

        Ok(())
    }

    pub async fn train_model(&self, model_id: Uuid, dataset: Vec<serde_json::Value>) -> Result<TrainingSession, Box<dyn std::error::Error + Send + Sync>> {
        let mut training_session = TrainingSession {
            id: Uuid::new_v4(),
            model_id,
            dataset_size: dataset.len(),
            epochs: 100,
            learning_rate: 0.001,
            validation_accuracy: 0.0,
            loss: 1.0,
            started_at: Utc::now(),
            completed_at: None,
            status: TrainingStatus::Queued,
        };

        // Add to training queue
        {
            let mut queue = self.training_queue.write().await;
            queue.push(training_session.clone());
        }

        // Start training in background
        self.execute_training(&mut training_session, dataset).await?;

        Ok(training_session)
    }

    async fn execute_training(&self, session: &mut TrainingSession, dataset: Vec<serde_json::Value>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        session.status = TrainingStatus::Running;
        
        // Simulate training process
        for epoch in 1..=session.epochs {
            // Mini-batch gradient descent simulation
            let batch_size = 32;
            let mut epoch_loss = 0.0;
            
            for batch_start in (0..dataset.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(dataset.len());
                let _batch = &dataset[batch_start..batch_end];
                
                // Forward pass + backward pass simulation
                let batch_loss = self.simulate_training_step().await?;
                epoch_loss += batch_loss;
            }
            
            session.loss = epoch_loss / (dataset.len() as f64 / batch_size as f64);
            
            // Validation every 10 epochs
            if epoch % 10 == 0 {
                session.validation_accuracy = self.validate_model(session.model_id, &dataset).await?;
                
                // Early stopping
                if session.validation_accuracy > 0.95 {
                    break;
                }
            }
        }
        
        session.status = TrainingStatus::Completed;
        session.completed_at = Some(Utc::now());
        
        // Update model in database
        self.update_model_after_training(session).await?;
        
        Ok(())
    }

    async fn neural_network_inference(&self, features: &[f64], parameters: &serde_json::Value) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate neural network forward pass
        let weights: Vec<Vec<f64>> = parameters["weights"].as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|w| w.as_array().unwrap_or(&vec![]).iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect())
            .collect();
        
        let biases: Vec<f64> = parameters["biases"].as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|b| b.as_f64().unwrap_or(0.0))
            .collect();

        // Three-layer network simulation
        let mut layer1_output = Vec::new();
        for i in 0..weights[0].len() {
            let mut sum = biases.get(i).copied().unwrap_or(0.0);
            for (j, &feature) in features.iter().enumerate() {
                sum += feature * weights[0].get(j).copied().unwrap_or(0.0);
            }
            layer1_output.push(self.sigmoid(sum));
        }

        let mut layer2_output = Vec::new();
        for i in 0..weights[1].len() {
            let mut sum = biases.get(10 + i).copied().unwrap_or(0.0);
            for (j, &output) in layer1_output.iter().enumerate() {
                sum += output * weights[1].get(j).copied().unwrap_or(0.0);
            }
            layer2_output.push(self.sigmoid(sum));
        }

        // Output layer
        let mut final_output = biases.get(20).copied().unwrap_or(0.0);
        for (i, &output) in layer2_output.iter().enumerate() {
            final_output += output * weights[2].get(i).copied().unwrap_or(0.0);
        }

        Ok(self.sigmoid(final_output))
    }

    async fn extract_fraud_features(&self, input: &FraudDetectionInput) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![
            input.transaction_amount / 1000.0, // Normalized
            input.account_balance / 10000.0,   // Normalized  
            input.time_since_last_transaction as f64 / 3600.0, // Hours
            input.account_age_days as f64 / 365.0, // Years
            input.average_transaction_amount / 1000.0,
            input.transaction_count_last_hour as f64,
            if input.is_weekend { 1.0 } else { 0.0 },
            if input.is_night_time { 1.0 } else { 0.0 },
            self.get_merchant_risk_score(&input.merchant_category).await?,
            self.get_location_risk_score(input.location.as_deref()).await?,
        ])
    }

    async fn extract_risk_features(&self, input: &RiskAssessmentInput) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![
            input.account_balance / 10000.0,
            input.credit_score.unwrap_or(650) as f64 / 850.0,
            input.annual_income.unwrap_or(50000.0) / 100000.0,
            input.debt_to_income_ratio.unwrap_or(0.3),
            input.number_of_accounts as f64 / 10.0,
            if input.kyc_status { 1.0 } else { 0.0 },
            input.country_risk_score,
            self.calculate_transaction_volatility(&input.transaction_history).await?,
        ])
    }

    async fn isolation_forest(&self, data: &[f64], _parameters: &serde_json::Value) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        // Simplified isolation forest implementation
        let mut anomaly_scores = Vec::new();
        
        for &value in data {
            let z_score = (value - self.mean(data)) / self.std_dev(data);
            let anomaly_score = 1.0 / (1.0 + (-z_score.abs()).exp());
            anomaly_scores.push(anomaly_score);
        }
        
        Ok(anomaly_scores)
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn mean(&self, data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn std_dev(&self, data: &[f64]) -> f64 {
        let mean = self.mean(data);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    async fn get_merchant_risk_score(&self, category: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // High-risk merchant categories
        let high_risk = ["gambling", "adult", "cryptocurrency", "cash_advance"];
        let medium_risk = ["jewelry", "electronics", "travel"];
        
        if high_risk.contains(&category) {
            Ok(0.8)
        } else if medium_risk.contains(&category) {
            Ok(0.5)  
        } else {
            Ok(0.2)
        }
    }

    async fn get_location_risk_score(&self, location: Option<&str>) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        match location {
            Some(loc) => {
                // High-risk countries/regions
                let high_risk_locations = ["XX", "YY"]; // Placeholder country codes
                if high_risk_locations.iter().any(|&hr| loc.contains(hr)) {
                    Ok(0.9)
                } else {
                    Ok(0.1)
                }
            },
            None => Ok(0.5), // Unknown location = medium risk
        }
    }

    fn categorize_risk(&self, score: f64) -> &'static str {
        if score > 0.8 { "HIGH" }
        else if score > 0.5 { "MEDIUM" }
        else { "LOW" }
    }

    fn recommend_action(&self, score: f64) -> &'static str {
        if score > 0.8 { "DENY_TRANSACTION" }
        else if score > 0.6 { "REQUIRE_ADDITIONAL_VERIFICATION" }
        else if score > 0.4 { "MONITOR_CLOSELY" }
        else { "APPROVE" }
    }

    async fn calculate_transaction_volatility(&self, transactions: &[f64]) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        if transactions.len() < 2 {
            return Ok(0.0);
        }
        
        let mean = self.mean(transactions);
        let variance = transactions.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (transactions.len() - 1) as f64;
        
        Ok(variance.sqrt() / mean)
    }

    async fn update_q_table(&self, _model_id: Uuid, _action: String, _reward: f64, _state: serde_json::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        // This is a placeholder for the actual Q-table update logic
        Ok(())
    }

    async fn simulate_training_step(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate a training step and return loss
        Ok(rand::random::<f64>() * 0.1) // Random loss between 0 and 0.1
    }

    async fn validate_model(&self, _model_id: Uuid, _dataset: &[serde_json::Value]) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate validation accuracy
        Ok(0.80 + rand::random::<f64>() * 0.15) // Random accuracy between 80% and 95%
    }

    async fn initialize_models(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Create default AI models
        let fraud_model = AIModel {
            id: Uuid::new_v4(),
            name: "Fraud Detection Neural Network".to_string(),
            model_type: ModelType::FraudDetection,
            version: "1.0.0".to_string(),
            parameters: serde_json::json!({
                "weights": [
                    vec![vec![0.1; 10]; 10], // Input to hidden layer
                    vec![vec![0.1; 8]; 10],  // Hidden to hidden layer  
                    vec![0.1; 8]             // Hidden to output layer
                ],
                "biases": vec![0.0; 30],
                "activation": "sigmoid",
                "learning_rate": 0.001
            }),
            training_data: Some("fraud_training_dataset_v1".to_string()),
            accuracy: 0.92,
            last_trained: Utc::now(),
            active: true,
        };

        let risk_model = AIModel {
            id: Uuid::new_v4(), 
            name: "Risk Assessment Model".to_string(),
            model_type: ModelType::RiskAssessment,
            version: "1.0.0".to_string(),
            parameters: serde_json::json!({
                "weights": [
                    vec![vec![0.1; 8]; 8],
                    vec![vec![0.1; 6]; 8],
                    vec![0.1; 6]
                ],
                "biases": vec![0.0; 22],
                "activation": "sigmoid",
                "learning_rate": 0.001
            }),
            training_data: Some("risk_assessment_dataset_v1".to_string()),
            accuracy: 0.88,
            last_trained: Utc::now(),
            active: true,
        };

        let mut models = self.models.write().await;
        models.insert(fraud_model.id, fraud_model);
        models.insert(risk_model.id, risk_model);

        Ok(())
    }

    async fn store_prediction(&self, prediction: &Prediction) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query!(
            "INSERT INTO ai_predictions 
             (id, model_id, input_data, prediction, confidence, created_at, feedback_received)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            prediction.id,
            prediction.model_id,
            prediction.input_data,
            prediction.prediction,
            prediction.confidence,
            prediction.created_at,
            prediction.feedback_received
        ).execute(self.db.as_ref()).await?;

        Ok(())
    }

    async fn update_model_after_training(&self, session: &TrainingSession) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query!(
            "UPDATE ai_models SET accuracy = $1, last_trained = $2 WHERE id = $3",
            session.validation_accuracy,
            Utc::now(),
            session.model_id
        ).execute(self.db.as_ref()).await?;

        Ok(())
    }
}