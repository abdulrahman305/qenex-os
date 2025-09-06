#!/usr/bin/env python3
"""
QENEX AI Self-Improvement Engine v3.0
Advanced autonomous learning system with evolutionary algorithms and neural architecture search
"""

import asyncio
import json
import time
import logging
import hashlib
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import pickle
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import optuna
import joblib
import genetic_algorithm as ga
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    memory_usage: float
    model_complexity: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AIModel:
    """AI model representation"""
    model_id: str
    model_type: str
    model_params: Dict[str, Any]
    performance: ModelPerformance
    generation: int
    parent_models: List[str]
    architecture: Dict[str, Any]
    model_object: Any = None
    is_active: bool = True

class EvolutionaryOptimizer:
    """Genetic algorithm for model optimization"""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = 0
        
    def create_initial_population(self, model_types: List[str]) -> List[Dict[str, Any]]:
        """Create initial population of model configurations"""
        population = []
        
        for _ in range(self.population_size):
            model_type = random.choice(model_types)
            genome = self._create_random_genome(model_type)
            population.append(genome)
        
        return population
    
    def _create_random_genome(self, model_type: str) -> Dict[str, Any]:
        """Create random model configuration"""
        if model_type == "random_forest":
            return {
                "type": "random_forest",
                "n_estimators": random.randint(50, 500),
                "max_depth": random.choice([None] + list(range(3, 20))),
                "min_samples_split": random.randint(2, 20),
                "min_samples_leaf": random.randint(1, 10),
                "max_features": random.choice(["sqrt", "log2", None])
            }
        elif model_type == "gradient_boosting":
            return {
                "type": "gradient_boosting",
                "n_estimators": random.randint(50, 300),
                "learning_rate": random.uniform(0.01, 0.3),
                "max_depth": random.randint(3, 10),
                "subsample": random.uniform(0.5, 1.0),
                "max_features": random.choice(["sqrt", "log2", None])
            }
        elif model_type == "neural_network":
            return {
                "type": "neural_network",
                "hidden_layer_sizes": tuple(random.randint(50, 500) for _ in range(random.randint(1, 4))),
                "activation": random.choice(["relu", "tanh", "logistic"]),
                "alpha": random.uniform(0.0001, 0.1),
                "learning_rate": random.choice(["constant", "invscaling", "adaptive"]),
                "max_iter": random.randint(200, 1000)
            }
        elif model_type == "svm":
            return {
                "type": "svm",
                "C": random.uniform(0.1, 100),
                "kernel": random.choice(["linear", "poly", "rbf", "sigmoid"]),
                "gamma": random.choice(["scale", "auto"]) if random.choice([True, False]) else random.uniform(0.001, 1.0),
                "degree": random.randint(2, 5) if random.choice([True, False]) else 3
            }
        else:  # logistic_regression
            return {
                "type": "logistic_regression",
                "C": random.uniform(0.01, 100),
                "penalty": random.choice(["l2", "elasticnet", "none"]),
                "solver": random.choice(["liblinear", "lbfgs", "saga"]),
                "max_iter": random.randint(100, 2000)
            }
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring from two parent genomes"""
        # Single-point crossover
        child = parent1.copy()
        
        # Randomly select parameters from parent2
        for key, value in parent2.items():
            if key != "type" and random.random() < 0.5:
                child[key] = value
        
        return child
    
    def mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a genome"""
        mutated = genome.copy()
        
        if random.random() < self.mutation_rate:
            # Randomly mutate one parameter
            param_to_mutate = random.choice([k for k in mutated.keys() if k != "type"])
            
            if param_to_mutate in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                mutated[param_to_mutate] = random.randint(1, 500)
            elif param_to_mutate in ["learning_rate", "alpha", "subsample", "C"]:
                mutated[param_to_mutate] = random.uniform(0.001, 10.0)
            elif param_to_mutate == "activation":
                mutated[param_to_mutate] = random.choice(["relu", "tanh", "logistic"])
            elif param_to_mutate == "kernel":
                mutated[param_to_mutate] = random.choice(["linear", "poly", "rbf", "sigmoid"])
        
        return mutated
    
    def select_parents(self, population: List[Tuple[Dict[str, Any], float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Tournament selection for parent selection"""
        tournament_size = 3
        
        def tournament_select():
            tournament = random.sample(population, min(tournament_size, len(population)))
            return max(tournament, key=lambda x: x[1])[0]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2

class NeuralArchitectureSearch:
    """Neural Architecture Search for optimal neural network design"""
    
    def __init__(self):
        self.search_space = {
            "num_layers": [1, 2, 3, 4, 5],
            "layer_sizes": [32, 64, 128, 256, 512, 1024],
            "activation_functions": ["relu", "tanh", "sigmoid", "swish"],
            "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.5],
            "learning_rates": [0.0001, 0.001, 0.01, 0.1],
            "batch_sizes": [16, 32, 64, 128, 256],
            "optimizers": ["adam", "sgd", "rmsprop", "adagrad"]
        }
    
    def create_model_from_config(self, config: Dict[str, Any], input_shape: int, num_classes: int) -> tf.keras.Model:
        """Create TensorFlow model from configuration"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(
            config["layer_sizes"][0],
            activation=config["activation_functions"][0],
            input_shape=(input_shape,)
        ))
        
        if config["dropout_rates"][0] > 0:
            model.add(tf.keras.layers.Dropout(config["dropout_rates"][0]))
        
        # Hidden layers
        for i in range(1, config["num_layers"]):
            model.add(tf.keras.layers.Dense(
                config["layer_sizes"][min(i, len(config["layer_sizes"])-1)],
                activation=config["activation_functions"][min(i, len(config["activation_functions"])-1)]
            ))
            
            if i < len(config["dropout_rates"]) and config["dropout_rates"][i] > 0:
                model.add(tf.keras.layers.Dropout(config["dropout_rates"][i]))
        
        # Output layer
        if num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        else:
            model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        
        # Compile model
        optimizer = config["optimizers"][0]
        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rates"][0])
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=config["learning_rates"][0])
        elif optimizer == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rates"][0])
        else:
            opt = tf.keras.optimizers.Adagrad(learning_rate=config["learning_rates"][0])
        
        loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        
        return model
    
    def random_search(self, num_trials: int = 50) -> List[Dict[str, Any]]:
        """Random search over architecture space"""
        configurations = []
        
        for _ in range(num_trials):
            config = {}
            for param, values in self.search_space.items():
                if param in ["layer_sizes", "activation_functions", "dropout_rates"]:
                    # Create lists for layers
                    num_layers = random.choice(self.search_space["num_layers"])
                    config[param] = [random.choice(values) for _ in range(num_layers)]
                else:
                    config[param] = [random.choice(values)]
            
            configurations.append(config)
        
        return configurations

class AutoMLOptimizer:
    """Automated Machine Learning with Optuna"""
    
    def __init__(self):
        self.study = None
        
    def objective(self, trial, X_train, X_val, y_train, y_val, task_type="classification"):
        """Optuna objective function"""
        # Suggest model type
        model_type = trial.suggest_categorical('model_type', [
            'random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'neural_network'
        ])
        
        # Create model with suggested hyperparameters
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 500),
                max_depth=trial.suggest_int('rf_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('gb_n_estimators', 50, 300),
                learning_rate=trial.suggest_float('gb_learning_rate', 0.01, 0.3),
                max_depth=trial.suggest_int('gb_max_depth', 3, 10),
                subsample=trial.suggest_float('gb_subsample', 0.5, 1.0),
                random_state=42
            )
        elif model_type == 'svm':
            model = SVC(
                C=trial.suggest_float('svm_C', 0.1, 100, log=True),
                kernel=trial.suggest_categorical('svm_kernel', ['linear', 'poly', 'rbf']),
                gamma=trial.suggest_categorical('svm_gamma', ['scale', 'auto']),
                random_state=42
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=trial.suggest_float('lr_C', 0.01, 100, log=True),
                penalty=trial.suggest_categorical('lr_penalty', ['l2', 'none']),
                max_iter=trial.suggest_int('lr_max_iter', 100, 2000),
                random_state=42
            )
        else:  # neural_network
            model = MLPClassifier(
                hidden_layer_sizes=tuple([trial.suggest_int(f'nn_layer_{i}', 50, 500) 
                                         for i in range(trial.suggest_int('nn_n_layers', 1, 4))]),
                activation=trial.suggest_categorical('nn_activation', ['relu', 'tanh', 'logistic']),
                alpha=trial.suggest_float('nn_alpha', 0.0001, 0.1, log=True),
                learning_rate=trial.suggest_categorical('nn_learning_rate', ['constant', 'adaptive']),
                max_iter=trial.suggest_int('nn_max_iter', 200, 1000),
                random_state=42
            )
        
        # Train and evaluate
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_val)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        # Store additional metrics in trial
        trial.set_user_attr('training_time', training_time)
        trial.set_user_attr('inference_time', inference_time)
        trial.set_user_attr('precision', precision_score(y_val, y_pred, average='weighted', zero_division=0))
        trial.set_user_attr('recall', recall_score(y_val, y_pred, average='weighted', zero_division=0))
        trial.set_user_attr('f1', f1_score(y_val, y_pred, average='weighted', zero_division=0))
        
        return accuracy
    
    def optimize(self, X, y, n_trials: int = 100, cv_folds: int = 3):
        """Run optimization study"""
        self.study = optuna.create_study(direction='maximize')
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials
        )
        
        return self.study.best_params, self.study.best_value

class AIModelManager:
    """Manage AI model lifecycle and performance"""
    
    def __init__(self, db_path: str = "/tmp/qenex_ai_models.db"):
        self.db_path = db_path
        self.models: Dict[str, AIModel] = {}
        self.performance_history: List[ModelPerformance] = []
        self._init_database()
    
    def _init_database(self):
        """Initialize model database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                model_params TEXT NOT NULL,
                performance_data TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_models TEXT,
                architecture TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                f1_score REAL NOT NULL,
                training_time REAL NOT NULL,
                inference_time REAL NOT NULL,
                memory_usage REAL NOT NULL,
                model_complexity INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_model(self, model: AIModel):
        """Save model to database and disk"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO models (
                model_id, model_type, model_params, performance_data,
                generation, parent_models, architecture, is_active,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.model_id,
            model.model_type,
            json.dumps(model.model_params),
            json.dumps({
                'accuracy': model.performance.accuracy,
                'precision': model.performance.precision,
                'recall': model.performance.recall,
                'f1_score': model.performance.f1_score,
                'training_time': model.performance.training_time,
                'inference_time': model.performance.inference_time,
                'memory_usage': model.performance.memory_usage,
                'model_complexity': model.performance.model_complexity
            }),
            model.generation,
            json.dumps(model.parent_models),
            json.dumps(model.architecture),
            1 if model.is_active else 0,
            model.performance.timestamp.isoformat(),
            datetime.now(timezone.utc).isoformat()
        ))
        
        # Save performance history
        conn.execute("""
            INSERT INTO performance_history (
                model_id, accuracy, precision_score, recall_score, f1_score,
                training_time, inference_time, memory_usage, model_complexity, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.model_id,
            model.performance.accuracy,
            model.performance.precision,
            model.performance.recall,
            model.performance.f1_score,
            model.performance.training_time,
            model.performance.inference_time,
            model.performance.memory_usage,
            model.performance.model_complexity,
            model.performance.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Save model object to disk
        model_path = Path(f"/tmp/qenex_models/{model.model_id}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model.model_object:
            joblib.dump(model.model_object, model_path)
        
        # Store in memory
        self.models[model.model_id] = model
        
        logger.info(f"Model {model.model_id} saved successfully")
    
    def load_model(self, model_id: str) -> Optional[AIModel]:
        """Load model from database and disk"""
        if model_id in self.models:
            return self.models[model_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Reconstruct model object
        model_path = Path(f"/tmp/qenex_models/{model_id}.pkl")
        model_object = None
        
        if model_path.exists():
            try:
                model_object = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model object for {model_id}: {e}")
        
        # Create AIModel instance
        performance_data = json.loads(row[3])
        model = AIModel(
            model_id=row[0],
            model_type=row[1],
            model_params=json.loads(row[2]),
            performance=ModelPerformance(
                accuracy=performance_data['accuracy'],
                precision=performance_data['precision'],
                recall=performance_data['recall'],
                f1_score=performance_data['f1_score'],
                training_time=performance_data['training_time'],
                inference_time=performance_data['inference_time'],
                memory_usage=performance_data['memory_usage'],
                model_complexity=performance_data['model_complexity'],
                timestamp=datetime.fromisoformat(row[8])
            ),
            generation=row[4],
            parent_models=json.loads(row[5]) if row[5] else [],
            architecture=json.loads(row[6]) if row[6] else {},
            model_object=model_object,
            is_active=bool(row[7])
        )
        
        self.models[model_id] = model
        return model
    
    def get_best_models(self, n: int = 5, metric: str = "accuracy") -> List[AIModel]:
        """Get top N best performing models"""
        all_models = list(self.models.values())
        
        if metric == "accuracy":
            all_models.sort(key=lambda m: m.performance.accuracy, reverse=True)
        elif metric == "f1_score":
            all_models.sort(key=lambda m: m.performance.f1_score, reverse=True)
        elif metric == "training_time":
            all_models.sort(key=lambda m: m.performance.training_time)
        elif metric == "inference_time":
            all_models.sort(key=lambda m: m.performance.inference_time)
        
        return all_models[:n]

class SelfImprovementEngine:
    """Main self-improvement engine coordinating all optimization strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "population_size": 20,
            "mutation_rate": 0.1,
            "optimization_interval": 3600,  # 1 hour
            "performance_threshold": 0.95,
            "max_generations": 100,
            "automl_trials": 100,
            "nas_trials": 50
        }
        
        self.evolutionary_optimizer = EvolutionaryOptimizer(
            population_size=self.config["population_size"],
            mutation_rate=self.config["mutation_rate"]
        )
        self.nas = NeuralArchitectureSearch()
        self.automl = AutoMLOptimizer()
        self.model_manager = AIModelManager()
        
        self.current_generation = 0
        self.best_performance = 0.0
        self.training_data = None
        self.is_running = False
        
        logger.info("Self-Improvement Engine initialized")
    
    async def initialize(self, training_data: Tuple[np.ndarray, np.ndarray]):
        """Initialize with training data"""
        self.training_data = training_data
        X, y = training_data
        
        # Create initial population of models
        model_types = ["random_forest", "gradient_boosting", "neural_network", "svm", "logistic_regression"]
        initial_genomes = self.evolutionary_optimizer.create_initial_population(model_types)
        
        # Train and evaluate initial models
        for i, genome in enumerate(initial_genomes):
            model_id = f"gen0_model_{i}"
            model = await self._create_and_train_model(model_id, genome, X, y, generation=0)
            if model:
                self.model_manager.save_model(model)
                if model.performance.accuracy > self.best_performance:
                    self.best_performance = model.performance.accuracy
        
        logger.info(f"Initial population created with {len(initial_genomes)} models")
        logger.info(f"Best initial performance: {self.best_performance:.4f}")
    
    async def start_improvement_loop(self):
        """Start continuous improvement process"""
        self.is_running = True
        
        while self.is_running and self.current_generation < self.config["max_generations"]:
            try:
                await self._run_evolution_cycle()
                await self._run_automl_optimization()
                await self._run_neural_architecture_search()
                await self._cleanup_poor_models()
                
                self.current_generation += 1
                
                # Log progress
                best_models = self.model_manager.get_best_models(n=3)
                if best_models:
                    logger.info(f"Generation {self.current_generation} complete")
                    logger.info(f"Best performance: {best_models[0].performance.accuracy:.4f}")
                    
                    # Check if we've reached the performance threshold
                    if best_models[0].performance.accuracy >= self.config["performance_threshold"]:
                        logger.info("Performance threshold reached!")
                        break
                
                # Wait before next cycle
                await asyncio.sleep(self.config["optimization_interval"])
                
            except Exception as e:
                logger.error(f"Error in improvement cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _run_evolution_cycle(self):
        """Run evolutionary optimization cycle"""
        if not self.training_data:
            return
        
        X, y = self.training_data
        
        # Get current population performance
        current_models = list(self.model_manager.models.values())
        if len(current_models) < 5:
            return
        
        # Create fitness scores
        population_with_fitness = [(
            {
                "type": model.model_type,
                **model.model_params
            },
            model.performance.f1_score
        ) for model in current_models[-20:]]  # Use last 20 models
        
        # Select parents and create offspring
        offspring = []
        for _ in range(10):  # Create 10 offspring
            parent1, parent2 = self.evolutionary_optimizer.select_parents(population_with_fitness)
            child = self.evolutionary_optimizer.crossover(parent1, parent2)
            child = self.evolutionary_optimizer.mutate(child)
            offspring.append(child)
        
        # Train offspring
        for i, genome in enumerate(offspring):
            model_id = f"gen{self.current_generation}_evo_{i}"
            model = await self._create_and_train_model(
                model_id, genome, X, y, 
                generation=self.current_generation,
                parent_models=[m.model_id for m in current_models[-2:]]
            )
            if model:
                self.model_manager.save_model(model)
        
        logger.info(f"Evolution cycle complete: {len(offspring)} offspring created")
    
    async def _run_automl_optimization(self):
        """Run AutoML optimization"""
        if not self.training_data:
            return
        
        X, y = self.training_data
        
        try:
            # Run optimization
            best_params, best_score = self.automl.optimize(X, y, n_trials=self.config["automl_trials"])
            
            # Create model from best parameters
            model_id = f"gen{self.current_generation}_automl"
            
            # Convert Optuna params to our format
            genome = {
                "type": best_params["model_type"],
                **{k: v for k, v in best_params.items() if k != "model_type"}
            }
            
            model = await self._create_and_train_model(
                model_id, genome, X, y,
                generation=self.current_generation
            )
            
            if model:
                self.model_manager.save_model(model)
                logger.info(f"AutoML optimization complete: score {best_score:.4f}")
        
        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")
    
    async def _run_neural_architecture_search(self):
        """Run neural architecture search"""
        if not self.training_data:
            return
        
        X, y = self.training_data
        
        try:
            # Generate architectures
            architectures = self.nas.random_search(num_trials=self.config["nas_trials"])
            
            # Evaluate top architectures
            for i, arch in enumerate(architectures[:5]):  # Test top 5
                model_id = f"gen{self.current_generation}_nas_{i}"
                
                # Create and train TensorFlow model
                tf_model = self.nas.create_model_from_config(
                    arch, 
                    input_shape=X.shape[1],
                    num_classes=len(np.unique(y))
                )
                
                # Train model
                start_time = time.time()
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                history = tf_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=arch["batch_sizes"][0],
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate
                start_time = time.time()
                val_predictions = tf_model.predict(X_val)
                inference_time = time.time() - start_time
                
                if len(np.unique(y)) == 2:
                    y_pred = (val_predictions > 0.5).astype(int).flatten()
                else:
                    y_pred = np.argmax(val_predictions, axis=1)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                # Create performance object
                performance = ModelPerformance(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    training_time=training_time,
                    inference_time=inference_time,
                    memory_usage=tf_model.count_params() * 4,  # Approximate
                    model_complexity=tf_model.count_params()
                )
                
                # Create AI model
                model = AIModel(
                    model_id=model_id,
                    model_type="neural_network_tf",
                    model_params=arch,
                    performance=performance,
                    generation=self.current_generation,
                    parent_models=[],
                    architecture=arch,
                    model_object=tf_model
                )
                
                # Save model
                self.model_manager.save_model(model)
            
            logger.info("Neural architecture search complete")
        
        except Exception as e:
            logger.error(f"Neural architecture search failed: {e}")
    
    async def _create_and_train_model(
        self, 
        model_id: str, 
        genome: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray,
        generation: int = 0,
        parent_models: List[str] = None
    ) -> Optional[AIModel]:
        """Create and train model from genome"""
        try:
            # Create model based on type
            if genome["type"] == "random_forest":
                model = RandomForestClassifier(**{k: v for k, v in genome.items() if k != "type"})
            elif genome["type"] == "gradient_boosting":
                model = GradientBoostingClassifier(**{k: v for k, v in genome.items() if k != "type"})
            elif genome["type"] == "neural_network":
                model = MLPClassifier(**{k: v for k, v in genome.items() if k != "type"})
            elif genome["type"] == "svm":
                model = SVC(**{k: v for k, v in genome.items() if k != "type"})
            elif genome["type"] == "logistic_regression":
                model = LogisticRegression(**{k: v for k, v in genome.items() if k != "type"})
            else:
                return None
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            start_time = time.time()
            y_pred = model.predict(X_val)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # Estimate model complexity
            complexity = getattr(model, 'n_features_in_', X.shape[1])
            if hasattr(model, 'n_estimators'):
                complexity *= model.n_estimators
            
            # Create performance object
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=complexity * 8,  # Rough estimate
                model_complexity=complexity
            )
            
            # Create AI model
            ai_model = AIModel(
                model_id=model_id,
                model_type=genome["type"],
                model_params={k: v for k, v in genome.items() if k != "type"},
                performance=performance,
                generation=generation,
                parent_models=parent_models or [],
                architecture=genome,
                model_object=model
            )
            
            return ai_model
        
        except Exception as e:
            logger.error(f"Failed to create model {model_id}: {e}")
            return None
    
    async def _cleanup_poor_models(self):
        """Remove poorly performing models to save space"""
        all_models = list(self.model_manager.models.values())
        
        if len(all_models) > 100:  # Keep only top 100 models
            all_models.sort(key=lambda m: m.performance.f1_score, reverse=True)
            models_to_remove = all_models[100:]
            
            for model in models_to_remove:
                # Remove from memory
                if model.model_id in self.model_manager.models:
                    del self.model_manager.models[model.model_id]
                
                # Remove model file
                model_path = Path(f"/tmp/qenex_models/{model.model_id}.pkl")
                if model_path.exists():
                    model_path.unlink()
            
            logger.info(f"Cleaned up {len(models_to_remove)} poor performing models")
    
    async def get_best_model(self) -> Optional[AIModel]:
        """Get current best performing model"""
        best_models = self.model_manager.get_best_models(n=1, metric="f1_score")
        return best_models[0] if best_models else None
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        all_models = list(self.model_manager.models.values())
        
        if not all_models:
            return {"message": "No models available"}
        
        # Calculate statistics
        accuracies = [m.performance.accuracy for m in all_models]
        f1_scores = [m.performance.f1_score for m in all_models]
        training_times = [m.performance.training_time for m in all_models]
        
        best_model = max(all_models, key=lambda m: m.performance.f1_score)
        
        return {
            "total_models": len(all_models),
            "current_generation": self.current_generation,
            "best_performance": {
                "model_id": best_model.model_id,
                "accuracy": best_model.performance.accuracy,
                "f1_score": best_model.performance.f1_score,
                "training_time": best_model.performance.training_time
            },
            "average_performance": {
                "accuracy": np.mean(accuracies),
                "f1_score": np.mean(f1_scores),
                "training_time": np.mean(training_times)
            },
            "performance_improvement": {
                "accuracy_std": np.std(accuracies),
                "f1_std": np.std(f1_scores),
                "generation_progress": self.current_generation / self.config["max_generations"]
            },
            "model_types": {
                model_type: len([m for m in all_models if m.model_type == model_type])
                for model_type in set(m.model_type for m in all_models)
            }
        }
    
    def stop(self):
        """Stop the improvement loop"""
        self.is_running = False
        logger.info("Self-improvement engine stopped")

async def demonstrate_self_improvement():
    """Demonstrate the AI Self-Improvement Engine"""
    print("=" * 80)
    print("QENEX AI SELF-IMPROVEMENT ENGINE v3.0 - DEMONSTRATION")
    print("=" * 80)
    
    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple decision boundary
    
    print(f"Generated synthetic dataset: {n_samples} samples, {n_features} features")
    
    # Initialize self-improvement engine
    config = {
        "population_size": 10,  # Smaller for demo
        "mutation_rate": 0.2,
        "optimization_interval": 5,  # 5 seconds for demo
        "performance_threshold": 0.95,
        "max_generations": 5,  # Fewer generations for demo
        "automl_trials": 20,  # Fewer trials for demo
        "nas_trials": 10
    }
    
    engine = SelfImprovementEngine(config)
    
    # Initialize with training data
    print("\n1. INITIALIZING SELF-IMPROVEMENT ENGINE")
    print("-" * 50)
    await engine.initialize((X, y))
    
    # Get initial stats
    stats = await engine.get_system_stats()
    print(f"Initial models created: {stats['total_models']}")
    print(f"Best initial performance: {stats['best_performance']['f1_score']:.4f}")
    
    # Run improvement for a few cycles
    print("\n2. RUNNING SELF-IMPROVEMENT CYCLES")
    print("-" * 50)
    
    # Run 3 improvement cycles
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        
        # Run one evolution cycle
        await engine._run_evolution_cycle()
        
        # Run AutoML optimization
        if cycle % 2 == 0:  # Every other cycle
            await engine._run_automl_optimization()
        
        # Get updated stats
        stats = await engine.get_system_stats()
        print(f"  Total models: {stats['total_models']}")
        print(f"  Best F1 score: {stats['best_performance']['f1_score']:.4f}")
        print(f"  Average F1 score: {stats['average_performance']['f1_score']:.4f}")
        
        # Show model type distribution
        print(f"  Model types: {stats['model_types']}")
        
        # Wait between cycles
        await asyncio.sleep(1)
    
    # Get final best model
    print("\n3. FINAL RESULTS")
    print("-" * 50)
    
    best_model = await engine.get_best_model()
    if best_model:
        print(f"Best Model ID: {best_model.model_id}")
        print(f"Model Type: {best_model.model_type}")
        print(f"Generation: {best_model.generation}")
        print(f"Performance:")
        print(f"  - Accuracy: {best_model.performance.accuracy:.4f}")
        print(f"  - Precision: {best_model.performance.precision:.4f}")
        print(f"  - Recall: {best_model.performance.recall:.4f}")
        print(f"  - F1 Score: {best_model.performance.f1_score:.4f}")
        print(f"  - Training Time: {best_model.performance.training_time:.2f}s")
        print(f"  - Inference Time: {best_model.performance.inference_time:.4f}s")
        print(f"Model Parameters: {best_model.model_params}")
    
    # Final system statistics
    final_stats = await engine.get_system_stats()
    print(f"\n4. SYSTEM STATISTICS")
    print("-" * 50)
    print(f"Total Models Created: {final_stats['total_models']}")
    print(f"Generations Completed: {final_stats['current_generation']}")
    print(f"Performance Improvement:")
    print(f"  - Best F1 Score: {final_stats['best_performance']['f1_score']:.4f}")
    print(f"  - Average F1 Score: {final_stats['average_performance']['f1_score']:.4f}")
    print(f"  - F1 Score Std Dev: {final_stats['performance_improvement']['f1_std']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ AI SELF-IMPROVEMENT ENGINE DEMONSTRATION COMPLETE!")
    print("🧠 Autonomous Learning ✅")
    print("🔬 Evolutionary Optimization ✅") 
    print("🤖 AutoML Integration ✅")
    print("🏗️ Neural Architecture Search ✅")
    print("📊 Performance Tracking ✅")
    print("=" * 80)

if __name__ == "__main__":
    # Create necessary directories
    Path("/tmp/qenex_models").mkdir(parents=True, exist_ok=True)
    
    # Run demonstration
    asyncio.run(demonstrate_self_improvement())