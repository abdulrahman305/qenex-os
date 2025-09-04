#!/usr/bin/env python3
"""
Self-healing Infrastructure with Automatic Failover and Recovery
Production-ready infrastructure management with predictive maintenance
"""

import asyncio
import psutil
import socket
import subprocess
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import yaml
import threading
from queue import PriorityQueue
import numpy as np
from collections import deque, defaultdict
import aiohttp
import signal
import os
import sys

# Configuration
HEALTH_CHECK_INTERVAL = 5  # seconds
METRIC_COLLECTION_INTERVAL = 1  # seconds
FAILURE_DETECTION_THRESHOLD = 3  # consecutive failures
RECOVERY_TIMEOUT = 60  # seconds
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 30
PREDICTIVE_MAINTENANCE_WINDOW = 3600  # 1 hour lookahead
MAX_RESTART_ATTEMPTS = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service operational states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of infrastructure failures"""
    CRASH = "crash"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"
    CPU_OVERLOAD = "cpu_overload"
    DISK_FULL = "disk_full"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIG_ERROR = "config_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for failures"""
    RESTART = "restart"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAK = "circuit_break"
    ROLLBACK = "rollback"
    MIGRATE = "migrate"
    MANUAL = "manual"


@dataclass
class ServiceHealth:
    """Service health metrics"""
    service_id: str
    state: ServiceState
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    request_rate: float
    error_rate: float
    response_time: float
    uptime: float
    last_check: float
    consecutive_failures: int = 0
    health_score: float = 100.0
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score"""
        scores = []
        
        # CPU score (inverse of usage)
        scores.append(max(0, 100 - self.cpu_usage))
        
        # Memory score
        scores.append(max(0, 100 - self.memory_usage))
        
        # Error rate score
        if self.request_rate > 0:
            error_percentage = (self.error_rate / self.request_rate) * 100
            scores.append(max(0, 100 - error_percentage))
        
        # Response time score (assuming < 100ms is perfect)
        response_score = max(0, 100 - (self.response_time / 10))
        scores.append(response_score)
        
        # Network latency score (assuming < 10ms is perfect)
        latency_score = max(0, 100 - (self.network_latency * 10))
        scores.append(latency_score)
        
        # Weight recent failures heavily
        if self.consecutive_failures > 0:
            failure_penalty = min(50, self.consecutive_failures * 10)
            scores.append(100 - failure_penalty)
        
        self.health_score = np.mean(scores) if scores else 0
        return self.health_score


@dataclass
class FailureEvent:
    """Infrastructure failure event"""
    event_id: str
    service_id: str
    failure_type: FailureType
    timestamp: float
    severity: int  # 1-5
    description: str
    metrics: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovered: bool = False
    recovery_time: Optional[float] = None


@dataclass
class ServiceDependency:
    """Service dependency mapping"""
    service_id: str
    depends_on: List[str]
    dependents: List[str]
    critical: bool
    health_propagation: float = 0.5  # How much health affects dependents


class CircuitBreaker:
    """Circuit breaker for failure isolation"""
    
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD,
                 timeout: int = CIRCUIT_BREAKER_TIMEOUT):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
    
    def record_success(self):
        """Record successful operation"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 3:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_proceed(self) -> bool:
        """Check if operation can proceed"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        
        return True  # HALF_OPEN allows traffic


class PredictiveMaintenance:
    """Predictive maintenance using ML"""
    
    def __init__(self):
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.failure_patterns = []
        self.predictions = {}
    
    def record_metrics(self, service_id: str, metrics: Dict[str, float]):
        """Record service metrics for analysis"""
        self.metric_history[service_id].append({
            'timestamp': time.time(),
            'metrics': metrics
        })
    
    def predict_failure(self, service_id: str) -> Tuple[bool, float, str]:
        """Predict potential failure"""
        if service_id not in self.metric_history:
            return False, 0, ""
        
        history = list(self.metric_history[service_id])
        if len(history) < 10:
            return False, 0, ""
        
        # Analyze trends
        recent_metrics = history[-10:]
        
        # CPU trend
        cpu_trend = [m['metrics'].get('cpu', 0) for m in recent_metrics]
        cpu_increasing = all(cpu_trend[i] <= cpu_trend[i+1] for i in range(len(cpu_trend)-1))
        
        # Memory trend
        mem_trend = [m['metrics'].get('memory', 0) for m in recent_metrics]
        mem_increasing = all(mem_trend[i] <= mem_trend[i+1] for i in range(len(mem_trend)-1))
        
        # Error rate trend
        error_trend = [m['metrics'].get('error_rate', 0) for m in recent_metrics]
        error_increasing = sum(error_trend[-5:]) > sum(error_trend[:5])
        
        # Predict failure probability
        failure_probability = 0
        failure_reason = []
        
        if cpu_increasing and cpu_trend[-1] > 80:
            failure_probability += 0.4
            failure_reason.append("CPU exhaustion")
        
        if mem_increasing and mem_trend[-1] > 85:
            failure_probability += 0.4
            failure_reason.append("Memory exhaustion")
        
        if error_increasing:
            failure_probability += 0.2
            failure_reason.append("Increasing errors")
        
        # Check for known patterns
        if self._matches_failure_pattern(recent_metrics):
            failure_probability = min(1.0, failure_probability + 0.3)
            failure_reason.append("Matches known failure pattern")
        
        return failure_probability > 0.5, failure_probability, ", ".join(failure_reason)
    
    def _matches_failure_pattern(self, metrics: List[Dict]) -> bool:
        """Check if metrics match known failure patterns"""
        # Simplified pattern matching
        # In production, use trained ML models
        
        # Pattern 1: Rapid memory growth (memory leak)
        mem_values = [m['metrics'].get('memory', 0) for m in metrics]
        if len(mem_values) >= 5:
            mem_growth = mem_values[-1] - mem_values[-5]
            if mem_growth > 20:  # 20% growth in 5 samples
                return True
        
        # Pattern 2: CPU spikes with high error rate
        cpu_values = [m['metrics'].get('cpu', 0) for m in metrics[-3:]]
        error_values = [m['metrics'].get('error_rate', 0) for m in metrics[-3:]]
        if np.mean(cpu_values) > 90 and np.mean(error_values) > 10:
            return True
        
        return False


class AutoScaler:
    """Automatic scaling based on load"""
    
    def __init__(self):
        self.scaling_policies = {}
        self.current_instances = defaultdict(int)
        self.scaling_history = []
        self.cooldown_period = 300  # 5 minutes
        self.last_scale_time = {}
    
    def register_policy(self, service_id: str, policy: Dict[str, Any]):
        """Register scaling policy for service"""
        self.scaling_policies[service_id] = policy
        self.current_instances[service_id] = policy.get('min_instances', 1)
    
    async def evaluate_scaling(self, service_id: str, metrics: Dict[str, float]) -> Optional[str]:
        """Evaluate if scaling is needed"""
        if service_id not in self.scaling_policies:
            return None
        
        policy = self.scaling_policies[service_id]
        current = self.current_instances[service_id]
        
        # Check cooldown
        if service_id in self.last_scale_time:
            if time.time() - self.last_scale_time[service_id] < self.cooldown_period:
                return None
        
        # Scale up conditions
        if metrics.get('cpu', 0) > policy.get('scale_up_cpu', 80):
            if current < policy.get('max_instances', 10):
                return "SCALE_UP"
        
        if metrics.get('memory', 0) > policy.get('scale_up_memory', 80):
            if current < policy.get('max_instances', 10):
                return "SCALE_UP"
        
        if metrics.get('request_rate', 0) > policy.get('scale_up_requests', 1000):
            if current < policy.get('max_instances', 10):
                return "SCALE_UP"
        
        # Scale down conditions
        if metrics.get('cpu', 100) < policy.get('scale_down_cpu', 20):
            if current > policy.get('min_instances', 1):
                return "SCALE_DOWN"
        
        if metrics.get('memory', 100) < policy.get('scale_down_memory', 20):
            if current > policy.get('min_instances', 1):
                return "SCALE_DOWN"
        
        return None
    
    async def scale_service(self, service_id: str, action: str) -> bool:
        """Scale service up or down"""
        current = self.current_instances[service_id]
        
        if action == "SCALE_UP":
            new_count = current + 1
            logger.info(f"Scaling up {service_id} from {current} to {new_count} instances")
        elif action == "SCALE_DOWN":
            new_count = current - 1
            logger.info(f"Scaling down {service_id} from {current} to {new_count} instances")
        else:
            return False
        
        # In production, would actually create/destroy instances
        # For now, simulate
        self.current_instances[service_id] = new_count
        self.last_scale_time[service_id] = time.time()
        
        self.scaling_history.append({
            'service_id': service_id,
            'action': action,
            'from': current,
            'to': new_count,
            'timestamp': time.time()
        })
        
        return True


class SelfHealingOrchestrator:
    """Main self-healing infrastructure orchestrator"""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.dependencies: Dict[str, ServiceDependency] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_history: List[FailureEvent] = []
        
        self.predictive_maintenance = PredictiveMaintenance()
        self.auto_scaler = AutoScaler()
        
        self.recovery_strategies: Dict[FailureType, RecoveryStrategy] = {
            FailureType.CRASH: RecoveryStrategy.RESTART,
            FailureType.TIMEOUT: RecoveryStrategy.CIRCUIT_BREAK,
            FailureType.MEMORY_LEAK: RecoveryStrategy.RESTART,
            FailureType.CPU_OVERLOAD: RecoveryStrategy.SCALE_UP,
            FailureType.DISK_FULL: RecoveryStrategy.MIGRATE,
            FailureType.NETWORK_PARTITION: RecoveryStrategy.FAILOVER,
            FailureType.DEPENDENCY_FAILURE: RecoveryStrategy.CIRCUIT_BREAK,
            FailureType.CONFIG_ERROR: RecoveryStrategy.ROLLBACK
        }
        
        self.running = False
        self.health_check_tasks = {}
        
        # Metrics
        self.metrics = {
            'total_failures': 0,
            'recovered_failures': 0,
            'mttr': 0,  # Mean time to recovery
            'availability': 100.0
        }
    
    async def start(self):
        """Start self-healing orchestrator"""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._failure_detector())
        asyncio.create_task(self._recovery_executor())
        asyncio.create_task(self._predictive_analyzer())
        asyncio.create_task(self._metrics_collector())
        
        logger.info("Self-healing orchestrator started")
    
    def register_service(self, service_id: str, config: Dict[str, Any]):
        """Register service for monitoring"""
        self.services[service_id] = ServiceHealth(
            service_id=service_id,
            state=ServiceState.HEALTHY,
            cpu_usage=0,
            memory_usage=0,
            disk_usage=0,
            network_latency=0,
            request_rate=0,
            error_rate=0,
            response_time=0,
            uptime=0,
            last_check=time.time()
        )
        
        # Register dependencies
        if 'dependencies' in config:
            self.dependencies[service_id] = ServiceDependency(
                service_id=service_id,
                depends_on=config['dependencies'],
                dependents=[],
                critical=config.get('critical', False)
            )
        
        # Create circuit breaker
        self.circuit_breakers[service_id] = CircuitBreaker()
        
        # Register scaling policy
        if 'scaling' in config:
            self.auto_scaler.register_policy(service_id, config['scaling'])
        
        logger.info(f"Registered service: {service_id}")
    
    async def _health_monitor(self):
        """Monitor service health"""
        while self.running:
            try:
                for service_id in list(self.services.keys()):
                    asyncio.create_task(self._check_service_health(service_id))
                
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _check_service_health(self, service_id: str):
        """Check health of individual service"""
        try:
            health = self.services[service_id]
            
            # Collect system metrics
            health.cpu_usage = psutil.cpu_percent()
            health.memory_usage = psutil.virtual_memory().percent
            health.disk_usage = psutil.disk_usage('/').percent
            
            # Check service-specific health endpoint
            health_ok = await self._check_health_endpoint(service_id)
            
            if health_ok:
                health.consecutive_failures = 0
                if health.state != ServiceState.HEALTHY:
                    health.state = ServiceState.HEALTHY
                    logger.info(f"Service {service_id} recovered")
                
                self.circuit_breakers[service_id].record_success()
            else:
                health.consecutive_failures += 1
                self.circuit_breakers[service_id].record_failure()
                
                if health.consecutive_failures >= FAILURE_DETECTION_THRESHOLD:
                    await self._handle_service_failure(service_id)
            
            # Calculate health score
            health.calculate_health_score()
            
            # Update last check time
            health.last_check = time.time()
            
            # Record metrics for predictive analysis
            self.predictive_maintenance.record_metrics(service_id, {
                'cpu': health.cpu_usage,
                'memory': health.memory_usage,
                'disk': health.disk_usage,
                'error_rate': health.error_rate,
                'response_time': health.response_time
            })
            
        except Exception as e:
            logger.error(f"Health check error for {service_id}: {e}")
    
    async def _check_health_endpoint(self, service_id: str) -> bool:
        """Check service health endpoint"""
        # In production, would make actual HTTP health check
        # For demo, simulate with random success
        import random
        
        # Simulate health check
        if service_id == "database":
            return random.random() > 0.05  # 95% healthy
        elif service_id == "api":
            return random.random() > 0.1  # 90% healthy
        else:
            return random.random() > 0.02  # 98% healthy
    
    async def _handle_service_failure(self, service_id: str):
        """Handle service failure"""
        health = self.services[service_id]
        
        if health.state not in [ServiceState.UNHEALTHY, ServiceState.FAILED]:
            health.state = ServiceState.UNHEALTHY
            
            # Determine failure type
            failure_type = self._diagnose_failure(health)
            
            # Create failure event
            failure = FailureEvent(
                event_id=f"FAIL_{int(time.time())}_{service_id}",
                service_id=service_id,
                failure_type=failure_type,
                timestamp=time.time(),
                severity=self._calculate_severity(service_id, failure_type),
                description=f"Service {service_id} failed: {failure_type.value}",
                metrics={
                    'cpu': health.cpu_usage,
                    'memory': health.memory_usage,
                    'error_rate': health.error_rate
                }
            )
            
            self.failure_history.append(failure)
            self.metrics['total_failures'] += 1
            
            logger.error(f"Service failure detected: {service_id} - {failure_type.value}")
            
            # Trigger recovery
            await self._initiate_recovery(failure)
    
    def _diagnose_failure(self, health: ServiceHealth) -> FailureType:
        """Diagnose type of failure"""
        if health.cpu_usage > 95:
            return FailureType.CPU_OVERLOAD
        elif health.memory_usage > 95:
            return FailureType.MEMORY_LEAK
        elif health.disk_usage > 95:
            return FailureType.DISK_FULL
        elif health.network_latency > 1000:
            return FailureType.NETWORK_PARTITION
        elif health.error_rate > health.request_rate * 0.5:
            return FailureType.DEPENDENCY_FAILURE
        else:
            return FailureType.CRASH
    
    def _calculate_severity(self, service_id: str, failure_type: FailureType) -> int:
        """Calculate failure severity (1-5)"""
        severity = 3  # Default medium
        
        # Critical services have higher severity
        if service_id in self.dependencies:
            if self.dependencies[service_id].critical:
                severity += 2
        
        # Certain failure types are more severe
        if failure_type in [FailureType.CRASH, FailureType.NETWORK_PARTITION]:
            severity += 1
        
        return min(5, severity)
    
    async def _initiate_recovery(self, failure: FailureEvent):
        """Initiate recovery for failure"""
        # Determine recovery strategy
        strategy = self.recovery_strategies.get(
            failure.failure_type,
            RecoveryStrategy.RESTART
        )
        
        failure.recovery_strategy = strategy
        failure.recovery_attempted = True
        
        logger.info(f"Initiating recovery for {failure.service_id}: {strategy.value}")
        
        # Execute recovery
        success = await self._execute_recovery(failure.service_id, strategy)
        
        if success:
            failure.recovered = True
            failure.recovery_time = time.time() - failure.timestamp
            self.metrics['recovered_failures'] += 1
            
            # Update MTTR
            self._update_mttr(failure.recovery_time)
            
            logger.info(f"Successfully recovered {failure.service_id} in {failure.recovery_time:.2f}s")
        else:
            # Escalate if recovery fails
            await self._escalate_failure(failure)
    
    async def _execute_recovery(self, service_id: str, strategy: RecoveryStrategy) -> bool:
        """Execute recovery strategy"""
        try:
            if strategy == RecoveryStrategy.RESTART:
                return await self._restart_service(service_id)
            
            elif strategy == RecoveryStrategy.FAILOVER:
                return await self._failover_service(service_id)
            
            elif strategy == RecoveryStrategy.SCALE_UP:
                return await self.auto_scaler.scale_service(service_id, "SCALE_UP")
            
            elif strategy == RecoveryStrategy.SCALE_DOWN:
                return await self.auto_scaler.scale_service(service_id, "SCALE_DOWN")
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                # Circuit breaker automatically handles this
                return True
            
            elif strategy == RecoveryStrategy.ROLLBACK:
                return await self._rollback_service(service_id)
            
            elif strategy == RecoveryStrategy.MIGRATE:
                return await self._migrate_service(service_id)
            
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
    
    async def _restart_service(self, service_id: str) -> bool:
        """Restart a service"""
        logger.info(f"Restarting service: {service_id}")
        
        # In production, would use orchestration API
        # For demo, simulate restart
        await asyncio.sleep(2)
        
        # Reset health metrics
        if service_id in self.services:
            self.services[service_id].state = ServiceState.RECOVERING
            self.services[service_id].consecutive_failures = 0
        
        return True
    
    async def _failover_service(self, service_id: str) -> bool:
        """Failover to backup service"""
        logger.info(f"Failing over service: {service_id}")
        
        # In production, would update load balancer
        # For demo, simulate failover
        await asyncio.sleep(1)
        
        return True
    
    async def _rollback_service(self, service_id: str) -> bool:
        """Rollback service to previous version"""
        logger.info(f"Rolling back service: {service_id}")
        
        # In production, would use deployment system
        # For demo, simulate rollback
        await asyncio.sleep(3)
        
        return True
    
    async def _migrate_service(self, service_id: str) -> bool:
        """Migrate service to different host"""
        logger.info(f"Migrating service: {service_id}")
        
        # In production, would use orchestration API
        # For demo, simulate migration
        await asyncio.sleep(5)
        
        return True
    
    async def _escalate_failure(self, failure: FailureEvent):
        """Escalate unrecovered failure"""
        logger.critical(
            f"ESCALATION REQUIRED: Service {failure.service_id} recovery failed. "
            f"Manual intervention needed."
        )
        
        # In production, would:
        # - Send alerts to on-call
        # - Create incident ticket
        # - Notify stakeholders
    
    async def _failure_detector(self):
        """Detect cascading failures"""
        while self.running:
            try:
                await asyncio.sleep(10)
                
                # Check for cascading failures
                unhealthy_services = [
                    s for s in self.services.values()
                    if s.state in [ServiceState.UNHEALTHY, ServiceState.FAILED]
                ]
                
                if len(unhealthy_services) > len(self.services) * 0.3:
                    logger.critical("Cascading failure detected! Multiple services affected.")
                    
                    # Initiate emergency response
                    await self._emergency_response()
                
            except Exception as e:
                logger.error(f"Failure detector error: {e}")
    
    async def _emergency_response(self):
        """Emergency response for cascading failures"""
        logger.critical("Initiating emergency response procedures")
        
        # 1. Circuit break all non-critical services
        for service_id in self.services:
            if service_id not in self.dependencies or not self.dependencies[service_id].critical:
                self.circuit_breakers[service_id].state = "OPEN"
        
        # 2. Scale up critical services
        for service_id in self.services:
            if service_id in self.dependencies and self.dependencies[service_id].critical:
                await self.auto_scaler.scale_service(service_id, "SCALE_UP")
        
        # 3. Alert all stakeholders
        # In production, would send emergency alerts
    
    async def _recovery_executor(self):
        """Execute recovery operations"""
        while self.running:
            try:
                await asyncio.sleep(5)
                
                # Check services in recovery
                recovering = [
                    s for s in self.services.values()
                    if s.state == ServiceState.RECOVERING
                ]
                
                for service in recovering:
                    # Check if recovery complete
                    if await self._check_health_endpoint(service.service_id):
                        service.state = ServiceState.HEALTHY
                        logger.info(f"Service {service.service_id} recovery complete")
                
            except Exception as e:
                logger.error(f"Recovery executor error: {e}")
    
    async def _predictive_analyzer(self):
        """Run predictive failure analysis"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for service_id in self.services:
                    will_fail, probability, reason = self.predictive_maintenance.predict_failure(service_id)
                    
                    if will_fail:
                        logger.warning(
                            f"Predicted failure for {service_id}: "
                            f"Probability={probability:.2f}, Reason={reason}"
                        )
                        
                        # Take preventive action
                        await self._preventive_action(service_id, reason)
                
            except Exception as e:
                logger.error(f"Predictive analyzer error: {e}")
    
    async def _preventive_action(self, service_id: str, reason: str):
        """Take preventive action to avoid predicted failure"""
        logger.info(f"Taking preventive action for {service_id}: {reason}")
        
        if "CPU exhaustion" in reason:
            await self.auto_scaler.scale_service(service_id, "SCALE_UP")
        
        elif "Memory exhaustion" in reason:
            # Schedule graceful restart
            asyncio.create_task(self._scheduled_restart(service_id, delay=300))
        
        elif "Increasing errors" in reason:
            # Enable circuit breaker proactively
            self.circuit_breakers[service_id].state = "HALF_OPEN"
    
    async def _scheduled_restart(self, service_id: str, delay: int):
        """Schedule service restart"""
        logger.info(f"Scheduling restart of {service_id} in {delay} seconds")
        await asyncio.sleep(delay)
        await self._restart_service(service_id)
    
    async def _metrics_collector(self):
        """Collect and report metrics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                # Calculate availability
                total_uptime = sum(
                    1 for s in self.services.values()
                    if s.state == ServiceState.HEALTHY
                )
                
                self.metrics['availability'] = (
                    (total_uptime / len(self.services)) * 100
                    if self.services else 100
                )
                
                logger.info(f"Infrastructure metrics: {self.metrics}")
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
    
    def _update_mttr(self, recovery_time: float):
        """Update mean time to recovery"""
        recovered = self.metrics['recovered_failures']
        if recovered > 0:
            current_mttr = self.metrics['mttr']
            self.metrics['mttr'] = (
                (current_mttr * (recovered - 1) + recovery_time) / recovered
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get infrastructure status"""
        return {
            'services': {
                sid: {
                    'state': s.state.value,
                    'health_score': s.health_score,
                    'cpu': s.cpu_usage,
                    'memory': s.memory_usage
                }
                for sid, s in self.services.items()
            },
            'metrics': self.metrics,
            'recent_failures': [
                {
                    'service': f.service_id,
                    'type': f.failure_type.value,
                    'recovered': f.recovered,
                    'recovery_time': f.recovery_time
                }
                for f in self.failure_history[-10:]
            ]
        }
    
    async def stop(self):
        """Stop orchestrator"""
        self.running = False
        logger.info("Self-healing orchestrator stopped")


async def main():
    """Test self-healing infrastructure"""
    orchestrator = SelfHealingOrchestrator()
    
    # Register services
    orchestrator.register_service("database", {
        'critical': True,
        'dependencies': [],
        'scaling': {
            'min_instances': 1,
            'max_instances': 5,
            'scale_up_cpu': 70,
            'scale_down_cpu': 30
        }
    })
    
    orchestrator.register_service("api", {
        'critical': True,
        'dependencies': ['database'],
        'scaling': {
            'min_instances': 2,
            'max_instances': 10,
            'scale_up_cpu': 60,
            'scale_up_requests': 1000,
            'scale_down_cpu': 20
        }
    })
    
    orchestrator.register_service("cache", {
        'critical': False,
        'dependencies': [],
        'scaling': {
            'min_instances': 1,
            'max_instances': 3,
            'scale_up_memory': 70,
            'scale_down_memory': 30
        }
    })
    
    # Start orchestrator
    await orchestrator.start()
    
    # Simulate running for a while
    for i in range(5):
        await asyncio.sleep(10)
        
        status = orchestrator.get_status()
        print(f"\n=== Infrastructure Status (T={i*10}s) ===")
        print(f"Services:")
        for service, info in status['services'].items():
            print(f"  {service}: {info['state']} (health={info['health_score']:.1f})")
        print(f"Metrics: {status['metrics']}")
        
        # Simulate some failures
        if i == 2:
            # Simulate API failure
            orchestrator.services['api'].consecutive_failures = 5
            await orchestrator._handle_service_failure('api')
    
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())