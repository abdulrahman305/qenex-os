#!/usr/bin/env python3
"""
Production Microservices Architecture with Service Mesh
Implements scalable, resilient microservices with proper service discovery
"""

import os
import json
import time
import uuid
import asyncio
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
from queue import Queue, Empty
import socket
from concurrent.futures import ThreadPoolExecutor

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"

@dataclass
class ServiceInfo:
    """Service registration information"""
    service_id: str
    name: str
    version: str
    host: str
    port: int
    protocol: str
    status: ServiceStatus
    metadata: Dict[str, Any]
    registered_at: datetime
    last_heartbeat: datetime
    health_check_url: str
    dependencies: List[str]

class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services = {}
        self.lock = threading.RLock()
        self.watchers = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure service registry logging"""
        logger = logging.getLogger('ServiceRegistry')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('service_registry.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def register_service(self, service: ServiceInfo) -> bool:
        """Register a new service"""
        with self.lock:
            key = f"{service.name}:{service.service_id}"
            
            if key in self.services:
                self.logger.warning(f"Service {key} already registered")
                return False
            
            self.services[key] = service
            self.logger.info(f"Registered service {key}")
            
            # Notify watchers
            self._notify_watchers(service.name, 'registered', service)
            
            return True
    
    def deregister_service(self, service_id: str, name: str) -> bool:
        """Remove service from registry"""
        with self.lock:
            key = f"{name}:{service_id}"
            
            if key not in self.services:
                return False
            
            service = self.services[key]
            del self.services[key]
            
            self.logger.info(f"Deregistered service {key}")
            self._notify_watchers(name, 'deregistered', service)
            
            return True
    
    def update_heartbeat(self, service_id: str, name: str) -> bool:
        """Update service heartbeat"""
        with self.lock:
            key = f"{name}:{service_id}"
            
            if key not in self.services:
                return False
            
            self.services[key].last_heartbeat = datetime.now()
            return True
    
    def discover_service(self, name: str, version: str = None) -> List[ServiceInfo]:
        """Discover available service instances"""
        with self.lock:
            instances = []
            
            for key, service in self.services.items():
                if service.name == name and service.status == ServiceStatus.HEALTHY:
                    if version is None or service.version == version:
                        instances.append(service)
            
            return instances
    
    def watch_service(self, name: str, callback: Callable):
        """Watch for service changes"""
        with self.lock:
            if name not in self.watchers:
                self.watchers[name] = []
            
            self.watchers[name].append(callback)
    
    def _notify_watchers(self, name: str, event: str, service: ServiceInfo):
        """Notify service watchers of changes"""
        if name in self.watchers:
            for callback in self.watchers[name]:
                try:
                    callback(event, service)
                except Exception as e:
                    self.logger.error(f"Watcher callback error: {e}")

class LoadBalancer:
    """Client-side load balancing"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index = {}
        self.request_counts = {}
        
    def select_instance(self, instances: List[ServiceInfo], key: str = None) -> Optional[ServiceInfo]:
        """Select service instance based on strategy"""
        if not instances:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(instances)
        elif self.strategy == "least_connections":
            return self._least_connections(instances)
        elif self.strategy == "random":
            import random
            return random.choice(instances)
        elif self.strategy == "consistent_hash":
            return self._consistent_hash(instances, key)
        else:
            return instances[0]
    
    def _round_robin(self, instances: List[ServiceInfo]) -> ServiceInfo:
        """Round-robin selection"""
        service_name = instances[0].name
        
        if service_name not in self.current_index:
            self.current_index[service_name] = 0
        
        index = self.current_index[service_name]
        instance = instances[index % len(instances)]
        
        self.current_index[service_name] = (index + 1) % len(instances)
        
        return instance
    
    def _least_connections(self, instances: List[ServiceInfo]) -> ServiceInfo:
        """Select instance with least active requests"""
        min_count = float('inf')
        selected = instances[0]
        
        for instance in instances:
            key = f"{instance.name}:{instance.service_id}"
            count = self.request_counts.get(key, 0)
            
            if count < min_count:
                min_count = count
                selected = instance
        
        return selected
    
    def _consistent_hash(self, instances: List[ServiceInfo], key: str) -> ServiceInfo:
        """Consistent hashing for sticky sessions"""
        if not key:
            key = str(time.time())
        
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_value % len(instances)
        
        return instances[index]

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class MessageBroker:
    """Asynchronous message broker for inter-service communication"""
    
    def __init__(self):
        self.topics = {}
        self.subscribers = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_topic(self, topic: str, max_size: int = 1000):
        """Create message topic"""
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = Queue(maxsize=max_size)
                self.subscribers[topic] = []
    
    def publish(self, topic: str, message: Dict) -> bool:
        """Publish message to topic"""
        with self.lock:
            if topic not in self.topics:
                self.create_topic(topic)
            
            try:
                self.topics[topic].put_nowait({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now().isoformat(),
                    'payload': message
                })
                
                # Notify subscribers asynchronously
                for subscriber in self.subscribers[topic]:
                    self.executor.submit(subscriber, message)
                
                return True
            except:
                return False
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        with self.lock:
            if topic not in self.subscribers:
                self.create_topic(topic)
            
            self.subscribers[topic].append(callback)
    
    def consume(self, topic: str, timeout: float = 1.0) -> Optional[Dict]:
        """Consume message from topic"""
        if topic not in self.topics:
            return None
        
        try:
            return self.topics[topic].get(timeout=timeout)
        except Empty:
            return None

class APIGateway:
    """API Gateway for routing and authentication"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.load_balancer = LoadBalancer()
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.routes = {}
        
    def register_route(
        self,
        path: str,
        service_name: str,
        methods: List[str] = None,
        auth_required: bool = True
    ):
        """Register API route"""
        self.routes[path] = {
            'service': service_name,
            'methods': methods or ['GET', 'POST'],
            'auth_required': auth_required
        }
    
    def route_request(
        self,
        path: str,
        method: str,
        headers: Dict,
        body: Any = None
    ) -> Dict:
        """Route request to appropriate service"""
        # Find matching route
        route = self._match_route(path)
        if not route:
            return {'status': 404, 'error': 'Route not found'}
        
        # Check method
        if method not in route['methods']:
            return {'status': 405, 'error': 'Method not allowed'}
        
        # Check authentication
        if route['auth_required'] and not self._check_auth(headers):
            return {'status': 401, 'error': 'Authentication required'}
        
        # Rate limiting
        if not self._check_rate_limit(headers.get('X-Client-ID')):
            return {'status': 429, 'error': 'Too many requests'}
        
        # Discover service
        instances = self.registry.discover_service(route['service'])
        if not instances:
            return {'status': 503, 'error': 'Service unavailable'}
        
        # Select instance
        instance = self.load_balancer.select_instance(instances)
        
        # Get or create circuit breaker
        breaker = self._get_circuit_breaker(instance.service_id)
        
        try:
            # Forward request (simulated)
            response = breaker.call(
                self._forward_request,
                instance,
                path,
                method,
                headers,
                body
            )
            return response
        except Exception as e:
            return {'status': 500, 'error': str(e)}
    
    def _match_route(self, path: str) -> Optional[Dict]:
        """Match request path to route"""
        # Exact match
        if path in self.routes:
            return self.routes[path]
        
        # Pattern matching (simplified)
        for route_path, route_config in self.routes.items():
            if path.startswith(route_path.rstrip('/')):
                return route_config
        
        return None
    
    def _check_auth(self, headers: Dict) -> bool:
        """Check authentication headers"""
        # Simplified auth check
        return 'Authorization' in headers
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit for client"""
        if not client_id:
            client_id = 'anonymous'
        
        current_time = time.time()
        
        if client_id not in self.rate_limiters:
            self.rate_limiters[client_id] = []
        
        # Clean old requests (1 minute window)
        self.rate_limiters[client_id] = [
            t for t in self.rate_limiters[client_id]
            if current_time - t < 60
        ]
        
        # Check limit (100 requests per minute)
        if len(self.rate_limiters[client_id]) >= 100:
            return False
        
        self.rate_limiters[client_id].append(current_time)
        return True
    
    def _get_circuit_breaker(self, service_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_id not in self.circuit_breakers:
            self.circuit_breakers[service_id] = CircuitBreaker()
        
        return self.circuit_breakers[service_id]
    
    def _forward_request(
        self,
        instance: ServiceInfo,
        path: str,
        method: str,
        headers: Dict,
        body: Any
    ) -> Dict:
        """Forward request to service instance"""
        # In production, use actual HTTP client
        # Simulated response
        return {
            'status': 200,
            'data': {
                'service': instance.name,
                'instance': instance.service_id,
                'path': path,
                'method': method
            }
        }

class Microservice(ABC):
    """Base class for microservices"""
    
    def __init__(
        self,
        name: str,
        version: str,
        port: int,
        registry: ServiceRegistry,
        broker: MessageBroker
    ):
        self.name = name
        self.version = version
        self.port = port
        self.registry = registry
        self.broker = broker
        self.service_id = str(uuid.uuid4())
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Configure service logging"""
        logger = logging.getLogger(f'Service_{self.name}')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f'service_{self.name}.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start(self):
        """Start microservice"""
        try:
            # Register with service registry
            service_info = ServiceInfo(
                service_id=self.service_id,
                name=self.name,
                version=self.version,
                host=socket.gethostname(),
                port=self.port,
                protocol='http',
                status=ServiceStatus.STARTING,
                metadata={'started_at': datetime.now().isoformat()},
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                health_check_url=f"http://localhost:{self.port}/health",
                dependencies=self.get_dependencies()
            )
            
            self.registry.register_service(service_info)
            
            # Start heartbeat
            self.running = True
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            
            # Initialize service
            self.initialize()
            
            # Update status
            service_info.status = ServiceStatus.HEALTHY
            
            self.logger.info(f"Service {self.name} started on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            raise
    
    def stop(self):
        """Stop microservice"""
        self.running = False
        self.shutdown()
        self.registry.deregister_service(self.service_id, self.name)
        self.logger.info(f"Service {self.name} stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            self.registry.update_heartbeat(self.service_id, self.name)
            time.sleep(30)  # 30 second heartbeat
    
    @abstractmethod
    def initialize(self):
        """Initialize service resources"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup service resources"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get service dependencies"""
        pass

class MicroservicesOrchestrator:
    """Orchestrate microservices deployment and management"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.broker = MessageBroker()
        self.gateway = APIGateway(self.registry)
        self.services = {}
        
    def deploy_service(self, service_class: type, **kwargs):
        """Deploy a microservice"""
        service = service_class(
            registry=self.registry,
            broker=self.broker,
            **kwargs
        )
        
        service.start()
        self.services[service.name] = service
        
        return service.service_id
    
    def scale_service(self, name: str, instances: int):
        """Scale service to specified instances"""
        current = len([s for s in self.services.values() if s.name == name])
        
        if instances > current:
            # Scale up
            for i in range(instances - current):
                # Deploy additional instance
                pass
        elif instances < current:
            # Scale down
            pass
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        for name, service in self.services.items():
            instances = self.registry.discover_service(name)
            health['services'][name] = {
                'instances': len(instances),
                'healthy': sum(1 for i in instances if i.status == ServiceStatus.HEALTHY),
                'unhealthy': sum(1 for i in instances if i.status == ServiceStatus.UNHEALTHY)
            }
        
        return health

# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = MicroservicesOrchestrator()
    
    # Configure API Gateway routes
    orchestrator.gateway.register_route('/api/users', 'user-service')
    orchestrator.gateway.register_route('/api/products', 'product-service')
    orchestrator.gateway.register_route('/api/orders', 'order-service')
    
    # Simulate request routing
    response = orchestrator.gateway.route_request(
        '/api/users/123',
        'GET',
        {'Authorization': 'Bearer token'}
    )
    
    print(f"Response: {response}")
    
    # Health check
    health = orchestrator.health_check()
    print(f"System health: {json.dumps(health, indent=2)}")