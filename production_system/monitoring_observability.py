#!/usr/bin/env python3
"""
Production Monitoring and Observability System
Implements comprehensive metrics, tracing, and alerting
"""

import os
import time
import json
import threading
import queue
import sqlite3
import logging
import psutil
import socket
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
import statistics

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: float

@dataclass
class Trace:
    """Distributed trace structure"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float]
    status: str
    tags: Dict[str, Any]
    logs: List[Dict]

@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    name: str
    severity: AlertSeverity
    condition: str
    message: str
    timestamp: float
    resolved: bool = False

class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logging.getLogger('MetricsCollector')
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter metric"""
        with self.lock:
            key = self._metric_key(name, labels)
            self.counters[key] += value
            self._record_metric(name, MetricType.COUNTER, self.counters[key], labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric value"""
        with self.lock:
            key = self._metric_key(name, labels)
            self.gauges[key] = value
            self._record_metric(name, MetricType.GAUGE, value, labels)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe value for histogram"""
        with self.lock:
            key = self._metric_key(name, labels)
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
            self._record_metric(name, MetricType.HISTOGRAM, value, labels)
    
    def _metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate unique key for metric"""
        if labels:
            label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name},{label_str}"
        return name
    
    def _record_metric(self, name: str, type: MetricType, value: float, labels: Dict[str, str] = None):
        """Record metric data point"""
        metric = Metric(
            name=name,
            type=type,
            value=value,
            labels=labels or {},
            timestamp=time.time()
        )
        self.metrics[name].append(metric)
    
    def get_metrics(self, name: str = None) -> List[Metric]:
        """Get collected metrics"""
        with self.lock:
            if name:
                return list(self.metrics.get(name, []))
            
            all_metrics = []
            for metric_list in self.metrics.values():
                all_metrics.extend(metric_list)
            return all_metrics
    
    def get_percentiles(self, name: str, percentiles: List[float] = None) -> Dict[float, float]:
        """Calculate percentiles for histogram metrics"""
        percentiles = percentiles or [0.5, 0.9, 0.95, 0.99]
        
        with self.lock:
            values = []
            for key, hist_values in self.histograms.items():
                if name in key:
                    values.extend(hist_values)
            
            if not values:
                return {}
            
            values.sort()
            result = {}
            for p in percentiles:
                index = int(len(values) * p)
                result[p] = values[min(index, len(values) - 1)]
            
            return result

class DistributedTracer:
    """Distributed tracing system"""
    
    def __init__(self):
        self.traces = {}
        self.active_spans = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger('DistributedTracer')
    
    def start_trace(self, operation: str, parent_span_id: Optional[str] = None) -> Trace:
        """Start new trace or span"""
        trace_id = hashlib.md5(f"{time.time()}:{operation}".encode()).hexdigest()[:16]
        span_id = hashlib.md5(f"{trace_id}:{time.time()}".encode()).hexdigest()[:16]
        
        trace = Trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            end_time=None,
            status="in_progress",
            tags={},
            logs=[]
        )
        
        with self.lock:
            self.active_spans[span_id] = trace
            if trace_id not in self.traces:
                self.traces[trace_id] = []
            self.traces[trace_id].append(trace)
        
        return trace
    
    def end_trace(self, span_id: str, status: str = "success"):
        """End active span"""
        with self.lock:
            if span_id in self.active_spans:
                trace = self.active_spans[span_id]
                trace.end_time = time.time()
                trace.status = status
                del self.active_spans[span_id]
    
    def add_tag(self, span_id: str, key: str, value: Any):
        """Add tag to active span"""
        with self.lock:
            if span_id in self.active_spans:
                self.active_spans[span_id].tags[key] = value
    
    def add_log(self, span_id: str, message: str, level: str = "info"):
        """Add log to active span"""
        with self.lock:
            if span_id in self.active_spans:
                log_entry = {
                    "timestamp": time.time(),
                    "level": level,
                    "message": message
                }
                self.active_spans[span_id].logs.append(log_entry)
    
    def get_trace(self, trace_id: str) -> List[Trace]:
        """Get complete trace by ID"""
        with self.lock:
            return self.traces.get(trace_id, [])

class SystemMonitor:
    """Monitor system resources and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.running = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start(self, interval: float = 60.0):
        """Start monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge("system_cpu_usage_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.set_gauge("system_memory_usage_percent", memory.percent)
        self.metrics.set_gauge("system_memory_available_bytes", memory.available)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.set_gauge("system_disk_usage_percent", disk.percent)
        self.metrics.set_gauge("system_disk_free_bytes", disk.free)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.metrics.increment_counter("system_network_bytes_sent", net_io.bytes_sent)
        self.metrics.increment_counter("system_network_bytes_recv", net_io.bytes_recv)
        
        # Process metrics
        self.metrics.set_gauge("process_cpu_percent", self.process.cpu_percent())
        self.metrics.set_gauge("process_memory_rss_bytes", self.process.memory_info().rss)
        self.metrics.set_gauge("process_num_threads", self.process.num_threads())
        
        # Open file descriptors
        try:
            self.metrics.set_gauge("process_open_fds", self.process.num_fds())
        except AttributeError:
            # Windows doesn't have num_fds
            pass

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts = {}
        self.alert_rules = []
        self.alert_history = deque(maxlen=1000)
        self.handlers = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger('AlertManager')
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity,
        message: str,
        cooldown: int = 300
    ):
        """Add alert rule"""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'cooldown': cooldown,
            'last_triggered': 0
        }
        self.alert_rules.append(rule)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.handlers.append(handler)
    
    def check_rules(self):
        """Check alert rules and trigger alerts"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            # Check cooldown
            if current_time - rule['last_triggered'] < rule['cooldown']:
                continue
            
            # Check condition
            try:
                if rule['condition']():
                    self._trigger_alert(rule)
                    rule['last_triggered'] = current_time
            except Exception as e:
                self.logger.error(f"Error checking rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict):
        """Trigger alert for rule"""
        alert_id = hashlib.md5(f"{rule['name']}:{time.time()}".encode()).hexdigest()[:16]
        
        alert = Alert(
            alert_id=alert_id,
            name=rule['name'],
            severity=rule['severity'],
            condition=str(rule['condition']),
            message=rule['message'],
            timestamp=time.time()
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        with self.lock:
            return [a for a in self.alerts.values() if not a.resolved]

class LogAggregator:
    """Centralized log aggregation"""
    
    def __init__(self, buffer_size: int = 10000):
        self.logs = deque(maxlen=buffer_size)
        self.handlers = []
        self.lock = threading.Lock()
        self.log_levels = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
    
    def add_log(
        self,
        message: str,
        level: str = 'INFO',
        source: str = None,
        metadata: Dict = None
    ):
        """Add log entry"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'source': source or 'system',
            'message': message,
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.logs.append(log_entry)
        
        # Process handlers
        for handler in self.handlers:
            try:
                handler(log_entry)
            except Exception as e:
                print(f"Log handler error: {e}")
    
    def search_logs(
        self,
        query: str = None,
        level: str = None,
        source: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search logs with filters"""
        with self.lock:
            results = []
            
            for log in reversed(self.logs):
                # Check filters
                if query and query.lower() not in log['message'].lower():
                    continue
                
                if level and self.log_levels.get(log['level'], 0) < self.log_levels.get(level, 0):
                    continue
                
                if source and log['source'] != source:
                    continue
                
                if start_time:
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if log_time < start_time:
                        continue
                
                if end_time:
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if log_time > end_time:
                        continue
                
                results.append(log)
                
                if len(results) >= limit:
                    break
            
            return results

class PrometheusExporter:
    """Export metrics in Prometheus format"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def export(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        
        # Export counters
        for key, value in self.metrics.counters.items():
            name, labels = self._parse_key(key)
            output.append(f"# TYPE {name} counter")
            label_str = self._format_labels(labels)
            output.append(f"{name}{label_str} {value}")
        
        # Export gauges
        for key, value in self.metrics.gauges.items():
            name, labels = self._parse_key(key)
            output.append(f"# TYPE {name} gauge")
            label_str = self._format_labels(labels)
            output.append(f"{name}{label_str} {value}")
        
        # Export histograms
        for key, values in self.metrics.histograms.items():
            if values:
                name, labels = self._parse_key(key)
                output.append(f"# TYPE {name} histogram")
                
                # Calculate buckets
                buckets = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                for bucket in buckets:
                    count = sum(1 for v in values if v <= bucket)
                    label_str = self._format_labels({**labels, 'le': str(bucket)})
                    output.append(f"{name}_bucket{label_str} {count}")
                
                label_str = self._format_labels({**labels, 'le': '+Inf'})
                output.append(f"{name}_bucket{label_str} {len(values)}")
                
                # Sum and count
                label_str = self._format_labels(labels)
                output.append(f"{name}_sum{label_str} {sum(values)}")
                output.append(f"{name}_count{label_str} {len(values)}")
        
        return '\n'.join(output)
    
    def _parse_key(self, key: str) -> tuple:
        """Parse metric key into name and labels"""
        parts = key.split(',')
        name = parts[0]
        labels = {}
        
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                labels[k] = v
        
        return name, labels
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return f"{{{','.join(label_pairs)}}}"

class ObservabilityPlatform:
    """Complete observability platform"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()
        self.monitor = SystemMonitor(self.metrics)
        self.alerts = AlertManager(self.metrics)
        self.logs = LogAggregator()
        self.exporter = PrometheusExporter(self.metrics)
        
        # Start monitoring
        self.monitor.start()
        
        # Add default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        
        # High CPU usage alert
        self.alerts.add_rule(
            name="high_cpu_usage",
            condition=lambda: self.metrics.gauges.get("system_cpu_usage_percent", 0) > 80,
            severity=AlertSeverity.WARNING,
            message="CPU usage above 80%"
        )
        
        # High memory usage alert
        self.alerts.add_rule(
            name="high_memory_usage",
            condition=lambda: self.metrics.gauges.get("system_memory_usage_percent", 0) > 90,
            severity=AlertSeverity.ERROR,
            message="Memory usage above 90%"
        )
        
        # Low disk space alert
        self.alerts.add_rule(
            name="low_disk_space",
            condition=lambda: self.metrics.gauges.get("system_disk_free_bytes", float('inf')) < 1e9,
            severity=AlertSeverity.CRITICAL,
            message="Less than 1GB disk space remaining"
        )
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        return {
            'metrics': {
                'cpu': self.metrics.gauges.get('system_cpu_usage_percent', 0),
                'memory': self.metrics.gauges.get('system_memory_usage_percent', 0),
                'disk': self.metrics.gauges.get('system_disk_usage_percent', 0)
            },
            'active_alerts': len(self.alerts.get_active_alerts()),
            'recent_logs': self.logs.search_logs(limit=10),
            'traces': len(self.tracer.traces)
        }

# Example usage and decorators
def traced(operation_name: str):
    """Decorator for tracing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = DistributedTracer()
            trace = tracer.start_trace(operation_name)
            
            try:
                result = func(*args, **kwargs)
                tracer.end_trace(trace.span_id, "success")
                return result
            except Exception as e:
                tracer.add_log(trace.span_id, str(e), "error")
                tracer.end_trace(trace.span_id, "error")
                raise
        
        return wrapper
    return decorator

def monitored(metric_name: str):
    """Decorator for monitoring function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = MetricsCollector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                metrics.observe_histogram(f"{metric_name}_duration_seconds", duration)
                metrics.increment_counter(f"{metric_name}_total")
                
                return result
            except Exception as e:
                metrics.increment_counter(f"{metric_name}_errors_total")
                raise
        
        return wrapper
    return decorator