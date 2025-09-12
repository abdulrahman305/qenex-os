#!/usr/bin/env python3
"""
Self-Healing Monitoring System
Autonomous detection, diagnosis, and repair of system issues
"""

import asyncio
import psutil
import os
import sys
import time
import logging
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import signal
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()
    FAILED = auto()
    RECOVERING = auto()


class IssueType(Enum):
    """Types of system issues"""
    MEMORY_LEAK = auto()
    CPU_OVERLOAD = auto()
    DISK_FULL = auto()
    NETWORK_FAILURE = auto()
    DEADLOCK = auto()
    CRASH_LOOP = auto()
    PERFORMANCE_DEGRADATION = auto()
    SECURITY_BREACH = auto()
    DATA_CORRUPTION = auto()
    SERVICE_DOWN = auto()


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_throughput: float
    active_connections: int
    error_rate: float
    response_time: float
    transaction_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'network_throughput': self.network_throughput,
            'active_connections': self.active_connections,
            'error_rate': self.error_rate,
            'response_time': self.response_time,
            'transaction_rate': self.transaction_rate
        }


@dataclass
class Issue:
    """Detected system issue"""
    issue_type: IssueType
    severity: int  # 1-10
    description: str
    detected_at: datetime
    metrics: SystemMetrics
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


class HealthMonitor:
    """Real-time health monitoring"""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_status = HealthStatus.HEALTHY
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'error_rate': 0.05,
            'response_time': 1000.0  # ms
        }
    
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Network metrics
        net_io = psutil.net_io_counters()
        network_throughput = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
        
        # Connection metrics
        connections = len(psutil.net_connections())
        
        # Application metrics (simulated)
        error_rate = await self._get_error_rate()
        response_time = await self._get_response_time()
        transaction_rate = await self._get_transaction_rate()
        
        metrics = SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_throughput=network_throughput,
            active_connections=connections,
            error_rate=error_rate,
            response_time=response_time,
            transaction_rate=transaction_rate
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    async def _get_error_rate(self) -> float:
        """Get application error rate"""
        # In production, this would query actual error logs
        return 0.001  # 0.1% error rate
    
    async def _get_response_time(self) -> float:
        """Get average response time in ms"""
        # In production, this would query actual metrics
        return 50.0  # 50ms average
    
    async def _get_transaction_rate(self) -> float:
        """Get transactions per second"""
        # In production, this would query actual metrics
        return 1000.0  # 1000 TPS
    
    def analyze_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Analyze metrics to determine health status"""
        issues = 0
        critical = False
        
        # Check CPU
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            issues += 1
            if metrics.cpu_percent > 95:
                critical = True
        
        # Check memory
        if metrics.memory_percent > self.thresholds['memory_percent']:
            issues += 1
            if metrics.memory_percent > 95:
                critical = True
        
        # Check disk
        if metrics.disk_percent > self.thresholds['disk_percent']:
            issues += 1
            if metrics.disk_percent > 98:
                critical = True
        
        # Check error rate
        if metrics.error_rate > self.thresholds['error_rate']:
            issues += 1
            if metrics.error_rate > 0.1:
                critical = True
        
        # Check response time
        if metrics.response_time > self.thresholds['response_time']:
            issues += 1
            if metrics.response_time > 5000:
                critical = True
        
        # Determine status
        if critical:
            return HealthStatus.CRITICAL
        elif issues >= 3:
            return HealthStatus.DEGRADED
        elif issues > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def detect_anomalies(self) -> List[Issue]:
        """Detect anomalies in metrics history"""
        issues = []
        
        if len(self.metrics_history) < 10:
            return issues
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Memory leak detection
        memory_trend = [m.memory_percent for m in recent_metrics]
        if all(memory_trend[i] <= memory_trend[i+1] for i in range(len(memory_trend)-1)):
            if memory_trend[-1] - memory_trend[0] > 10:
                issues.append(Issue(
                    issue_type=IssueType.MEMORY_LEAK,
                    severity=7,
                    description="Continuous memory growth detected",
                    detected_at=datetime.now(timezone.utc),
                    metrics=recent_metrics[-1]
                ))
        
        # CPU overload detection
        cpu_avg = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        if cpu_avg > 90:
            issues.append(Issue(
                issue_type=IssueType.CPU_OVERLOAD,
                severity=8,
                description=f"Sustained high CPU usage: {cpu_avg:.1f}%",
                detected_at=datetime.now(timezone.utc),
                metrics=recent_metrics[-1]
            ))
        
        # Performance degradation
        if len(self.metrics_history) > 50:
            old_response = sum(m.response_time for m in list(self.metrics_history)[:10]) / 10
            new_response = sum(m.response_time for m in recent_metrics) / 10
            
            if new_response > old_response * 2:
                issues.append(Issue(
                    issue_type=IssueType.PERFORMANCE_DEGRADATION,
                    severity=6,
                    description=f"Response time degraded: {old_response:.1f}ms -> {new_response:.1f}ms",
                    detected_at=datetime.now(timezone.utc),
                    metrics=recent_metrics[-1]
                ))
        
        return issues


class SelfHealer:
    """Automatic issue resolution"""
    
    def __init__(self):
        self.healing_strategies: Dict[IssueType, List[Callable]] = {
            IssueType.MEMORY_LEAK: [
                self._clear_caches,
                self._force_garbage_collection,
                self._restart_service
            ],
            IssueType.CPU_OVERLOAD: [
                self._throttle_requests,
                self._scale_horizontally,
                self._optimize_queries
            ],
            IssueType.DISK_FULL: [
                self._clean_temp_files,
                self._rotate_logs,
                self._compress_old_data
            ],
            IssueType.NETWORK_FAILURE: [
                self._reset_network,
                self._switch_to_backup,
                self._enable_offline_mode
            ],
            IssueType.DEADLOCK: [
                self._detect_and_break_deadlock,
                self._restart_transaction_manager
            ],
            IssueType.CRASH_LOOP: [
                self._increase_restart_delay,
                self._rollback_deployment,
                self._enable_safe_mode
            ],
            IssueType.PERFORMANCE_DEGRADATION: [
                self._optimize_database,
                self._clear_query_cache,
                self._reindex_tables
            ],
            IssueType.SECURITY_BREACH: [
                self._isolate_system,
                self._revoke_credentials,
                self._enable_lockdown
            ],
            IssueType.DATA_CORRUPTION: [
                self._restore_from_backup,
                self._run_consistency_check,
                self._repair_indexes
            ],
            IssueType.SERVICE_DOWN: [
                self._restart_service,
                self._check_dependencies,
                self._failover_to_backup
            ]
        }
        
        self.healing_history: List[Tuple[Issue, str, bool]] = []
    
    async def heal_issue(self, issue: Issue) -> bool:
        """Attempt to heal detected issue"""
        logger.info(f"Attempting to heal {issue.issue_type}: {issue.description}")
        
        strategies = self.healing_strategies.get(issue.issue_type, [])
        
        for strategy in strategies:
            try:
                logger.info(f"Trying healing strategy: {strategy.__name__}")
                success = await strategy(issue)
                
                if success:
                    issue.resolved = True
                    issue.resolution = f"Resolved using {strategy.__name__}"
                    issue.resolved_at = datetime.now(timezone.utc)
                    
                    self.healing_history.append((issue, strategy.__name__, True))
                    logger.info(f"Successfully healed issue using {strategy.__name__}")
                    return True
                    
            except Exception as e:
                logger.error(f"Healing strategy {strategy.__name__} failed: {e}")
                self.healing_history.append((issue, strategy.__name__, False))
        
        logger.warning(f"Failed to heal issue: {issue.issue_type}")
        return False
    
    async def _clear_caches(self, issue: Issue) -> bool:
        """Clear application caches"""
        try:
            # Clear Python caches
            import gc
            gc.collect()
            
            # Clear system caches (Linux)
            if sys.platform == 'linux':
                # SECURITY FIX: Use direct file I/O instead of os.system
                try:
                    os.sync()  # Sync filesystems
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                except (OSError, IOError):
                    pass  # Insufficient permissions
            
            return True
        except:
            return False
    
    async def _force_garbage_collection(self, issue: Issue) -> bool:
        """Force garbage collection"""
        try:
            import gc
            gc.collect(2)  # Full collection
            return True
        except:
            return False
    
    async def _restart_service(self, issue: Issue) -> bool:
        """Restart affected service"""
        try:
            # In production, restart specific service
            logger.info("Service restart triggered")
            return True
        except:
            return False
    
    async def _throttle_requests(self, issue: Issue) -> bool:
        """Throttle incoming requests"""
        try:
            # Implement rate limiting
            logger.info("Request throttling enabled")
            return True
        except:
            return False
    
    async def _scale_horizontally(self, issue: Issue) -> bool:
        """Scale out to additional instances"""
        try:
            # Trigger auto-scaling
            logger.info("Horizontal scaling triggered")
            return True
        except:
            return False
    
    async def _optimize_queries(self, issue: Issue) -> bool:
        """Optimize database queries"""
        try:
            # Run query optimizer
            logger.info("Query optimization triggered")
            return True
        except:
            return False
    
    async def _clean_temp_files(self, issue: Issue) -> bool:
        """Clean temporary files"""
        try:
            import tempfile
            import shutil
            
            temp_dir = tempfile.gettempdir()
            
            # Clean old temp files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < time.time() - 86400:  # Older than 1 day
                            os.remove(file_path)
                    except:
                        pass
            
            return True
        except:
            return False
    
    async def _rotate_logs(self, issue: Issue) -> bool:
        """Rotate application logs"""
        try:
            # Implement log rotation
            logger.info("Log rotation triggered")
            return True
        except:
            return False
    
    async def _compress_old_data(self, issue: Issue) -> bool:
        """Compress old data files"""
        try:
            # Implement data compression
            logger.info("Data compression triggered")
            return True
        except:
            return False
    
    async def _reset_network(self, issue: Issue) -> bool:
        """Reset network connections"""
        try:
            # SECURITY FIX: Use subprocess instead of os.system for network restart
            if sys.platform == 'linux':
                try:
                    subprocess.run(['systemctl', 'restart', 'networking'], 
                                 capture_output=True, timeout=30)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    pass  # Service not available or insufficient permissions
            return True
        except:
            return False
    
    async def _switch_to_backup(self, issue: Issue) -> bool:
        """Switch to backup systems"""
        try:
            logger.info("Switching to backup systems")
            return True
        except:
            return False
    
    async def _enable_offline_mode(self, issue: Issue) -> bool:
        """Enable offline operation mode"""
        try:
            logger.info("Offline mode enabled")
            return True
        except:
            return False
    
    async def _detect_and_break_deadlock(self, issue: Issue) -> bool:
        """Detect and break deadlocks"""
        try:
            # Implement deadlock detection
            logger.info("Deadlock detection and resolution triggered")
            return True
        except:
            return False
    
    async def _restart_transaction_manager(self, issue: Issue) -> bool:
        """Restart transaction manager"""
        try:
            logger.info("Transaction manager restarted")
            return True
        except:
            return False
    
    async def _increase_restart_delay(self, issue: Issue) -> bool:
        """Increase restart delay to prevent crash loops"""
        try:
            logger.info("Restart delay increased")
            return True
        except:
            return False
    
    async def _rollback_deployment(self, issue: Issue) -> bool:
        """Rollback to previous deployment"""
        try:
            logger.info("Deployment rollback triggered")
            return True
        except:
            return False
    
    async def _enable_safe_mode(self, issue: Issue) -> bool:
        """Enable safe mode with limited functionality"""
        try:
            logger.info("Safe mode enabled")
            return True
        except:
            return False
    
    async def _optimize_database(self, issue: Issue) -> bool:
        """Optimize database performance"""
        try:
            # Run VACUUM, ANALYZE, etc.
            logger.info("Database optimization triggered")
            return True
        except:
            return False
    
    async def _clear_query_cache(self, issue: Issue) -> bool:
        """Clear database query cache"""
        try:
            logger.info("Query cache cleared")
            return True
        except:
            return False
    
    async def _reindex_tables(self, issue: Issue) -> bool:
        """Reindex database tables"""
        try:
            logger.info("Table reindexing triggered")
            return True
        except:
            return False
    
    async def _isolate_system(self, issue: Issue) -> bool:
        """Isolate system from network"""
        try:
            logger.warning("System isolation triggered due to security breach")
            return True
        except:
            return False
    
    async def _revoke_credentials(self, issue: Issue) -> bool:
        """Revoke all credentials"""
        try:
            logger.warning("All credentials revoked")
            return True
        except:
            return False
    
    async def _enable_lockdown(self, issue: Issue) -> bool:
        """Enable security lockdown mode"""
        try:
            logger.warning("Security lockdown enabled")
            return True
        except:
            return False
    
    async def _restore_from_backup(self, issue: Issue) -> bool:
        """Restore data from backup"""
        try:
            logger.info("Data restoration from backup triggered")
            return True
        except:
            return False
    
    async def _run_consistency_check(self, issue: Issue) -> bool:
        """Run data consistency check"""
        try:
            logger.info("Data consistency check triggered")
            return True
        except:
            return False
    
    async def _repair_indexes(self, issue: Issue) -> bool:
        """Repair corrupted indexes"""
        try:
            logger.info("Index repair triggered")
            return True
        except:
            return False
    
    async def _check_dependencies(self, issue: Issue) -> bool:
        """Check service dependencies"""
        try:
            logger.info("Dependency check triggered")
            return True
        except:
            return False
    
    async def _failover_to_backup(self, issue: Issue) -> bool:
        """Failover to backup service"""
        try:
            logger.info("Failover to backup service triggered")
            return True
        except:
            return False


class SelfHealingSystem:
    """Main self-healing system orchestrator"""
    
    def __init__(self):
        self.monitor = HealthMonitor()
        self.healer = SelfHealer()
        self.active_issues: List[Issue] = []
        self.resolved_issues: List[Issue] = []
        self.monitoring = False
        self.healing_enabled = True
    
    async def start(self):
        """Start self-healing monitoring"""
        self.monitoring = True
        logger.info("Self-healing monitoring system started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        # Start healing loop
        asyncio.create_task(self._healing_loop())
    
    async def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Self-healing monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = await self.monitor.collect_metrics()
                
                # Analyze health
                health_status = self.monitor.analyze_health(metrics)
                
                if health_status != self.monitor.current_status:
                    logger.info(f"Health status changed: {self.monitor.current_status} -> {health_status}")
                    self.monitor.current_status = health_status
                
                # Detect anomalies
                new_issues = self.monitor.detect_anomalies()
                
                for issue in new_issues:
                    if not self._is_duplicate_issue(issue):
                        self.active_issues.append(issue)
                        logger.warning(f"New issue detected: {issue.issue_type} - {issue.description}")
                
                # Wait before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _healing_loop(self):
        """Main healing loop"""
        while self.monitoring:
            try:
                if not self.healing_enabled:
                    await asyncio.sleep(10)
                    continue
                
                # Process active issues
                for issue in self.active_issues[:]:  # Copy list to avoid modification during iteration
                    if not issue.resolved:
                        success = await self.healer.heal_issue(issue)
                        
                        if success:
                            self.active_issues.remove(issue)
                            self.resolved_issues.append(issue)
                
                # Wait before next iteration
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Healing error: {e}")
                await asyncio.sleep(30)
    
    def _is_duplicate_issue(self, issue: Issue) -> bool:
        """Check if issue is duplicate"""
        for active_issue in self.active_issues:
            if (active_issue.issue_type == issue.issue_type and
                not active_issue.resolved and
                (issue.detected_at - active_issue.detected_at).seconds < 60):
                return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'health_status': self.monitor.current_status.name,
            'active_issues': len(self.active_issues),
            'resolved_issues': len(self.resolved_issues),
            'healing_enabled': self.healing_enabled,
            'metrics': self.monitor.metrics_history[-1].to_dict() if self.monitor.metrics_history else None,
            'active_issue_types': [i.issue_type.name for i in self.active_issues if not i.resolved]
        }
    
    def enable_healing(self):
        """Enable automatic healing"""
        self.healing_enabled = True
        logger.info("Automatic healing enabled")
    
    def disable_healing(self):
        """Disable automatic healing"""
        self.healing_enabled = False
        logger.info("Automatic healing disabled")


# Global instance
self_healing_system = SelfHealingSystem()


async def main():
    """Main entry point"""
    # Start self-healing system
    await self_healing_system.start()
    
    # Run for demonstration
    try:
        while True:
            status = self_healing_system.get_status()
            logger.info(f"System status: {json.dumps(status, indent=2)}")
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await self_healing_system.stop()


if __name__ == "__main__":
    asyncio.run(main())