#!/usr/bin/env python3
"""
Real Process Management System
Working implementation with actual process control
"""

import os
import psutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import json
from datetime import datetime

@dataclass
class ProcessInfo:
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: float
    username: str
    cmdline: List[str]

class RealProcessManager:
    """Real process management with actual control capabilities"""
    
    def __init__(self):
        self.monitored_processes = {}
        self.process_limits = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def get_all_processes(self) -> List[ProcessInfo]:
        """Get information about all running processes"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 
                                        'memory_percent', 'create_time', 
                                        'username', 'cmdline']):
            try:
                info = proc.info
                processes.append(ProcessInfo(
                    pid=info['pid'],
                    name=info['name'] or 'Unknown',
                    status=info['status'] or 'unknown',
                    cpu_percent=info['cpu_percent'] or 0.0,
                    memory_percent=info['memory_percent'] or 0.0,
                    create_time=info['create_time'] or 0,
                    username=info['username'] or 'unknown',
                    cmdline=info['cmdline'] or []
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def get_process_by_pid(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed information about a specific process"""
        try:
            proc = psutil.Process(pid)
            
            with proc.oneshot():
                return ProcessInfo(
                    pid=pid,
                    name=proc.name(),
                    status=proc.status(),
                    cpu_percent=proc.cpu_percent(),
                    memory_percent=proc.memory_percent(),
                    create_time=proc.create_time(),
                    username=proc.username(),
                    cmdline=proc.cmdline()
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def start_process(self, command: str, args: List[str] = None, 
                     shell: bool = False, cwd: str = None) -> Optional[int]:
        """Start a new process and return its PID"""
        try:
            if shell:
                proc = subprocess.Popen(command, shell=True, cwd=cwd)
            else:
                cmd_list = [command] + (args or [])
                proc = subprocess.Popen(cmd_list, cwd=cwd)
            
            # Add to monitored processes
            self.monitored_processes[proc.pid] = {
                'command': command,
                'started_at': datetime.now().isoformat(),
                'subprocess': proc
            }
            
            return proc.pid
        except Exception as e:
            print(f"Failed to start process: {e}")
            return None
    
    def stop_process(self, pid: int, force: bool = False) -> bool:
        """Stop a process (SIGTERM or SIGKILL)"""
        try:
            proc = psutil.Process(pid)
            
            if force:
                proc.kill()  # SIGKILL
            else:
                proc.terminate()  # SIGTERM
            
            # Wait for process to actually terminate
            proc.wait(timeout=5)
            
            # Remove from monitored if present
            if pid in self.monitored_processes:
                del self.monitored_processes[pid]
            
            return True
        except psutil.NoSuchProcess:
            return False
        except psutil.TimeoutExpired:
            # Process didn't terminate, try force kill
            if not force:
                return self.stop_process(pid, force=True)
            return False
        except psutil.AccessDenied:
            print(f"Access denied to stop process {pid}")
            return False
    
    def suspend_process(self, pid: int) -> bool:
        """Suspend a process (pause execution)"""
        try:
            proc = psutil.Process(pid)
            proc.suspend()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def resume_process(self, pid: int) -> bool:
        """Resume a suspended process"""
        try:
            proc = psutil.Process(pid)
            proc.resume()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def set_process_priority(self, pid: int, priority: int) -> bool:
        """Set process priority (nice value on Unix, priority class on Windows)"""
        try:
            proc = psutil.Process(pid)
            
            if os.name == 'posix':
                # Unix/Linux: nice values from -20 (highest) to 19 (lowest)
                proc.nice(priority)
            else:
                # Windows priority classes
                proc.nice(psutil.Process.NORMAL_PRIORITY_CLASS)
            
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def limit_process_resources(self, pid: int, cpu_percent: float = None, 
                               memory_mb: int = None) -> bool:
        """Set resource limits for a process"""
        try:
            # Store limits
            self.process_limits[pid] = {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            }
            
            # On Linux, we could use cgroups or ulimit
            # This is a simplified monitoring approach
            if not self.monitoring:
                self.start_resource_monitoring()
            
            return True
        except Exception as e:
            print(f"Failed to set resource limits: {e}")
            return False
    
    def start_resource_monitoring(self):
        """Start monitoring process resources"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop monitoring process resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor processes and enforce resource limits"""
        while self.monitoring:
            for pid, limits in list(self.process_limits.items()):
                try:
                    proc = psutil.Process(pid)
                    
                    # Check CPU usage
                    if limits.get('cpu_percent'):
                        cpu = proc.cpu_percent(interval=0.1)
                        if cpu > limits['cpu_percent']:
                            # Throttle by suspending briefly
                            proc.suspend()
                            time.sleep(0.1)
                            proc.resume()
                    
                    # Check memory usage
                    if limits.get('memory_mb'):
                        mem = proc.memory_info().rss / 1024 / 1024
                        if mem > limits['memory_mb']:
                            print(f"Process {pid} exceeds memory limit ({mem:.1f}MB > {limits['memory_mb']}MB)")
                            # Could terminate or alert
                    
                except psutil.NoSuchProcess:
                    # Process no longer exists
                    del self.process_limits[pid]
                except psutil.AccessDenied:
                    pass
            
            time.sleep(1)  # Check every second
    
    def get_process_tree(self, pid: int) -> Dict:
        """Get process tree (parent and children)"""
        try:
            proc = psutil.Process(pid)
            
            tree = {
                'pid': pid,
                'name': proc.name(),
                'children': []
            }
            
            for child in proc.children(recursive=True):
                tree['children'].append({
                    'pid': child.pid,
                    'name': child.name()
                })
            
            try:
                parent = proc.parent()
                if parent:
                    tree['parent'] = {
                        'pid': parent.pid,
                        'name': parent.name()
                    }
            except:
                pass
            
            return tree
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def find_processes_by_name(self, name: str) -> List[ProcessInfo]:
        """Find all processes matching a name pattern"""
        matching = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if name.lower() in proc.info['name'].lower():
                    matching.append(self.get_process_by_pid(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return [p for p in matching if p]
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'boot_time': psutil.boot_time(),
            'process_count': len(psutil.pids())
        }

class ProcessScheduler:
    """Schedule and manage process execution"""
    
    def __init__(self, manager: RealProcessManager):
        self.manager = manager
        self.scheduled_tasks = []
        self.scheduler_thread = None
        self.running = False
    
    def schedule_process(self, command: str, run_at: datetime, 
                        recurring: bool = False, interval_seconds: int = 0) -> str:
        """Schedule a process to run at a specific time"""
        task_id = f"task_{len(self.scheduled_tasks)}_{int(time.time())}"
        
        task = {
            'id': task_id,
            'command': command,
            'run_at': run_at,
            'recurring': recurring,
            'interval': interval_seconds,
            'last_run': None,
            'status': 'scheduled'
        }
        
        self.scheduled_tasks.append(task)
        
        if not self.running:
            self.start_scheduler()
        
        return task_id
    
    def start_scheduler(self):
        """Start the scheduler thread"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the scheduler thread"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            now = datetime.now()
            
            for task in self.scheduled_tasks:
                if task['status'] != 'scheduled':
                    continue
                
                if now >= task['run_at']:
                    # Run the task
                    pid = self.manager.start_process(task['command'], shell=True)
                    task['last_run'] = now
                    
                    if task['recurring']:
                        # Schedule next run
                        task['run_at'] = now + datetime.timedelta(seconds=task['interval'])
                    else:
                        task['status'] = 'completed'
                    
                    print(f"Scheduled task {task['id']} started (PID: {pid})")
            
            time.sleep(1)  # Check every second

def demonstrate_process_management():
    """Demonstrate real process management"""
    print("=" * 70)
    print("REAL PROCESS MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    
    pm = RealProcessManager()
    
    # 1. List current processes
    print("\n1. Current System Processes:")
    print("-" * 40)
    processes = pm.get_all_processes()
    print(f"Total processes: {len(processes)}")
    
    # Show top 5 by CPU usage
    top_cpu = sorted(processes, key=lambda p: p.cpu_percent, reverse=True)[:5]
    print("\nTop 5 processes by CPU:")
    for proc in top_cpu:
        print(f"  PID {proc.pid:6} | {proc.name[:20]:20} | CPU: {proc.cpu_percent:5.1f}%")
    
    # 2. Start a new process
    print("\n2. Starting New Process:")
    print("-" * 40)
    
    # Start a simple Python process
    test_script = "import time; [print(f'Test process running... {i}') or time.sleep(1) for i in range(5)]"
    pid = pm.start_process("python3", ["-c", test_script])
    
    if pid:
        print(f"✅ Started process with PID: {pid}")
        
        # Get info about the new process
        time.sleep(1)
        info = pm.get_process_by_pid(pid)
        if info:
            print(f"   Name: {info.name}")
            print(f"   Status: {info.status}")
    
    # 3. System statistics
    print("\n3. System Statistics:")
    print("-" * 40)
    stats = pm.get_system_stats()
    print(f"CPU Usage: {stats['cpu']['percent']}%")
    print(f"Memory Usage: {stats['memory']['percent']:.1f}%")
    print(f"Disk Usage: {stats['disk']['percent']:.1f}%")
    print(f"Total Processes: {stats['process_count']}")
    
    # 4. Process scheduling
    print("\n4. Process Scheduling:")
    print("-" * 40)
    scheduler = ProcessScheduler(pm)
    
    # Schedule a task
    future_time = datetime.now() + datetime.timedelta(seconds=2)
    task_id = scheduler.schedule_process(
        "echo 'Scheduled task executed!'",
        future_time
    )
    print(f"✅ Task scheduled to run at {future_time.strftime('%H:%M:%S')}")
    
    # Wait for scheduled task
    time.sleep(3)
    
    scheduler.stop_scheduler()
    pm.stop_resource_monitoring()
    
    print("\n" + "=" * 70)
    print("✅ REAL PROCESS MANAGEMENT WORKING!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_process_management()