#!/usr/bin/env python3
"""
QENEX OS Kernel - Core system management
"""

import asyncio
import os
import psutil
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor

@dataclass
class Process:
    """Represents a system process"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    created_time: float
    priority: int = 0

class ProcessManager:
    """Manages system processes"""
    
    def __init__(self):
        self.processes: Dict[int, Process] = {}
        self.scheduler_queue: List[Process] = []
        
    def create_process(self, name: str, priority: int = 0) -> Process:
        """Create a new process"""
        pid = os.getpid()
        process = Process(
            pid=pid,
            name=name,
            status="running",
            cpu_percent=0.0,
            memory_percent=0.0,
            created_time=time.time(),
            priority=priority
        )
        self.processes[pid] = process
        return process
    
    def kill_process(self, pid: int) -> bool:
        """Kill a process"""
        try:
            os.kill(pid, signal.SIGTERM)
            if pid in self.processes:
                self.processes[pid].status = "terminated"
            return True
        except ProcessLookupError:
            return False
    
    def get_process_list(self) -> List[Process]:
        """Get list of all processes"""
        process_list = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
            try:
                process_list.append(Process(
                    pid=proc.info['pid'],
                    name=proc.info['name'],
                    status=proc.info['status'],
                    cpu_percent=proc.info['cpu_percent'] or 0.0,
                    memory_percent=proc.info['memory_percent'] or 0.0,
                    created_time=proc.create_time()
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return process_list
    
    def schedule_process(self, process: Process):
        """Add process to scheduler queue"""
        self.scheduler_queue.append(process)
        self.scheduler_queue.sort(key=lambda p: p.priority, reverse=True)

class MemoryManager:
    """Manages system memory"""
    
    def __init__(self):
        self.page_size = 4096  # 4KB pages
        self.pages: Dict[int, bytes] = {}
        self.free_pages: List[int] = list(range(1000))  # Start with 1000 free pages
        
    def allocate_memory(self, size: int) -> Optional[List[int]]:
        """Allocate memory pages"""
        pages_needed = (size + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < pages_needed:
            return None
        
        allocated_pages = []
        for _ in range(pages_needed):
            page = self.free_pages.pop(0)
            allocated_pages.append(page)
            self.pages[page] = bytes(self.page_size)
        
        return allocated_pages
    
    def free_memory(self, pages: List[int]):
        """Free memory pages"""
        for page in pages:
            if page in self.pages:
                del self.pages[page]
                self.free_pages.append(page)
    
    def get_memory_info(self) -> Dict:
        """Get memory information"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "pages_allocated": len(self.pages),
            "pages_free": len(self.free_pages)
        }

class FileSystem:
    """Simple file system implementation"""
    
    def __init__(self, root_path: str = "/qenex-os/fs"):
        self.root_path = root_path
        os.makedirs(root_path, exist_ok=True)
        
    def create_file(self, path: str, content: bytes = b"") -> bool:
        """Create a file"""
        full_path = os.path.join(self.root_path, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def read_file(self, path: str) -> Optional[bytes]:
        """Read a file"""
        full_path = os.path.join(self.root_path, path)
        try:
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def delete_file(self, path: str) -> bool:
        """Delete a file"""
        full_path = os.path.join(self.root_path, path)
        try:
            os.remove(full_path)
            return True
        except Exception:
            return False
    
    def list_files(self, path: str = "") -> List[str]:
        """List files in a directory"""
        full_path = os.path.join(self.root_path, path)
        try:
            return os.listdir(full_path)
        except Exception:
            return []

class Kernel:
    """QENEX OS Kernel"""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.memory_manager = MemoryManager()
        self.file_system = FileSystem()
        self.running = False
        self.boot_time = None
        self.system_calls: Dict[str, Callable] = {}
        self.interrupt_handlers: Dict[int, Callable] = {}
        
    async def boot(self):
        """Boot the kernel"""
        print("🚀 QENEX OS Kernel booting...")
        self.boot_time = time.time()
        self.running = True
        
        # Initialize system calls
        self._init_system_calls()
        
        # Start kernel services
        asyncio.create_task(self.scheduler())
        asyncio.create_task(self.memory_monitor())
        
        print(f"✅ QENEX OS Kernel booted in {time.time() - self.boot_time:.2f} seconds")
        
    async def shutdown(self):
        """Shutdown the kernel"""
        print("🔴 QENEX OS Kernel shutting down...")
        self.running = False
        
        # Kill all processes
        for pid in list(self.process_manager.processes.keys()):
            self.process_manager.kill_process(pid)
        
        print("✅ QENEX OS Kernel shutdown complete")
    
    def _init_system_calls(self):
        """Initialize system calls"""
        self.system_calls = {
            "open": self.syscall_open,
            "read": self.syscall_read,
            "write": self.syscall_write,
            "close": self.syscall_close,
            "fork": self.syscall_fork,
            "exec": self.syscall_exec,
            "exit": self.syscall_exit,
            "malloc": self.syscall_malloc,
            "free": self.syscall_free
        }
    
    async def syscall_open(self, path: str, mode: str = "r") -> int:
        """Open file system call"""
        # Simple file descriptor simulation
        return hash(path) % 1000
    
    async def syscall_read(self, fd: int, size: int) -> bytes:
        """Read system call"""
        # Simplified read operation
        return b"data" * (size // 4)
    
    async def syscall_write(self, fd: int, data: bytes) -> int:
        """Write system call"""
        return len(data)
    
    async def syscall_close(self, fd: int) -> bool:
        """Close file system call"""
        return True
    
    async def syscall_fork(self) -> int:
        """Fork process system call"""
        return os.fork() if hasattr(os, 'fork') else -1
    
    async def syscall_exec(self, program: str, args: List[str]) -> bool:
        """Execute program system call"""
        try:
            os.execvp(program, args)
            return True
        except Exception:
            return False
    
    async def syscall_exit(self, code: int):
        """Exit system call"""
        sys.exit(code)
    
    async def syscall_malloc(self, size: int) -> Optional[List[int]]:
        """Allocate memory system call"""
        return self.memory_manager.allocate_memory(size)
    
    async def syscall_free(self, pages: List[int]):
        """Free memory system call"""
        self.memory_manager.free_memory(pages)
    
    async def scheduler(self):
        """Process scheduler"""
        while self.running:
            await asyncio.sleep(0.1)
            
            # Simple round-robin scheduling
            if self.process_manager.scheduler_queue:
                process = self.process_manager.scheduler_queue.pop(0)
                # Simulate process execution
                await asyncio.sleep(0.01)
                # Re-add to queue if still running
                if process.status == "running":
                    self.process_manager.scheduler_queue.append(process)
    
    async def memory_monitor(self):
        """Monitor memory usage"""
        while self.running:
            await asyncio.sleep(5)
            
            mem_info = self.memory_manager.get_memory_info()
            if mem_info["percent"] > 90:
                print(f"⚠️ High memory usage: {mem_info['percent']:.1f}%")
                # Trigger garbage collection or memory optimization
    
    def handle_interrupt(self, interrupt_num: int, *args):
        """Handle hardware/software interrupts"""
        if interrupt_num in self.interrupt_handlers:
            return self.interrupt_handlers[interrupt_num](*args)
        else:
            print(f"Unhandled interrupt: {interrupt_num}")
    
    def register_interrupt_handler(self, interrupt_num: int, handler: Callable):
        """Register an interrupt handler"""
        self.interrupt_handlers[interrupt_num] = handler
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "kernel_version": "1.0.0",
            "uptime": time.time() - self.boot_time if self.boot_time else 0,
            "processes": len(self.process_manager.processes),
            "memory": self.memory_manager.get_memory_info(),
            "cpu": {
                "percent": psutil.cpu_percent(),
                "cores": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        }

# Singleton kernel instance
kernel = Kernel()

async def main():
    """Main function for testing"""
    await kernel.boot()
    
    # Create a test process
    process = kernel.process_manager.create_process("test_process", priority=5)
    print(f"Created process: {process}")
    
    # Allocate memory
    pages = await kernel.syscall_malloc(8192)
    print(f"Allocated memory pages: {pages}")
    
    # Create a file
    kernel.file_system.create_file("test.txt", b"Hello, QENEX OS!")
    content = kernel.file_system.read_file("test.txt")
    print(f"File content: {content}")
    
    # Get system info
    info = kernel.get_system_info()
    print(f"System info: {info}")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Free memory
    if pages:
        await kernel.syscall_free(pages)
        print("Memory freed")
    
    await kernel.shutdown()

if __name__ == "__main__":
    asyncio.run(main())