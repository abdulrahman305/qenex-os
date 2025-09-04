/*
 * QENEX Cross-Platform Compatibility Layer
 * Enables operation on Windows, Linux, macOS, BSD, and embedded systems
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

#ifdef _WIN32
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#elif __linux__
    #include <sys/mman.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
    #include <dlfcn.h>
#elif __APPLE__
    #include <mach/mach.h>
    #include <mach/vm_map.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
#elif __FreeBSD__
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
#endif

namespace QENEX {
namespace Platform {

/* Platform detection */
enum class OSType {
    Windows,
    Linux,
    macOS,
    FreeBSD,
    Android,
    iOS,
    Embedded,
    Unknown
};

enum class Architecture {
    x86,
    x86_64,
    ARM32,
    ARM64,
    RISCV64,
    PowerPC,
    MIPS,
    Unknown
};

class PlatformInfo {
public:
    static OSType GetOS() {
        #ifdef _WIN32
            return OSType::Windows;
        #elif __linux__
            #ifdef __ANDROID__
                return OSType::Android;
            #else
                return OSType::Linux;
            #endif
        #elif __APPLE__
            #include <TargetConditionals.h>
            #if TARGET_OS_IPHONE
                return OSType::iOS;
            #else
                return OSType::macOS;
            #endif
        #elif __FreeBSD__
            return OSType::FreeBSD;
        #else
            return OSType::Unknown;
        #endif
    }
    
    static Architecture GetArchitecture() {
        #if defined(__x86_64__) || defined(_M_X64)
            return Architecture::x86_64;
        #elif defined(__i386__) || defined(_M_IX86)
            return Architecture::x86;
        #elif defined(__aarch64__) || defined(_M_ARM64)
            return Architecture::ARM64;
        #elif defined(__arm__) || defined(_M_ARM)
            return Architecture::ARM32;
        #elif defined(__riscv) && __riscv_xlen == 64
            return Architecture::RISCV64;
        #elif defined(__powerpc64__)
            return Architecture::PowerPC;
        #elif defined(__mips__)
            return Architecture::MIPS;
        #else
            return Architecture::Unknown;
        #endif
    }
    
    static size_t GetPageSize() {
        #ifdef _WIN32
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            return si.dwPageSize;
        #else
            return sysconf(_SC_PAGESIZE);
        #endif
    }
    
    static size_t GetProcessorCount() {
        return std::thread::hardware_concurrency();
    }
};

/* Memory abstraction layer */
class Memory {
public:
    static void* Allocate(size_t size, bool executable = false) {
        #ifdef _WIN32
            DWORD protect = executable ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE;
            return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, protect);
        #else
            int prot = PROT_READ | PROT_WRITE;
            if (executable) prot |= PROT_EXEC;
            
            void* ptr = mmap(NULL, size, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            return (ptr == MAP_FAILED) ? nullptr : ptr;
        #endif
    }
    
    static void Free(void* ptr, size_t size) {
        #ifdef _WIN32
            VirtualFree(ptr, 0, MEM_RELEASE);
        #else
            munmap(ptr, size);
        #endif
    }
    
    static bool Protect(void* ptr, size_t size, bool read, bool write, bool execute) {
        #ifdef _WIN32
            DWORD protect = 0;
            if (execute && write) protect = PAGE_EXECUTE_READWRITE;
            else if (execute && read) protect = PAGE_EXECUTE_READ;
            else if (write) protect = PAGE_READWRITE;
            else if (read) protect = PAGE_READONLY;
            else protect = PAGE_NOACCESS;
            
            DWORD oldProtect;
            return VirtualProtect(ptr, size, protect, &oldProtect) != 0;
        #else
            int prot = 0;
            if (read) prot |= PROT_READ;
            if (write) prot |= PROT_WRITE;
            if (execute) prot |= PROT_EXEC;
            
            return mprotect(ptr, size, prot) == 0;
        #endif
    }
    
    /* Shared memory for inter-process communication */
    static void* CreateSharedMemory(const char* name, size_t size) {
        #ifdef _WIN32
            HANDLE hMapFile = CreateFileMappingA(
                INVALID_HANDLE_VALUE,
                NULL,
                PAGE_READWRITE,
                0,
                (DWORD)size,
                name
            );
            
            if (hMapFile == NULL) return nullptr;
            
            return MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        #else
            int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
            if (fd == -1) return nullptr;
            
            if (ftruncate(fd, size) == -1) {
                close(fd);
                return nullptr;
            }
            
            void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            
            return (ptr == MAP_FAILED) ? nullptr : ptr;
        #endif
    }
};

/* Thread abstraction */
class Thread {
public:
    using ThreadFunc = std::function<void()>;
    
private:
    #ifdef _WIN32
        HANDLE handle_;
    #else
        pthread_t thread_;
    #endif
    ThreadFunc func_;
    std::atomic<bool> running_;
    
public:
    Thread(ThreadFunc func) : func_(func), running_(false) {}
    
    bool Start(int priority = 0) {
        running_ = true;
        
        #ifdef _WIN32
            handle_ = CreateThread(
                NULL,
                0,
                [](LPVOID param) -> DWORD {
                    Thread* self = static_cast<Thread*>(param);
                    self->func_();
                    return 0;
                },
                this,
                0,
                NULL
            );
            
            if (handle_ && priority != 0) {
                SetThreadPriority(handle_, priority);
            }
            
            return handle_ != NULL;
        #else
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            
            if (priority != 0) {
                struct sched_param param;
                param.sched_priority = priority;
                pthread_attr_setschedparam(&attr, &param);
            }
            
            int result = pthread_create(&thread_, &attr, 
                [](void* param) -> void* {
                    Thread* self = static_cast<Thread*>(param);
                    self->func_();
                    return nullptr;
                }, this);
            
            pthread_attr_destroy(&attr);
            return result == 0;
        #endif
    }
    
    void Join() {
        #ifdef _WIN32
            WaitForSingleObject(handle_, INFINITE);
            CloseHandle(handle_);
        #else
            pthread_join(thread_, nullptr);
        #endif
        running_ = false;
    }
    
    static void SetAffinity(int cpu) {
        #ifdef _WIN32
            SetThreadAffinityMask(GetCurrentThread(), 1 << cpu);
        #elif __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        #endif
    }
};

/* Network abstraction */
class Network {
private:
    static bool initialized_;
    
public:
    static bool Initialize() {
        if (initialized_) return true;
        
        #ifdef _WIN32
            WSADATA wsaData;
            int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
            initialized_ = (result == 0);
        #else
            initialized_ = true;
        #endif
        
        return initialized_;
    }
    
    static void Cleanup() {
        #ifdef _WIN32
            WSACleanup();
        #endif
        initialized_ = false;
    }
    
    class Socket {
    private:
        #ifdef _WIN32
            SOCKET sock_;
        #else
            int sock_;
        #endif
        
    public:
        Socket(int family, int type, int protocol) {
            sock_ = socket(family, type, protocol);
        }
        
        ~Socket() {
            Close();
        }
        
        bool Bind(uint16_t port) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            
            return bind(sock_, (struct sockaddr*)&addr, sizeof(addr)) == 0;
        }
        
        bool Listen(int backlog = 128) {
            return listen(sock_, backlog) == 0;
        }
        
        Socket* Accept() {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            #ifdef _WIN32
                SOCKET client = accept(sock_, (struct sockaddr*)&client_addr, &client_len);
                if (client == INVALID_SOCKET) return nullptr;
            #else
                int client = accept(sock_, (struct sockaddr*)&client_addr, &client_len);
                if (client < 0) return nullptr;
            #endif
            
            Socket* client_socket = new Socket(0, 0, 0);
            client_socket->sock_ = client;
            return client_socket;
        }
        
        int Send(const void* data, size_t length, int flags = 0) {
            return send(sock_, (const char*)data, (int)length, flags);
        }
        
        int Receive(void* buffer, size_t length, int flags = 0) {
            return recv(sock_, (char*)buffer, (int)length, flags);
        }
        
        void Close() {
            #ifdef _WIN32
                if (sock_ != INVALID_SOCKET) {
                    closesocket(sock_);
                    sock_ = INVALID_SOCKET;
                }
            #else
                if (sock_ >= 0) {
                    close(sock_);
                    sock_ = -1;
                }
            #endif
        }
        
        bool SetNonBlocking(bool enable) {
            #ifdef _WIN32
                u_long mode = enable ? 1 : 0;
                return ioctlsocket(sock_, FIONBIO, &mode) == 0;
            #else
                int flags = fcntl(sock_, F_GETFL, 0);
                if (flags < 0) return false;
                
                if (enable) flags |= O_NONBLOCK;
                else flags &= ~O_NONBLOCK;
                
                return fcntl(sock_, F_SETFL, flags) == 0;
            #endif
        }
    };
};

bool Network::initialized_ = false;

/* File system abstraction */
class FileSystem {
public:
    static std::string GetSeparator() {
        #ifdef _WIN32
            return "\\";
        #else
            return "/";
        #endif
    }
    
    static std::string GetTempPath() {
        #ifdef _WIN32
            char buffer[MAX_PATH];
            GetTempPathA(MAX_PATH, buffer);
            return std::string(buffer);
        #else
            const char* tmp = getenv("TMPDIR");
            if (!tmp) tmp = getenv("TMP");
            if (!tmp) tmp = getenv("TEMP");
            if (!tmp) tmp = "/tmp";
            return std::string(tmp);
        #endif
    }
    
    static bool CreateDirectory(const std::string& path) {
        #ifdef _WIN32
            return CreateDirectoryA(path.c_str(), NULL) != 0;
        #else
            return mkdir(path.c_str(), 0755) == 0;
        #endif
    }
    
    static bool FileExists(const std::string& path) {
        #ifdef _WIN32
            DWORD attrs = GetFileAttributesA(path.c_str());
            return attrs != INVALID_FILE_ATTRIBUTES;
        #else
            return access(path.c_str(), F_OK) == 0;
        #endif
    }
};

/* Dynamic library loading */
class DynamicLibrary {
private:
    #ifdef _WIN32
        HMODULE handle_;
    #else
        void* handle_;
    #endif
    
public:
    DynamicLibrary(const std::string& path) {
        #ifdef _WIN32
            handle_ = LoadLibraryA(path.c_str());
        #else
            handle_ = dlopen(path.c_str(), RTLD_LAZY);
        #endif
    }
    
    ~DynamicLibrary() {
        if (handle_) {
            #ifdef _WIN32
                FreeLibrary(handle_);
            #else
                dlclose(handle_);
            #endif
        }
    }
    
    void* GetSymbol(const std::string& name) {
        if (!handle_) return nullptr;
        
        #ifdef _WIN32
            return GetProcAddress(handle_, name.c_str());
        #else
            return dlsym(handle_, name.c_str());
        #endif
    }
    
    bool IsLoaded() const {
        return handle_ != nullptr;
    }
};

/* Hardware abstraction */
class Hardware {
public:
    struct CPUInfo {
        std::string vendor;
        std::string brand;
        int cores;
        int threads;
        uint64_t features;
        bool has_aes;
        bool has_avx;
        bool has_avx2;
        bool has_avx512;
        bool has_sgx;
    };
    
    static CPUInfo GetCPUInfo() {
        CPUInfo info;
        
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            uint32_t regs[4];
            
            // Get vendor
            __cpuid(0, regs[0], regs[1], regs[2], regs[3]);
            char vendor[13];
            memcpy(vendor, &regs[1], 4);
            memcpy(vendor + 4, &regs[3], 4);
            memcpy(vendor + 8, &regs[2], 4);
            vendor[12] = '\0';
            info.vendor = vendor;
            
            // Get features
            __cpuid(1, regs[0], regs[1], regs[2], regs[3]);
            info.has_aes = (regs[2] & (1 << 25)) != 0;
            
            // Check extended features
            __cpuid(7, regs[0], regs[1], regs[2], regs[3]);
            info.has_avx = (regs[1] & (1 << 5)) != 0;
            info.has_avx2 = (regs[1] & (1 << 5)) != 0;
            info.has_sgx = (regs[1] & (1 << 2)) != 0;
        #endif
        
        info.cores = PlatformInfo::GetProcessorCount();
        info.threads = info.cores;
        
        return info;
    }
    
    static uint64_t GetMemorySize() {
        #ifdef _WIN32
            MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            return status.ullTotalPhys;
        #elif __linux__
            long pages = sysconf(_SC_PHYS_PAGES);
            long page_size = sysconf(_SC_PAGE_SIZE);
            return pages * page_size;
        #elif __APPLE__
            int mib[2] = {CTL_HW, HW_MEMSIZE};
            uint64_t memsize;
            size_t len = sizeof(memsize);
            sysctl(mib, 2, &memsize, &len, NULL, 0);
            return memsize;
        #else
            return 0;
        #endif
    }
};

/* Virtualization layer */
class Virtualization {
public:
    enum class HypervisorType {
        None,
        VMware,
        VirtualBox,
        HyperV,
        KVM,
        Xen,
        QEMU,
        Docker,
        Unknown
    };
    
    static bool IsVirtualized() {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            uint32_t regs[4];
            __cpuid(1, regs[0], regs[1], regs[2], regs[3]);
            return (regs[2] & (1 << 31)) != 0;  // Hypervisor present bit
        #else
            return false;
        #endif
    }
    
    static HypervisorType GetHypervisor() {
        if (!IsVirtualized()) return HypervisorType::None;
        
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            uint32_t regs[4];
            __cpuid(0x40000000, regs[0], regs[1], regs[2], regs[3]);
            
            char vendor[13];
            memcpy(vendor, &regs[1], 4);
            memcpy(vendor + 4, &regs[2], 4);
            memcpy(vendor + 8, &regs[3], 4);
            vendor[12] = '\0';
            
            std::string vendorStr(vendor);
            
            if (vendorStr == "VMwareVMware") return HypervisorType::VMware;
            if (vendorStr == "VBoxVBoxVBox") return HypervisorType::VirtualBox;
            if (vendorStr == "Microsoft Hv") return HypervisorType::HyperV;
            if (vendorStr == "KVMKVMKVM") return HypervisorType::KVM;
            if (vendorStr == "XenVMMXenVMM") return HypervisorType::Xen;
        #endif
        
        return HypervisorType::Unknown;
    }
    
    /* Container detection */
    static bool IsInDocker() {
        #ifdef __linux__
            return FileSystem::FileExists("/.dockerenv");
        #else
            return false;
        #endif
    }
    
    static bool IsInKubernetes() {
        #ifdef __linux__
            return getenv("KUBERNETES_SERVICE_HOST") != nullptr;
        #else
            return false;
        #endif
    }
};

/* Banking-specific hardware support */
class BankingHardware {
public:
    /* Hardware Security Module (HSM) interface */
    class HSM {
    private:
        DynamicLibrary* pkcs11_;
        
    public:
        HSM(const std::string& libraryPath) {
            pkcs11_ = new DynamicLibrary(libraryPath);
        }
        
        ~HSM() {
            delete pkcs11_;
        }
        
        bool Initialize() {
            if (!pkcs11_->IsLoaded()) return false;
            
            // Initialize PKCS#11
            typedef int (*C_Initialize)(void*);
            C_Initialize init = (C_Initialize)pkcs11_->GetSymbol("C_Initialize");
            if (!init) return false;
            
            return init(nullptr) == 0;
        }
        
        bool GenerateKeyPair(uint8_t* publicKey, size_t* pubKeyLen,
                            uint32_t* privateKeyHandle) {
            // PKCS#11 key generation
            return false;  // Simplified
        }
        
        bool Sign(uint32_t keyHandle, const uint8_t* data, size_t dataLen,
                 uint8_t* signature, size_t* sigLen) {
            // PKCS#11 signing
            return false;  // Simplified
        }
    };
    
    /* Trusted Platform Module (TPM) interface */
    class TPM {
    public:
        static bool IsAvailable() {
            #ifdef _WIN32
                // Check for TPM on Windows
                return false;  // Simplified
            #elif __linux__
                return FileSystem::FileExists("/dev/tpm0");
            #else
                return false;
            #endif
        }
        
        static bool SealData(const uint8_t* data, size_t dataLen,
                            uint8_t* sealed, size_t* sealedLen) {
            // TPM sealing operation
            return false;  // Simplified
        }
    };
    
    /* Secure Enclave support */
    class SecureEnclave {
    public:
        static bool IsSupported() {
            Hardware::CPUInfo cpu = Hardware::GetCPUInfo();
            return cpu.has_sgx;
        }
        
        static void* CreateEnclave(size_t size) {
            if (!IsSupported()) return nullptr;
            
            // Intel SGX enclave creation
            return nullptr;  // Simplified
        }
    };
};

/* Cross-platform unified interface */
class UnifiedPlatform {
private:
    static UnifiedPlatform* instance_;
    OSType os_;
    Architecture arch_;
    Hardware::CPUInfo cpu_;
    std::map<std::string, std::unique_ptr<DynamicLibrary>> libraries_;
    std::mutex mutex_;
    
    UnifiedPlatform() {
        os_ = PlatformInfo::GetOS();
        arch_ = PlatformInfo::GetArchitecture();
        cpu_ = Hardware::GetCPUInfo();
        Network::Initialize();
    }
    
public:
    static UnifiedPlatform* GetInstance() {
        if (!instance_) {
            instance_ = new UnifiedPlatform();
        }
        return instance_;
    }
    
    /* Platform capabilities */
    bool SupportsAES() const { return cpu_.has_aes; }
    bool SupportsAVX() const { return cpu_.has_avx; }
    bool SupportsSecureEnclave() const { return cpu_.has_sgx; }
    bool IsVirtualized() const { return Virtualization::IsVirtualized(); }
    bool IsContainerized() const { 
        return Virtualization::IsInDocker() || Virtualization::IsInKubernetes();
    }
    
    /* Load platform-specific banking module */
    bool LoadBankingModule(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string libName;
        
        switch (os_) {
            case OSType::Windows:
                libName = name + ".dll";
                break;
            case OSType::Linux:
            case OSType::FreeBSD:
                libName = "lib" + name + ".so";
                break;
            case OSType::macOS:
            case OSType::iOS:
                libName = "lib" + name + ".dylib";
                break;
            default:
                return false;
        }
        
        auto lib = std::make_unique<DynamicLibrary>(libName);
        if (!lib->IsLoaded()) return false;
        
        libraries_[name] = std::move(lib);
        return true;
    }
    
    /* Get banking module function */
    void* GetBankingFunction(const std::string& module, const std::string& function) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = libraries_.find(module);
        if (it == libraries_.end()) return nullptr;
        
        return it->second->GetSymbol(function);
    }
    
    /* Allocate secure memory for sensitive data */
    void* AllocateSecureMemory(size_t size) {
        void* ptr = Memory::Allocate(size);
        
        if (ptr) {
            // Lock memory to prevent swapping
            #ifdef _WIN32
                VirtualLock(ptr, size);
            #elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
                mlock(ptr, size);
            #endif
        }
        
        return ptr;
    }
    
    void FreeSecureMemory(void* ptr, size_t size) {
        if (ptr) {
            // Clear memory before freeing
            volatile uint8_t* vptr = static_cast<volatile uint8_t*>(ptr);
            for (size_t i = 0; i < size; i++) {
                vptr[i] = 0;
            }
            
            #ifdef _WIN32
                VirtualUnlock(ptr, size);
            #elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
                munlock(ptr, size);
            #endif
            
            Memory::Free(ptr, size);
        }
    }
    
    ~UnifiedPlatform() {
        Network::Cleanup();
    }
};

UnifiedPlatform* UnifiedPlatform::instance_ = nullptr;

} // namespace Platform
} // namespace QENEX