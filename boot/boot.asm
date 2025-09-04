# QENEX Banking OS Boot Assembly
# Real bootable kernel entry point with hardware initialization

.global _start
.global kernel_main_rust
.global efi_kernel_main
.global efi64_kernel_main

# Include multiboot2 header
.include "multiboot2_header.asm"

# Boot stack (16KB for early initialization)
.section .bootstrap_stack, "aw", @nobits
stack_bottom:
    .skip 16384
stack_top:

# Early page tables for long mode transition
.section .bss, "aw", @nobits
.align 4096

# PML4 table (512 entries * 8 bytes = 4KB)
boot_pml4:
    .skip 4096

# PDPT table
boot_pdpt:
    .skip 4096

# Page Directory
boot_pd:
    .skip 4096

# GDT for long mode
.section .rodata
.align 16
gdt64:
    .quad 0                             # Null descriptor
    .quad 0x00AF9A000000FFFF            # Code segment (64-bit)
    .quad 0x00AF92000000FFFF            # Data segment
gdt64_end:

.align 8
gdt64_pointer:
    .word gdt64_end - gdt64 - 1         # Limit
    .quad gdt64                         # Base

# Boot error messages
boot_error_no_multiboot:
    .ascii "FATAL: Not booted by Multiboot2 compliant bootloader"
boot_error_no_cpuid:
    .ascii "FATAL: CPUID instruction not supported"
boot_error_no_long_mode:
    .ascii "FATAL: Long mode not supported by CPU"

.section .text
.code32

# Main kernel entry point (called by GRUB)
_start:
    # Set up stack
    mov $stack_top, %esp
    
    # Clear direction flag
    cld
    
    # Store multiboot2 information
    push %ebx                           # Multiboot2 info structure
    push %eax                           # Multiboot2 magic number
    
    # Verify multiboot2 magic number
    cmp $0x36d76289, %eax
    jne .no_multiboot2
    
    # Display boot message
    call display_boot_message
    
    # Check CPU capabilities required for banking OS
    call check_multiboot2
    call check_cpuid
    call check_long_mode
    call check_banking_cpu_features
    
    # Set up paging for long mode
    call setup_page_tables
    call enable_paging
    
    # Load GDT and switch to long mode
    lgdt gdt64_pointer
    
    # Enable long mode
    mov %cr4, %eax
    or $0x20, %eax                      # Set PAE bit
    mov %eax, %cr4
    
    # Enable long mode in EFER
    mov $0xC0000080, %ecx
    rdmsr
    or $0x100, %eax                     # Set LM bit
    wrmsr
    
    # Enable paging
    mov %cr0, %eax
    or $0x80000000, %eax                # Set PG bit
    mov %eax, %cr0
    
    # Far jump to 64-bit code
    ljmp $0x08, $long_mode_start

# Display early boot message
display_boot_message:
    push %esi
    push %edi
    
    # VGA text mode buffer at 0xB8000
    mov $0xB8000, %edi
    mov $boot_message, %esi
    mov $0x0F, %ah                      # White on black
    
.display_loop:
    lodsb
    test %al, %al
    jz .display_done
    
    stosw                               # Store char + attribute
    jmp .display_loop
    
.display_done:
    pop %edi
    pop %esi
    ret

boot_message:
    .asciz "QENEX Banking OS v1.0 - Secure Boot Initializing..."

# Check multiboot2 compliance
check_multiboot2:
    cmp $0x36d76289, %eax
    jne .no_multiboot2
    ret

.no_multiboot2:
    mov $boot_error_no_multiboot, %esi
    call display_error
    jmp halt_system

# Check CPUID availability
check_cpuid:
    # Try to flip ID bit in EFLAGS
    pushfl
    pushfl
    xor $0x200000, (%esp)
    popfl
    pushfl
    pop %eax
    xor (%esp), %eax
    popfl
    
    and $0x200000, %eax
    jz .no_cpuid
    ret

.no_cpuid:
    mov $boot_error_no_cpuid, %esi
    call display_error
    jmp halt_system

# Check for long mode support
check_long_mode:
    # Check if extended processor info is available
    mov $0x80000000, %eax
    cpuid
    cmp $0x80000001, %eax
    jb .no_long_mode
    
    # Check for long mode
    mov $0x80000001, %eax
    cpuid
    test $0x20000000, %edx
    jz .no_long_mode
    ret

.no_long_mode:
    mov $boot_error_no_long_mode, %esi
    call display_error
    jmp halt_system

# Check banking-specific CPU features
check_banking_cpu_features:
    # Check for required features: SSE2, AES-NI, RDRAND, RDSEED
    mov $1, %eax
    cpuid
    
    # Check SSE2 (bit 26)
    test $0x4000000, %edx
    jz .missing_banking_features
    
    # Check AES-NI (bit 25)
    test $0x2000000, %ecx
    jz .missing_banking_features
    
    # Check RDRAND (bit 30)
    test $0x40000000, %ecx
    jz .missing_banking_features
    
    # Check extended features
    mov $7, %eax
    mov $0, %ecx
    cpuid
    
    # Check RDSEED (bit 18)
    test $0x40000, %ebx
    jz .missing_banking_features
    
    ret

.missing_banking_features:
    mov $boot_error_banking_features, %esi
    call display_error
    jmp halt_system

boot_error_banking_features:
    .ascii "FATAL: CPU lacks required banking security features (AES-NI, RDRAND, RDSEED)"

# Set up identity paging for first 2MB
setup_page_tables:
    # Clear page tables
    mov $boot_pml4, %edi
    mov $0x3000, %ecx                   # 3 * 4KB = 12KB to clear
    xor %eax, %eax
    rep stosl
    
    # Set up PML4
    mov $boot_pml4, %edi
    mov $boot_pdpt, %eax
    or $0x3, %eax                       # Present + Writable
    mov %eax, (%edi)
    
    # Set up PDPT
    mov $boot_pdpt, %edi
    mov $boot_pd, %eax
    or $0x3, %eax
    mov %eax, (%edi)
    
    # Set up Page Directory (2MB pages)
    mov $boot_pd, %edi
    mov $0, %eax                        # Start at physical address 0
    mov $512, %ecx                      # 512 entries
    
.setup_pd_loop:
    mov %eax, %edx
    or $0x83, %edx                      # Present + Writable + Large page
    mov %edx, (%edi)
    add $0x200000, %eax                 # Next 2MB page
    add $8, %edi
    loop .setup_pd_loop
    
    ret

# Enable paging
enable_paging:
    # Load CR3 with PML4 address
    mov $boot_pml4, %eax
    mov %eax, %cr3
    ret

# Display error message and halt
display_error:
    # Display error in red
    push %esi
    push %edi
    
    mov $0xB8000, %edi
    add $160, %edi                      # Second line
    mov $0x4F, %ah                      # White on red
    
.error_loop:
    lodsb
    test %al, %al
    jz .error_done
    stosw
    jmp .error_loop
    
.error_done:
    pop %edi
    pop %esi
    ret

# System halt on error
halt_system:
    cli
    hlt
    jmp halt_system

# Long mode entry point
.code64
long_mode_start:
    # Set up 64-bit segments
    mov $0x10, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %fs
    mov %ax, %gs
    mov %ax, %ss
    
    # Set up 64-bit stack
    mov $stack_top, %rsp
    
    # Clear frame pointer
    xor %rbp, %rbp
    
    # Jump to C kernel  
    # Restore multiboot parameters for C kernel
    pop %rdi                            # Multiboot info pointer
    pop %rsi                            # Multiboot magic
    call kernel_main
    
    # Should never return
    jmp halt_system

# Kernel main for 64-bit mode
kernel_main_64:
    # This will call into Rust kernel code
    # For now, just display success message
    
    # Display success message in green
    mov $0xB8000, %rdi
    add $320, %rdi                      # Third line
    mov $success_message, %rsi
    mov $0x2F, %ah                      # White on green
    
.success_loop:
    lodsb
    test %al, %al
    jz .success_done
    stosw
    jmp .success_loop
    
.success_done:
    # Initialize banking kernel subsystems
    call init_banking_kernel
    
    # Enter main banking kernel loop
    call banking_kernel_main
    
    # Should never return
    jmp halt_system

success_message:
    .asciz "QENEX Banking OS - Long Mode Initialized Successfully"

# EFI entry points
efi_kernel_main:
    # 32-bit EFI entry point
    jmp _start

efi64_kernel_main:
    # 64-bit EFI entry point
    jmp long_mode_start

# Initialize banking kernel subsystems
init_banking_kernel:
    # This will be implemented to initialize:
    # - Hardware Security Modules
    # - Trusted Platform Module
    # - Banking-specific hardware
    # - Security subsystems
    ret

# Main banking kernel loop
banking_kernel_main:
    # Main kernel loop for banking operations
    # This will handle interrupts, scheduling, etc.
    
.kernel_loop:
    hlt                                 # Wait for interrupt
    jmp .kernel_loop

# Memory layout symbols (defined in linker script)
.global _kernel_start
.global _data_end
.global _bss_end