/* QENEX Banking OS Linker Script */
/* Production-ready memory layout for secure banking operations */

ENTRY(_start)
OUTPUT_FORMAT("elf64-x86-64")

SECTIONS
{
    /* Kernel loads at 1MB physical address (standard multiboot) */
    . = 0x100000;
    _kernel_start = .;
    
    /* Multiboot header must be early in kernel */
    .multiboot_header : ALIGN(8)
    {
        *(.multiboot_header)
    }
    
    /* Boot code section */
    .boot : ALIGN(4K)
    {
        *(.text.boot)
        *(.text._start)
    }
    
    /* Main kernel code */
    .text : ALIGN(4K)
    {
        *(.text)
        *(.text.*)
    }
    
    /* Read-only data */
    .rodata : ALIGN(4K)
    {
        *(.rodata)
        *(.rodata.*)
        
        /* Banking configuration section */
        . = ALIGN(16);
        __banking_config_start = .;
        *(.banking.config)
        __banking_config_end = .;
        
        /* Cryptographic constants */
        . = ALIGN(32);
        __crypto_constants_start = .;
        *(.crypto.constants)
        __crypto_constants_end = .;
        
        /* Regulatory compliance data */
        . = ALIGN(16);
        __compliance_data_start = .;
        *(.compliance.data)
        __compliance_data_end = .;
    }
    
    /* Initialized data */
    .data : ALIGN(4K)
    {
        *(.data)
        *(.data.*)
        
        /* Global Descriptor Table */
        . = ALIGN(16);
        __gdt_start = .;
        *(.gdt)
        __gdt_end = .;
        
        /* Interrupt Descriptor Table */
        . = ALIGN(16);
        __idt_start = .;
        *(.idt)
        __idt_end = .;
        
        /* Page tables */
        . = ALIGN(4096);
        __page_tables_start = .;
        *(.page_tables)
        __page_tables_end = .;
    }
    _data_end = .;
    
    /* Uninitialized data */
    .bss : ALIGN(4K)
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        
        /* Bootstrap stack */
        . = ALIGN(16);
        __bootstrap_stack_bottom = .;
        . += 0x4000;                    # 16KB bootstrap stack
        __bootstrap_stack_top = .;
        
        /* Banking kernel stacks */
        . = ALIGN(4096);
        __kernel_stack_bottom = .;
        . += 0x10000;                   # 64KB main kernel stack
        __kernel_stack_top = .;
        
        . = ALIGN(4096);
        __interrupt_stack_bottom = .;
        . += 0x8000;                    # 32KB interrupt stack
        __interrupt_stack_top = .;
        
        __bss_end = .;
    }
    
    /* Banking-specific memory regions */
    
    /* Secure transaction processing buffers */
    .transaction_memory : ALIGN(4K)
    {
        __transaction_memory_start = .;
        . += 0x200000;                  # 2MB for transaction buffers
        __transaction_memory_end = .;
    }
    
    /* Cryptographic working memory (isolated) */
    .crypto_workspace : ALIGN(4K)
    {
        __crypto_workspace_start = .;
        . += 0x400000;                  # 4MB for crypto operations
        __crypto_workspace_end = .;
    }
    
    /* Hardware Security Module communication buffers */
    .hsm_buffers : ALIGN(4K)
    {
        __hsm_buffers_start = .;
        . += 0x20000;                   # 128KB for HSM communication
        __hsm_buffers_end = .;
    }
    
    /* Compliance and audit log storage */
    .audit_storage : ALIGN(4K)
    {
        __audit_storage_start = .;
        . += 0x100000;                  # 1MB for audit logs
        __audit_storage_end = .;
    }
    
    /* Regulatory reporting buffers */
    .regulatory_buffers : ALIGN(4K)
    {
        __regulatory_buffers_start = .;
        . += 0x80000;                   # 512KB for regulatory reports
        __regulatory_buffers_end = .;
    }
    
    /* AI/ML model storage and working memory */
    .ai_memory : ALIGN(4K)
    {
        __ai_memory_start = .;
        . += 0x800000;                  # 8MB for AI/ML operations
        __ai_memory_end = .;
    }
    
    /* Network protocol buffers */
    .network_buffers : ALIGN(4K)
    {
        __network_buffers_start = .;
        . += 0x100000;                  # 1MB for network operations
        __network_buffers_end = .;
    }
    
    /* Dynamic memory heap */
    .heap : ALIGN(4K)
    {
        __heap_start = .;
        . += 0x2000000;                 # 32MB heap for dynamic allocation
        __heap_end = .;
    }
    
    /* High availability cluster communication */
    .cluster_memory : ALIGN(4K)
    {
        __cluster_memory_start = .;
        . += 0x200000;                  # 2MB for cluster operations
        __cluster_memory_end = .;
    }
    
    /* Emergency recovery and backup buffers */
    .recovery_buffers : ALIGN(4K)
    {
        __recovery_buffers_start = .;
        . += 0x400000;                  # 4MB for recovery operations
        __recovery_buffers_end = .;
    }
    
    __kernel_end = .;
    
    /* Discard debug information */
    /DISCARD/ :
    {
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.comment)
        *(.note.*)
        *(.debug*)
    }
}

/* Memory layout assertions for banking compliance */
ASSERT(__kernel_end < 0x40000000, "Kernel image exceeds 1GB limit")
ASSERT((__bss_end - __bss_start) < 0x1000000, "BSS section exceeds 16MB")
ASSERT((__heap_end - __heap_start) == 0x2000000, "Heap size verification failed")

/* Banking-specific memory region validation */
ASSERT(__transaction_memory_start % 4096 == 0, "Transaction memory not page-aligned")
ASSERT(__crypto_workspace_start % 4096 == 0, "Crypto workspace not page-aligned")
ASSERT(__hsm_buffers_start % 4096 == 0, "HSM buffers not page-aligned")
ASSERT(__audit_storage_start % 4096 == 0, "Audit storage not page-aligned")
ASSERT(__regulatory_buffers_start % 4096 == 0, "Regulatory buffers not page-aligned")
ASSERT(__ai_memory_start % 4096 == 0, "AI memory not page-aligned")
ASSERT(__network_buffers_start % 4096 == 0, "Network buffers not page-aligned")
ASSERT(__cluster_memory_start % 4096 == 0, "Cluster memory not page-aligned")
ASSERT(__recovery_buffers_start % 4096 == 0, "Recovery buffers not page-aligned")

/* Security validation - ensure no overlap in critical sections */
ASSERT(__crypto_workspace_end <= __hsm_buffers_start, "Crypto workspace overlaps HSM buffers")
ASSERT(__transaction_memory_end <= __crypto_workspace_start, "Transaction memory overlaps crypto workspace")
ASSERT(__audit_storage_end <= __ai_memory_start, "Audit storage overlaps AI memory")

/* Size validation for banking operations */
ASSERT((__transaction_memory_end - __transaction_memory_start) >= 0x200000, "Transaction memory too small")
ASSERT((__crypto_workspace_end - __crypto_workspace_start) >= 0x400000, "Crypto workspace too small")
ASSERT((__audit_storage_end - __audit_storage_start) >= 0x100000, "Audit storage too small")