/* Linker script for QENEX Banking OS Kernel */
/* Designed for x86_64 bare metal execution with banking security requirements */

ENTRY(_start)

SECTIONS
{
    /* Kernel loads at 1MB physical address */
    . = 0x100000;
    
    /* Kernel code section - read-only and executable */
    .text ALIGN(4K) : AT(ADDR(.text))
    {
        *(.text.boot)  /* Boot assembly code */
        *(.text)       /* Main kernel code */
        *(.text.*)
    }
    
    /* Read-only data section */
    .rodata ALIGN(4K) : AT(ADDR(.rodata))
    {
        *(.rodata)
        *(.rodata.*)
        
        /* Banking configuration data */
        __banking_config_start = .;
        *(.banking_config)
        __banking_config_end = .;
        
        /* Security policy data */
        __security_policy_start = .;
        *(.security_policy)
        __security_policy_end = .;
    }
    
    /* Initialized data section */
    .data ALIGN(4K) : AT(ADDR(.data))
    {
        *(.data)
        *(.data.*)
        
        /* Global Descriptor Table */
        __gdt_start = .;
        *(.gdt)
        __gdt_end = .;
        
        /* Interrupt Descriptor Table */
        __idt_start = .;
        *(.idt)
        __idt_end = .;
    }
    
    /* Uninitialized data section */
    .bss ALIGN(4K) : AT(ADDR(.bss))
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    }
    
    /* Stack section for kernel */
    .stack ALIGN(4K) : AT(ADDR(.stack))
    {
        __stack_bottom = .;
        . += 0x10000; /* 64KB stack */
        __stack_top = .;
    }
    
    /* Banking transaction buffers - secure memory region */
    .transaction_buffers ALIGN(4K) : AT(ADDR(.transaction_buffers))
    {
        __transaction_buffers_start = .;
        . += 0x100000; /* 1MB for transaction buffers */
        __transaction_buffers_end = .;
    }
    
    /* Cryptographic working memory - isolated region */
    .crypto_memory ALIGN(4K) : AT(ADDR(.crypto_memory))
    {
        __crypto_memory_start = .;
        . += 0x200000; /* 2MB for cryptographic operations */
        __crypto_memory_end = .;
    }
    
    /* HSM communication buffers */
    .hsm_buffers ALIGN(4K) : AT(ADDR(.hsm_buffers))
    {
        __hsm_buffers_start = .;
        . += 0x10000; /* 64KB for HSM communication */
        __hsm_buffers_end = .;
    }
    
    /* Audit log buffers - tamper-resistant region */
    .audit_logs ALIGN(4K) : AT(ADDR(.audit_logs))
    {
        __audit_logs_start = .;
        . += 0x80000; /* 512KB for audit logs */
        __audit_logs_end = .;
    }
    
    /* Kernel heap for dynamic allocation */
    .heap ALIGN(4K) : AT(ADDR(.heap))
    {
        __heap_start = .;
        . += 0x1000000; /* 16MB heap */
        __heap_end = .;
    }
    
    /* End of kernel image */
    __kernel_end = .;
    
    /* Discard debug sections */
    /DISCARD/ :
    {
        *(.eh_frame)
        *(.comment)
        *(.note.*)
    }
}

/* Memory layout validation */
ASSERT(__kernel_end < 0x80000000, "Kernel too large - exceeds 2GB limit")
ASSERT((__bss_end - __bss_start) < 0x10000000, "BSS section too large")
ASSERT((__heap_end - __heap_start) == 0x1000000, "Heap size mismatch")

/* Banking security validation */
ASSERT(__transaction_buffers_start % 4096 == 0, "Transaction buffers not page-aligned")
ASSERT(__crypto_memory_start % 4096 == 0, "Crypto memory not page-aligned")
ASSERT(__audit_logs_start % 4096 == 0, "Audit logs not page-aligned")