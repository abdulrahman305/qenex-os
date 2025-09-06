# Multiboot2 Header for QENEX Banking OS
# Compliant with Multiboot2 specification for real hardware booting

.section .multiboot_header
.align 8

# Multiboot2 header magic numbers and structure
multiboot_header_start:
    .long 0xe85250d6                    # Multiboot2 magic
    .long 0                             # Architecture: i386
    .long multiboot_header_end - multiboot_header_start
    .long -(0xe85250d6 + 0 + (multiboot_header_end - multiboot_header_start))

# Information request tag
information_request_tag_start:
    .short 1                            # Type: information request
    .short 0                            # Flags
    .long information_request_tag_end - information_request_tag_start
    .long 1                             # Basic memory information
    .long 4                             # Boot device info
    .long 6                             # Memory map
    .long 8                             # VBE info
    .long 9                             # Framebuffer info
information_request_tag_end:

# Address tag for loading
address_tag_start:
    .short 2                            # Type: address
    .short 0                            # Flags
    .long address_tag_end - address_tag_start
    .long multiboot_header_start        # Header address
    .long _kernel_start                 # Load address
    .long _data_end                     # Load end address
    .long _bss_end                      # BSS end address
address_tag_end:

# Entry address tag
entry_address_tag_start:
    .short 3                            # Type: entry address
    .short 0                            # Flags
    .long entry_address_tag_end - entry_address_tag_start
    .long kernel_main                   # Entry point
entry_address_tag_end:

# Console flags tag for early output
console_flags_tag_start:
    .short 4                            # Type: console flags
    .short 0                            # Flags
    .long console_flags_tag_end - console_flags_tag_start
    .long 3                             # Console flags (EGA text + console required)
console_flags_tag_end:

# Framebuffer tag for graphics
framebuffer_tag_start:
    .short 5                            # Type: framebuffer
    .short 0                            # Flags
    .long framebuffer_tag_end - framebuffer_tag_start
    .long 1024                          # Width
    .long 768                           # Height
    .long 32                            # Depth (bits per pixel)
framebuffer_tag_end:

# Module alignment tag
module_alignment_tag_start:
    .short 6                            # Type: module alignment
    .short 0                            # Flags
    .long module_alignment_tag_end - module_alignment_tag_start
module_alignment_tag_end:

# EFI boot services tag
efi_bs_tag_start:
    .short 7                            # Type: EFI boot services
    .short 0                            # Flags
    .long efi_bs_tag_end - efi_bs_tag_start
efi_bs_tag_end:

# EFI 32-bit entry point
efi_i386_entry_tag_start:
    .short 8                            # Type: EFI i386 entry
    .short 0                            # Flags
    .long efi_i386_entry_tag_end - efi_i386_entry_tag_start
    .long efi_kernel_main               # EFI entry point
efi_i386_entry_tag_end:

# EFI 64-bit entry point
efi_amd64_entry_tag_start:
    .short 9                            # Type: EFI amd64 entry
    .short 0                            # Flags
    .long efi_amd64_entry_tag_end - efi_amd64_entry_tag_start
    .long efi64_kernel_main             # EFI64 entry point
efi_amd64_entry_tag_end:

# Relocatable tag for ASLR support
relocatable_tag_start:
    .short 10                           # Type: relocatable
    .short 0                            # Flags
    .long relocatable_tag_end - relocatable_tag_start
    .long 0x100000                      # Min address
    .long 0xFFFFFFFF                    # Max address
    .long 0x1000                        # Alignment (4KB)
    .long 1                             # Preference (load at min address)
relocatable_tag_end:

# End tag
.short 0                                # Type: end
.short 0                                # Flags
.long 8                                 # Size

multiboot_header_end: