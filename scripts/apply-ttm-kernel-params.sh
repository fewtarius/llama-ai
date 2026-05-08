#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
# =============================================================================
# Configure GPU Memory (GTT) Allocation for AMD APUs
# =============================================================================
# Supports two methods:
#   1. amd-smi set (runtime, no reboot needed on ROCm 7.2+)
#   2. Kernel parameters via bootloader:
#      - systemd-boot (SteamFork 3.8+): edits loader entry
#      - GRUB (SteamFork 3.7): edits /etc/default/grub + steamfork-grub-update
#
# Usage:
#   sudo ./scripts/apply-ttm-kernel-params.sh [GB]
#   Default: auto-detected based on system RAM
#
# Kernel parameters set (all sizes in MB except ttm which uses 4KB pages):
#   amdgpu.vis_vramlimit  - visible VRAM limit
#   amdgpu.gttsize       - GTT (GPU-accessible system RAM) size
#   ttm.pages_limit       - TTM page pool limit
#   ttm.page_pool_size   - TTM page pool size
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_ROOT/scripts/detect-gpu.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error(){ echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Memory detection
# =============================================================================

# Get physical system RAM from DMI (actual hardware, not kernel-adjusted)
get_physical_ram_mb() {
    local total_mb=0

    # Method 1: dmidecode (most reliable)
    if command -v dmidecode &>/dev/null; then
        local dimm_size
        while read -r dimm_size; do
            [[ "$dimm_size" =~ Size:\ ([0-9]+)\ ([GM]B) ]] || continue
            local size="${BASH_REMATCH[1]}"
            local unit="${BASH_REMATCH[2]}"
            case "$unit" in
                GB) (( total_mb += size * 1024 )) ;;
                MB) (( total_mb += size )) ;;
            esac
        done < <(dmidecode -t memory 2>/dev/null | grep "^\s*Size:" | grep -v "No Module")
    fi

    # Method 2: fallback to /sys/class/drm + estimate
    if (( total_mb == 0 )) && [[ -d /sys/class/drm ]]; then
        local gtt_total vis_vram
        gtt_total=$(cat /sys/class/drm/card0/device/mem_info_gtt_total 2>/dev/null || echo 0)
        vis_vram=$(cat /sys/class/drm/card0/device/mem_info_vis_vram_total 2>/dev/null || echo 0)
        if (( gtt_total > 0 || vis_vram > 0 )); then
            # Sum GPU-accessible memory + buffer for OS
            total_mb=$(( (gtt_total + vis_vram) / 1024 / 1024 + 6144 ))
        fi
    fi

    echo "$total_mb"
}

# Get current GPU memory allocation from sysfs
get_current_gpu_alloc_mb() {
    local gtt vis_vram total
    gtt=$(cat /sys/class/drm/card0/device/mem_info_gtt_total 2>/dev/null || echo 0)
    vis_vram=$(cat /sys/class/drm/card0/device/mem_info_vis_vram_total 2>/dev/null || echo 0)
    total=$((gtt + vis_vram))
    echo $(( total / 1024 / 1024 ))
}

# =============================================================================
# Boot loader detection
# =============================================================================

detect_boot_method() {
    if [[ -d /boot/loader/entries ]] || [[ -d /boot/EFI/steamfork/loader/entries ]]; then
        echo "systemd-boot"
        return 0
    fi

    if [[ -f /etc/default/grub ]] || [[ -f /boot/grub/grub.cfg ]]; then
        echo "grub"
        return 0
    fi

    if command -v steamfork-grub-update &>/dev/null; then
        echo "grub"
        return 0
    fi

    echo "unknown"
}

# =============================================================================
# amd-smi method (runtime, no reboot on ROCm 7.2+)
# =============================================================================

try_amd_smi() {
    local amd_smi_path="$PROJECT_ROOT/deps/bin/amd-smi"
    if [[ ! -x "$amd_smi_path" ]]; then
        amd_smi_path=$(command -v amd-smi 2>/dev/null || true)
    fi

    if [[ -z "$amd_smi_path" ]]; then
        log_warn "amd-smi not found"
        return 1
    fi

    log_info "Setting GTT to ${GTT_SIZE_GB}GB using amd-smi..."
    if "$amd_smi_path" set -G "$GTT_SIZE_GB" 2>&1; then
        log_ok "GTT hint set to ${GTT_SIZE_GB}GB via amd-smi"
        return 0
    else
        log_warn "amd-smi set failed"
        return 1
    fi
}

# =============================================================================
# systemd-boot method (SteamFork 3.8+)
# =============================================================================

apply_systemd_boot() {
    log_info "Using systemd-boot method..."

    local efi_mount="/boot"
    local efi_part
    efi_part=$(findmnt -n -o SOURCE -t vfat /boot 2>/dev/null) \
        || efi_part=$(findmnt -n -o SOURCE -t vfat /boot/efi 2>/dev/null) \
        || efi_part=$(findmnt -n -o SOURCE -t vfat /efi 2>/dev/null) \
        || true

    if [[ -z "$efi_part" ]]; then
        efi_part=$(lsblk -l -n -o NAME,FSTYPE,SIZE,MOUNTPOINT 2>/dev/null \
            | awk '$2=="vfat" && $3+0 < 1024 {print "/dev/"$1; exit}') \
            || true
    fi

    if [[ -z "$efi_part" ]]; then
        log_error "Could not auto-detect EFI partition"
        return 1
    fi

    if ! mountpoint -q "$efi_mount"; then
        mkdir -p "$efi_mount"
        mount "$efi_part" "$efi_mount"
        trap "umount $efi_mount 2>/dev/null || true" EXIT
    fi

    local loader_entry=""
    for dir in "$efi_mount/loader/entries" "$efi_mount/EFI/steamfork/loader/entries"; do
        [[ -d "$dir" ]] || continue
        for f in "$dir"/*.conf; do
            [[ -f "$f" ]] || continue
            local name=$(basename "$f")
            [[ "$name" =~ fallback|previous|verbose ]] && continue
            loader_entry="$f"
            break
        done
        [[ -n "$loader_entry" ]] && break
    done

    if [[ -z "$loader_entry" || ! -f "$loader_entry" ]]; then
        log_error "Could not find systemd-boot loader entry"
        return 1
    fi

    apply_params_to_entry "$loader_entry"
}

# =============================================================================
# GRUB method (SteamFork 3.7 and others)
# =============================================================================

apply_grub() {
    log_info "Using GRUB method..."

    local grub_default="/etc/default/grub"

    if [[ ! -f "$grub_default" ]]; then
        log_error "/etc/default/grub not found"
        return 1
    fi

    current_cmdline=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$grub_default" | head -1 \
        | sed 's/GRUB_CMDLINE_LINUX_DEFAULT=//;s/"//g')

    log_info "Current GRUB_CMDLINE_LINUX_DEFAULT:"
    echo "  $current_cmdline"

    apply_params_to_grub "$grub_default"

    # Update GRUB
    if command -v steamfork-grub-update &>/dev/null; then
        log_info "Running steamfork-grub-update..."
        steamfork-grub-update
    elif command -v grub-mkconfig &>/dev/null; then
        log_info "Running grub-mkconfig..."
        grub-mkconfig -o /boot/grub/grub.cfg
    else
        log_warn "Could not update GRUB automatically"
        log_info "Run: sudo grub-mkconfig -o /boot/grub/grub.cfg"
    fi
}

# =============================================================================
# Apply parameters to systemd-boot entry
# =============================================================================

apply_params_to_entry() {
    local entry="$1"
    local params="amdgpu.vis_vramlimit=${VIS_VRAM_LIMIT_MB} amdgpu.gttsize=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${SIZE_PAGES}"

    log_info "Target entry: $entry"

    # Backup
    local backup="${entry}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$entry" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    # Remove existing TTM params
    sed -i 's/amdgpu\.vis_vramlimit=[0-9]*//g; s/amdgpu\.gttsize=[0-9]*//g; s/ttm\.pages_limit=[0-9]*//g; s/ttm\.page_pool_size=[0-9]*//g' "$entry"
    # Clean up multiple spaces
    sed -i 's/  */ /g; s/^ //; s/ $//' "$entry"

    # Append new params to options line
    sed -i "s/options /options ${params} /" "$entry"

    verify_and_report "$entry"
}

# =============================================================================
# Apply parameters to GRUB default
# =============================================================================

apply_params_to_grub() {
    local file="$1"
    local params="amdgpu.vis_vramlimit=${VIS_VRAM_LIMIT_MB} amdgpu.gttsize=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${SIZE_PAGES}"

    log_info "Target file: $file"

    # Backup
    local backup="${file}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$file" "$backup"
    log_info "Backed up to: $(basename "$backup")"

    # Remove existing TTM params
    sed -i 's/amdgpu\.vis_vramlimit=[0-9]*//g; s/amdgpu\.gttsize=[0-9]*//g; s/ttm\.pages_limit=[0-9]*//g; s/ttm\.page_pool_size=[0-9]*//g' "$file"
    # Clean up
    sed -i 's/  */ /g; s/^ //; s/ $//' "$file"

    # Update GRUB_CMDLINE_LINUX_DEFAULT
    local new_cmdline="${params} ${current_cmdline}"
    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${new_cmdline}\"|" "$file"

    verify_and_report "$file"
}

# =============================================================================
# Verify and report
# =============================================================================

verify_and_report() {
    local file="$1"

    if grep -q "amdgpu.gttsize=${SIZE_MB}" "$file" 2>/dev/null && grep -q "amdgpu.vis_vramlimit=${VIS_VRAM_LIMIT_MB}" "$file" 2>/dev/null; then
        log_ok "Parameters applied:"
        grep -oE "amdgpu.vis_vramlimit=[0-9]+|amdgpu.gttsize=[0-9]+|ttm.pages_limit=[0-9]+|ttm.page_pool_size=[0-9]+" "$file" | while read -r line; do
            echo "  $line"
        done
        echo ""
        log_warn "Reboot required for changes to take effect"
        log_warn "Run: sudo reboot"
        return 0
    else
        log_error "Failed to apply parameters"
        log_error "Restoring backup..."
        cp "${file}.bak-"* "$file" 2>/dev/null || true
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (sudo)."
    exit 1
fi

# Get requested size (or auto-detect)
GTT_SIZE_GB="${1:-}"

# Detect system RAM
PHYSICAL_RAM_MB=$(get_physical_ram_mb)
CURRENT_GPU_ALLOC_MB=$(get_current_gpu_alloc_mb)

log_info "Physical system RAM: $((PHYSICAL_RAM_MB / 1024))GB"
log_info "Current GPU allocation: ${CURRENT_GPU_ALLOC_MB}MB"

if [[ -z "$GTT_SIZE_GB" ]]; then
    # Default: full physical RAM minus small OS reserve
    local os_reserve_mb=2048
    if (( PHYSICAL_RAM_MB > 16384 )); then
        os_reserve_mb=4096
    fi
    GTT_SIZE_GB=$(( (PHYSICAL_RAM_MB - os_reserve_mb) / 1024 ))
fi

# Calculate parameters
# vis_vramlimit: cap visible VRAM at 6GB (firmware allocation) to prevent double-counting
# gttsize: the actual GTT size you want from system RAM
VIS_VRAM_LIMIT_MB=6144  # 6GB cap for firmware VRAM
SIZE_MB=$((GTT_SIZE_GB * 1024))
SIZE_PAGES=$((GTT_SIZE_GB * 1024 * 1024 / 4))  # 4KB pages

log_info "Target GPU memory: ${GTT_SIZE_GB}GB GTT + ${VIS_VRAM_LIMIT_MB}MB vis = $((GTT_SIZE_GB + VIS_VRAM_LIMIT_MB/1024))GB total"
log_info "Parameters:"
echo "  amdgpu.vis_vramlimit=${VIS_VRAM_LIMIT_MB} (cap firmware allocation)"
echo "  amdgpu.gttsize=${SIZE_MB}"
echo "  ttm.pages_limit=${SIZE_PAGES}"
echo "  ttm.page_pool_size=${SIZE_PAGES}"
echo ""

# Try amd-smi as a hint (does not persist across reboots on its own)
try_amd_smi || true

# Always write kernel params to bootloader - this is what actually persists
BOOT_METHOD=$(detect_boot_method)
log_info "Detected boot method: $BOOT_METHOD"

case "$BOOT_METHOD" in
    systemd-boot)
        apply_systemd_boot
        ;;
    grub)
        apply_grub
        ;;
    *)
        log_error "Unknown boot method"
        log_info "Set parameters manually:"
        echo "  amdgpu.vis_vramlimit=${SIZE_MB} amdgpu.gttsize=${SIZE_MB} ttm.pages_limit=${SIZE_PAGES} ttm.page_pool_size=${SIZE_PAGES}"
        exit 1
        ;;
esac