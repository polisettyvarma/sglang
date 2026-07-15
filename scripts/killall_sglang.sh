#!/bin/bash

# DEPRECATED: This script will be migrated to python/sglang/cli/killall.py.
# CI mode is already handled there. This script remains for local/non-CI usage.
#
# TODO: Migrate remaining modes (rocm, xpu, xpus, all, gpus) to killall.py and remove this file.
#
# Usage:
#   ./killall_sglang.sh              - Kill SGLang processes only (NVIDIA mode)
#   ./killall_sglang.sh rocm         - Kill SGLang processes only (ROCm mode)
#   ./killall_sglang.sh xpu          - Kill SGLang processes only (Intel XPU mode)
#   ./killall_sglang.sh xpu all      - Kill all Intel XPU processes
#   ./killall_sglang.sh xpus 0,1,2,3 - Kill all processes on specific XPU devices
#   ./killall_sglang.sh all          - Kill all GPU processes (NVIDIA mode)
#   ./killall_sglang.sh gpus 0,1,2,3 - Kill all processes on specific GPUs

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat <<'EOF'
Usage: ./killall_sglang.sh [MODE] [ARG]

Modes:
  (none)            Kill SGLang processes only (NVIDIA/CUDA mode)
  all               Kill SGLang processes + all NVIDIA GPU processes
  gpus <ids>        Kill all processes on specific NVIDIA GPUs
                      <ids>: comma-separated GPU indices, e.g. 0,1,2,3
  rocm              Kill SGLang processes only (AMD ROCm mode)
  xpu               Kill SGLang processes only (Intel XPU mode)
  xpu all           Kill SGLang processes + all Intel XPU processes
  xpus <ids>        Kill all processes on specific Intel XPU devices
                      <ids>: comma-separated XPU device indices, e.g. 0,1,2,3
                      (maps to /dev/dri/renderD128, renderD129, ...)

Options:
  -h, --help        Show this help message and exit

Examples:
  ./killall_sglang.sh
  ./killall_sglang.sh all
  ./killall_sglang.sh gpus 0,1
  ./killall_sglang.sh rocm
  ./killall_sglang.sh xpu
  ./killall_sglang.sh xpu all
  ./killall_sglang.sh xpus 0,1

Note: For CI usage, prefer python/sglang/cli/killall.py (handles NVIDIA and XPU).
EOF
    exit 0

elif [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

elif [ "$1" = "xpu" ]; then
    echo "Running in Intel XPU mode"

    # Show current XPU status
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi discovery
    else
        echo "xpu-smi not found; listing /dev/dri/ instead:"
        ls /dev/dri/ 2>/dev/null || echo "No /dev/dri/ devices found"
    fi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # Kill all XPU processes if "all" argument is provided
    if [ "$2" = "all" ]; then
        lsof /dev/dri/renderD* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null
    fi

    # Show XPU status after clean up
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi discovery
    fi

elif [ "$1" = "xpus" ] && [ -n "$2" ]; then
    # Kill all processes on specific XPU devices only
    # renderD128 = XPU device 0, renderD129 = device 1, etc.
    echo "Killing all processes on XPU devices: $2"

    # Show current XPU status
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi discovery
    else
        echo "xpu-smi not found; listing /dev/dri/ instead:"
        ls /dev/dri/ 2>/dev/null || echo "No /dev/dri/ devices found"
    fi

    # Build render-node list from device indices (e.g., "0,1,2,3" -> "/dev/dri/renderD128 /dev/dri/renderD129 ...")
    devices=$(echo "$2" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk '{printf "/dev/dri/renderD%d ", $1 + 128}')
    echo "Targeting devices: $devices"

    # Kill all processes using the specified XPU render nodes
    [ -n "$devices" ] && lsof $devices 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null

    # Show XPU status after clean up
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi discovery
    fi

elif [ "$1" = "gpus" ] && [ -n "$2" ]; then
    # Kill all processes on specific GPUs only
    echo "Killing all processes on GPUs: $2"

    # Show current GPU status
    nvidia-smi

    # Build device file list from GPU IDs (e.g., "0,1,2,3" -> "/dev/nvidia0 /dev/nvidia1 ...")
    devices=$(echo "$2" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's|^|/dev/nvidia|' | tr '\n' ' ')
    echo "Targeting devices: $devices"

    # Kill all processes using specified GPU devices
    [ -n "$devices" ] && lsof $devices 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null

    # Show GPU status after clean up
    nvidia-smi

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # Clean all GPU processes if "all" argument is provided
    if [ "$1" = "all" ]; then
        # Check if sudo is available
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y lsof
        else
            apt-get update
            apt-get install -y lsof
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    # Show GPU status after clean up
    nvidia-smi
fi
