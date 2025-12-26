#!/bin/bash
# Мониторинг GPU во время обучения

echo "🔥 GPU Training Monitor"
echo "======================"
echo ""

while true; do
    clear
    echo "🔥 GPU Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
    echo ""

    # GPU temperature and memory
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status:"
        nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "   GPU %s: %s\n   Temperature: %s°C\n   Memory: %s/%s MB\n   Utilization: %s%%\n\n", $1, $2, $3, $4, $5, $6}'

        # Warning if too hot
        temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        if [ "$temp" -gt 80 ]; then
            echo "⚠️  WARNING: GPU temperature is $temp°C (high!)"
            echo "   Consider cooling or reducing batch size"
            echo ""
        fi
    else
        echo "⚠️  nvidia-smi not available"
        echo ""
    fi

    # Python processes
    echo "🐍 Training Processes:"
    ps aux | grep -E "python.*run_full_combo|python.*train" | grep -v grep | \
    awk '{printf "   PID: %s, CPU: %s%%, Mem: %s%%, Command: %s\n", $2, $3, $4, substr($0, index($0,$11))}'

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo ""

    sleep 5
done
