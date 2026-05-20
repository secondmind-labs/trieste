#!/bin/bash
MAX_GPU=0
MAX_MEM=0
PID=$1
while kill -0 $PID 2>/dev/null; do
    LINE=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU=$(echo $LINE | cut -d',' -f1 | tr -d ' ')
    MEM=$(echo $LINE | cut -d',' -f2 | tr -d ' ')
    [ "${GPU:-0}" -gt "$MAX_GPU" ] && MAX_GPU=$GPU
    [ "${MEM:-0}" -gt "$MAX_MEM" ] && MAX_MEM=$MEM
    sleep 1
done
echo "MAX_GPU=$MAX_GPU MAX_MEM=$MAX_MEM"
