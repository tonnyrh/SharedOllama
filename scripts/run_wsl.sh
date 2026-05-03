#!/bin/bash
# Start SharedOllama proxy and admin services on WSL
set -e

cd /mnt/c/vscode/SharedOllama
VENV="$HOME/.sharedollama_user/.venv"

# Kill any existing processes
pkill -f "monitor\\.app" || true
pkill -f "monitor\\.admin" || true
sleep 1

# Start proxy
echo "[SharedOllama] Starting proxy on port 11434..."
nohup $VENV/bin/python -m monitor.app > /tmp/proxy.log 2>&1 &
PROXY_PID=$!
echo "[SharedOllama] Proxy PID: $PROXY_PID"

# Start admin
echo "[SharedOllama] Starting admin on port 11444..."
nohup $VENV/bin/python -m monitor.admin > /tmp/admin.log 2>&1 &
ADMIN_PID=$!
echo "[SharedOllama] Admin PID: $ADMIN_PID"

# Wait for services to start
sleep 3

# Test endpoints
echo "[SharedOllama] Testing proxy health..."
if curl -s http://127.0.0.1:11434/health > /dev/null; then
    echo "[SharedOllama] ✓ Proxy responding on 11434"
else
    echo "[SharedOllama] ✗ Proxy not responding"
    cat /tmp/proxy.log
fi

echo "[SharedOllama] Testing admin health..."
if curl -s http://127.0.0.1:11444/monitor/api/admin/state > /dev/null 2>&1; then
    echo "[SharedOllama] ✓ Admin responding on 11444"
else
    echo "[SharedOllama] ✗ Admin not responding"
    cat /tmp/admin.log
fi

echo "[SharedOllama] Services started. Use 'ps aux | grep monitor' to check status."
