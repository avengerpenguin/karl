# --- Secure Host-to-Docker Beeper MCP Tunnel ---
BEEPER_LOCAL_PORT=8500   # The actual port your local Beeper MCP is bound to
BEEPER_PROXY_PORT=18500  # The proxy port host.docker.internal targets

on_unload() {
  echo "Closing Beeper MCP proxy tunnel..."
  if [ -f .bpr_proxy.pid ]; then
    kill $(cat .bpr_proxy.pid) 2>/dev/null
    rm .bpr_proxy.pid
  fi
}

if ! [ -f .bpr_proxy.pid ]; then
  echo "Opening secure Beeper tunnel via socat..."
  if ! command -v socat &> /dev/null; then
    echo "⚠️ Error: 'socat' not found on host Mac. Run 'brew install socat'."
  else
    # Binds strictly to loopback interface to prevent Wi-Fi leaking
    socat TCP-LISTEN:$BEEPER_PROXY_PORT,bind=127.0.0.1,fork TCP:127.0.0.1:$BEEPER_LOCAL_PORT & echo $! >> .bpr_proxy.pid
  fi
fi
