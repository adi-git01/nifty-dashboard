#!/usr/bin/env bash
# One-time setup of the fixed-IP exit on a fresh Ubuntu/Debian server. Run as root:
#     curl -fsSL https://raw.githubusercontent.com/adi-git01/nifty-dashboard/main/deploy/order-proxy/setup.sh | sudo bash
# It installs proxy.py as a service that starts on boot, makes a random password, and prints the two
# things you need: the IP address to register with Dhan, and the line to paste into the app's secrets.
set -euo pipefail
PORT="${PROXY_PORT:-8443}"
DIR=/opt/order-proxy
RAW=https://raw.githubusercontent.com/adi-git01/nifty-dashboard/main/deploy/order-proxy/proxy.py

command -v python3 >/dev/null || { apt-get update -y && apt-get install -y python3; }
mkdir -p "$DIR"
if [ -f "$(dirname "${BASH_SOURCE[0]:-/nonexistent}")/proxy.py" ]; then
  cp "$(dirname "${BASH_SOURCE[0]}")/proxy.py" "$DIR/proxy.py"
else
  curl -fsSL "$RAW" -o "$DIR/proxy.py"
fi
if [ ! -f "$DIR/env" ]; then
  PASS="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
  printf 'PROXY_USER=dash\nPROXY_PASS=%s\nPROXY_PORT=%s\n' "$PASS" "$PORT" > "$DIR/env"
  chmod 600 "$DIR/env"
fi
. "$DIR/env"

cat > /etc/systemd/system/order-proxy.service <<UNIT
[Unit]
Description=Fixed-IP exit for dashboard orders
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=$DIR/env
ExecStart=/usr/bin/python3 $DIR/proxy.py
Restart=always
RestartSec=3
DynamicUser=yes
NoNewPrivileges=yes
ProtectSystem=strict

[Install]
WantedBy=multi-user.target
UNIT
chmod 644 "$DIR/proxy.py"; chmod 755 "$DIR"
systemctl daemon-reload
systemctl enable --now order-proxy.service
command -v ufw >/dev/null && ufw status | grep -q "Status: active" && ufw allow "$PORT/tcp" || true

IP="$(curl -fsS https://api.ipify.org || echo '<this server public IP>')"
echo
echo "=============================================================="
echo " 1. Register this IP with Dhan:   $IP"
echo "    web.dhan.co -> Profile -> DhanHQ Trading APIs -> Add IP"
echo
echo " 2. Add these lines under [dhan] in the Streamlit app's secrets:"
echo "    orders_enabled = true"
echo "    order_proxy = \"http://$PROXY_USER:$PROXY_PASS@$IP:$PORT\""
echo
echo " 3. Open port $PORT (TCP) in the cloud provider's firewall."
echo "=============================================================="
