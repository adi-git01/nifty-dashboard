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
command -v openssl >/dev/null || { apt-get update -y && apt-get install -y openssl; }
id -u orderproxy >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin orderproxy
mkdir -p "$DIR"
if [ -f "$(dirname "${BASH_SOURCE[0]:-/nonexistent}")/proxy.py" ]; then
  cp "$(dirname "${BASH_SOURCE[0]}")/proxy.py" "$DIR/proxy.py"
else
  curl -fsSL "$RAW" -o "$DIR/proxy.py"
fi
if [ ! -f "$DIR/cert.pem" ] || [ ! -f "$DIR/key.pem" ]; then
  # Self-signed TLS cert (or swap in Let's Encrypt fullchain.pem / privkey.pem if using a domain)
  openssl req -x509 -newkey rsa:2048 -keyout "$DIR/key.pem" -out "$DIR/cert.pem" \
    -days 825 -nodes -subj "/CN=order-proxy" >/dev/null 2>&1
fi
chown orderproxy:orderproxy "$DIR/cert.pem" "$DIR/key.pem"
chmod 644 "$DIR/cert.pem"; chmod 600 "$DIR/key.pem"
if [ ! -f "$DIR/env" ]; then
  PASS="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
  printf 'PROXY_USER=dash\nPROXY_PASS=%s\nPROXY_PORT=%s\nPROXY_TLS_CERT=%s/cert.pem\nPROXY_TLS_KEY=%s/key.pem\nPROXY_REQUIRE_TLS=1\n' \
    "$PASS" "$PORT" "$DIR" "$DIR" > "$DIR/env"
  chown orderproxy:orderproxy "$DIR/env"
  chmod 600 "$DIR/env"
fi
. "$DIR/env"

cat > /etc/systemd/system/order-proxy.service <<UNIT
[Unit]
Description=Fixed-IP TLS exit for dashboard orders
After=network-online.target
Wants=network-online.target

[Service]
User=orderproxy
Group=orderproxy
EnvironmentFile=$DIR/env
ExecStart=/usr/bin/python3 $DIR/proxy.py
Restart=always
RestartSec=3
NoNewPrivileges=yes
ProtectSystem=strict
ReadOnlyPaths=$DIR

[Install]
WantedBy=multi-user.target
UNIT
chmod 644 "$DIR/proxy.py"; chmod 755 "$DIR"
systemctl daemon-reload
systemctl enable --now order-proxy.service
command -v ufw >/dev/null && ufw status | grep -q "Status: active" && ufw allow "$PORT/tcp" || true
# Oracle's Ubuntu images block new ports in iptables as well as in the cloud firewall
if command -v iptables >/dev/null; then
  iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport "$PORT" -j ACCEPT || true
  command -v netfilter-persistent >/dev/null && netfilter-persistent save >/dev/null 2>&1 || true
fi

IP="$(curl -fsS https://api.ipify.org || echo '<this server public IP>')"
echo
echo "=============================================================="
echo " 1. Register this IP with Dhan:   $IP"
echo "    web.dhan.co -> Profile -> DhanHQ Trading APIs -> Add IP"
echo
echo " 2. Add these lines under [dhan] in the Streamlit app's secrets:"
echo "    orders_enabled = true"
echo "    order_proxy = \"https://$PROXY_USER:$PROXY_PASS@$IP:$PORT\""
echo "    # If using the self-signed cert without a domain, upload $DIR/cert.pem or set REQUESTS_CA_BUNDLE"
echo
echo " 3. Open port $PORT (TCP) in the cloud provider's firewall."
echo "=============================================================="
