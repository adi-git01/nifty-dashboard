#!/usr/bin/env python3
"""
A fixed-IP exit for the dashboard's Dhan orders. Runs on a small server whose public IP address
never changes; the dashboard sends its order calls through it, so Dhan sees that one address.

It is an HTTPS tunnel (HTTP CONNECT) and nothing else:
  - only to the hosts in ALLOW (Dhan's API, and an address-echo service used by the "check" button)
  - only with the right user name and password
  - it cannot read what passes through: the dashboard's connection to Dhan is encrypted end to end,
    so your Dhan token never appears here, in memory or in logs
Standard library only. Settings come from the environment: PROXY_USER, PROXY_PASS, PROXY_PORT (8443).
"""
import asyncio
import base64
import hmac
import os
import sys

ALLOW = {"api.dhan.co", "api.ipify.org"}
USER, PASS = os.environ.get("PROXY_USER", "dash"), os.environ.get("PROXY_PASS", "")
PORT = int(os.environ.get("PROXY_PORT", "8443"))
ALLOW |= {h.strip().lower() for h in os.environ.get("PROXY_ALLOW", "").split(",") if h.strip()}


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def handle(reader, writer):
    async def refuse(code, text, extra=""):
        writer.write(f"HTTP/1.1 {code} {text}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        writer.close()

    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
    except Exception:
        writer.close()
        return
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split()
    hdr = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
    want = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()
    if not PASS or not hmac.compare_digest(hdr.get("proxy-authorization", ""), want):
        return await refuse(407, "Proxy Authentication Required", 'Proxy-Authenticate: Basic realm="orders"\r\n')
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        return await refuse(405, "Only CONNECT")
    host, _, port = parts[1].rpartition(":")
    if host.lower() not in ALLOW or port != os.environ.get("PROXY_TARGET_PORT", "443"):
        return await refuse(403, "Host not allowed")
    try:
        r2, w2 = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 15)
    except Exception:
        return await refuse(502, "Upstream unreachable")
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(pipe(reader, w2), pipe(r2, writer))


async def main():
    if not PASS:
        sys.exit("PROXY_PASS is not set")
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    print(f"order proxy listening on :{PORT}, allowed: {sorted(ALLOW)}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
