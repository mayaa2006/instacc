"""Find the addresses a phone can use to reach this PC (no packets are sent)."""
from __future__ import annotations

import ipaddress
import socket


def _kind(ip: str) -> str | None:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if a.version != 4 or a.is_loopback or a.is_link_local:
        return None
    if a in ipaddress.ip_network("100.64.0.0/10"):
        return "tailscale"
    if a.is_private:
        return "wifi"
    return None


def phone_addresses() -> list[dict]:
    """[{ip, kind}] with home-network (wifi) addresses first, then Tailscale."""
    found: dict[str, str] = {}
    # UDP "connect" picks the outgoing interface without sending anything
    for probe in ("10.255.255.255", "192.168.255.255", "100.100.100.100"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 1))
                ip = s.getsockname()[0]
                if (k := _kind(ip)):
                    found.setdefault(ip, k)
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if (k := _kind(ip)):
                found.setdefault(ip, k)
    except OSError:
        pass
    order = {"wifi": 0, "tailscale": 1}
    return sorted(({"ip": ip, "kind": k} for ip, k in found.items()), key=lambda d: order[d["kind"]])
