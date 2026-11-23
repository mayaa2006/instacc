"""One-time pairing codes: scan a QR on the PC and the phone is signed in, no password typing.

The code travels in the URL *fragment* (#pair=...), so it never reaches server logs or proxies,
works once, and expires after 10 minutes.
"""
from __future__ import annotations

import io
import secrets
import time

TTL_S = 600
MAX_FAILS_PER_MIN = 10


class PairingError(ValueError):
    pass


class Pairing:
    def __init__(self, api_token: str | None, port: int = 8000, scheme: str = "http"):
        self.api_token = api_token or ""
        self.port = port
        self.scheme = scheme
        self._codes: dict[str, float] = {}
        self._fails: list[float] = []

    def links(self, code: str | None = None) -> list[dict]:
        """Addresses a phone can open, best first. With a code, each link signs the phone in."""
        from agent.net import phone_addresses

        frag = f"#pair={code}" if code else ""
        out = [{"label": "Home Wi-Fi" if a["kind"] == "wifi" else "Tailscale (anywhere)",
                "url": f"{self.scheme}://{a['ip']}:{self.port}/{frag}"} for a in phone_addresses()]
        return out

    def pc_link(self, code: str | None = None) -> str:
        frag = f"#pair={code}" if code else ""
        return f"{self.scheme}://127.0.0.1:{self.port}/{frag}"

    def new_code(self) -> tuple[str, float]:
        now = time.time()
        self._codes = {c: exp for c, exp in self._codes.items() if exp > now}
        code = secrets.token_urlsafe(9)
        self._codes[code] = now + TTL_S
        return code, now + TTL_S

    def claim(self, code: str) -> str:
        now = time.time()
        self._fails = [t for t in self._fails if t > now - 60]
        if len(self._fails) >= MAX_FAILS_PER_MIN:
            raise PairingError("Too many attempts. Wait a minute and try again.")
        exp = self._codes.pop(code, None)
        if exp is None or exp < now:
            self._fails.append(now)
            raise PairingError("This link has expired or was already used. Make a new one on your PC.")
        return self.api_token


def qr_svg(data: str) -> bytes:
    import qrcode
    import qrcode.image.svg

    q = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(data)
    q.make(fit=True)
    buf = io.BytesIO()
    q.make_image(image_factory=qrcode.image.svg.SvgPathFillImage).save(buf)
    return buf.getvalue()


def qr_terminal(data: str) -> str:
    import qrcode

    q = qrcode.QRCode(border=1)
    q.add_data(data)
    q.make(fit=True)
    out = io.StringIO()
    q.print_ascii(out=out, invert=True)
    return out.getvalue()
