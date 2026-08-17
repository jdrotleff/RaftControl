from __future__ import annotations

import json
import struct


def encode_message(message: dict) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


class FrameDecoder:
    def __init__(self, max_payload: int = 1_000_000):
        self.buffer = bytearray()
        self.max_payload = max_payload

    def feed(self, data: bytes) -> list[dict]:
        self.buffer.extend(data)
        messages = []
        while len(self.buffer) >= 4:
            length = struct.unpack(">I", self.buffer[:4])[0]
            if length > self.max_payload:
                raise ValueError("protocol payload too large")
            if len(self.buffer) < 4 + length:
                break
            payload = bytes(self.buffer[4:4 + length])
            del self.buffer[:4 + length]
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("protocol payload must be a JSON object")
            messages.append(value)
        return messages

