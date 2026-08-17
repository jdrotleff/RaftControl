from __future__ import annotations

import socket
import threading
import uuid

from .protocol import FrameDecoder, encode_message


class RaftControlClient:
    """Small dependency-free client usable from Rafts/Linux or another caller."""

    def __init__(self, host: str, port: int = 6340, timeout: float = 5.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.socket: socket.socket | None = None
        self.decoder = FrameDecoder()
        self._lock = threading.Lock()

    def connect(self):
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        return self.request({"type": "hello", "protocol": "raft-control", "version": 1})

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def request(self, message: dict) -> dict:
        if self.socket is None:
            raise RuntimeError("client is not connected")
        message = dict(message)
        message.setdefault("request_id", uuid.uuid4().hex)
        with self._lock:
            self.socket.sendall(encode_message(message))
            while True:
                data = self.socket.recv(65536)
                if not data:
                    raise ConnectionError("RaftControl server closed the connection")
                messages = self.decoder.feed(data)
                if messages:
                    return messages[0]

    def enable(self):
        return self.request({"type": "enable"})

    def disable(self):
        return self.request({"type": "disable"})

    def heartbeat(self):
        return self.request({"type": "heartbeat"})

    def send(self, action: dict) -> dict:
        return self.request({"type": "send", "action": action})

    def status(self, action_id: str) -> dict:
        return self.request({"type": "status", "action_id": action_id})

    def recent(self) -> dict:
        return self.request({"type": "recent"})

    def stop(self, action_id: str | None = None):
        message = {"type": "stop"}
        if action_id is not None:
            message["action_id"] = action_id
        return self.request(message)
