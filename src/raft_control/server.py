from __future__ import annotations

import argparse
import json
import socket
import threading
import time

from .config import load_config
from .controller import FieldController
from .protocol import FrameDecoder, encode_message


class RaftControlServer:
    def __init__(self, controller, host: str, port: int):
        self.controller, self.host, self.port = controller, host, port
        self._stop = threading.Event()
        self._server = None
        self._client_lock = threading.Lock()
        self._last_heartbeat: float | None = None

    def serve_forever(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(4)
        print(f"RaftControl listening on {self.host}:{self.port}", flush=True)
        threading.Thread(target=self._watchdog, name="raft-control-watchdog", daemon=True).start()
        while not self._stop.is_set():
            self._server.settimeout(1.0)
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def close(self):
        self._stop.set()
        if self._server:
            self._server.close()
        self.controller.close()

    def _client(self, conn):
        decoder = FrameDecoder()
        controls_hardware = False
        try:
            with conn:
                while not self._stop.is_set():
                    data = conn.recv(65536)
                    if not data:
                        return
                    for request in decoder.feed(data):
                        if request.get("type") in {"enable", "disable", "send", "stop", "heartbeat"}:
                            controls_hardware = True
                        try:
                            response = self._dispatch(request)
                        except Exception as exc:
                            response = {"type": "error", "request_id": request.get("request_id"), "error": str(exc)}
                        conn.sendall(encode_message(response))
        finally:
            # A read-only viewer must never stop the experiment when it exits.
            if controls_hardware:
                with self._client_lock:
                    self._last_heartbeat = None
                self.controller.stop()
                self.controller.disable()

    def _watchdog(self):
        while not self._stop.wait(0.25):
            with self._client_lock:
                last = self._last_heartbeat
            if last is not None and time.monotonic() - last > self.controller.config.heartbeat_timeout_s:
                self.controller.stop()
                self.controller.disable()
                with self._client_lock:
                    self._last_heartbeat = None

    def _dispatch(self, request):
        kind = request.get("type")
        request_id = request.get("request_id")
        if kind == "hello":
            return {"type": "hello_ack", "request_id": request_id, "protocol": "raft-control", "version": 1}
        if kind == "heartbeat":
            with self._client_lock:
                self._last_heartbeat = time.monotonic()
            return {"type": "heartbeat_ack", "request_id": request_id}
        if kind == "enable":
            with self._client_lock:
                self._last_heartbeat = time.monotonic()
            self.controller.enable()
            return {"type": "ack", "request_id": request_id}
        if kind == "disable":
            self.controller.disable()
            return {"type": "ack", "request_id": request_id}
        if kind == "send":
            record = self.controller.send(request["action"])
            return {"type": "action_ack", "request_id": request_id, "action": record.summary()}
        if kind == "status":
            return {"type": "status", "request_id": request_id, "action": self.controller.status(request["action_id"]).summary()}
        if kind == "recent":
            with self.controller._lock:
                records = list(self.controller._records.values())[-20:]
            actions = []
            for record in records:
                item = record.as_dict()
                item.pop("calculated_currents_a", None)
                item.pop("transmitted_currents_a", None)
                actions.append(item)
            return {"type": "recent", "request_id": request_id, "actions": actions}
        if kind == "stop":
            self.controller.stop(request.get("action_id"))
            return {"type": "ack", "request_id": request_id}
        raise ValueError(f"unknown request type: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    controller = FieldController(config)
    server = RaftControlServer(controller, config.bind_address, config.python_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
