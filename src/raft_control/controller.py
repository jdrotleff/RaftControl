from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping

from .backends import Backend, NIDAQmxBackend, SimulationBackend
from .config import ControllerConfig
from .logging import EventLogger, utc_now
from .models import ActionRecord, ActionRequest, ActionStatus, PreviewResult
from .waveform import build_waveform


class FieldController:
    def __init__(self, config: ControllerConfig, backend: Backend | None = None):
        self.config = config
        self.backend = backend or (SimulationBackend() if config.backend == "simulation" else NIDAQmxBackend(config.channels, config.daq_device, config.current_limit_a))
        self.logger = EventLogger(config.log_path)
        self.enabled = False
        self._records: dict[str, ActionRecord] = {}
        self._active: str | None = None
        self._pending: str | None = None
        self._lock = threading.RLock()

    def _request(self, value):
        return value if isinstance(value, ActionRequest) else ActionRequest.from_mapping(value)

    def preview(self, request) -> PreviewResult:
        request = self._request(request)
        return PreviewResult(request, build_waveform(request, self.config), self.backend.name, self.config.channel_mapping)

    def enable(self):
        with self._lock:
            self.enabled = True
            self.logger.write("enabled")

    def disable(self):
        self.stop()
        with self._lock:
            self.enabled = False
            self.logger.write("disabled")

    def send(self, request) -> ActionRecord:
        action_id = uuid.uuid4().hex
        try:
            request = self._request(request)
            waveform = build_waveform(request, self.config)
        except Exception as exc:
            record = ActionRecord(action_id, request if isinstance(request, ActionRequest) else ActionRequest(0, 0, 0, 0, 0, 0, 1), ActionStatus.REJECTED, self.backend.name, self.config.channel_mapping, self.config.calibration_version, error=str(exc))
            self._store_event(record, ActionStatus.REJECTED)
            return record
        with self._lock:
            if not self.enabled:
                record = self._record(action_id, request, waveform, ActionStatus.REJECTED, "controller is disabled")
                self._store_event(record, ActionStatus.REJECTED)
                return record
            if self._pending is not None:
                record = self._record(action_id, request, waveform, ActionStatus.REJECTED, "one replacement action is already queued")
                self._store_event(record, ActionStatus.REJECTED)
                return record
            record = self._record(action_id, request, waveform, ActionStatus.QUEUED)
            self._records[action_id] = record
            self._emit(record, ActionStatus.VALIDATED)
            if self._active is None:
                self._active = action_id
                self._emit(record, ActionStatus.QUEUED)
                self.backend.start(waveform, lambda: self._started(action_id), lambda: self._duration(action_id), lambda e: self._done(action_id, e))
            else:
                active = self._records[self._active]
                if active.duration_elapsed:
                    self._emit(record, ActionStatus.QUEUED)
                    self._replace_active(action_id)
                else:
                    self._pending = action_id
                    self._emit(record, ActionStatus.QUEUED)
            return record

    def status(self, action_id: str) -> ActionRecord:
        with self._lock:
            return self._records[action_id]

    def stop(self, action_id: str | None = None):
        with self._lock:
            target = action_id or self._active
            self.backend.stop()
            if target and target in self._records:
                record = self._records[target]
                if record.status not in (ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.REJECTED, ActionStatus.STOPPED):
                    self._emit(record, ActionStatus.STOPPED)
            self._active = None
            self._pending = None

    def close(self):
        self.disable()
        self.backend.close()

    def _record(self, action_id, request, waveform, status, error=None):
        values = waveform.clipped_currents_a
        return ActionRecord(action_id, request, status, self.backend.name, self.config.channel_mapping, waveform.calibration_version, values.copy(), values.copy(), bool(waveform.clipped.any()), values.min(axis=1).tolist(), values.max(axis=1).tolist(), error=error)

    def _store_event(self, record, status):
        with self._lock:
            self._records[record.action_id] = record
            self._emit(record, status)

    def _emit(self, record, status):
        record.status = status
        record.timestamps[status.value] = utc_now()
        self.logger.write(status.value, action=record.as_dict())

    def _started(self, action_id):
        with self._lock:
            self._emit(self._records[action_id], ActionStatus.STARTED)

    def _duration(self, action_id):
        with self._lock:
            record = self._records[action_id]
            record.duration_elapsed = True
            record.tail_active = True
            self._emit(record, ActionStatus.DURATION_ELAPSED)
            self.logger.write("tail_started", action_id=action_id)
            if self._pending is not None:
                self._replace_active(self._pending)
                self._pending = None

    def _replace_active(self, action_id):
        old_id = self._active
        if old_id is None:
            return
        old = self._records[old_id]
        new = self._records[action_id]
        old.tail_active = False
        old.replacement_action_id = action_id
        self._emit(old, ActionStatus.COMPLETED)
        self._emit(old, ActionStatus.REPLACED)
        self._active = action_id
        new_waveform = build_waveform(new.request, self.config)
        self.backend.replace(new_waveform, lambda: self._started(action_id), lambda: self._duration(action_id), lambda e: self._done(action_id, e))

    def _done(self, action_id, error):
        with self._lock:
            record = self._records[action_id]
            record.tail_active = False
            if error and error != "stopped":
                record.error = error
                self._emit(record, ActionStatus.FAILED)
            elif record.status != ActionStatus.STOPPED:
                self._emit(record, ActionStatus.COMPLETED)
            if self._active == action_id:
                self._active = None
                pending = self._pending
                self._pending = None
                if pending is not None:
                    self._replace_active(pending)
