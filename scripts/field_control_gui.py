"""Manual RaftControl client and waveform-preview GUI.

The GUI performs previews locally through the simulation backend. Hardware
commands go only through the public RaftControl TCP client API.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from raft_control import FieldController, RaftControlClient
from raft_control.config import load_config
from raft_control.models import ActionRequest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUI_CONFIG = ROOT / "configs" / "gui.json"


def load_gui_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "server_host": str(data.get("server_host", "134.105.56.173")),
        "server_port": int(data.get("server_port", 6340)),
        "preview_config": ROOT / data.get("preview_config", "configs/simulation.json"),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_GUI_CONFIG)
    parser.add_argument("--mode", choices=("controller", "viewer"), default="controller")
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Controller mode: own the hardware controller in this GUI process (default)",
    )
    connection.add_argument(
        "--remote",
        dest="local",
        action="store_false",
        help="Controller mode: connect to a separately running RaftControl server",
    )
    parser.set_defaults(local=True)
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Controller mode: queue actions until the active action duration elapses",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Controller mode: show a button for generating random field actions",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.queue and args.mode != "controller":
        parser.error("--queue is only available in controller mode")
    if args.shuffle and args.mode != "controller":
        parser.error("--shuffle is only available in controller mode")
    gui_config = load_gui_config(args.config)
    host = args.host or gui_config["server_host"]
    port = args.port or gui_config["server_port"]

    preview_controller = FieldController(load_config(gui_config["preview_config"]))
    local_controller = None
    if args.local and args.mode == "controller":
        local_controller = FieldController(load_config(ROOT / "configs" / "windows.json"))
    client: RaftControlClient | None = None
    enabled = False
    active_action_id: str | None = None
    active_action_status: str | None = None
    queued_actions: deque[ActionRequest] = deque()
    viewed_action_id: str | None = None

    root = tk.Tk()
    root.title("RaftControl — Manual Field Control")
    root.geometry("1100x780")
    mode_note_font = tkfont.nametofont("TkDefaultFont").copy()
    mode_note_font.configure(slant="italic")

    fields = ["bx", "by", "fx", "fy", "FX", "FY", "duration"]
    labels = {
        "bx": "B_x (G)", "by": "B_y (G)", "fx": "f_x (Hz)",
        "fy": "f_y (Hz)", "FX": "grad_X B (G/mm)",
        "FY": "grad_Y B (G/mm)", "duration": "Duration (s)",
    }
    defaults = ["50", "50", "25", "25", "0", "0", "3"]
    entries: dict[str, ttk.Entry] = {}

    buttons = ttk.Frame(root, padding=8)
    buttons.pack(side="top", fill="x")
    controls = ttk.Frame(root, padding=8)
    controls.pack(side="top", fill="x")
    for index, (name, default) in enumerate(zip(fields, defaults)):
        ttk.Label(controls, text=labels[name]).grid(row=0, column=index * 2, sticky="w")
        entry = ttk.Entry(controls, width=9)
        entry.insert(0, default)
        entry.grid(row=0, column=index * 2 + 1, padx=3)
        entries[name] = entry

    status = tk.StringVar(value=f"{args.mode.upper()} — DISCONNECTED — target {host}:{port}")
    warning = tk.StringVar(value="No clipping")
    summary = tk.StringVar()
    ttk.Label(root, textvariable=status).pack()
    warning_label = ttk.Label(root, textvariable=warning, padding=4)
    warning_label.pack()
    ttk.Label(root, textvariable=summary).pack()

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def get_action() -> ActionRequest | None:
        try:
            return ActionRequest(**{name: float(entry.get() or 0.0) for name, entry in entries.items()})
        except ValueError as exc:
            messagebox.showerror("Invalid action", str(exc))
            return None

    def update_preview(action: ActionRequest | None = None):
        action = action or get_action()
        if action is None:
            return
        try:
            result = preview_controller.preview(action)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return
        axes[0].clear()
        axes[1].clear()
        colors = ["#962428", "#ff4242", "#0eaa00", "#42aee8"]
        for index, current in enumerate(result.waveform.clipped_currents_a):
            axes[0].plot(result.waveform.time_s, current, color=colors[index], label=f"Coil {index + 1}")
        axes[0].set_title("Predicted coil currents")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Current (A)")
        for index, gradient in enumerate(result.waveform.requested_components[2:4]):
            axes[1].plot(result.waveform.time_s, gradient, color=colors[index], label=["grad_X B", "grad_Y B"][index])
        axes[1].set_title("Requested magnetic-field gradients")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Gradient (G/mm)")
        for axis in axes:
            axis.legend()
            axis.grid(True)
        clipped = bool(result.waveform.clipped.any())
        warning.set("⚠ CURRENT CLIPPED ⚠" if clipped else "No clipping")
        warning_label.configure(background="#75b9ff" if clipped else root.cget("background"))
        data = result.waveform.clipped_currents_a
        summary.set(f"Preview min: {np.min(data, axis=1).round(3).tolist()} A    max: {np.max(data, axis=1).round(3).tolist()} A")
        canvas.draw_idle()

    def shuffle_and_send_action():
        values = {
            "bx": random.uniform(0.0, 80.0),
            "by": random.uniform(0.0, 80.0),
            "fx": random.uniform(0.0, 60.0),
            "fy": random.uniform(0.0, 60.0),
            "FX": 0.0,
            "FY": 0.0,
        }
        for name, value in values.items():
            entries[name].delete(0, tk.END)
            entries[name].insert(0, f"{value:.3f}")
        send_action()

    def connect(show_error: bool = True) -> bool:
        nonlocal client
        if local_controller is not None:
            return True
        if client is not None:
            return True
        try:
            client = RaftControlClient(host, port)
            client.connect()
            status.set(f"CONNECTED / DISABLED — {host}:{port}")
            return True
        except Exception as exc:
            client = None
            if show_error:
                messagebox.showerror("Connection failed", str(exc))
            else:
                status.set(f"{args.mode.upper()} — disconnected: {exc}")
            return False

    def enable_hardware():
        nonlocal enabled
        if local_controller is not None:
            local_controller.enable()
            enabled = True
            status.set("CONTROLLER / ENABLED / LOCAL")
        elif connect():
            try:
                client.enable()
                enabled = True
                status.set(f"CONNECTED / ENABLED — {host}:{port}")
            except Exception as exc:
                messagebox.showerror("Enable failed", str(exc))

    def disable_hardware():
        nonlocal enabled, active_action_id, active_action_status
        queued_actions.clear()
        if local_controller is not None:
            local_controller.disable()
        elif client is not None:
            try:
                client.disable()
            except Exception as exc:
                messagebox.showerror("Disable failed", str(exc))
        enabled = False
        active_action_id = None
        active_action_status = None
        status.set("CONTROLLER / DISABLED / LOCAL" if local_controller is not None else f"CONNECTED / DISABLED — {host}:{port}")

    def transmit_action(action: ActionRequest):
        nonlocal active_action_id, active_action_status
        try:
            response = local_controller.send(action) if local_controller is not None else client.send(action.as_dict())
            record = response.as_dict() if local_controller is not None else response["action"]
            active_action_id = record["action_id"]
            active_action_status = record["status"]
            status.set(f"ACTION {active_action_id[:8]} — {record['status']}")
            if active_action_status in {"queued", "started"}:
                root.after(100, lambda action_id=active_action_id: poll_status(action_id))
        except Exception as exc:
            messagebox.showerror("Send failed", str(exc))

    def send_action():
        action = get_action()
        if action is None:
            return
        update_preview(action)
        if not enabled or not connect():
            if not enabled:
                messagebox.showwarning("Hardware disabled", "Press Enable before sending an action.")
            return
        if args.queue and active_action_status in {"queued", "started"}:
            queued_actions.append(action)
            active = active_action_id[:8] if active_action_id else "none"
            status.set(f"QUEUED LOCALLY ({len(queued_actions)}) - active {active}")
            return
        transmit_action(action)

    def poll_status(action_id: str):
        nonlocal active_action_status
        if action_id != active_action_id or (client is None and local_controller is None):
            return
        try:
            record = local_controller.status(action_id).as_dict() if local_controller is not None else client.status(action_id)["action"]
            active_action_status = record["status"]
            status.set(f"ACTION {active_action_id[:8]} — {record['status']}")
            if active_action_status in {"queued", "started"}:
                root.after(100, lambda: poll_status(action_id))
            elif active_action_status == "duration_elapsed" and args.queue and queued_actions:
                transmit_action(queued_actions.popleft())
        except Exception as exc:
            status.set(f"STATUS ERROR — {exc}")

    def stop_hardware():
        nonlocal active_action_id, active_action_status, enabled
        queued_actions.clear()
        if local_controller is not None:
            local_controller.stop(active_action_id)
            local_controller.disable()
        elif client is not None:
            try:
                client.stop(active_action_id)
                client.disable()
            except Exception as exc:
                messagebox.showerror("Stop failed", str(exc))
        active_action_id = None
        active_action_status = None
        enabled = False
        status.set("STOPPED / DISABLED")

    def keep_remote_controller_alive():
        if args.mode == "controller" and local_controller is None and client is not None:
            try:
                client.heartbeat()
            except Exception as exc:
                status.set(f"HEARTBEAT ERROR — {exc}")
        root.after(1000, keep_remote_controller_alive)

    def on_close():
        try:
            if args.mode == "controller":
                stop_hardware()
        finally:
            if client is not None:
                client.close()
            if local_controller is not None:
                local_controller.close()
            preview_controller.close()
            plt.close(figure)
            root.destroy()

    ttk.Button(buttons, text="Update preview", command=update_preview).pack(side="left", padx=4)
    if args.mode == "controller":
        if args.shuffle:
            ttk.Button(buttons, text="Shuffle and send", command=shuffle_and_send_action).pack(side="left", padx=4)
        ttk.Button(buttons, text="Enable", command=enable_hardware).pack(side="left", padx=4)
        ttk.Button(buttons, text="Disable", command=disable_hardware).pack(side="left", padx=4)
        ttk.Button(buttons, text="Send action", command=send_action).pack(side="left", padx=4)
        ttk.Button(buttons, text="Stop", command=stop_hardware).pack(side="left", padx=4)
        mode_note = (
            "Local controller — actions can be interrupted immediately"
            if local_controller is not None
            else "Remote controller — actions can be interrupted immediately"
        )
        if args.queue:
            location = "Local" if local_controller is not None else "Remote"
            mode_note = f"{location} controller - actions run sequentially"
        ttk.Label(buttons, text=mode_note, font=mode_note_font).pack(side="left", padx=10)
    else:
        ttk.Label(buttons, text="READ-ONLY VIEWER").pack(side="left", padx=12)
        ttk.Label(
            buttons,
            text="Viewer — actions can only be viewed",
            font=mode_note_font,
        ).pack(side="left", padx=10)

        def refresh_viewer():
            nonlocal viewed_action_id
            if connect(show_error=False):
                try:
                    response = client.recent()
                    records = response["actions"]
                    if records:
                        active_id = response.get("active_action_id")
                        latest = next(
                            (record for record in records if record["action_id"] == active_id),
                            records[-1],
                        )
                        status.set(f"VIEWER — latest {latest['action_id'][:8]} — {latest['status']}")
                        controller_state = "ENABLED" if response.get("enabled") else "DISABLED"
                        action_state = "active" if active_id else "latest"
                        status.set(
                            f"VIEWER - {controller_state} - {action_state} "
                            f"{latest['action_id'][:8]} - {latest['status']}"
                        )
                        summary.set("RL actions: " + " | ".join(f"{r['action_id'][:8]}:{r['status']}" for r in records[-5:]))
                        if latest["action_id"] != viewed_action_id:
                            request = latest["request"]
                            for name, entry in entries.items():
                                entry.delete(0, tk.END)
                                entry.insert(0, str(request[name]))
                            viewed_action_id = latest["action_id"]
                            update_preview()
                except Exception as exc:
                    status.set(f"VIEWER ERROR — {exc}")
            root.after(250, refresh_viewer)

        root.after(100, refresh_viewer)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(1000, keep_remote_controller_alive)
    update_preview()
    root.mainloop()


if __name__ == "__main__":
    main()
