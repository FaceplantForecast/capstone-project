# ==================================================================================
# Faceplant Forecast, 2026
# This code creates a GUI to test 10 simulated devices connecting to the GCP server
# and sending fall flags in order to verify the functionality of the website and
# app with multiple connections. Config for each simulated device is stored within
# devices.json
# ==================================================================================

import asyncio
import json
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import websockets


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CONFIG_PATH = "devices.json"
DEFAULT_BASE_GCP_URL = "gcr-ws-482782751069.us-central1.run.app/ws"
INCLUDE_ROLE_QUERY_PARAM = True
GUI_POLL_MS = 100
CONNECT_TIMEOUT_SEC = 8


# ============================================================
# MESSAGE HELPERS
# ============================================================

def make_timestamp(ts: Optional[float] = None) -> str:
    """Return timestamp in the same format as your current server code."""
    if ts is None:
        ts = time.time()
    return time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(ts))


def build_base_message(msg_type: str, device_id: str, account_id: str, payload: dict) -> dict:
    """
    Build a message in the same top-level shape as your uploaded server.py:
    {
        "msg_type": "...",
        "ts_send": "...",
        "payload": {...}
    }
    """
    return {
        "msg_type": msg_type,
        "ts_send": make_timestamp(),
        "device_id": device_id,
        "account_id": account_id,
        "payload": payload,
    }


# ============================================================
# CONFIG / STATE
# ============================================================

@dataclass
class DeviceConfig:
    """Editable configuration for one simulated device."""
    index: int
    name: str
    token: str
    device_id: str
    account_id: str
    base_url: str


@dataclass
class DeviceRuntimeState:
    """Runtime state for one device."""
    connected: bool = False
    connecting: bool = False
    stop_requested: bool = False
    ws: Optional[object] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    thread: Optional[threading.Thread] = None
    last_error: str = ""
    sent_count: int = 0
    recv_count: int = 0


# ============================================================
# JSON CONFIG LOADING
# ============================================================

def load_config(config_path: str) -> tuple[str, list[DeviceConfig]]:
    """
    Load simulator configuration from JSON.

    Expected structure:
    {
      "base_url": "gcr-ws-482782751069.us-central1.run.app/ws",
      "devices": [
        {
          "name": "Device 1",
          "token": "...",
          "device_id": "sim-pi-01",
          "account_id": "account-1"
        }
      ]
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")

    base_url = raw.get("base_url", DEFAULT_BASE_GCP_URL)
    devices_raw = raw.get("devices")

    if not isinstance(devices_raw, list) or len(devices_raw) == 0:
        raise ValueError("Config must contain a non-empty 'devices' array.")

    devices = []
    for i, item in enumerate(devices_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Device entry at index {i} must be a JSON object.")

        name = str(item.get("name", f"Device {i+1}"))
        token = str(item.get("token", "")).strip()
        device_id = str(item.get("device_id", f"sim-pi-{i+1:02d}")).strip()
        account_id = str(item.get("account_id", f"account-{i+1}")).strip()

        if not token:
            raise ValueError(f"Device '{name}' is missing a token.")

        devices.append(DeviceConfig(
            index=i,
            name=name,
            token=token,
            device_id=device_id,
            account_id=account_id,
            base_url=base_url,
        ))

    return base_url, devices


# ============================================================
# SIMULATED DEVICE
# ============================================================

class SimulatedDevice:
    """
    Represents one independently controlled simulated device.

    Each device runs its websocket connection in its own background thread
    with its own asyncio event loop so the GUI remains responsive.
    """

    def __init__(self, cfg: DeviceConfig, ui_queue: queue.Queue):
        self.cfg = cfg
        self.state = DeviceRuntimeState()
        self.ui_queue = ui_queue

    def _push_log(self, message: str) -> None:
        self.ui_queue.put(("log", self.cfg.index, f"[{self.cfg.name} | {self.cfg.device_id}] {message}"))

    def _push_status(self) -> None:
        self.ui_queue.put(("status", self.cfg.index, {
            "connected": self.state.connected,
            "connecting": self.state.connecting,
            "last_error": self.state.last_error,
            "sent_count": self.state.sent_count,
            "recv_count": self.state.recv_count,
        }))

    def websocket_url(self) -> str:
        """
        Build websocket URL like:
            wss://.../ws?role=pi&token=...
        """
        base = f"wss://{self.cfg.base_url}"
        if INCLUDE_ROLE_QUERY_PARAM:
            return f"{base}?role=pi&token={self.cfg.token}"
        return f"{base}?token={self.cfg.token}"

    def build_connection_message(self) -> dict:
        return {
            "type": "status",
            "event": "boot_connected",
            "status": "connected",
            "device": self.cfg.device_id,
            "timestamp": make_timestamp(time.time())
        }

    def build_fall_message(self, probability: float, frame_id: int, event_ts: float) -> dict:
        return build_base_message("fall_event", self.cfg.device_id, self.cfg.account_id, {
            "fall_detected": 1,
            "probability": float(probability),
            "frame_id": int(frame_id),
            "ts_fall": make_timestamp(event_ts),
        })

    def build_system_event(self, event_type: str, message: str) -> dict:
        return build_base_message("system_event", self.cfg.device_id, self.cfg.account_id, {
            "event_type": event_type,
            "message": message,
        })

    def connect(self) -> None:
        """Start background thread and connect if not already running."""
        if self.state.connected or self.state.connecting:
            return

        self.state.stop_requested = False
        self.state.thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name=f"SimDevice-{self.cfg.index}"
        )
        self.state.thread.start()

    def disconnect(self) -> None:
        """Request disconnect from websocket and stop the loop."""
        self.state.stop_requested = True

        if self.state.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.state.loop)
            except Exception as exc:
                self.state.last_error = f"disconnect scheduling error: {exc}"
                self._push_status()

    def send_fall(self, probability: Optional[float] = None) -> None:
        """Send a manual fall flag."""
        if not self.state.connected or self.state.loop is None:
            self._push_log("Cannot send fall flag: device is not connected.")
            return

        if probability is None:
            probability = round(random.uniform(0.80, 0.99), 2)

        frame_id = random.randint(1000, 999999)
        event_ts = time.time()

        msg = self.build_fall_message(probability=probability, frame_id=frame_id, event_ts=event_ts)

        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_send(msg), self.state.loop)
            fut.result(timeout=5)
            self._push_log(f"Sent fall_event: probability={probability}, frame_id={frame_id}")
        except Exception as exc:
            self.state.last_error = f"send fall error: {exc}"
            self._push_log(self.state.last_error)
            self._push_status()

    def send_system_event(self, event_type: str = "manual_test", message: str = "Manual simulator test event") -> None:
        """Send a manual system event."""
        if not self.state.connected or self.state.loop is None:
            self._push_log("Cannot send system_event: device is not connected.")
            return

        msg = self.build_system_event(event_type, message)

        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_send(msg), self.state.loop)
            fut.result(timeout=5)
            self._push_log(f"Sent system_event: {event_type}")
        except Exception as exc:
            self.state.last_error = f"send system_event error: {exc}"
            self._push_log(self.state.last_error)
            self._push_status()

    def _thread_main(self) -> None:
        """Create a dedicated event loop for this device thread."""
        loop = asyncio.new_event_loop()
        self.state.loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connection_task())
        except Exception as exc:
            self.state.last_error = f"thread main error: {exc}"
            self._push_log(self.state.last_error)
        finally:
            self.state.connected = False
            self.state.connecting = False
            self.state.ws = None
            self._push_status()

            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass

            loop.close()
            self.state.loop = None

    async def _connection_task(self) -> None:
        """Open websocket, send register, and listen for incoming commands."""
        self.state.connecting = True
        self.state.last_error = ""
        self._push_status()
        self._push_log(f"Connecting to {self.websocket_url()}")

        try:
            async with websockets.connect(self.websocket_url()) as ws:
                self.state.ws = ws
                self.state.connecting = False
                self.state.connected = True
                self._push_status()
                self._push_log("Connected.")

                register_msg = self.build_connection_message()
                await ws.send(json.dumps(register_msg))
                self.state.sent_count += 1
                self._push_log("Sent register message.")
                self._push_status()

                while not self.state.stop_requested:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=0.25)
                        self.state.recv_count += 1
                        self._handle_incoming(message)
                        self._push_status()
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed as exc:
                        self.state.last_error = f"connection closed: code={exc.code}, reason={exc.reason}"
                        self._push_log(self.state.last_error)
                        break

        except TimeoutError:
            self.state.last_error = "connection timeout"
            self._push_log(self.state.last_error)
        except Exception as exc:
            self.state.last_error = f"connect/listen error: {exc}"
            self._push_log(self.state.last_error)
        finally:
            self.state.connected = False
            self.state.connecting = False
            self.state.ws = None
            self._push_status()
            self._push_log("Disconnected.")

    async def _async_send(self, msg: dict) -> None:
        """Send a JSON message over the open websocket."""
        if self.state.ws is None:
            raise RuntimeError("websocket is not connected")

        await self.state.ws.send(json.dumps(msg))
        self.state.sent_count += 1
        self._push_status()

    async def _async_disconnect(self) -> None:
        """Gracefully close websocket if open."""
        if self.state.ws is not None:
            try:
                await self.state.ws.close()
            except Exception:
                pass

    def _handle_incoming(self, raw_msg: str) -> None:
        """Log incoming messages or commands from the backend."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            self._push_log(f"Received non-JSON message: {raw_msg}")
            return

        if isinstance(data, dict) and "command" in data:
            self._push_log(f"Received command: {data['command']}")
        else:
            self._push_log(f"Received message: {json.dumps(data)}")


# ============================================================
# TKINTER GUI
# ============================================================

class DeviceRow:
    """GUI widgets for one device row."""

    def __init__(self, master, device: SimulatedDevice):
        self.device = device
        self.frame = ttk.Frame(master, padding=4)

        self.name_label = ttk.Label(self.frame, text=device.cfg.name, width=14)
        self.name_label.grid(row=0, column=0, padx=3)

        self.device_id_var = tk.StringVar(value=device.cfg.device_id)
        self.account_id_var = tk.StringVar(value=device.cfg.account_id)

        self.device_id_entry = ttk.Entry(self.frame, textvariable=self.device_id_var, width=16)
        self.account_id_entry = ttk.Entry(self.frame, textvariable=self.account_id_var, width=16)

        self.device_id_entry.grid(row=0, column=1, padx=3)
        self.account_id_entry.grid(row=0, column=2, padx=3)

        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var, width=16)
        self.status_label.grid(row=0, column=3, padx=6)

        self.stats_var = tk.StringVar(value="TX:0 RX:0")
        self.stats_label = ttk.Label(self.frame, textvariable=self.stats_var, width=12)
        self.stats_label.grid(row=0, column=4, padx=6)

        self.connect_btn = ttk.Button(self.frame, text="Connect", command=self.on_connect)
        self.disconnect_btn = ttk.Button(self.frame, text="Disconnect", command=self.on_disconnect)
        self.fall_btn = ttk.Button(self.frame, text="Send Fall", command=self.on_send_fall)
        self.system_btn = ttk.Button(self.frame, text="Send System", command=self.on_send_system)

        self.connect_btn.grid(row=0, column=5, padx=2)
        self.disconnect_btn.grid(row=0, column=6, padx=2)
        self.fall_btn.grid(row=0, column=7, padx=2)
        self.system_btn.grid(row=0, column=8, padx=2)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def sync_config_to_device(self):
        """Copy editable GUI field values into the device config."""
        self.device.cfg.device_id = self.device_id_var.get().strip()
        self.device.cfg.account_id = self.account_id_var.get().strip()

    def on_connect(self):
        self.sync_config_to_device()
        self.device.connect()

    def on_disconnect(self):
        self.device.disconnect()

    def on_send_fall(self):
        self.sync_config_to_device()
        self.device.send_fall()

    def on_send_system(self):
        self.sync_config_to_device()
        self.device.send_system_event()

    def update_status(self, connected: bool, connecting: bool, last_error: str, sent_count: int, recv_count: int):
        if connecting:
            self.status_var.set("Connecting...")
        elif connected:
            self.status_var.set("Connected")
        else:
            self.status_var.set("Disconnected")

        if last_error:
            self.status_var.set(self.status_var.get() + " (!)")
        self.stats_var.set(f"TX:{sent_count} RX:{recv_count}")


class SimulatorApp:
    """Main GUI application."""

    def __init__(self, root: tk.Tk, config_path: str):
        self.root = root
        self.root.title("Faceplant Forecast - Multi Device Simulator")
        self.root.geometry("1350x780")

        self.ui_queue: queue.Queue = queue.Queue()
        self.devices: list[SimulatedDevice] = []
        self.rows: list[DeviceRow] = []

        self.config_path = config_path
        self.base_url, device_configs = load_config(config_path)

        self._build_ui()
        self._create_devices(device_configs)

        self.root.after(GUI_POLL_MS, self.process_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Base URL:").pack(side="left", padx=(0, 4))
        self.base_url_var = tk.StringVar(value=self.base_url)
        self.base_url_entry = ttk.Entry(top, textvariable=self.base_url_var, width=60)
        self.base_url_entry.pack(side="left", padx=(0, 12))

        ttk.Button(top, text="Connect All", command=self.connect_all).pack(side="left", padx=4)
        ttk.Button(top, text="Disconnect All", command=self.disconnect_all).pack(side="left", padx=4)
        ttk.Button(top, text="Send Fall From All", command=self.send_fall_all).pack(side="left", padx=4)
        ttk.Button(top, text="Reload Config", command=self.reload_config).pack(side="left", padx=4)

        header = ttk.Frame(self.root, padding=(10, 0))
        header.pack(fill="x")
        headers = [
            ("Name", 0),
            ("Device ID", 1),
            ("Account ID", 2),
            ("Status", 3),
            ("Stats", 4),
        ]
        for text, col in headers:
            ttk.Label(header, text=text).grid(row=0, column=col, padx=8, sticky="w")

        self.device_frame = ttk.Frame(self.root, padding=10)
        self.device_frame.pack(fill="x")

        log_frame = ttk.LabelFrame(self.root, text="Event Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, wrap="word", height=24)
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _create_devices(self, device_configs: list[DeviceConfig]):
        for i, cfg in enumerate(device_configs):
            cfg.index = i
            dev = SimulatedDevice(cfg, self.ui_queue)
            row = DeviceRow(self.device_frame, dev)
            row.grid(row=i, column=0, sticky="w", pady=2)

            self.devices.append(dev)
            self.rows.append(row)

    def connect_all(self):
        for row in self.rows:
            row.device.cfg.base_url = self.base_url_var.get().strip()
            row.on_connect()

    def disconnect_all(self):
        for dev in self.devices:
            dev.disconnect()

    def send_fall_all(self):
        for row in self.rows:
            row.device.cfg.base_url = self.base_url_var.get().strip()
            row.sync_config_to_device()
            row.device.send_fall()

    def reload_config(self):
        """
        Reload configuration from disk.

        For simplicity, this only reloads if everything is disconnected.
        That avoids replacing active device objects mid-connection.
        """
        if any(dev.state.connected or dev.state.connecting for dev in self.devices):
            messagebox.showwarning("Reload blocked", "Disconnect all devices before reloading config.")
            return

        try:
            self.base_url, device_configs = load_config(self.config_path)
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return

        self.base_url_var.set(self.base_url)

        for row in self.rows:
            row.frame.destroy()

        self.devices.clear()
        self.rows.clear()
        self._create_devices(device_configs)

        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')}  Reloaded config from {self.config_path}\n")
        self.log_text.see("end")

    def process_ui_queue(self):
        """Process log and status updates from device worker threads."""
        try:
            while True:
                item = self.ui_queue.get_nowait()
                kind = item[0]

                if kind == "log":
                    _, dev_index, message = item
                    timestamp = time.strftime("%H:%M:%S")
                    self.log_text.insert("end", f"{timestamp}  {message}\n")
                    self.log_text.see("end")

                elif kind == "status":
                    _, dev_index, data = item
                    self.rows[dev_index].update_status(
                        connected=data["connected"],
                        connecting=data["connecting"],
                        last_error=data["last_error"],
                        sent_count=data["sent_count"],
                        recv_count=data["recv_count"],
                    )
        except queue.Empty:
            pass

        self.root.after(GUI_POLL_MS, self.process_ui_queue)

    def on_close(self):
        """Disconnect all devices before closing GUI."""
        self.disconnect_all()
        self.root.after(500, self.root.destroy)


# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    try:
        app = SimulatorApp(root, DEFAULT_CONFIG_PATH)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()