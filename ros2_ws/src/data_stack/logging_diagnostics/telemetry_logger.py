# src/aimbot_data/telemetry_logger.py
import csv
import time
import queue
import threading
from typing import List, Any

class AsyncTelemetryLogger:
    def __init__(self, output_path: str = "telemetry_log.csv"):
        self.queue: queue.Queue = queue.Queue(maxsize=10000)
        self.output_path = output_path
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def record(self, target_id: str, dx: float, dy: float, range_m: float):
        ts = time.time()
        try:
            self.queue.put_nowait([ts, target_id, dx, dy, range_m])
        except queue.Full:
            pass  # Drop oldest or alert downstream

    def _worker_loop(self):
        with open(self.output_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Target_ID', 'Offset_X', 'Offset_Y', 'Range_Meters'])
            while self.running or not self.queue.empty():
                try:
                    entry = self.queue.get(timeout=0.05)
                    writer.writerow(entry)
                except queue.Empty:
                    continue

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)