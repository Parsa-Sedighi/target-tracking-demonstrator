"""PyQt6 operator interface (FR-05, FR-06, FR-09, ES-10.1).

Works with the in-process Engine or with RemoteEngine (gateway client): both expose
``snapshot()`` and ``command()``. Keyboard: Esc = E-STOP, Space = start/stop,
arrows = jog, H = home, R = record, O = OCR.
"""

from __future__ import annotations

import html
import math
import sys
import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
                             QSplitter, QStatusBar, QVBoxLayout, QWidget)

from .video_widget import STATE_COLORS, VideoWidget

STYLE = """
QWidget { background: #14171c; color: #d8dde3; font-family: 'Segoe UI', 'Ubuntu', sans-serif; font-size: 10pt; }
QGroupBox { border: 1px solid #2a3038; border-radius: 6px; margin-top: 14px; padding: 6px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8fa3b8; font-weight: 600; }
QPushButton { background: #232a33; border: 1px solid #36404c; border-radius: 5px; padding: 6px 10px; }
QPushButton:hover { background: #2c3541; }
QPushButton:pressed { background: #1b2027; }
QPushButton:checked { background: #1f4d33; border-color: #2f8a55; }
QPushButton:disabled { color: #59626d; }
QPushButton#estop { background: #a3161b; border: 2px solid #ff4d52; color: white; font-size: 14pt; font-weight: 800;
                    padding: 12px; }
QPushButton#estop:hover { background: #c41c22; }
QPushButton#start { background: #1d5c37; border-color: #2f8a55; font-weight: 700; }
QComboBox, QDoubleSpinBox { background: #1d232b; border: 1px solid #36404c; border-radius: 4px; padding: 3px 6px; }
QPlainTextEdit { background: #0f1216; border: 1px solid #2a3038; font-family: Consolas, 'DejaVu Sans Mono', monospace;
                 font-size: 9pt; }
QLabel#state { font-size: 18pt; font-weight: 800; padding: 8px; border-radius: 6px; }
QLabel.k { color: #8fa3b8; }
QScrollArea { border: none; }
QStatusBar { background: #0f1216; color: #8fa3b8; }
"""


def _fmt(v: Any, nd: int = 2, unit: str = "") -> str:
    try:
        if v in ("", None):
            return "-"
        x = float(v)
        if math.isnan(x):
            return "-"
        return f"{x:.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


class MainWindow(QMainWindow):
    def __init__(self, engine, fps: float = 60.0):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("AimBot - Target Tracking Demonstrator (GA-EMS / SDSU #23)")
        self.resize(1500, 900)
        self._last_frame_id = -1
        self._last_slow = 0.0
        self._last_events_len = 0
        self._jog_keys: dict[int, tuple[float, float]] = {}
        self._snap: dict = {}

        self.video = VideoWidget(self._designate, self._set_aim)
        panel = self._build_panel()
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setMinimumWidth(440)
        scroll.setMinimumWidth(470)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.video)
        split.addWidget(scroll)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(5, int(1000 / fps)))
        for key, fn in ((Qt.Key.Key_Escape, self._estop), (Qt.Key.Key_H, lambda: self._cmd("home")),
                        (Qt.Key.Key_R, lambda: self._cmd("record")), (Qt.Key.Key_O, lambda: self._cmd("ocr"))):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act.triggered.connect(fn)
            self.addAction(act)

    # ------------------------------------------------------------------ layout
    def _build_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)
        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setObjectName("state")
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_state)
        self.lbl_reason = QLabel("")
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setStyleSheet("color:#8fa3b8; font-size:9pt;")
        v.addWidget(self.lbl_reason)

        row = QHBoxLayout()
        self.btn_start = QPushButton("Start tracking")
        self.btn_start.setObjectName("start")
        self.btn_start.setToolTip("Start a tracking session (Space). The system seeks the selected target class.")
        self.btn_start.clicked.connect(lambda: self._cmd("start"))
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._pause_resume)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(lambda: self._cmd("stop"))
        for b in (self.btn_start, self.btn_pause, self.btn_stop):
            row.addWidget(b)
        v.addLayout(row)
        self.btn_estop = QPushButton("E-STOP  (Esc)")
        self.btn_estop.setObjectName("estop")
        self.btn_estop.setToolTip("Software emergency halt: motion stops, motors off where supported. "
                                  "The hardware E-stop is the independent safety device.")
        self.btn_estop.clicked.connect(self._estop)
        v.addWidget(self.btn_estop)
        self.btn_reset = QPushButton("Reset E-stop / fault")
        self.btn_reset.clicked.connect(lambda: self._cmd("reset"))
        v.addWidget(self.btn_reset)

        g = QGroupBox("Target")
        gl = QGridLayout(g)
        gl.addWidget(QLabel("Seek class"), 0, 0)
        self.cmb_class = QComboBox()
        self.cmb_class.currentTextChanged.connect(lambda t: t and self._cmd("set_target_class", cls=t))
        gl.addWidget(self.cmb_class, 0, 1)
        btn_clear = QPushButton("Release target")
        btn_clear.clicked.connect(lambda: self._cmd("clear_target"))
        gl.addWidget(btn_clear, 1, 0)
        btn_ocr = QPushButton("Read card (OCR)")
        btn_ocr.clicked.connect(lambda: self._cmd("ocr"))
        gl.addWidget(btn_ocr, 1, 1)
        self.lbl_ocr = QLabel("OCR: -")
        self.lbl_ocr.setWordWrap(True)
        gl.addWidget(self.lbl_ocr, 2, 0, 1, 2)
        hint = QLabel("Click a box to lock it, click anywhere / drag a box to track an unclassified region. "
                      "Right-click sets the aim point.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7d8894; font-size:8.5pt;")
        gl.addWidget(hint, 3, 0, 1, 2)
        v.addWidget(g)

        g = QGroupBox("Gimbal")
        gl = QGridLayout(g)
        self.jog_speed = QSlider(Qt.Orientation.Horizontal)
        self.jog_speed.setRange(1, 60)
        self.jog_speed.setValue(15)
        gl.addWidget(QLabel("Jog speed"), 0, 0)
        gl.addWidget(self.jog_speed, 0, 1, 1, 2)
        self.lbl_jog = QLabel("15 deg/s")
        self.jog_speed.valueChanged.connect(lambda x: self.lbl_jog.setText(f"{x} deg/s"))
        gl.addWidget(self.lbl_jog, 0, 3)
        for (txt, r, c, dp, dt) in (("▲", 1, 1, 0, 1), ("◀", 2, 0, -1, 0), ("▶", 2, 2, 1, 0),
                                    ("▼", 3, 1, 0, -1)):
            b = QPushButton(txt)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(0)
            b.setAutoRepeatInterval(60)
            b.pressed.connect(lambda dp=dp, dt=dt: self._jog(dp, dt))
            b.clicked.connect(lambda _=False, dp=dp, dt=dt: self._jog(dp, dt))
            b.released.connect(lambda: self._jog(0, 0))
            gl.addWidget(b, r, c)
        btn_home = QPushButton("Home (H)")
        btn_home.clicked.connect(lambda: self._cmd("home"))
        gl.addWidget(btn_home, 2, 1)
        btn_st = QPushButton("Self-test")
        btn_st.clicked.connect(lambda: self._cmd("selftest"))
        gl.addWidget(btn_st, 3, 3)
        v.addWidget(g)

        g = QGroupBox("Calibration")
        gl = QGridLayout(g)
        gl.addWidget(QLabel("Target board distance"), 0, 0)
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(2.0, 40.0)
        self.spin_dist.setSingleStep(0.5)
        self.spin_dist.setValue(10.0)
        self.spin_dist.setSuffix(" ft")
        gl.addWidget(self.spin_dist, 0, 1)
        btn_cal = QPushButton("Calibrate (zero gimbal + LiDAR)")
        btn_cal.clicked.connect(lambda: self._cmd("calibrate", distance_m=self.spin_dist.value() * 0.3048))
        gl.addWidget(btn_cal, 1, 0, 1, 2)
        btn_aim = QPushButton("Set aim point (click)")
        btn_aim.clicked.connect(self._aim_mode)
        gl.addWidget(btn_aim, 2, 0)
        btn_aim_r = QPushButton("Reset aim point")
        btn_aim_r.clicked.connect(lambda: self._cmd("reset_boresight"))
        gl.addWidget(btn_aim_r, 2, 1)
        self.lbl_cal = QLabel("")
        self.lbl_cal.setWordWrap(True)
        self.lbl_cal.setStyleSheet("font-size:9pt;")
        gl.addWidget(self.lbl_cal, 3, 0, 1, 2)
        v.addWidget(g)

        g = QGroupBox("Sensor / scenario / recording")
        gl = QGridLayout(g)
        gl.addWidget(QLabel("Camera"), 0, 0)
        self.cmb_sensor = QComboBox()
        self.cmb_sensor.activated.connect(lambda i: self._cmd("switch_sensor", name=self.cmb_sensor.itemText(i)))
        gl.addWidget(self.cmb_sensor, 0, 1)
        gl.addWidget(QLabel("Sim scenario"), 1, 0)
        self.cmb_scen = QComboBox()
        self.cmb_scen.activated.connect(lambda i: self._cmd("set_scenario", name=self.cmb_scen.itemText(i)))
        gl.addWidget(self.cmb_scen, 1, 1)
        self.btn_rec = QPushButton("Record")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(lambda on: self._cmd("record", on=bool(on)))
        gl.addWidget(self.btn_rec, 2, 0)
        self.lbl_log = QLabel("")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setStyleSheet("font-size:8.5pt; color:#8fa3b8;")
        gl.addWidget(self.lbl_log, 2, 1)
        v.addWidget(g)

        g = QGroupBox("Telemetry")
        gl = QGridLayout(g)
        self.tel: dict[str, QLabel] = {}
        names = ["Gimbal pan / tilt", "Command", "Target az / el", "Target rate", "Error", "Jitter @ 20 ft",
                 "LiDAR range", "Target XYZ", "Latency p50 / p95", "Camera / vision", "Detector", "Control / display",
                 "Locked target"]
        for i, n in enumerate(names):
            k = QLabel(n)
            k.setStyleSheet("color:#8fa3b8;")
            val = QLabel("-")
            val.setFont(QFont("Consolas", 9))
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            gl.addWidget(k, i, 0)
            gl.addWidget(val, i, 1)
            self.tel[n] = val
        v.addWidget(g)

        g = QGroupBox("Self-test")
        gl = QVBoxLayout(g)
        self.lbl_selftest = QLabel("not run")
        self.lbl_selftest.setWordWrap(True)
        self.lbl_selftest.setTextFormat(Qt.TextFormat.RichText)
        gl.addWidget(self.lbl_selftest)
        v.addWidget(g)

        g = QGroupBox("Events")
        gl = QVBoxLayout(g)
        self.txt_events = QPlainTextEdit()
        self.txt_events.setReadOnly(True)
        self.txt_events.setMaximumBlockCount(300)
        self.txt_events.setMinimumHeight(160)
        gl.addWidget(self.txt_events)
        v.addWidget(g)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ actions
    def _cmd(self, name: str, **kw) -> None:
        try:
            r = self.engine.command({"cmd": name, **kw})
            if isinstance(r, dict) and not r.get("ok", True) and r.get("msg"):
                self.status.showMessage(f"{name}: {r['msg']}", 4000)
        except Exception as exc:
            self.status.showMessage(f"{name} failed: {exc}", 6000)

    def _estop(self) -> None:
        self._cmd("estop")

    def _pause_resume(self) -> None:
        st = (self._snap.get("state") or {}).get("state")
        self._cmd("resume" if st == "PAUSED" else "pause")

    def _designate(self, x, y, w, h) -> None:
        self._cmd("designate", x=x, y=y, w=w, h=h)

    def _set_aim(self, x, y) -> None:
        self._cmd("set_boresight", x=x, y=y)

    def _aim_mode(self) -> None:
        self.video.aim_mode = True

    def _jog(self, dp: int, dt: int) -> None:
        s = float(self.jog_speed.value())
        self._cmd("jog", pan=dp * s, tilt=dt * s)

    def keyPressEvent(self, e) -> None:
        keys = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, 1), Qt.Key.Key_Down: (0, -1)}
        if e.key() in keys:
            self._jog_keys[e.key()] = keys[e.key()]
            dp = sum(v[0] for v in self._jog_keys.values())
            dt = sum(v[1] for v in self._jog_keys.values())
            self._jog(dp, dt)
            return
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            st = (self._snap.get("state") or {}).get("state")
            self._cmd("stop" if st in ("SEARCHING", "TRACKING", "LOST", "PAUSED") else "start")
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() in self._jog_keys and not e.isAutoRepeat():
            self._jog_keys.pop(e.key(), None)
            if not self._jog_keys:
                self._jog(0, 0)
            return
        super().keyReleaseEvent(e)

    # ------------------------------------------------------------------ refresh
    def _tick(self) -> None:
        try:
            snap = self.engine.snapshot(with_frame=True)
        except Exception as exc:
            self.status.showMessage(f"engine snapshot failed: {exc}")
            return
        self._snap = snap
        fid = snap.get("frame_id", -1)
        frame = snap.get("frame") if fid != self._last_frame_id else None
        if frame is not None:
            self._last_frame_id = fid
        self.video.set_frame(frame, snap)
        now = time.perf_counter()
        if now - self._last_slow > 0.1:      # text widgets at 10 Hz
            self._last_slow = now
            self._update_panel(snap)

    def _sync_combo(self, cmb: QComboBox, items, current) -> None:
        items = [str(i) for i in (items or [])]
        if [cmb.itemText(i) for i in range(cmb.count())] != items:
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItems(items)
            cmb.blockSignals(False)
        if current is not None and cmb.currentText() != str(current) and not cmb.view().isVisible():
            cmb.blockSignals(True)
            cmb.setCurrentText(str(current))
            cmb.blockSignals(False)

    def _update_panel(self, s: dict) -> None:
        st = s.get("state") or {}
        state = st.get("state", "?")
        col = STATE_COLORS.get(state)
        self.lbl_state.setText(state)
        if col is not None:
            self.lbl_state.setStyleSheet(f"background: rgba({col.red()},{col.green()},{col.blue()},60);"
                                         f"color: rgb({col.red()},{col.green()},{col.blue()}); border: 1px solid "
                                         f"rgb({col.red()},{col.green()},{col.blue()});")
        self.lbl_reason.setText(html.escape(str(st.get("reason", ""))))
        self.btn_pause.setText("Resume" if state == "PAUSED" else "Pause")
        self.btn_start.setEnabled(state in ("IDLE", "LOST", "PAUSED"))
        self.btn_pause.setEnabled(state in ("SEARCHING", "TRACKING", "LOST", "PAUSED"))
        self.btn_stop.setEnabled(state in ("SEARCHING", "TRACKING", "LOST", "PAUSED", "CALIBRATING"))
        self.btn_reset.setEnabled(state in ("ESTOP", "FAULT"))
        self._sync_combo(self.cmb_class, s.get("seek_classes"), s.get("seek_class"))
        self._sync_combo(self.cmb_sensor, s.get("sensors"), s.get("active_sensor"))
        scen = s.get("scenarios")
        self.cmb_scen.setEnabled(bool(scen))
        self._sync_combo(self.cmb_scen, scen or ["(real hardware)"], s.get("scenario") or "(real hardware)")
        tel = s.get("telemetry") or {}
        lat = s.get("latency") or {}
        jit = s.get("jitter") or {}
        rates = s.get("rates") or {}
        det = s.get("detector") or {}
        g = s.get("gimbal") or {}
        T = self.tel
        T["Gimbal pan / tilt"].setText(f"{_fmt(tel.get('pan_deg'))} / {_fmt(tel.get('tilt_deg'))} deg  "
                                       f"[{g.get('kind', '?')} {_fmt(g.get('feedback_hz'), 0)} Hz]")
        mode = tel.get("cmd_mode") or "-"
        unit = " deg/s" if mode == "rate" else " deg"
        T["Command"].setText(f"{mode}: {_fmt(tel.get('cmd_pan'))} / {_fmt(tel.get('cmd_tilt'))}{unit if mode != '-' else ''}")
        T["Target az / el"].setText(f"{_fmt(tel.get('kf_az_deg'))} / {_fmt(tel.get('kf_el_deg'))} deg")
        T["Target rate"].setText(f"{_fmt(tel.get('kf_az_rate_dps'), 1)} / {_fmt(tel.get('kf_el_rate_dps'), 1)} deg/s")
        if tel.get("err_az_deg"):
            e = math.hypot(float(tel["err_az_deg"]), float(tel["err_el_deg"]))
            T["Error"].setText(f"{_fmt(tel['err_az_deg'], 3)} / {_fmt(tel['err_el_deg'], 3)} deg  "
                               f"(dx {_fmt(tel.get('dx'), 3)}, dy {_fmt(tel.get('dy'), 3)})")
        else:
            T["Error"].setText("-")
        T["Jitter @ 20 ft"].setText(f"{_fmt(jit.get('jitter_dia_in_20ft'))} in dia   (RMS {_fmt(jit.get('rms_deg'), 3)} deg)")
        li = s.get("lidar") or {}
        T["LiDAR range"].setText(f"{_fmt(tel.get('range_m'), 3)} m  {'ON TARGET' if tel.get('range_on_target') == 1 else ''}"
                                 f"  [{li.get('kind', '-')} {_fmt(li.get('rate_hz'), 0)} Hz, cksum err "
                                 f"{li.get('checksum_errors', 0)}]")
        T["Target XYZ"].setText(f"{_fmt(tel.get('target_x_m'))}, {_fmt(tel.get('target_y_m'))}, {_fmt(tel.get('target_z_m'))} m")
        T["Latency p50 / p95"].setText(f"{_fmt(lat.get('cmd_p50'), 1)} / {_fmt(lat.get('cmd_p95'), 1)} ms (budget 100)")
        T["Camera / vision"].setText(f"{_fmt(rates.get('camera'), 1)} fps / {_fmt(rates.get('vision'), 1)} Hz "
                                     f"({_fmt(rates.get('vision_ms'), 1)} ms)  {s.get('active_sensor', '')}")
        assist = f" + {det['assist']}" if det.get("assist") else ""
        T["Detector"].setText(f"{det.get('name', '?')} {det.get('model', '')}{assist} {_fmt(det.get('hz'), 1)} Hz "
                              f"{_fmt(det.get('infer_ms'), 1)} ms {det.get('device', '')}")
        tip = []
        if det.get("classes"):
            tip.append("classes: " + ", ".join(det["classes"][:20]) + (" ..." if len(det["classes"]) > 20 else ""))
        if det.get("trained_on"):
            tip.append("trained on: " + det["trained_on"])
        T["Detector"].setToolTip("\n".join(tip))
        T["Control / display"].setText(f"{_fmt(rates.get('control'), 1)} Hz / {self.video.display_fps:.1f} fps "
                                       f"(render {self.video.render_ms:.1f} ms)")
        lk = (s.get("overlay") or {}).get("lock")
        T["Locked target"].setText(f"{lk['cls']} #{lk['tid']} ({lk['mode']}) score {lk['score']:.2f}" if lk else "-")
        lg = s.get("logger") or {}
        self.btn_rec.blockSignals(True)
        self.btn_rec.setChecked(bool(lg.get("active")))
        self.btn_rec.blockSignals(False)
        self.lbl_log.setText((f"REC {lg.get('rows', 0)} rows, {lg.get('frames_saved', 0)} frames\n{lg.get('dir', '')}"
                              if lg.get("active") else f"not recording - {lg.get('free_gb', '?')} GB free")
                             + (f"\n{lg['warning']}" if lg.get("warning") else ""))
        ocr = s.get("ocr") or {}
        if ocr:
            self.lbl_ocr.setText(f"OCR: <b>{html.escape(ocr.get('text') or '')}</b> "
                                 f"{html.escape(ocr.get('error', ''))} ({ocr.get('ms', '-')} ms)")
        elif not s.get("ocr_available", True):
            self.lbl_ocr.setText("OCR: Tesseract not installed (see README)")
        cal = s.get("calibration_result") or {}
        calib = s.get("calibration") or {}
        parts = []
        if cal:
            parts.append(("<b style='color:#3c3'>PASS</b>" if cal.get("ok") else "<b style='color:#f55'>FAIL</b>")
                         + " " + "; ".join(f"{x['step']}: {'ok' if x.get('ok') else 'FAIL'}" for x in cal.get("steps", [])))
        if calib:
            parts.append(f"zero pan {_fmt(calib.get('zero_pan_deg'), 3)} tilt {_fmt(calib.get('zero_tilt_deg'), 3)} deg, "
                         f"LiDAR offset {_fmt(calib.get('range_offset_m'), 3)} m, aim "
                         f"{calib.get('boresight_px', 'centre')}")
        self.lbl_cal.setText("<br>".join(parts))
        stt = s.get("selftest") or {}
        if stt:
            rows = [f"<span style='color:{'#3c3' if c['ok'] else '#f55'}'>{'PASS' if c['ok'] else 'FAIL'}</span> "
                    f"<b>{html.escape(c['name'])}</b>: {html.escape(str(c['detail']))}" for c in stt.get("checks", [])]
            self.lbl_selftest.setText("<br>".join(rows))
        ev = s.get("events") or []
        if ev and (len(ev) != self._last_events_len or ev[-1]["t"] != getattr(self, "_last_ev_t", None)):
            self._last_events_len = len(ev)
            self._last_ev_t = ev[-1]["t"]
            self.txt_events.setPlainText("\n".join(time.strftime("%H:%M:%S", time.localtime(e["t"])) +
                                                   f" [{e['kind']}] {e['text']}" for e in ev))
            self.txt_events.verticalScrollBar().setValue(self.txt_events.verticalScrollBar().maximum())
        alerts = s.get("alerts") or {}
        self.status.showMessage(f"{s.get('mode', 'local')} | config {s.get('config', '')} | AimBot {s.get('version', '')}"
                                + (" | ALERT: " + " ; ".join(alerts.values()) if alerts else ""))


def run_gui(engine, fps: float = 60.0) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow(engine, fps)
    win.show()
    return app.exec()
