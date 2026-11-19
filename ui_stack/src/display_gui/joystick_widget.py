import math

from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget


class JoystickWidget(QWidget):
    # Emitted whenever the knob moves. Both values range from -1.0 to 1.0.
    # pan: -1 = full left, +1 = full right
    # tilt: -1 = full down, +1 = full up
    moved = Signal(float, float)

    SIZE = 160
    KNOB_RADIUS = 20
    MAX_TRAVEL = SIZE / 2 - 4 - KNOB_RADIUS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._offset = QPointF(0, 0)
        self._dragging = False

    def reset(self):
        self._dragging = False
        self._set_offset(QPointF(0, 0))

    def _center(self):
        return QPointF(self.width() / 2, self.height() / 2)

    def _set_offset(self, offset):
        self._offset = offset
        self.update()
        pan = offset.x() / self.MAX_TRAVEL
        tilt = -offset.y() / self.MAX_TRAVEL
        self.moved.emit(pan, tilt)

    def _follow_mouse(self, mouse_pos):
        offset = mouse_pos - self._center()
        length = math.hypot(offset.x(), offset.y())
        if length > self.MAX_TRAVEL:
            offset = offset * (self.MAX_TRAVEL / length)
        self._set_offset(offset)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._follow_mouse(event.position())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._follow_mouse(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else 0.35)

        center = self._center()
        base_radius = self.SIZE / 2 - 4

        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setBrush(QColor(50, 50, 50))
        painter.drawEllipse(center, base_radius, base_radius)

        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawLine(center + QPointF(-base_radius, 0), center + QPointF(base_radius, 0))
        painter.drawLine(center + QPointF(0, -base_radius), center + QPointF(0, base_radius))

        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setBrush(QColor(110, 110, 110))
        painter.drawEllipse(center + self._offset, self.KNOB_RADIUS, self.KNOB_RADIUS)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    joystick = JoystickWidget()
    joystick.moved.connect(lambda pan, tilt: print(f"pan={pan:.2f} tilt={tilt:.2f}"))
    joystick.show()
    sys.exit(app.exec())