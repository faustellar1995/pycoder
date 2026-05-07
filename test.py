import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer
from datetime import datetime


class ClockWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时时钟")
        self.resize(300, 120)

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setStyleSheet("font-size: 36px; font-family: monospace;")
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 每秒更新一次

        self.update_time()

    def update_time(self):
        now = datetime.now()
        self.label.setText(now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClockWindow()
    win.show()
    sys.exit(app.exec_())
