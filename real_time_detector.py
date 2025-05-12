import sys
import time
import psutil
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton, QLabel,
                             QHeaderView, QHBoxLayout, QSizePolicy, QDialog)
from PyQt5.QtCore import QTimer, Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from advanced_anomaly_detector import AdvancedAnomalyDetector
import json

class WorkerSignals(QObject):
    finished = pyqtSignal(dict)

class StatsWorker(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.signals = WorkerSignals()
        self.signals.finished.connect(callback)

    @pyqtSlot()
    def run(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            load_avg = psutil.getloadavg()
            num_procs = len(psutil.pids())

            try:
                freq = psutil.cpu_freq()
                cpu_freq = freq.current if freq else 0
            except Exception:
                cpu_freq = 0

            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'status', 'create_time', 'cmdline']):
                try:
                    pinfo = proc.info
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    created = datetime.fromtimestamp(pinfo['create_time']).strftime('%Y-%m-%d %H:%M:%S')

                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'username': pinfo.get('username', 'N/A'),
                        'cpu': pinfo['cpu_percent'],
                        'memory': memory_mb,
                        'status': pinfo['status'],
                        'created': created,
                        'cmdline': pinfo.get('cmdline', [])
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.signals.finished.emit({
                'cpu_percent': cpu_percent,
                'memory': memory,
                'load_avg': load_avg,
                'cpu_freq': cpu_freq,  
                'num_procs': num_procs,
                'processes': processes
            })

        except Exception as e:
            print("Error in background worker:", e)


class ProcessTableWindow(QDialog):
    def __init__(self, processes):
        super().__init__()
        self.setWindowTitle("All Running Processes")
        self.resize(1000, 600)

        layout = QVBoxLayout(self)

        self.process_table = QTableWidget()
        self.process_table.setColumnCount(7)
        self.process_table.setHorizontalHeaderLabels([
            "PID", "Name", "Username", "CPU (%)", "Memory (MB)", "Status", "Start Time"
        ])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.process_table)

        self.populate_table(processes)

    def populate_table(self, processes):
        self.process_table.setRowCount(len(processes))
        for row, proc in enumerate(processes):
            items = [
                QTableWidgetItem(str(proc['pid'])),
                QTableWidgetItem(proc['name']),
                QTableWidgetItem(proc['username']),
                QTableWidgetItem(f"{proc['cpu']:.1f}"),
                QTableWidgetItem(f"{proc['memory']:.1f}"),
                QTableWidgetItem(proc['status']),
                QTableWidgetItem(proc['created'])
            ]
            for col, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.process_table.setItem(row, col, item)

class ProcessMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real Time Anomaly Detector")
        self.setGeometry(100, 100, 1200, 800)

        self.anomaly_detector = AdvancedAnomalyDetector()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        header_layout = QHBoxLayout()
        self.system_info_label = QLabel()
        self.system_info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.system_info_label)
        layout.addLayout(header_layout)

        self.figure, self.axes = plt.subplots(5, 1, figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)

        self.data_history = {
            'time': [],
            'sensor_01': [],
            'sensor_02': [],
            'sensor_03': [],
            'sensor_04': [],
            'sensor_05': []
        }

        control_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setStyleSheet("background-color: red; color: white;")
        self.refresh_button.clicked.connect(self.update_data)
        control_layout.addWidget(self.refresh_button)

        self.auto_refresh_button = QPushButton("Auto Refresh")
        self.auto_refresh_button.setStyleSheet("background-color: blue; color: white;")
        self.auto_refresh_button.setCheckable(True)
        self.auto_refresh_button.toggled.connect(self.toggle_auto_refresh)
        control_layout.addWidget(self.auto_refresh_button)

        self.detect_anomalies_button = QPushButton("Detect Anomalies")
        self.detect_anomalies_button.setStyleSheet("background-color: green; color: white;")
        self.detect_anomalies_button.clicked.connect(self.check_anomalies)
        control_layout.addWidget(self.detect_anomalies_button)

        self.save_snapshot_button = QPushButton("Save Snapshot")
        self.save_snapshot_button.setStyleSheet("background-color: yellow; color: black;")
        self.save_snapshot_button.clicked.connect(self.save_resource_snapshot)
        control_layout.addWidget(self.save_snapshot_button)

        self.show_process_table_button = QPushButton("Show Process Table")
        self.show_process_table_button.setStyleSheet("background-color: gray; color: white;")
        self.show_process_table_button.clicked.connect(self.show_process_table)
        control_layout.addWidget(self.show_process_table_button)

        layout.addLayout(control_layout)
        layout.addWidget(QLabel("<b>Sensor Graphs</b>"))
        layout.addWidget(self.canvas)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.threadpool = QThreadPool()

        self.update_data()

    def toggle_auto_refresh(self, checked):
        if checked:
            self.timer.start(3000)
            self.refresh_button.setEnabled(False)
        else:
            self.timer.stop()
            self.refresh_button.setEnabled(True)

    def update_data(self):
        worker = StatsWorker(self.on_data_ready)
        self.threadpool.start(worker)

    def on_data_ready(self, data):
        current_time = time.time() - self.start_time
        self.data_history['time'].append(current_time)
        self.data_history['sensor_01'].append(data['cpu_percent'])
        self.data_history['sensor_02'].append(data['memory'].percent)
        self.data_history['sensor_03'].append(data['cpu_freq'])
        self.data_history['sensor_04'].append(data['load_avg'][0])
        self.data_history['sensor_05'].append(data['num_procs'])
        if len(self.data_history['time']) > 50:
            for key in self.data_history:
                self.data_history[key].pop(0)

        self.system_info_label.setText(
            f"Sensor_01: {data['cpu_percent']}% | "
            f"Sensor_02: {data['memory'].percent}% | "
            f"Sensor_03 (CPU_Freq): {data['cpu_freq']}MHz | "
            f"Sensor_04 (Load): {data['load_avg'][0]:.2f} | "
            f"Sensor_05 (Procs): {data['num_procs']}"
        )

        for ax, key, color in zip(self.axes, list(self.data_history.keys())[1:], ['blue', 'red', 'orange', 'green', 'purple']):
            ax.clear()
            ax.plot(self.data_history['time'], self.data_history[key], color=color, label=key)
            ax.set_title(key)
            ax.legend()
            ax.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def save_resource_snapshot(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sensor_snapshot_{timestamp}.png"
            self.figure.savefig(filename)
            self.status_label.setText(f"Snapshot saved as {filename}")
            self.status_label.setStyleSheet("color: #4CAF50;")
        except Exception as e:
            self.status_label.setText(f"Error saving snapshot: {str(e)}")
            self.status_label.setStyleSheet("color: #F44336;")

    def check_anomalies(self):
        try:
            self.anomaly_detector.update_history()
            if not self.anomaly_detector.is_trained:
                self.status_label.setText("Training anomaly detection model...")
                self.status_label.setStyleSheet("color: #2196F3;")
                if not self.anomaly_detector.train_model():
                    self.status_label.setText("Need more data to train model")
                    return

            current_metrics = self.anomaly_detector.collect_process_metrics()
            anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
            report = self.anomaly_detector.generate_report(anomalies)

            report_file = f'anomaly_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.status_label.setText(
                f"Found {len(anomalies)} anomalous processes. Report saved to {report_file}"
            )
            self.status_label.setStyleSheet("color: #4CAF50;")

        except Exception as e:
            self.status_label.setText(f"Error detecting anomalies: {str(e)}")
            self.status_label.setStyleSheet("color: #F44336;")

    def show_process_table(self):
        def callback(data):
            self.process_window = ProcessTableWindow(data['processes'])
            self.process_window.exec_()

        worker = StatsWorker(callback)
        self.threadpool.start(worker)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProcessMonitorUI()
    window.show()
    sys.exit(app.exec_())
