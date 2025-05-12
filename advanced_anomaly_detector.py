import numpy as np
import psutil
import pandas as pd
from collections import deque
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class AdvancedAnomalyDetector:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.process_history = deque(maxlen=history_size)
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.is_trained = False
        self.reconstruction_threshold = None

        # Initialize logging
        logging.basicConfig(
            filename=f'process_monitor_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def collect_process_metrics(self):
        """Collect detailed metrics for all running processes"""
        process_metrics = []

        # Get CPU frequency (MHz)
        try:
            freq = psutil.cpu_freq()
            core_freq = freq.current if freq else 0
        except Exception as e:
            logging.warning(f"Could not get CPU frequency: {e}")
            core_freq = 0

        # Get 1-minute load average
        try:
            load_avg = psutil.getloadavg()[0]
        except Exception as e:
            logging.warning(f"Could not get load average: {e}")
            load_avg = 0

        # Get number of processes
        try:
            num_processes = len(psutil.pids())
        except Exception as e:
            logging.warning(f"Could not get process count: {e}")
            num_processes = 0

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info

                metrics = {
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'sensor_01': pinfo.get('cpu_percent', 0) or 0,
                    'sensor_02': pinfo.get('memory_percent', 0) or 0,
                    'sensor_03': core_freq,
                    'sensor_04': load_avg,
                    'sensor_05': num_processes
                }

                process_metrics.append(metrics)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logging.warning(f"Error collecting metrics for process: {e}")
                continue

        return process_metrics

    def update_history(self):
        current_metrics = self.collect_process_metrics()
        self.process_history.append(current_metrics)
        logging.info(f"Updated process history. Current size: {len(self.process_history)}")

    def prepare_training_data(self):
        if not self.process_history:
            return None

        all_data = []
        for metrics in self.process_history:
            for proc in metrics:
                all_data.append([
                    proc['sensor_01'],
                    proc['sensor_02'],
                    proc['sensor_03'],
                    proc['sensor_04'],
                    proc['sensor_05']
                ])

        return np.array(all_data)

    def build_autoencoder(self, input_dim):
        inp = Input(shape=(input_dim,))
        x = Dense(16, activation='relu')(inp)
        x = Dense(8, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        out = Dense(input_dim, activation='linear')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self):
        data = self.prepare_training_data()
        if data is None or len(data) < self.history_size:
            logging.warning("Insufficient data for training")
            return False

        try:
            scaled_data = self.scaler.fit_transform(data)
            input_dim = scaled_data.shape[1]
            self.autoencoder = self.build_autoencoder(input_dim)
            self.autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
            reconstructed = self.autoencoder.predict(scaled_data)
            reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)
            self.reconstruction_threshold = np.percentile(reconstruction_error, 95)  # 95th percentile
            self.is_trained = True
            logging.info("Successfully trained autoencoder-based anomaly detection model")
            return True
        except Exception as e:
            logging.error(f"Error training autoencoder: {e}")
            return False

    def detect_anomalies(self, processes):
        if not self.is_trained or self.autoencoder is None:
            logging.warning("Autoencoder model not trained")
            return []

        try:
            current_data = []
            for proc in processes:
                current_data.append([
                    proc['sensor_01'],
                    proc['sensor_02'],
                    proc['sensor_03'],
                    proc['sensor_04'],
                    proc['sensor_05']
                ])

            scaled_data = self.scaler.transform(current_data)
            reconstructed = self.autoencoder.predict(scaled_data)
            reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)

            anomalies = []
            for i, err in enumerate(reconstruction_error):
                if err > self.reconstruction_threshold:  # Using dynamic threshold
                    processes[i]['anomaly_reason'] = "Anomaly detected based on reconstruction error"
                    anomalies.append(processes[i])

            logging.info(f"Detected {len(anomalies)} anomalous processes")
            return anomalies

        except Exception as e:
            logging.error(f"Error during anomaly detection: {e}")
            return []

    def generate_report(self, anomalies):
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_processes': len(self.process_history[-1]) if self.process_history else 0,
            'anomaly_count': len(anomalies),
            'anomalies': []
        }

        for proc in anomalies:
            report['anomalies'].append({
                'id': proc['pid'],
                'sensor': proc['name'],
                'reason': proc['anomaly_reason'],
                'sensor_01': proc['sensor_01'],
                'sensor_02': proc['sensor_02'],
                'sensor_03': proc['sensor_03'],
                'sensor_04': proc['sensor_04'],
                'sensor_05': proc['sensor_05']
            })

        return report
