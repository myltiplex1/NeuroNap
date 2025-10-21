import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch, hilbert
from scipy.interpolate import interp1d
import yasa
import mne
import antropy as ant
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
import os

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuronap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    logger.info(f"Created bandpass filter: {lowcut}-{highcut} Hz, order={order}")
    return b, a

def detect_spindles(epoch, fs, low=12, high=16, threshold=1.0, min_duration=0.3):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    sig = filtfilt(b, a, epoch)
    env = np.abs(hilbert(sig))
    thresh_val = np.mean(env) + threshold * np.std(env)
    mask = env > thresh_val
    spindle_times = []
    in_spindle = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_spindle:
            start = i
            in_spindle = True
        elif not val and in_spindle:
            end = i
            if (end - start) / fs >= min_duration:
                spindle_times.append(start)
            in_spindle = False
    logger.info(f"Detected {len(spindle_times)} spindles in epoch")
    return len(spindle_times)

def detect_slow_waves(epoch, fs, low=0.5, high=2, threshold=1.0, min_duration=0.2):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    sig = filtfilt(b, a, epoch)
    amp = np.abs(sig)
    thresh_val = np.mean(amp) + threshold * np.std(amp)
    mask = amp > thresh_val
    slow_wave_times = []
    in_wave = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_wave:
            start = i
            in_wave = True
        elif not val and in_wave:
            end = i
            if (end - start) / fs >= min_duration:
                slow_wave_times.append(start)
            in_wave = False
    logger.info(f"Detected {len(slow_wave_times)} slow waves in epoch")
    return len(slow_wave_times)

def extract_features(epochs, fs):
    logger.info(f"Extracting features from {len(epochs)} epochs")
    features = []
    for ep in epochs:
        f, psd = welch(ep, fs=fs, nperseg=fs*4)
        total_power = np.sum(psd)
        delta = np.sum(psd[(f >= 0.5) & (f < 4)])
        theta = np.sum(psd[(f >= 4) & (f < 8)])
        alpha = np.sum(psd[(f >= 8) & (f < 13)])
        sigma = np.sum(psd[(f >= 12) & (f < 16)])
        beta = np.sum(psd[(f >= 13) & (f < 32)])
        rel_delta = delta / total_power if total_power else 0
        rel_theta = theta / total_power if total_power else 0
        rel_alpha = alpha / total_power if total_power else 0
        rel_sigma = sigma / total_power if total_power else 0
        rel_beta = beta / total_power if total_power else 0
        samp_entropy = ant.sample_entropy(ep)
        spindle_count = detect_spindles(ep, fs)
        slow_wave_count = detect_slow_waves(ep, fs)
        features.append([rel_delta, rel_theta, rel_alpha, rel_sigma, rel_beta, samp_entropy, spindle_count, slow_wave_count])
    feature_names = ['rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta', 'sample_entropy', 'spindle_count', 'slow_wave_count']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df.to_csv('features.csv', index=False)
    logger.info("Saved features to features.csv")
    return features_df

def process_eeg(file_path):
    logger.info(f"Processing EEG file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path)
    if 'time_ms' not in df.columns or 'adc_value' not in df.columns:
        logger.error("CSV must contain 'time_ms' and 'adc_value' columns")
        raise ValueError("CSV must contain 'time_ms' and 'adc_value' columns")
    adc_values = pd.to_numeric(df['adc_value'], errors='coerce').values
    time_ms = pd.to_numeric(df['time_ms'], errors='coerce').values
    mask = ~np.isnan(adc_values) & ~np.isnan(time_ms)
    adc_values = adc_values[mask]
    time_ms = time_ms[mask]
    logger.info(f"Loaded {len(adc_values)} valid data points")
    v_ref = 3.3
    gain = 15000
    voltage_raw = adc_values * (v_ref / 4095)
    estimated_bias = np.mean(voltage_raw)
    voltage = voltage_raw - estimated_bias
    eeg_uv = voltage * 1e6 / gain
    df_res = pd.DataFrame({'time_ms': time_ms, 'adc_vals': adc_values})
    df_clean = df_res.groupby('time_ms', as_index=False).agg({'adc_vals': 'mean'})
    sample_rate_hz = 256
    delta_ms = 1000 / sample_rate_hz
    min_time = df_clean['time_ms'].min()
    max_time = df_clean['time_ms'].max()
    new_times = np.arange(min_time, max_time + delta_ms, delta_ms)
    interpolator = interp1d(df_clean['time_ms'], df_clean['adc_vals'], kind='linear', fill_value='extrapolate')
    new_adc_vals = interpolator(new_times)
    df_res = pd.DataFrame({'time_ms': new_times, 'adc_vals': new_adc_vals})
    voltage = (df_res['adc_vals'] * (v_ref / 4095)) - estimated_bias
    eeg_uv = voltage * 1e6 / gain
    df_res['time_s'] = df_res['time_ms'] / 1000
    df_res['eeg_uv'] = eeg_uv
    fs = 256.0
    lowcut = 0.5
    highcut = 30.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    eeg_uv_bp = filtfilt(b, a, eeg_uv)
    logger.info("Applied bandpass filter (0.5-30 Hz)")
    notch_freq = 50.0
    q = 30.0
    b_notch, a_notch = iirnotch(notch_freq, q, fs)
    eeg_uv_notched = filtfilt(b_notch, a_notch, eeg_uv_bp)
    logger.info("Applied notch filter (50 Hz)")
    clean_eeg = pd.DataFrame({'time_s': df_res['time_s'], 'eeg_uv': eeg_uv, 'eeg_bpf': eeg_uv_bp, 'eeg_notch': eeg_uv_notched})
    clean_eeg.to_csv('clean_eeg.csv', index=False)
    logger.info("Saved clean EEG data to clean_eeg.csv")
    info = mne.create_info(ch_names=['Fz'], sfreq=fs, ch_types=['eeg'])
    raw = mne.io.RawArray(eeg_uv_notched.reshape(1, -1), info)
    sl = yasa.SleepStaging(raw, eeg_name='Fz')
    hypno = sl.predict()
    logger.info("Completed YASA sleep staging")
    epoch_length = int(30 * fs)
    num_epochs = len(eeg_uv_notched) // epoch_length
    epochs = eeg_uv_notched[:num_epochs * epoch_length].reshape(num_epochs, epoch_length)
    features_df = extract_features(epochs, fs)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    pca = PCA(n_components=5)
    features_pca = pca.fit_transform(features_scaled)
    logger.info("Applied PCA with 5 components")
    model = GaussianHMM(n_components=5, covariance_type="full", n_iter=2000, tol=1e-6, random_state=42)
    model.fit(features_pca)
    hmm_labels = model.predict(features_pca)
    logger.info("Completed HMM clustering")
    clustered_features = features_df.copy()
    clustered_features['cluster'] = hmm_labels
    clustered_features.to_csv('eeg_clusters.csv', index=False)
    logger.info("Saved clustered features to eeg_clusters.csv")
    hypno_int = yasa.hypno_str_to_int(hypno)
    sf_hyp = 1 / 30
    stats = yasa.sleep_statistics(hypno_int, sf_hyp)
    logger.info("Computed sleep statistics")
    return eeg_uv, eeg_uv_bp, eeg_uv_notched, hypno, stats, features_df, hmm_labels, fs, df_res['time_s'].values