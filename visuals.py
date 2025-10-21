import matplotlib.pyplot as plt
import numpy as np
import yasa
import io
from PIL import Image
from processing import butter_bandpass, filtfilt
import logging

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

def plot_eeg_signals(time_s, eeg_uv, eeg_uv_bp, eeg_uv_notched):
    logger.info("Generating EEG signals plot")
    plt.figure(figsize=(12, 8))
    plt.plot(time_s, eeg_uv, label='Raw EEG (µV)', alpha=0.5)
    plt.plot(time_s, eeg_uv_bp, label='Bandpass EEG (0.5-30 Hz)', linewidth=2)
    plt.plot(time_s, eeg_uv_notched, label='Notched EEG (50 Hz)', linewidth=2, linestyle='--')
    plt.title("EEG Signal: Raw vs Filtered", fontsize=42)
    plt.xlabel("Time (s)", fontsize=36)
    plt.ylabel("Amplitude (µV)", fontsize=36)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid(True)
    plt.legend(fontsize=18.5, loc='upper right')  # Explicit loc to avoid warning
    plt.tight_layout()
    plt.savefig("eeg_signal_plot.pdf", format='pdf', dpi=300)
    logger.info("Saved EEG signals plot to eeg_signal_plot.pdf")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_frequency_spectra(eeg_uv_notched, fs):
    logger.info("Generating frequency spectra plot")
    from scipy.signal import welch
    f_welch, psd_welch = welch(eeg_uv_notched, fs, nperseg=1024, noverlap=512)
    plt.figure(figsize=(12, 12))
    plt.subplot(2,1,1)
    n = len(eeg_uv_notched)
    freqs_fft = np.fft.fftfreq(n, d=1/fs)
    fft_magnitude = np.abs(np.fft.fft(eeg_uv_notched)) / n
    plt.plot(freqs_fft[:n//2], fft_magnitude[:n//2] * 2, linewidth=2)
    plt.title("FFT Spectrum", fontsize=28)
    plt.xlabel("Frequency (Hz)", fontsize=24)
    plt.ylabel("Magnitude (µV)", fontsize=24)
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.semilogy(f_welch, psd_welch, linewidth=2)
    plt.title("Welch PSD", fontsize=28)
    plt.xlabel("Frequency (Hz)", fontsize=24)
    plt.ylabel("Power/Frequency (µV²/Hz)", fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eeg_fft_welch.pdf", format='pdf', dpi=300)
    logger.info("Saved frequency spectra plot to eeg_fft_welch.pdf")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_band_waveforms(eeg_uv_notched, fs, start_s=3000, end_s=3020):
    logger.info("Generating frequency bands plot")
    time_s = np.arange(len(eeg_uv_notched)) / fs
    mask = (time_s >= start_s) & (time_s <= end_s)
    window = eeg_uv_notched[mask]
    time_window = time_s[mask]
    eeg_delta = filtfilt(*butter_bandpass(0.5, 4, fs), window)
    eeg_theta = filtfilt(*butter_bandpass(4, 8, fs), window)
    eeg_alpha = filtfilt(*butter_bandpass(8, 13, fs), window)
    eeg_beta = filtfilt(*butter_bandpass(13, 30, fs), window)
    plt.figure(figsize=(16, 18))
    plt.subplot(4,1,1); plt.plot(time_window, eeg_delta, linewidth=2); plt.title("Delta (0.5-4 Hz)", fontsize=28)
    plt.subplot(4,1,2); plt.plot(time_window, eeg_theta, linewidth=2); plt.title("Theta (4-8 Hz)", fontsize=28)
    plt.subplot(4,1,3); plt.plot(time_window, eeg_alpha, linewidth=2); plt.title("Alpha (8-13 Hz)", fontsize=28)
    plt.subplot(4,1,4); plt.plot(time_window, eeg_beta, linewidth=2); plt.title("Beta (13-30 Hz)", fontsize=28)
    plt.tight_layout()
    plt.savefig("eeg_bands_plot.pdf", format='pdf', dpi=300)
    logger.info("Saved frequency bands plot to eeg_bands_plot.pdf")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_hypnogram(hypno):
    logger.info("Generating hypnogram plot")
    hypno_int = yasa.hypno_str_to_int(hypno)
    time_minutes = np.arange(len(hypno_int)) * 0.5
    start_min, end_min = 40, 170
    start_epoch = int(start_min / 0.5)
    end_epoch = int(end_min / 0.5)
    window_time = time_minutes[start_epoch:end_epoch]
    window_hypno = hypno_int[start_epoch:end_epoch]
    plt.figure(figsize=(16, 6))
    plt.step(window_time, window_hypno, where='post', color='navy', linewidth=3)
    plt.gca().invert_yaxis()
    plt.yticks([4, 3, 2, 1, 0], ['W', 'N1', 'N2', 'N3', 'R'], fontsize=28)
    plt.xlabel('Time (minutes)', fontsize=32)
    plt.ylabel('Sleep Stage', fontsize=32)
    plt.title(f'Hypnogram ({start_min}-{end_min} min)', fontsize=36)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("hypnogram_plot.pdf", format='pdf', dpi=300)
    logger.info("Saved hypnogram to hypnogram_plot.pdf")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img