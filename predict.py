import torch
import librosa
import numpy as np
from model import HybridDeepfakeDetector

def normalize_rms(waveform, target_rms=0.1):
    """Нормализация аудиосигнала по среднеквадратичному значению"""
    rms = np.sqrt(np.mean(waveform**2))
    if rms > 1e-8:
        return waveform * (target_rms / rms)
    return waveform

def load_audio(file_path, sr=16000, max_len=5, normalize=True):
    """Загрузка и обработка аудиофайла, извлечение коэффициентов"""
    waveform, sr = librosa.load(file_path, sr=sr, mono=True)
    if normalize:
        waveform = normalize_rms(waveform)
        waveform = waveform - np.mean(waveform)
    target_len = max_len * sr
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    coefs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    coefs_tensor = torch.from_numpy(coefs.T).float().unsqueeze(0)
    return audio_tensor, coefs_tensor

def predict(file_path, model_path="checkpoints/ready_model.pth", device="cpu"):
    """Загрузка модели и предсказание"""
    model = HybridDeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    audio, coefc = load_audio(file_path)
    audio, coefc = audio.to(device), coefc.to(device)
    with torch.no_grad():
        prob = model(audio, coefc).item()
    return prob, "FAKE" if prob > 0.5 else "REAL"