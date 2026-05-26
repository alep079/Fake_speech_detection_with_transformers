import torch
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
import nlpaug.augmenter.audio as naa
import torchaudio.transforms as T
import torchaudio
import random

class RawBoostAugmentation:
    """Базовые искажения: изменение громкости + гауссов шум"""
    @staticmethod
    def apply(waveform, prob=0.5):
        if torch.rand(1).item() > prob:
            return waveform
        # Изменение громкости
        gain = np.random.uniform(0.8, 1.2)
        waveform = waveform * gain
        # Гауссов шум
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        return torch.clamp(waveform, -1.0, 1.0)
    
class NLPAugNoiseAugmentation:
    """Генерация цветных шумов через nlpaug (белый, розовый, коричневый, случайный)"""
    def __init__(self, prob=0.5, noise_color='random'):
        self.prob = prob
        self.augmenter = naa.NoiseAug(zone=(0.0, 1.0), coverage=1.0, color=noise_color)

    def apply(self, waveform):
        if torch.rand(1).item() > self.prob:
            return waveform
        original_device = waveform.device
        waveform_np = waveform.cpu().numpy()
        try:
            augmented_np = self.augmenter.augment(waveform_np)
            augmented = torch.from_numpy(augmented_np).float().to(original_device)
            return torch.clamp(augmented, -1.0, 1.0)
        except Exception:
            return waveform

def normalize_rms(waveform, target_rms=0.1):
    """Нормализация среднеквадратичного уровня сигнала"""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 1e-8:
        return waveform * (target_rms / rms)
    return waveform

class DeepfakeDataset(Dataset):
    """Класс загрузки и обработки датасета"""
    
    def __init__(self, root_dir, augment=True, sr=16000, max_len=5,
             normalize_audio=True, noise_type='nlpaug',
             nlpaug_color='random', nlpaug_prob=0.5):
        self.sr = sr
        self.max_len = max_len
        self.augment = augment
        self.normalize = normalize_audio
        self.files = []
        self.labels = []
        # Собираем файлы из подпапок real / fake
        for label, subdir in enumerate(["real", "fake"]):
            path = Path(root_dir) / subdir
            if not path.exists():
                raise FileNotFoundError(f"Папка {path} не найдена!")
            for f in path.glob("*.*"):
                if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                    self.files.append(f)
                    self.labels.append(label)

        # Инициируем добавление шума
        self.rawboost = RawBoostAugmentation() if augment else None
        if augment and noise_type == 'nlpaug':
            self.noise_aug = NLPAugNoiseAugmentation(prob=nlpaug_prob, noise_color=nlpaug_color)
        elif augment and noise_type == 'realistic':
            self.noise_dir = "musan"  # <-- УКАЖИТЕ ВАШ ПУТЬ К ПАПКЕ С ШУМАМИ (MUSAN или другой)
            self.realistic_noise_files = list(Path(self.noise_dir).rglob('*.wav'))
            self.noise_aug = 'realistic'
        else:
            self.noise_aug = None

    def __len__(self):
        """Получение длины"""
        return len(self.files)

    def __getitem__(self, idx):
    
        # Загрузка аудио через librosa
        audio, _ = librosa.load(self.files[idx], sr=self.sr, mono=True)
        audio = torch.from_numpy(audio).float()

        # Обрезаем/дополняем до фиксированной длины
        target_len = self.max_len * self.sr
        if len(audio) < target_len:
            audio = F.pad(audio, (0, target_len - len(audio)), mode='constant', value=0)
        else:
            audio = audio[:target_len]

        # Нормализация среднеквадратичного уровня сигнала
        if self.normalize:
            audio = normalize_rms(audio)

        # Применяем аугментацию
        if self.augment:
            audio = self.rawboost.apply(audio)
            if self.noise_aug == 'realistic':
                audio = self.add_realistic_noise(audio)
            elif self.noise_aug is not None:
                audio = self.noise_aug.apply(audio)

        #mfcc = librosa.feature.mfcc(y=audio.numpy(), sr=self.sr, n_mfcc=13)
        #mfcc = torch.from_numpy(mfcc).float().T   # [time_frames, 13]

        # Подготовка для извлеченияя признаков (MFCC и LFCC)
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()
        sample_rate = self.sr
        audio_input = audio.unsqueeze(0)

        # Получение MFCC
        n_mfcc = 40
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False}
            )
        mfcc = mfcc_transform(audio_input)
        mfcc = mfcc.squeeze(0).T

        # Получение LFCC
        n_lfcc = 40
        lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            log_lf=True,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False}
            )
        lfcc = lfcc_transform(audio_input)
        lfcc = lfcc.squeeze(0).T

        # Объединяем признаки
        #combined = mfcc
        combined = torch.cat([mfcc, lfcc], dim=1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio, combined, label
    
    def _get_noise_snr(self, audio, noise, snr_db):
        """Вычисляет коэффициент усиления шума для заданного SNR (dB)"""
        snr = 10 ** (snr_db / 10)
        signal_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        if signal_power <= 1e-10 or noise_power <= 1e-10:
            return 0.0
        return torch.sqrt(signal_power / (noise_power * snr))

    def add_realistic_noise(self, audio):
        """Добавляет реальный шум из папки с вероятностью 0.6 и случайным SNR 5-20 dB"""
    
        # 60% шанс добавить шум
        if random.random() > 0.6:
            return audio
    
        # Выбираем случайный шумовой файл
        noise_path = random.choice(self.realistic_noise_files)
        try:
            noise, sr_noise = torchaudio.load(noise_path)
        except Exception:
            return audio
    
        # Приводим к моно
        if noise.shape[0] > 1:
            noise = torch.mean(noise, dim=0, keepdim=True)
        noise = noise.squeeze(0)
    
        # Приводим частоту дискретизации
        if sr_noise != self.sr:
            resampler = torchaudio.transforms.Resample(sr_noise, self.sr)
            noise = resampler(noise)
    
        # Обрезаем или зацикливаем шум до длины аудио
        if noise.shape[0] < audio.shape[0]:
            repeats = (audio.shape[0] // noise.shape[0]) + 1
            noise = noise.repeat(repeats)
        noise = noise[:audio.shape[0]]
    
        # Случайное отношение сигнал/шум (dB)
        snr_db = random.uniform(5, 20)
        noise_factor = self._get_noise_snr(audio, noise, snr_db)
    
        # Смешиваем и клиппируем
        noisy_audio = audio + noise_factor * noise
        return torch.clamp(noisy_audio, -1.0, 1.0)