import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AttentionPooling(nn.Module):
    """Внимание с обучаемыми весами для агрегации последовательности"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.Tanh(),
            nn.Linear(dim//2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return (x * weights).sum(dim=1)

class PyAraEncoder(nn.Module):
    """Модуль для обработки MFCC/LFCC признаков с помощью BiLSTM
    Вход: [batch, time, features]
    Выход: [batch, time, hidden*2]
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, proj_dim=768):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.proj = nn.Linear(hidden_dim * 2, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, input_dim]
        lstm_out, _ = self.bilstm(x)           
        lstm_out = self.dropout(lstm_out)
        projected = self.proj(lstm_out)      
        return projected

class CrossAttentionFusion(nn.Module):
    """Двунаправленное кросс-внимание между двумя ветками"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn_left = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_right = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, left, right):
        # Слева направо
        attended_left, _ = self.cross_attn_left(left, right, right)
        attended_left = self.norm(left + attended_left)
        # Справа налево
        attended_right, _ = self.cross_attn_right(right, left, left)
        attended_right = self.norm(right + attended_right)

        # Конкатенация по последнему измерению -> [B, T, dim*2]
        fused_seq = torch.cat([attended_left, attended_right], dim=-1)
        return fused_seq   # последовательность

class HybridDeepfakeDetector(nn.Module):
    """Общий класс нейронной сети, реализующий 2 ветки"""
    def __init__(self, mfcc_dim=80, bilstm_hidden=128, bilstm_layers=2,
                 wav2vec_model_name="facebook/wav2vec2-base-960h", freeze_wav2vec=True):
        super().__init__()

        # Левая ветка - BiLSTM модуль
        self.bilstm_encoder = PyAraEncoder(input_dim=mfcc_dim, hidden_dim=bilstm_hidden, num_layers=bilstm_layers, proj_dim=768)

        # Правая ветка - Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.freeze_wav2vec = freeze_wav2vec
        if freeze_wav2vec:
        # Полная заморозка
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        else:
        # Заморозить все параметры Wav2Vec2
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            # Разморозить последние 2 слоя трансформера (индексы 10 и 11)
            for name, param in self.wav2vec.named_parameters():
                if 'encoder.layers' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            layer_idx = int(part)
                            if layer_idx >= 10:
                                param.requires_grad = True
                            break
        self.wav2vec_proj = nn.Linear(768, 768)

        # Кросс-внимание
        self.cross_attn = CrossAttentionFusion(768, num_heads=8)

        # Attention pooling после кросс-внимания
        self.fusion_pool = AttentionPooling(768 * 2)

        # LayerNorm перед классификатором
        self.norm = nn.LayerNorm(768 * 2)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, audio_values, c_features):
        # Ветка 1: MFCC -> BiLSTM -> проекция
        left_seq = self.bilstm_encoder(c_features)

        # Ветка 2: Wav2Vec2 -> проекция -> пулинг
        w2v_out = self.wav2vec(audio_values).last_hidden_state
        w2v_proj = self.wav2vec_proj(w2v_out)

        # Обрезаем последовательности до минимальной длины
        min_len = min(left_seq.size(1), w2v_proj.size(1))
        left_trunc = left_seq[:, :min_len, :]
        right_trunc = w2v_proj[:, :min_len, :]

        # Кросс-внимание: теперь возвращает последовательность [B, T, 1536]
        fused_seq = self.cross_attn(left_trunc, right_trunc)

        # Attention pooling: агрегируем по временной оси → [B, 1536]
        fused_vec = self.fusion_pool(fused_seq)

        # LayerNorm
        fused_vec = self.norm(fused_vec)

        # Классификатор
        logits = self.classifier(fused_vec)
        return logits.squeeze(1)   # логиты без sigmoid