import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from dataset import DeepfakeDataset
from model import HybridDeepfakeDetector

class FocalLoss(nn.Module):
    """Класс Фокальной потери"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.sigmoid(logits)
        p_t = p_t * targets + (1 - p_t) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

def compute_min_dcf(target_scores, nontarget_scores, c_miss=1, c_fa=1, p_target=0.5):
    """ Функция, которая вычисляет минимальное значение Detection Cost Function (min DCF)"""

    # Объединяем все скоры и метки
    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])
    idx = np.argsort(scores, kind='mergesort')[::-1]
    labels_sorted = labels[idx]

    n_target = len(target_scores)
    n_nontarget = len(nontarget_scores)
    fa = np.cumsum(1 - labels_sorted)          # ложные принятия
    miss = np.cumsum(labels_sorted)            # пропуски цели

    fa_rate = fa / n_nontarget
    miss_rate = miss / n_target

    # Вычисляем DCF для каждого порога (точки на DET кривой)
    dcf = c_miss * miss_rate * p_target + c_fa * fa_rate * (1 - p_target)
    min_dcf = np.min(dcf)

    return min_dcf

def compute_eer(labels, scores):
    """Функция, вычисляющая Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Ищем точку, где fpr == fnr (или минимальное различие)
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer

def validate(model, loader, criterion, device):
    """
    Функция валидации с необходимыми метриками
    Вход: model - модель нейронной сети, loader - загрузчик данных, criterion - функция потерь, device - устройство (cuda или cpu)
    Выход: loss, accuracy, auc, f1, eer, min_dcf
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []
    total_correct = 0
    total_samples = 0
    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for audio, coefc, labels in loader:
            audio, coefc, labels = audio.to(device), coefc.to(device), labels.to(device)
            outputs = model(audio, coefc)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels_np)
            total_correct += (preds == labels_np).sum()
            total_samples += len(labels_np)
            for score, label in zip(scores, labels_np):
                if label == 0:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    auc = roc_auc_score(all_labels, all_scores)
    f1 = f1_score(all_labels, (np.array(all_scores) > 0.5).astype(int))
    eer = compute_eer(all_labels, all_scores)
    min_dcf = compute_min_dcf(target_scores=bonafide_scores, nontarget_scores=spoof_scores, c_miss=1, c_fa=10, p_target=0.5)

    return avg_loss, accuracy, auc, f1, eer, min_dcf

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Функция обучения
    Вход: model - модель нейронной сети, loader - загрузчик данных, optimizer - оптимизатор, criterion - функция потерь, device - устройство (cuda или cpu)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for audio, coefc, labels in tqdm(loader):
        audio, coefc, labels = audio.to(device), coefc.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio, coefc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def print_model_details(model):
    """Подробная информация о модели"""
    print("\n" + "="*70)
    print(f"МОДЕЛЬ: {model.__class__.__name__}")
    print("="*70)
    
    total_params = 0
    trainable_params = 0
    print(f"{'Модуль':<40} {'Параметры':>15} {'Обучаемые':>10}")
    print("-"*70)
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += mod_params
        trainable_params += mod_trainable
        print(f"{name:<40} {mod_params:>15,} {mod_trainable:>10,}")
    
    frozen_params = total_params - trainable_params
    print("-"*70)
    print(f"{'ИТОГО':<40} {total_params:>15,} {trainable_params:>10,}")
    print(f"Замороженных параметров: {frozen_params:,}")
    print("="*70)

def plot_history(train_losses, val_losses, train_accs, val_accs, metrics_history, save_path="training_curves.png"):
    """
    Функция для построения графиков
    metrics_history: dict с ключами 'auc', 'f1', 'eer', "min_DCF" - каждый список значений по эпохам
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Динамика обучения", fontsize=14)
    
    # Для функции потерь
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Потери')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Для точности
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Точность')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Для AUC
    axes[0, 2].plot(epochs, metrics_history['auc'], 'g-', label='AUC')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('AUC ROC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Для F1
    axes[1, 0].plot(epochs, metrics_history['f1'], 'm-', label='F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].set_title('F1-мера')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Для EER
    axes[1, 1].plot(epochs, metrics_history['eer'], 'c-', label='EER')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('EER')
    axes[1, 1].set_title('Equal Error Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Для min-DCF
    axes[1, 2].plot(epochs, metrics_history['min-DCF'], 'c-', label='min-DCF')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('min-DCF')
    axes[1, 2].set_title('min-DCF')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Графики сохранены в {save_path}")


def test_model(model, test_dir, device, model_path="checkpoints/best_model_acc.pth"):
    """Функция, которая загружает лучшую модель, тестирует и выводит метрики + графики"""
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Создание тестового датасета
    test_dataset = DeepfakeDataset(
        root_dir=test_dir,
        augment=False,
        normalize_audio=True
    )
    total_size = len(test_dataset)
    other_size = int(0.8 * total_size)
    test_size = int(total_size - other_size)
    test_data, val_data = random_split(test_dataset, [test_size, other_size])
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Сбор предсказаний и метрик
    all_labels = []
    all_scores = []
    total_loss = 0.0
    criterion = FocalLoss(alpha=0.6, gamma=2.0)
    bonafide_scores = []
    spoof_scores = []
    
    with torch.no_grad():
        for audio, mfcc, labels in test_loader:
            audio, mfcc, labels = audio.to(device), mfcc.to(device), labels.to(device)
            outputs = model(audio, mfcc)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            scores = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels_np)
            for score, label in zip(scores, labels_np):
                if label == 0:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    preds_binary = (all_scores > 0.5).astype(int)
    
    accuracy = np.mean(preds_binary == all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    f1 = f1_score(all_labels, preds_binary)
    eer = compute_eer(all_labels, all_scores)
    avg_loss = total_loss / len(test_loader)
    min_dcf = np.nan
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    min_dcf = compute_min_dcf(bonafide_scores, spoof_scores, c_miss=1, c_fa=10, p_target=0.5)
    
    print("\n========== РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ==========")
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"EER:      {eer:.4f}")
    print(f"min DCF:  {min_dcf:.4f}")
    
    # Построение ROC-кривая
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая (тест)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_roc.png')
    plt.show()
    
    # Построение матрицы ошибок
    cm = confusion_matrix(all_labels, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot()
    plt.title('Матрица ошибок (тест)')
    plt.savefig('test_cm.png')
    plt.show()

def train(device, model, optimizer):
    """Функция обучения"""
    # Загрузка и разбиение данных
    full_dataset = DeepfakeDataset(
        root_dir="data",
        augment=True,
        noise_type='realistic',
        nlpaug_color='random',
        nlpaug_prob=0.5,
        normalize_audio=True
    )
    total_size = len(full_dataset)
    val_size = int(0.02 * total_size)
    train_size = int(total_size - val_size)
    train_dataset, val_dataset   = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Инициализация оптимизатора и функции потерь и переменных для графиков
    criterion = FocalLoss(alpha=0.6, gamma=2.0)
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs   = []
    metrics_history = {'auc': [], 'f1': [], 'eer': [], 'min-DCF': []}

    for epoch in range(10):
        # Обучение и валидация
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_f1, val_eer, val_min_dcf = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        metrics_history['auc'].append(val_auc)
        metrics_history['f1'].append(val_f1)
        metrics_history['eer'].append(val_eer)
        metrics_history['min-DCF'].append(val_min_dcf)        
    
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"             Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | EER: {val_eer:.4f} | min-DCF: {val_min_dcf:.4f}")
    
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model_acc.pth")
            print(f"Лучшая модель сохранена! ( Accuracy={best_acc:.4f} )")

    return(train_losses, val_losses, train_accs, val_accs, metrics_history)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Создание модели
    model = HybridDeepfakeDetector(mfcc_dim=80, freeze_wav2vec=False).to(device)

    # Разделяем параметры на две группы
    wav2vec_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'wav2vec' in name:
                wav2vec_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.Adam([
        {'params': wav2vec_params, 'lr': 1e-5},   # в 10 раз меньше для дообучения Wav2Vec2
        {'params': other_params, 'lr': 1e-4}
    ])

    # Вывод параметров модели
    print_model_details(model)

    train_losses, val_losses, train_accs, val_accs, metrics_history = train(device, model, optimizer)
    plot_history(train_losses, val_losses, train_accs, val_accs, metrics_history)

    test_model(model, test_dir="data", device=device, model_path="checkpoints/best_model_acc.pth")

if __name__ == "__main__":
    main()