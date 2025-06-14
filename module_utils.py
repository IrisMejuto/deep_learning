# module_utils.py
"""
Utilidades comunes para la práctica "Predicción de Engagement Turístico".
"""

# ============================================================================
# CONFIGURACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================================

import random
import numpy as np
import torch
import os

RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    """Configura todas las semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Semillas configuradas con valor: {seed}")

# ============================================================================
# IMPORTACIONES PRINCIPALES
# ============================================================================

# Sistema y utilidades
import os
import sys
import warnings
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Google Colab
from google.colab import drive

# Manipulación de datos
import pandas as pd
import numpy as np

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Visualización geográfica
import folium

# Procesamiento de imágenes
import cv2
from PIL import Image, ImageEnhance, ImageFilter

# PyTorch - Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import models

# Optimización de hiperparámetros
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Métricas y evaluación
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

# TensorBoard para visualización
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

# Configuración de warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

# Matplotlib y Seaborn
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Configuración para gráficos de alta resolución
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# CONFIGURACIÓN DE PYTORCH Y GPU
# ============================================================================

def setup_device():
    """Configura y verifica el dispositivo (GPU/CPU) para PyTorch"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"   - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   - Memoria libre: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("GPU no disponible, usando CPU")
    
    return device

def setup_colab_drive():
    """Monta Google Drive en Colab"""
    try:
        drive.mount('/content/drive')
        print("Google Drive montado correctamente")
        return True
    except Exception as e:
        print(f"Error montando Google Drive: {e}")
        return False

# ============================================================================
# VERIFICACIÓN DEL ENTORNO
# ============================================================================

def check_environment():
    """Verifica que el entorno esté correctamente configurado"""
    print("=" * 60)
    print(" VERIFICACIÓN DEL ENTORNO DE DESARROLLO")
    print("=" * 60)
    
    # Información del sistema
    print(f"Python: {sys.version.split()[0]}")
    
    # Librerías principales
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Matplotlib: {plt.matplotlib.__version__}")
    print(f"Seaborn: {sns.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Optuna: {optuna.__version__}")
    
    # Configuración del dispositivo
    device = setup_device()
    
    # Configuración de semillas
    set_seeds(RANDOM_SEED)
    
    print("=" * 60)
    print("ENTORNO CONFIGURADO CORRECTAMENTE")
    print("=" * 60)
    
    return device

# ============================================================================
# CONFIGURACIÓN ESPECÍFICA DEL PROYECTO
# ============================================================================

# Rutas principales (ajustar según tu configuración)
PROJECT_NAME = "prediccion_engagement_turistico"

# Rutas para Google Colab
COLAB_PROJECT_PATH = f"/content/drive/MyDrive/{PROJECT_NAME}"
COLAB_DATA_PATH = f"{COLAB_PROJECT_PATH}/data"
COLAB_MODELS_PATH = f"{COLAB_PROJECT_PATH}/models"
COLAB_RESULTS_PATH = f"{COLAB_PROJECT_PATH}/results"

# Rutas locales (si trabajas en local)
LOCAL_PROJECT_PATH = "Z:/Iris/01-Learning-Datos/Big-Data/0_Repositorios-Git-Hub/Deep-Learning/deep_learning"
LOCAL_DATA_PATH = f"{LOCAL_PROJECT_PATH}/data"
LOCAL_MODELS_PATH = f"{LOCAL_PROJECT_PATH}/models"

# Parámetros del modelo
IMG_SIZE = (224, 224)  # Tamaño estándar para modelos pre-entrenados
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # Para DataLoader

# Parámetros de data augmentation
AUGMENTATION_PARAMS = {
    'rotation_degrees': 15,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}

# ============================================================================
# FUNCIONES AUXILIARES PARA EL PROYECTO
# ============================================================================

def create_project_directories(base_path: str):
    """Crea las carpetas necesarias para el proyecto"""
    directories = ['data', 'models', 'results', 'logs']
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Carpeta creada/verificada: {dir_path}")

def save_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                         epoch: int, loss: float, filepath: str):
    """Guarda un checkpoint del modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint guardado: {filepath}")

def load_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filepath: str):
    """Carga un checkpoint del modelo"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint cargado desde: {filepath}")
    return model, optimizer, epoch, loss

def count_parameters(model: nn.Module) -> int:
    """Cuenta el número de parámetros entrenables en un modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module):
    """Imprime un resumen del modelo"""
    total_params = count_parameters(model)
    print(f"Resumen del modelo:")
    print(f"   - Parámetros entrenables: {total_params:,}")
    print(f"   - Arquitectura: {model.__class__.__name__}")

# ============================================================================
# TRANSFORMACIONES DE IMÁGENES
# ============================================================================

def get_train_transforms(img_size: Tuple[int, int] = IMG_SIZE):
    """Transformaciones para el conjunto de entrenamiento (con augmentation)"""
    return transforms.Compose([
        transforms.Resize((img_size[0] + 32, img_size[1] + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=AUGMENTATION_PARAMS['rotation_degrees']),
        transforms.ColorJitter(
            brightness=AUGMENTATION_PARAMS['brightness'],
            contrast=AUGMENTATION_PARAMS['contrast'],
            saturation=AUGMENTATION_PARAMS['saturation'],
            hue=AUGMENTATION_PARAMS['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

def get_val_transforms(img_size: Tuple[int, int] = IMG_SIZE):
    """Transformaciones para validación y test (sin augmentation)"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN ESPECÍFICAS
# ============================================================================

def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         train_accs: List[float] = None, val_accs: List[float] = None):
    """Visualiza la historia del entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_title('Evolución de la Pérdida')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy (si se proporciona)
    if train_accs and val_accs:
        axes[1].plot(train_accs, label='Train Acc', color='blue')
        axes[1].plot(val_accs, label='Val Acc', color='red')
        axes[1].set_title('Evolución de la Precisión')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Precisión')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model: nn.Module, dataloader: DataLoader, 
                         device: torch.device, num_samples: int = 8):
    """Visualiza predicciones del modelo"""
    model.eval()
    
    # Obtener un batch de datos
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Hacer predicciones
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Visualizar
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Desnormalizar imagen para visualización
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'Real: {labels[i].item()}, Pred: {predicted[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# INICIALIZACIÓN AUTOMÁTICA
# ============================================================================

def initialize_project(use_colab: bool = True):
    """Inicializa todo el proyecto"""
    print("Inicializando proyecto de Deep Learning...")
    
    # Configurar entorno
    device = check_environment()
    
    # Configurar Google Drive si estamos en Colab
    if use_colab:
        setup_colab_drive()
        base_path = COLAB_PROJECT_PATH
    else:
        base_path = LOCAL_PROJECT_PATH
    
    # Crear directorios
    create_project_directories(base_path)
    
    print(f"Proyecto inicializado correctamente en: {base_path}")
    return device, base_path

# ============================================================================
# CONFIGURACIÓN POR DEFECTO AL IMPORTAR
# ============================================================================

# Configurar semillas por defecto
set_seeds(RANDOM_SEED)

print(" module_utils.py cargado correctamente")
print(" Uso recomendado:")
print(" from module_utils import *")
print(" device, project_path = initialize_project(use_colab=True)")
