# module_utils.py
"""
Utilidades para el proyecto de Predicción de Engagement Turístico con Deep Learning
"""

import os
import sys
import warnings
import random
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import folium
import folium.plugins as plugins

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Configuración global
warnings.filterwarnings('ignore')
RANDOM_SEED = 42

# Configuración de matplotlib
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def install_requirements():
    """Instala automáticamente las dependencias necesarias"""
    packages = [
        'torch', 'torchvision', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'opencv-python', 'folium', 'scikit-learn', 'Pillow'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Instalando {package}...")
            os.system(f"pip install {package} --quiet")
    
    print("Dependencias verificadas/instaladas")

def set_seeds(seed=RANDOM_SEED):
    """Configura todas las semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Semillas configuradas con valor: {seed}")

def setup_device():
    """Configura y verifica el dispositivo GPU/CPU"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU no disponible, usando CPU")
    return device

def setup_colab_drive():
    """Monta Google Drive en Colab si es necesario"""
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            print("Google Drive montado correctamente")
        else:
            print("Google Drive ya estaba montado")
        return True
    except Exception as e:
        print(f"Error montando Google Drive: {e}")
        return False

def check_environment():
    """Verifica el entorno y muestra información del sistema"""
    print("=== INFORMACIÓN DEL ENTORNO ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    
    device = setup_device()
    set_seeds(RANDOM_SEED)
    
    print("Entorno configurado correctamente")
    return device

# Constantes del proyecto
PROJECT_CONFIG = {
    'BASE_PATH': os.getcwd(),
    'DATA_FILE': 'data/poi_dataset.csv',
    'IMG_SIZE': (224, 224),
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 0.0001,
    'NUM_EPOCHS': 10,
    'PATIENCE': 3,
    'EVAL_EVERY': 2
}

def create_engagement_metric(df):
    """Crea la métrica de engagement ponderada"""
    features = ['Likes', 'Bookmarks', 'Dislikes']
    scaler = MinMaxScaler()
    
    # Normalización
    df[[f'{col}_norm' for col in features]] = scaler.fit_transform(df[features])
    
    # Cálculo de engagement
    df['engagement_score'] = (
        df['Likes_norm'] * 0.6 +
        df['Bookmarks_norm'] * 0.4 -
        df['Dislikes_norm'] * 0.2
    )
    
    # Discretización en 5 niveles
    quantiles = df['engagement_score'].quantile([0.2, 0.4, 0.6, 0.8])
    bins = [-float('inf')] + list(quantiles) + [float('inf')]
    labels = ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Excepcional']
    
    df['engagement_level'] = pd.cut(df['engagement_score'], bins=bins, labels=labels, include_lowest=True)
    
    print("Métrica de engagement creada")
    return df, scaler

def calculate_image_quality_score(df, base_path, sample_size=100):
    """Calcula score de calidad de imagen para una muestra"""
    subset = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
    image_quality_metrics = []
    
    for _, row in subset.iterrows():
        try:
            img_path = os.path.join(base_path, 'data', row['main_image_path'])
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                contrast = np.std(gray)
                sharpness = np.var(cv2.Laplacian(gray, cv2.CV_64F))
                quality_score = (contrast + sharpness) / 2
                image_quality_metrics.append(quality_score)
            else:
                image_quality_metrics.append(0)
        except:
            image_quality_metrics.append(0)
    
    df.loc[subset.index, 'image_quality_score'] = image_quality_metrics
    print(f"Calidad de imagen calculada para {len(image_quality_metrics)} muestras")
    return df

def preprocess_data(df):
    """Preprocesamiento completo de datos"""
    # Variables numéricas a normalizar
    numeric_cols = ["Likes", "Dislikes", "Bookmarks", "xps", "image_quality_score", "tier"]
    
    # Parsear listas desde strings
    df['categories_list'] = df['categories'].apply(ast.literal_eval)
    df['tags_list'] = df['tags'].apply(ast.literal_eval)
    
    # Top categorías y tags
    all_categories = [cat for cat_list in df['categories_list'] for cat in cat_list]
    category_counts = Counter(all_categories)
    top_categories = [cat for cat, _ in category_counts.most_common(10)]
    df['filtered_categories'] = df['categories_list'].apply(
        lambda lst: [c for c in lst if c in top_categories]
    )
    
    all_tags = [tag for tag_list in df['tags_list'] for tag in tag_list]
    tag_counts = Counter(all_tags)
    top_tags = [tag for tag, _ in tag_counts.most_common(15)]
    df['filtered_tags'] = df['tags_list'].apply(
        lambda lst: [t for t in lst if t in top_tags]
    )
    
    # One-hot encoding
    mlb_cat = MultiLabelBinarizer()
    categories_encoded = pd.DataFrame(
        mlb_cat.fit_transform(df['filtered_categories']),
        columns=[f"cat_{c}" for c in mlb_cat.classes_],
        index=df.index
    )
    
    mlb_tag = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(
        mlb_tag.fit_transform(df['filtered_tags']),
        columns=[f"tag_{t}" for t in mlb_tag.classes_],
        index=df.index
    )
    
    # Codificación del target
    le_target = LabelEncoder()
    df['engagement_level_encoded'] = le_target.fit_transform(df['engagement_level'])
    
    # Concatenar encodings
    df = pd.concat([df, categories_encoded, tags_encoded], axis=1)
    
    print("Preprocesamiento completado")
    
    return df, mlb_cat, mlb_tag, le_target, numeric_cols

def split_and_normalize_data(df, numeric_cols, target_col='engagement_level_encoded'):
    """División estratificada y normalización sin data leakage"""
    # División estratificada
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[target_col], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=RANDOM_SEED
    )
    
    # Normalización sin data leakage
    scaler_numeric = MinMaxScaler()
    scaler_numeric.fit(train_df[numeric_cols])
    
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[[col + "_norm" for col in numeric_cols]] = scaler_numeric.transform(train_df[numeric_cols])
    val_df_scaled[[col + "_norm" for col in numeric_cols]] = scaler_numeric.transform(val_df[numeric_cols])
    test_df_scaled[[col + "_norm" for col in numeric_cols]] = scaler_numeric.transform(test_df[numeric_cols])
    
    print(f"División completada - Train: {len(train_df_scaled)}, Val: {len(val_df_scaled)}, Test: {len(test_df_scaled)}")
    
    return train_df_scaled, val_df_scaled, test_df_scaled, scaler_numeric

def prepare_features_and_tensors(train_df, val_df, test_df, numeric_cols):
    """Prepara features y convierte a tensores PyTorch"""
    # Selección de features
    numeric_features = [col + "_norm" for col in numeric_cols]
    category_features = [col for col in train_df.columns if col.startswith('cat_')]
    tag_features = [col for col in train_df.columns if col.startswith('tag_')]
    model_features = numeric_features + category_features + tag_features
    
    # Conversión a tensores
    def df_to_tensors(df_subset, features, target_col='engagement_level_encoded'):
        X = df_subset[features].values.astype(np.float32)
        y = df_subset[target_col].values.astype(np.int64)
        
        # Limpiar NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return torch.tensor(X), torch.tensor(y)
    
    X_train, y_train = df_to_tensors(train_df, model_features)
    X_val, y_val = df_to_tensors(val_df, model_features)
    X_test, y_test = df_to_tensors(test_df, model_features)
    
    print(f"Tensores creados - Features: {len(model_features)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, model_features

class POIDataset(Dataset):
    """Dataset personalizado para datos multimodales"""
    def __init__(self, X_metadata, y_labels, image_paths, base_path, transform=None):
        self.X_metadata = X_metadata
        self.y_labels = y_labels
        self.image_paths = image_paths
        self.base_path = base_path
        self.transform = transform
        self.default_image = torch.zeros(3, PROJECT_CONFIG['IMG_SIZE'][0], PROJECT_CONFIG['IMG_SIZE'][1])
        
    def __len__(self):
        return len(self.X_metadata)
    
    def __getitem__(self, idx):
        metadata = self.X_metadata[idx]
        label = self.y_labels[idx]
        
        # Cargar imagen
        img_path = os.path.join(self.base_path, 'data', self.image_paths[idx])
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            else:
                image = self.default_image.clone()
        except:
            image = self.default_image.clone()
        
        return {
            'image': image,
            'metadata': torch.tensor(metadata, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalPOIModel(nn.Module):
    """Modelo multimodal CNN + MLP"""
    def __init__(self, metadata_size, num_classes, dropout_rate=0.2):
        super(MultimodalPOIModel, self).__init__()
        
        # CNN para imágenes
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        cnn_features = 512
        
        # Congelar capas iniciales
        for param in list(self.cnn.parameters())[:-10]:
            param.requires_grad = False
        
        # MLP para metadatos
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Clasificador final
        combined_features = cnn_features + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.metadata_mlp(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        return output

def get_transforms():
    """Obtiene transformaciones para imágenes"""
    img_size = PROJECT_CONFIG['IMG_SIZE']
    
    transform_train = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Función de entrenamiento por época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        try:
            images = batch['image'].to(device, non_blocking=True)
            metadata = batch['metadata'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            if torch.isnan(images).any() or torch.isnan(metadata).any():
                continue
            
            optimizer.zero_grad()
            outputs = model(images, metadata)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                continue
            
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        except:
            continue
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Función de validación por época"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                images = batch['image'].to(device, non_blocking=True)
                metadata = batch['metadata'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            except:
                continue
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def create_folium_map(df):
    """Crea mapa interactivo con POIs"""
    center_lat = df['locationLat'].mean()
    center_lon = df['locationLon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    for idx, row in df.iterrows():
        popup_text = f"""
        <b>{row['name']}</b><br>
        Tier: {row['tier']}<br>
        Categories: {', '.join(ast.literal_eval(row['categories']))}<br>
        Visits: {row['Visits']}<br>
        Likes: {row['Likes']}<br>
        """
        folium.Marker(
            location=[row['locationLat'], row['locationLon']],
            popup=popup_text,
            icon=folium.Icon(color='red' if row['tier'] == 1 else 'blue')
        ).add_to(m)
    
    return m

def plot_training_history(train_losses, val_losses, train_accs, val_accs, eval_every=1):
    """Visualiza la historia del entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_train = range(1, len(train_losses) + 1)
    epochs_val = range(1, len(val_losses) * eval_every + 1, eval_every)
    
    # Loss
    axes[0].plot(epochs_train, train_losses, 'b-', label='Train Loss', alpha=0.8)
    axes[0].plot(epochs_val, val_losses, 'r-', label='Val Loss', alpha=0.8)
    axes[0].set_title('Evolución de la Pérdida')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs_train, train_accs, 'b-', label='Train Accuracy', alpha=0.8)
    axes[1].plot(epochs_val, val_accs, 'r-', label='Val Accuracy', alpha=0.8)
    axes[1].set_title('Evolución de la Precisión')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataloader, device, le_target):
    """Evaluación completa del modelo"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, metadata)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas
    test_accuracy = accuracy_score(all_labels, all_predictions)
    class_names = le_target.classes_
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()
    
    return test_accuracy

def initialize_project():
    """Inicializa todo el proyecto"""
    print("Inicializando proyecto de Deep Learning...")
    install_requirements()
    device = check_environment()
    setup_colab_drive()
    
    # Detectar y configurar la ruta base automáticamente
    base_path = os.getcwd()
    PROJECT_CONFIG['BASE_PATH'] = base_path
    
    print(f"Proyecto inicializado en: {base_path}")
    print(f"Estructura detectada:")
    print(f"  - Datos: {os.path.exists(os.path.join(base_path, 'data'))}")
    print(f"  - Notebook: {os.path.exists(os.path.join(base_path, 'notebooks'))}")
    
    return device, base_path

print("module_utils.py cargado correctamente")
print("Uso: from module_utils import *")
print("Inicializar: device, base_path = initialize_project()")