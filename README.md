# 🎯 Tourist POI Engagement Prediction with Deep Learning

[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/IrisMejuto/deep_learning/blob/main/notebooks/02_simplified_analysis.ipynb)

## 📋 Description

Multimodal Deep Learning model that predicts engagement levels for tourist POIs (Points of Interest) by combining image analysis using CNN ResNet18 and metadata processing with MLP. The system automatically classifies each POI into 5 engagement levels: **Very Low**, **Low**, **Medium**, **High**, and **Exceptional**.

## 🚀 How to Use in Google Colab

### Step-by-step Instructions

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Create a new notebook**: Click on "New notebook"

3. **Paste this code** in the first cell:

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Go to Drive
%cd /content/drive/MyDrive

# Clone repository
repo_url = "https://github.com/IrisMejuto/deep_learning.git"
repo_name = "deep_learning"

if os.path.exists(repo_name):
    !rm -rf {repo_name}

!git clone {repo_url}

# Enter the cloned project
%cd {repo_name}

# Verify project structure
print("Project structure:")
!ls -la
print("Data content:")
!ls -la data/
```

4. **Execute the cell** - this automatically clones the entire project

5. **Open the project notebook**:
   * In Colab's left sidebar, click on "Files"
   * Navigate to: `drive/MyDrive/deep_learning/notebooks/`
   * Double-click on `02_simplified_analysis.ipynb`
   * The notebook will open ready to execute

6. **Execute the notebook cells** in order - no need to copy anything else

### 🔄 Complete Process:
Once the repository is cloned, the code automatically executes:
1. Mounts Google Drive
2. Automatically clones this complete repository to your Drive
3. Installs all dependencies (PyTorch, OpenCV, scikit-learn, etc.)
4. Loads and processes the complete dataset of 1,569 POIs
5. Creates engagement metric based on Likes, Bookmarks, and Dislikes
6. Trains the multimodal model for 10 epochs
7. Evaluates performance and generates visualizations
8. Saves the trained model for future use

**Just execute the cells - no manual configuration required.**

## 📁 Project Structure

```
├── data/
│   ├── poi_dataset.csv          # Main dataset (1,569 POIs)
│   └── data_main/               # POI images organized by ID
├── notebooks/
│   ├── 01_full_analysis.ipynb   # Complete analysis with experimentation
│   └── 02_simplified_analysis.ipynb # Colab-ready version (recommended)
├── src/
│   └── module_utils.py          # Project functions and classes
├── README.md                    # This documentation
└── memoria_tecnica.pdf          # Complete technical documentation
```

## 📊 Dataset

The dataset contains **1,569 tourist POIs** with:

**Metadata:**
* Geolocation (latitude, longitude)
* Categories (History, Culture, Architecture, Heritage, etc.)
* Specific descriptive tags
* Importance tier (1-4)
* Experience points (XPs)

**Engagement metrics:**
* Visits: Number of visits
* Likes: Positive ratings
* Dislikes: Negative ratings  
* Bookmarks: Saved as favorites

**Visual resources:**
* One main image per POI
* Automatic visual quality score

## 🧠 Model Architecture

**MultimodalPOIModel**
* **CNN Branch (Pretrained ResNet18)**
  * Visual feature extraction
  * Fine-tuning of final layers
  * Output: 512 features

* **MLP Branch**
  * Normalized metadata processing
  * Dense layers with BatchNorm and Dropout
  * Output: 128 features

* **Classifier**
  * Feature fusion (640 total features)
  * Classification layers with regularization
  * Output: 5 engagement classes

## 🔬 Methodology

**Preprocessing:**
* MinMax normalization of numerical variables
* One-hot encoding of main categories and tags
* Weighted engagement metric creation
* Balanced quintile discretization

**Training:**
* Stratified split: 70% train, 15% val, 15% test
* Adam optimizer with weight decay
* CrossEntropyLoss with label smoothing
* Early stopping based on validation loss
* Data augmentation for images

**Evaluation:**
* Accuracy, precision, recall per class
* Confusion matrix
* Image-engagement correlation analysis

## 📈 Results

**Model performance:**
* Test Accuracy: **54.47%**
* Best performance on extreme classes (Very Low/Exceptional)
* Stable convergence in 10 epochs
* Image-engagement correlation: **0.536**

**Balanced class distribution:**
* Very Low: 20.01%
* Low: 20.01%  
* Medium: 19.95%
* High: 20.01%
* Exceptional: 20.01%

**Key insights:**
* Higher quality images correlate positively with engagement
* Tier 1 POIs tend to have higher engagement
* History and Culture categories dominate the dataset
* Geolocation influences engagement patterns

## 📚 Additional Documentation

* **Colab Notebook**: `notebooks/02_simplified_analysis.ipynb` is optimized for direct execution in Google Colab with automatic configuration.

* **Complete Analysis**: See `notebooks/01_full_analysis.ipynb` for detailed data exploration, hyperparameter experimentation, and extended results analysis (requires manual configuration).

* **Technical Report**: Consult `memoria_tecnica.pdf` for complete methodology, literature review, technical decision justification, and limitation analysis.

* **Source Code**: All functions are documented in `src/module_utils.py` with usage examples.

## ⚙️ Main Dependencies

* `torch >= 2.0.0`
* `torchvision >= 0.15.0`
* `pandas >= 1.5.0`
* `scikit-learn >= 1.1.0`
* `opencv-python >= 4.6.0`
* `folium >= 0.14.0`

*Dependencies are automatically installed in the notebook.*

## 📞 Contact

**Repository**: [https://github.com/IrisMejuto/deep_learning](https://github.com/IrisMejuto/deep_learning)

---

**If you find this project useful, consider giving it a star on GitHub.**