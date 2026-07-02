# 🔬 Détection et classification automatiques des dislocations sur des images ECCI

## Présentation

Ce projet a été réalisé dans le cadre de mon stage de **Master 1 MISTRE** au **Laboratoire des Technologies de la Microélectronique (LTM)** de l'Université Grenoble Alpes.

L'objectif est de développer une chaîne complète de traitement permettant de détecter automatiquement les dislocations présentes dans des images **ECCI (Electron Channeling Contrast Imaging)**, puis de les classifier en trois catégories (**blanches**, **noires** et **mixtes**) grâce à des techniques de **Deep Learning**.

Le pipeline combine un réseau **U-Net** pour la segmentation des dislocations et un réseau **ResNet34** pour leur classification.

---

# 📁 Structure du projet

```text
📦 Detection-dislocations-ECCI
│
├── README.md
├── requirements.txt
├── ApplicationLTM.py
├── index3.html
│
├── code/
│   ├── split_dataset_colab.ipynb
│   ├── 03_train_unet_stable.ipynb
│   ├── create_classification_dataset.py
│   ├── equilibrage_claude.ipynb
│   ├── TRAIN_RESNET_FINAL.ipynb
│   └── ...
│
├── models/
│   ├── unet_binary_dislocation_stable_A.pth
│   └── resnet34_ecci_1.pth
│
├── uploads/
│
└── results/
```

---

# ⚙️ Pipeline de traitement

Le pipeline est constitué des étapes suivantes :

1. Préparation du jeu de données
2. Entraînement du modèle U-Net (segmentation binaire)
3. Extraction automatique des patches
4. Création du jeu de données de classification
5. Entraînement du modèle ResNet34
6. Développement de l'application web Flask
7. Calcul automatique de la densité de dislocations
8. Génération d'un rapport PDF
9. Évaluation de la robustesse du système (bruit et résolution)

---

# 🛠 Technologies utilisées

- Python
- PyTorch
- Torchvision
- Segmentation Models PyTorch
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Pillow
- Flask
- ReportLab

---

# 💻 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/asmaabas22-web/D-tection-automatique-de-dislocations-par-deep-learning.git
cd D-tection-automatique-de-dislocations-par-deep-learning
```
```bash
## 2. Créer un environnement

conda env create -f environment.yaml
```
```bash
## 3.Activer l'environnement

conda activate dislocation-env

# 🚀 Lancer l'application
```
```bash
python ApplicationLTM.py
```

Puis ouvrir dans le navigateur :

```
http://127.0.0.1:5000
```

---

# 🌐 Utilisation

1. Charger une image ECCI.
2. Lancer la détection automatique.
3. Le modèle U-Net génère un masque de segmentation.
4. Les dislocations détectées sont extraites automatiquement.
5. Le modèle ResNet34 classe chaque dislocation.
6. La densité de dislocations est calculée automatiquement.
7. Un rapport PDF est généré.

---

# 📊 Fonctionnalités

✔ Détection automatique des dislocations

✔ Segmentation par U-Net

✔ Extraction automatique des patches

✔ Classification (blanche, noire, mixte)

✔ Calcul automatique de la densité

✔ Génération d'un rapport PDF

✔ Interface Web Flask

✔ Compatible CPU et GPU

---

# 📩 Auteur

**Asmaa BASSOU**

Master 1 MISTRE

Université Grenoble Alpes

2026
