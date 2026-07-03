# 🔬 Détection et classification automatiques des dislocations dans des images ECCI par Deep Learning

## 📖 Présentation

Ce projet a été réalisé dans le cadre de mon stage de **Master 1 MISTRE** au **Laboratoire des Technologies de la Microélectronique (LTM)** de l'Université Grenoble Alpes.

L'objectif est de développer une chaîne complète d'analyse permettant de détecter automatiquement les dislocations présentes dans des images **ECCI (Electron Channeling Contrast Imaging)**, de les classifier en trois catégories (blanches, noires et mixtes), puis de calculer automatiquement leur densité.

Le pipeline repose sur deux modèles de Deep Learning :

- **U-Net** : segmentation binaire des dislocations.
- **ResNet34** : classification des dislocations détectées.

Une interface Web développée avec **Flask** permet d'utiliser facilement l'application sans avoir à manipuler directement le code.

---

# 📁 Structure du projet

```
D-tection-automatique-de-dislocations-par-deep-learning
│
├── README.md
├── requirements.txt
├── environment.yml
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

### Description des principaux fichiers

| Fichier | Description |
|----------|-------------|
| ApplicationLTM.py | Application Flask |
| index3.html | Interface utilisateur |
| code/ | Scripts d'entraînement des modèles |
| models/ | Modèles U-Net et ResNet34 entraînés |
| uploads/ | Images importées par l'utilisateur |
| results/ | Résultats et rapports PDF générés |

---

# 📂 Jeu de données

Le jeu de données est constitué d'images **ECCI annotées manuellement**.

Les annotations servent :

- à entraîner le modèle **U-Net** pour la segmentation ;
- à extraire automatiquement les patches des dislocations ;
- à créer le jeu de données utilisé pour entraîner **ResNet34**.

---

# ⚙️ Pipeline de traitement

Le pipeline est constitué des étapes suivantes :

1. Préparation des images ECCI.
2. Annotation et création des masques.
3. Entraînement du modèle U-Net.
4. Segmentation automatique des dislocations.
5. Extraction automatique des patches.
6. Création du jeu de données de classification.
7. Entraînement du modèle ResNet34.
8. Développement de l'application Flask.
9. Calcul automatique de la densité de dislocations.
10. Génération automatique d'un rapport PDF.
11. Évaluation de la robustesse du système (résolution et bruit).

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

## 1. Cloner le dépôt

```bash
git clone https://github.com/asmaabas22-web/D-tection-automatique-de-dislocations-par-deep-learning.git
cd D-tection-automatique-de-dislocations-par-deep-learning
```

## 2. Créer l'environnement Conda

```bash
conda env create -f environment.yml
```

## 3. Activer l'environnement

Le nom de l'environnement est défini dans le fichier **environment.yml**.

Par exemple :

```bash
conda activate dislocation-env
```

Si Conda n'est pas disponible :

```bash
pip install -r requirements.txt
```

---

# 🚀 Lancer l'application

Exécuter :

```bash
python ApplicationLTM.py
```

Puis ouvrir le navigateur :

```
http://127.0.0.1:5000
```

---

# 🌐 Utilisation

L'application permet :

1. Charger une image ECCI (.tif).
2. Renseigner la largeur physique de l'image.
3. Choisir le seuil du U-Net.
4. Lancer la détection automatique.

L'application génère automatiquement :

- le masque de segmentation ;
- l'image annotée ;
- le nombre de dislocations détectées ;
- la classification des dislocations (blanche, noire ou mixte) ;
- la densité de dislocations ;
- un rapport PDF récapitulatif.

---

# 📊 Fonctionnalités

✔ Détection automatique des dislocations

✔ Segmentation par U-Net

✔ Extraction automatique des patches

✔ Classification des dislocations (blanche, noire, mixte)

✔ Calcul automatique de la densité

✔ Génération automatique d'un rapport PDF

✔ Interface Web Flask

✔ Compatible CPU et GPU

---

# 📈 Résultats

Le système permet :

- la détection automatique des dislocations présentes dans une image ECCI ;
- la classification de chaque dislocation en trois catégories ;
- le calcul automatique de la densité de dislocations ;
- la génération d'un rapport PDF ;
- l'évaluation de la robustesse du pipeline face aux variations de résolution et au bruit.

---

# ⚠️ Limites actuelles

Malgré les bonnes performances obtenues, plusieurs limitations subsistent :

- certaines très petites dislocations ne sont pas détectées par le modèle U-Net ;
- des confusions persistent entre les classes **blanche**, **noire** et **mixte**, notamment lorsque le contraste est faible ;
- les performances diminuent lorsque les images présentent un niveau de bruit important ;
- les résultats restent dépendants du jeu de données utilisé pour l'entraînement.

---

# 🔬 Perspectives

Les pistes d'amélioration envisagées sont :

- enrichir la base de données avec davantage d'images annotées ;
- améliorer la détection des très petites dislocations ;
- réduire les confusions entre les différentes classes ;
- tester des architectures de segmentation plus récentes (U-Net++, Attention U-Net, DeepLabV3+, etc.) ;
- intégrer une étape de débruitage avant la segmentation afin d'améliorer la robustesse sur les images bruitées ;
- optimiser automatiquement le seuil de segmentation.

---

# 👩‍💻 Auteur

**Asmaa Bassou**

Master 1 MISTRE

Université Grenoble Alpes

Laboratoire des Technologies de la Microélectronique (LTM)

---

# 📄 Licence

Ce projet a été développé dans le cadre d'un stage académique de Master 1 au Laboratoire des Technologies de la Microélectronique (LTM), Université Grenoble Alpes.
