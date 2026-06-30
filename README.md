# Détection et classification automatiques des dislocations sur des images ECCI

## Présentation

Ce projet a été réalisé dans le cadre de mon stage de Master 1 au Laboratoire des Technologies de la Microélectronique (LTM).

L'objectif est de développer une chaîne complète de traitement permettant de détecter automatiquement les dislocations présentes dans des images ECCI (*Electron Channeling Contrast Imaging*), puis de les classifier en trois catégories (blanches, noires et mixtes) grâce à des techniques de Deep Learning.

---

## Pipeline de traitement

1. Préparation du jeu de données
2. Entraînement du modèle U-Net (segmentation binaire)
3. Création du jeu de données pour la classification
4. Entraînement du modèle ResNet34
5. Développement d'une application web avec Flask
6. Calcul automatique de la densité de dislocations
7. Évaluation de la robustesse du système

---

## Technologies utilisées

- Python
- PyTorch
- Segmentation Models PyTorch
- OpenCV
- Albumentations
- Flask
- NumPy
- Matplotlib

---

## Structure du projet

```
code/
│
├── split_dataset_colab.ipynb
├── 03_train_unet_stable.ipynb
├── create_classification_dataset.py
├── equilibrage_claude.ipynb
└── TRAIN_RESNET_FINAL.ipynb

ApplicationLTM.py
index3.html
```

---

## Fonctionnement

Le pipeline développé comprend les étapes suivantes :

- Prétraitement des images ECCI (CLAHE)
- Détection automatique des dislocations par U-Net
- Extraction des objets détectés
- Classification des dislocations (blanche, noire ou mixte) avec ResNet34
- Calcul automatique de la densité de dislocations
- Génération d'un rapport PDF des résultats
- Visualisation des résultats via une interface web Flask

---

## Résultats obtenus

Le système développé permet de :

- détecter automatiquement les dislocations sur des images ECCI ;
- classifier chaque dislocation en trois catégories (blanche, noire ou mixte) ;
- calculer automatiquement la densité de dislocations ;
- générer un rapport PDF contenant les résultats de l'analyse ;
- évaluer la robustesse du pipeline face au bruit et à la diminution de la résolution des images.

---

## Auteur

**Asmaa BASSOU**

Stage de Master 1 MISTRE  
Laboratoire des Technologies de la Microélectronique (LTM)  
Université Grenoble Alpes 
