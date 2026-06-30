# -*- coding: utf-8 -*-
"""
create_classification_dataset.py
Création automatique du dataset de classification
à partir des images ECCI et des masques annotés.
Version corrigée — supporte .npy et .png, patch 96
"""

import os
import cv2
import numpy as np
from skimage import io
import glob

# ============================================================
# PARAMÈTRES
# ============================================================
IMAGE_DIR  = r"C:\Users\Asmaa\Documents\dislocation_2026/images"
MASK_DIR   = r"C:\Users\Asmaa\Documents\dislocation_2026/masks"
OUTPUT_DIR = r"C:\Users\Asmaa\Documents\dislocation_2026/dataset_classification"

PATCH_SIZE = 96  # ← corrigé : 140 au lieu de 64

CLASSES = {
    1: "blanc",
    2: "noir",
    3: "mixte"
}

# ============================================================
# CRÉATION DOSSIERS
# ============================================================
for class_name in CLASSES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

# ============================================================
# CHARGEMENT DU MASQUE — supporte .npy ET .png
# ============================================================
def load_mask(base_name):
    """Charge le masque en .npy si disponible, sinon .png."""
    mask_npy = os.path.join(MASK_DIR, f"{base_name}_mask.npy")
    mask_png = os.path.join(MASK_DIR, f"{base_name}_mask.png")

    if os.path.exists(mask_npy):
        mask = np.load(mask_npy).astype(np.uint8)
        print(f"   ✅ Masque .npy chargé")
        return mask
    elif os.path.exists(mask_png):
        mask = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
        print(f"   ✅ Masque .png chargé")
        return mask
    else:
        print(f"   ❌ Masque introuvable pour {base_name}")
        return None

# ============================================================
# EXTRACTION PATCH ROBUSTE
# ============================================================
def extract_patch(image, center_x, center_y, patch_size):
    """Extrait un patch centré avec padding si bord d'image."""
    half = patch_size // 2
    h, w = image.shape[:2]

    x1 = center_x - half
    x2 = center_x + half
    y1 = center_y - half
    y2 = center_y + half

    pad_left = pad_right = pad_top = pad_bottom = 0

    if x1 < 0:
        pad_left = -x1
        x1 = 0
    if y1 < 0:
        pad_top = -y1
        y1 = 0
    if x2 > w:
        pad_right = x2 - w
        x2 = w
    if y2 > h:
        pad_bottom = y2 - h
        y2 = h

    patch = image[y1:y2, x1:x2]

    patch = cv2.copyMakeBorder(
        patch,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )

    # Resize uniquement si nécessaire
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))

    return patch

# ============================================================
# CENTRAGE ROBUSTE
# ============================================================
def get_refined_center(cnt, mask_class):
    """Calcule le centre précis de la dislocation."""
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

    x, y, w, h = cv2.boundingRect(cnt)
    roi = mask_class[y:y+h, x:x+w]
    if roi.size > 0:
        coords = cv2.findNonZero(roi)
        if coords is not None:
            coords = coords.reshape(-1, 2)
            cx = int(np.mean(coords[:, 0])) + x
            cy = int(np.mean(coords[:, 1])) + y

    return cx, cy

# ============================================================
# TRAITEMENT D'UNE IMAGE
# ============================================================
def process_image(image_path, patch_counter_start=0):
    base_name = os.path.basename(image_path).split('.')[0]
    print(f"\n📷 Traitement : {base_name}")

    # Charger l'image
    image = io.imread(image_path)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Recadrage barre noire
    h = image.shape[0]
    image = image[0:int(h * 0.93), :]

    # Charger le masque
    mask = load_mask(base_name)
    if mask is None:
        return patch_counter_start

    patch_counter = patch_counter_start

    for class_id, class_name in CLASSES.items():
        print(f"   → Classe : {class_name}")

        class_mask = np.uint8(mask == class_id) * 255

        contours, _ = cv2.findContours(
            class_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"      Objets trouvés : {len(contours)}")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3:
                continue

            cx, cy = get_refined_center(cnt, class_mask)
            patch  = extract_patch(image, cx, cy, PATCH_SIZE)

            patch_name = (
                f"{class_name}_{patch_counter:05d}.png"
            )
            save_path = os.path.join(OUTPUT_DIR, class_name, patch_name)
            cv2.imwrite(save_path, patch)
            patch_counter += 1

    print(f"   ✅ {patch_counter - patch_counter_start} patches sauvegardés")
    return patch_counter

# ============================================================
# MAIN
# ============================================================
def main():
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.tif"))

    if not image_paths:
        print("❌ Aucune image trouvée")
        return

    print(f"\n🔍 {len(image_paths)} images trouvées")

    total = 0
    for image_path in image_paths:
        total = process_image(image_path, total)

    print(f"\n{'='*40}")
    print(f"✅ DATASET TERMINÉ — {total} patches au total")
    print(f"{'='*40}")
    print(f"\n📋 ÉTAPE SUIVANTE :")
    print(f"   Vérifie les patches dans {OUTPUT_DIR}")
    print(f"   blanc/ → {len(glob.glob(OUTPUT_DIR+'/blanc/*.png'))} patches")
    print(f"   noir/  → {len(glob.glob(OUTPUT_DIR+'/noir/*.png'))} patches")
    print(f"   mixte/ → {len(glob.glob(OUTPUT_DIR+'/mixte/*.png'))} patches")

if __name__ == "__main__":
    main()