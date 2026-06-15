# -*- coding: utf-8 -*-
"""
ApplicationLTM.py
Application Flask :
1. Charger une image ECCI/SEM
2. Détecter les dislocations avec U-Net binaire
3. Régler le seuil U-Net depuis l'interface
4. Extraire les objets détectés
5. Classifier chaque dislocation avec ResNet34 : blanc / mixte / noir
6. Calculer la densité
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import os
import cv2
import io
import base64
import numpy as np

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision import models, transforms

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from flask import send_file
# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)
last_pdf_data = {}
CORS(app)


# ============================================================
# CHEMINS DES MODÈLES
# ============================================================

UNET_PATH = "unet_binary_dislocation_stable_A.pth"
#UNET_PATH = "unet_robuste_retrained.pth"
RESNET_PATH = "resnet34_ecci_1.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device :", DEVICE)


# ============================================================
# PARAMÈTRES
# ============================================================

PATCH_SIZE_RESNET = 96

MIN_AREA = 4
MAX_AREA = 500

CLASSES = ["blanc", "mixte", "noir"]

COLORS = {
    "blanc": (0, 255, 0),     # vert
    "mixte": (255, 0, 0),     # bleu
    "noir":  (0, 0, 255),     # rouge
    "incertain": (0, 255, 255)  # incertitude 
}


# ============================================================
# CHARGEMENT U-NET BINAIRE
# ============================================================

print("🔄 Chargement U-Net binaire...")

unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation=None
)

unet_model.load_state_dict(
    torch.load(UNET_PATH, map_location=DEVICE)
)

unet_model = unet_model.to(DEVICE)
unet_model.eval()

print("✅ U-Net chargé")


# ============================================================
# CHARGEMENT RESNET34
# ============================================================

print("🔄 Chargement ResNet34...")

resnet_model = models.resnet34(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 3)

resnet_model.load_state_dict(
    torch.load(RESNET_PATH, map_location=DEVICE)
)

resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

print("✅ ResNet34 chargé")


# ============================================================
# TRANSFORM RESNET
# ============================================================

resnet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(image):
    """Convertir une image numpy en base64 pour l'envoyer à l'interface."""

    if len(image.shape) == 2:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


# ============================================================
# CHARGEMENT ET PRÉTRAITEMENT IMAGE
# ============================================================

def load_image_from_bytes(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)

    image_original = cv2.imdecode(
        nparr,
        cv2.IMREAD_GRAYSCALE
    )

    if image_original is None:
        raise ValueError("Impossible de décoder l'image")

    h = image_original.shape[0]

    image_original = image_original[
        0:int(h * 0.93),
        :
    ]

    image_original = image_original.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image_clahe = clahe.apply(image_original)

    return image_original, image_clahe


# ============================================================
# PRÉDICTION U-NET
# ============================================================

def predict_unet(image, threshold):
    """
    Détection binaire des dislocations avec U-Net.
    Important : normalisation identique à l'entraînement :
    A.Normalize(mean=0.5, std=0.5)
    """

    img_norm = image.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5

    tensor = torch.tensor(
        img_norm,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = unet_model(tensor)
        prob = torch.sigmoid(output)

    prob = prob.squeeze().cpu().numpy()

    binary_mask = (prob > threshold).astype(np.uint8)

    return binary_mask, prob


# ============================================================
# EXTRACTION DES OBJETS
# ============================================================

def extract_objects(mask, image):
    """
    Extraire les centres des objets détectés par U-Net.
    Filtre :
    - taille de l'objet
    - zone trop noire de l'image
    """

    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        

        objects.append((cx, cy, area))

    return objects

# ============================================================
# EXTRACTION PATCH POUR RESNET
# ============================================================

def crop_patch_safe(image, cx, cy, patch_size):
    """Extraire un patch 96×96 centré sur la dislocation."""

    half = patch_size // 2
    h, w = image.shape

    x1 = cx - half
    x2 = cx + half
    y1 = cy - half
    y2 = cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    patch = image[y1:y2, x1:x2]

    patch = cv2.copyMakeBorder(
        patch,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT
    )

    patch = cv2.resize(patch, (patch_size, patch_size))

    return patch


# ============================================================
# CLASSIFICATION RESNET34
# ============================================================

def classify_patch(patch):
    """Classer un patch en blanc / mixte / noir."""

    pil_img = Image.fromarray(patch).convert("L")

    tensor = resnet_transform(pil_img)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = resnet_model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = CLASSES[pred.item()]
    confidence = conf.item()

    return label, confidence


# ============================================================
# DENSITÉ
# ============================================================

def calculate_density(dislocation_count, image_shape, image_width_um):
    """Calculer surface analysée et densité."""

    h, w = image_shape
    print(f"Largeur pixels : {w}")
    print(f"Hauteur pixels : {h}")

    pixel_size_um = image_width_um / w
    surface_um2 = w * h * (pixel_size_um ** 2)

    if surface_um2 > 0:
        density = dislocation_count / surface_um2
    else:
        density = 0

    return surface_um2, density, pixel_size_um


# ============================================================
# OVERLAY FINAL
# ============================================================

def draw_results(image, results):
    """
    Dessiner les résultats classés sur l'image.
    Version plus visible :
    - cercle plus grand
    - texte plus lisible
    - point rempli au centre
    """

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for r in results:

        x = int(r["x"])
        y = int(r["y"])
        label = r["label"]
        confidence = r["confidence"]

        color = COLORS[label]

        # point central rempli
        cv2.circle(
            overlay,
            (x, y),
            3,
            color,
            -1
        )

        # cercle autour de la dislocation
        cv2.circle(
            overlay,
            (x, y),
            10,
            color,
            2
        )

        # position du texte protégée pour ne pas sortir de l'image
        text = f"{label} {confidence:.2f}"
        text_x = min(x + 10, image.shape[1] - 90)
        text_y = max(y - 10, 15)

      #  cv2.putText(
        #    overlay,
         #   text,
          #  (text_x, text_y),
           # cv2.FONT_HERSHEY_SIMPLEX,
            #0.45,
            #color,
            #1,
         #   cv2.LINE_AA
        #)

    return overlay

# ============================================================
# ROUTES FLASK
# ============================================================

@app.route("/")
def index():
    return send_file("index3.html")


@app.route("/process", methods=["POST"])
def process_image():

    try:
        print("📥 Réception image...")

        if "image" not in request.files:
            return jsonify({"error": "Aucun fichier image fourni"}), 400

        file = request.files["image"]
        
        image_name = file.filename

        if file.filename == "":
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Largeur physique
        image_width_um = request.form.get("image_width_um")

        if image_width_um is None:
            return jsonify({"error": "Largeur physique obligatoire"}), 400

        image_width_um = float(image_width_um)

        if image_width_um <= 0:
            return jsonify({"error": "La largeur doit être positive"}), 400

        # Seuil U-Net reçu depuis le curseur
        threshold = request.form.get("threshold", 0.15)
        threshold = float(threshold)

        # Sécuriser le seuil
        threshold = max(0.05, min(threshold, 0.30))

        print(f"🎚️ Seuil utilisé : {threshold}")

        # Charger image
        image_bytes = file.read()
        image_original, image = load_image_from_bytes(
        image_bytes
)
        print("Image Flask shape :", image.shape)
        print("Image Flask min/max/mean :", image.min(), image.max(), image.mean())
        cv2.imwrite("debug_image_flask.png", image)




        print(f"✅ Image chargée : {image.shape}")

        # U-Net
        mask, prob = predict_unet(image, threshold)
        print("Pixels U-Net :", np.sum(mask))
        # Overlay U-Net seul

        overlay_unet = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay_unet[mask > 0] = [0, 0, 255]

        overlay_unet = cv2.addWeighted(
            cv2.cvtColor(image_original, cv2.COLOR_GRAY2BGR),
            0.7,
            overlay_unet,
            0.3,
            0
        )

        overlay_unet_rgb = cv2.cvtColor(
            overlay_unet,
            cv2.COLOR_BGR2RGB
        )

        overlay_unet_b64 = image_to_base64(
            overlay_unet_rgb
        )
        # Extraction objets
       # objects = extract_objects(mask)
        objects = extract_objects(mask, image)
        print("\n====================")
        print("EXTRACTION OBJETS")
        print("====================")
        print("Objets extraits :", len(objects))
        print(f"🔎 Objets détectés : {len(objects)}")
        print("OBJETS U-NET :", len(objects))
        # Classification
        results = []

        for cx, cy, area in objects:

            patch = crop_patch_safe(
                image,
                cx,
                cy,
                PATCH_SIZE_RESNET
            )

            label, confidence = classify_patch(patch)

            # Si ResNet n'est pas assez sûr, on marque la classe comme incertaine
            if confidence < 0.40:
                label = "incertain"

            results.append({
                "x": cx,
                "y": cy,
                "area": float(area),
                "label": label,
                "confidence": float(confidence)
            })
        print("\n====================")
        print("CLASSIFICATION RESNET")
        print("====================")
        print("Objets classifiés :", len(results))
        # Comptage
        print("OBJETS CLASSIFIÉS :", len(results))
        n_blanc = sum(1 for r in results if r["label"] == "blanc")
        n_noir = sum(1 for r in results if r["label"] == "noir")
        n_mixte = sum(1 for r in results if r["label"] == "mixte")
        n_incertain = sum(1 for r in results if r["label"] == "incertain")
        total_dislo = len(results)
        print("\n====================")
        print("RÉSULTATS")
        print("====================")
        print("Blanches :", n_blanc)
        print("Noires   :", n_noir)
        print("Mixtes   :", n_mixte)
        print("Incertain :", n_incertain)
        print("Total    :", total_dislo)
        # Densité
        surface_um2, density, pixel_size_um = calculate_density(
            total_dislo,
            image.shape,
            image_width_um
        )

        # Images de sortie
        overlay = draw_results( image_original,results)

        mask_vis = (mask * 255).astype(np.uint8)

        original_b64 = image_to_base64(image_original)
        mask_b64 = image_to_base64(mask_vis)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_b64 = image_to_base64(overlay_rgb)
        
        global last_pdf_data

        pdf_path = os.path.join("results", "overlay_result.png")
        cv2.imwrite(pdf_path, overlay)

        last_pdf_data = {
            "threshold": threshold,
            "image_width_um": image_width_um,
            "surface_mm2": round(surface_um2 / 1e6, 6),
            "pixel_size_um": round(pixel_size_um, 4),
            "density_mm2": f"{density * 1e6:.2e}",
            "total_dislo": total_dislo,
            "dislo_blanche": n_blanc,
            "dislo_noire": n_noir,
            "dislo_mixte": n_mixte,
            "prob_min": float(prob.min()),
            "prob_max": float(prob.max()),
            "prob_mean": float(prob.mean()),
            "image_name": image_name,
            "overlay_path": pdf_path
        }

        return jsonify({
            "success": True,

            "image_name": image_name,
            "threshold": threshold,

            "total_dislo": total_dislo,
            "dislo_blanche": n_blanc,
            "dislo_noire": n_noir,
            "dislo_mixte": n_mixte,
            "dislo_incertain": n_incertain,

            "surface_um2": round(surface_um2, 2),
            "surface_mm2": round(surface_um2 / 1e6, 6),
            "density_um2": f"{density:.2e}",
            "density_mm2": f"{density * 1e6:.2e}",
            "pixel_size_um": round(pixel_size_um, 4),

            "image_width_um": image_width_um,
            "image_dimensions": {
                "width": image.shape[1],
                "height": image.shape[0]
            },

            "prob_min": float(prob.min()),
            "prob_max": float(prob.max()),
            "prob_mean": float(prob.mean()),

            "original_image": original_b64,
            "mask_image": mask_b64,
            "overlay_image": overlay_b64,
            "overlay_unet_image": overlay_unet_b64,
            "objects": results
        })

    except Exception as e:
        print("❌ Erreur :", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/download_pdf")
def download_pdf():
    global last_pdf_data

    if not last_pdf_data:
        return jsonify({"error": "Aucun résultat disponible"}), 400

    RESULTS_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results"
    )
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    pdf_file = os.path.join(RESULTS_FOLDER, "rapport_ECCI.pdf")

    c = canvas.Canvas(pdf_file, pagesize=A4)
    width, height = A4

    y = height - 2 * cm

    # ==============================
    # TITRE
    # ==============================
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "Rapport d'analyse ECCI - LTM")

    y -= 1 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Detection U-Net + Classification ResNet34")

    # ==============================
    # PARAMETRES
    # ==============================
    y -= 1.2 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Parametres d'analyse")

    y -= 0.7 * cm
    c.setFont("Helvetica", 10)

    params = [
        ("Seuil U-Net", str(last_pdf_data["threshold"])),
        ("Largeur physique", f"{last_pdf_data['image_width_um']} um"),
        ("Surface analysee", f"{last_pdf_data['surface_mm2']} mm2"),
        ("Taille pixel", f"{last_pdf_data['pixel_size_um']} um/pixel"),
        ("Densite", f"{last_pdf_data['density_mm2']} dislocations/mm2"),
    ]

    for label, value in params:
        c.drawString(2 * cm, y, f"{label} : {value}")
        y -= 0.5 * cm

    # ==============================
    # TABLEAU RESULTATS
    # ==============================
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Tableau recapitulatif")

    y -= 0.7 * cm

    table_x = 2 * cm
    table_y = y
    row_h = 0.65 * cm
    col1_w = 7 * cm
    col2_w = 4 * cm

    rows = [
        ("Classe", "Nombre"),
        ("Dislocations blanches", str(last_pdf_data["dislo_blanche"])),
        ("Dislocations noires", str(last_pdf_data["dislo_noire"])),
        ("Dislocations mixtes", str(last_pdf_data["dislo_mixte"])),
        ("Total", str(last_pdf_data["total_dislo"])),
    ]

    for i, (classe, nombre) in enumerate(rows):
        y_row = table_y - i * row_h

        if i == 0:
            c.setFillColorRGB(0, 0.33, 0.64)
            c.rect(table_x, y_row - row_h, col1_w + col2_w, row_h, fill=1)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFillColorRGB(1, 1, 1)
            c.rect(table_x, y_row - row_h, col1_w + col2_w, row_h, fill=0)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 10)

        c.drawString(table_x + 0.3 * cm, y_row - 0.45 * cm, classe)
        c.drawString(table_x + col1_w + 0.3 * cm, y_row - 0.45 * cm, nombre)

        # lignes verticales
        c.line(table_x + col1_w, y_row, table_x + col1_w, y_row - row_h)

    y = table_y - len(rows) * row_h - 0.8 * cm

    # ==============================
    # PROBABILITES U-NET
    # ==============================
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Probabilites U-Net")

    y -= 0.7 * cm
    c.setFont("Helvetica", 10)

    c.drawString(2 * cm, y, f"Probabilite min : {last_pdf_data['prob_min']:.3e}")
    y -= 0.5 * cm
    c.drawString(2 * cm, y, f"Probabilite max : {last_pdf_data['prob_max']:.3f}")
    y -= 0.5 * cm
    c.drawString(2 * cm, y, f"Probabilite moyenne : {last_pdf_data['prob_mean']:.3e}")

    # ==============================
    # IMAGE RESULTAT
    # ==============================
    y -= 1 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Image resultat")

    overlay_path = last_pdf_data["overlay_path"]

    if os.path.exists(overlay_path):
        c.drawImage(
            overlay_path,
            2 * cm,
            2 * cm,
            width=16 * cm,
            height=7 * cm,
            preserveAspectRatio=True,
            mask="auto"
        )
    else:
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y - 0.7 * cm, "Image resultat introuvable.")

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True,
        download_name="rapport_ECCI.pdf"
    )


@app.route("/logo_ltm.png")
def serve_logo():
    return send_file("logo_ltm.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = 8082
    print(f"🚀 Serveur disponible : http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)