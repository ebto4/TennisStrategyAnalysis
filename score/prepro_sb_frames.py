import os
import numpy as np
from PIL import Image

# Dossier source et destination
input_folder = "cerv/sb_frames1set"
output_folder = "cerv/sb_frames_prepro1set"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Valeur du seuil
threshold_value = 128

# Parcourir toutes les images du dossier
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Vérifier l'extension
        image_path = os.path.join(input_folder, filename)

        # Charger l'image
        image = Image.open(image_path)

        # Convertir en niveaux de gris
        image_bw = image.convert('L')

        # Convertir en tableau NumPy
        image_np = np.array(image_bw)

        # Appliquer le thresholding
        image_thresholded = np.where(image_np > threshold_value, 255, 0).astype(np.uint8)

        # Convertir en image PIL
        image_result = Image.fromarray(image_thresholded)

        # Enregistrer l'image traitée
        output_path = os.path.join(output_folder, filename)
        image_result.save(output_path)

print("Traitement terminé. Images enregistrées dans", output_folder)
