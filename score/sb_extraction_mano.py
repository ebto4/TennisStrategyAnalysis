import os
import cv2
import numpy as np
import easyocr
from PIL import Image

def extract_text_from_images(image_folder, output_folder):
    """Extrait le texte OCR des images après débruitage et l'affiche dans le terminal."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = easyocr.Reader(['en'])  

    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        
        # Charger l'image avec PIL puis convertir en NumPy
        
        image = Image.open(img_path)
        image_np = np.array(image)

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Agrandir l'image pour améliorer la détection
        scale_factor = 3
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Améliorer le contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enlarged)

        # Binarisation avec Otsu
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilatation pour renforcer les chiffres
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Débruitage (utilisation de fastNlMeansDenoising au lieu de fastNlMeansDenoisingColored)
        image_denoised = cv2.fastNlMeansDenoising(dilated, h=190)

        # Initialiser le lecteur EasyOCR
        reader = easyocr.Reader(['en'])

        allowlist = "DJOKOVICMEDVEDEV01234567AD153040"
        # Appliquer la reconnaissance de texte
        results = reader.readtext(image_denoised, text_threshold=0.4, link_threshold=0.2, width_ths=0.4, add_margin=0.1, allowlist=allowlist, detail=1)

        # Vérifier si un des résultats contient des caractères interdits
        for result in results:
            if any(char in result[1] for char in ['/', '-', '#', '%', '~', ')', '(', '[', ']']):
                # Si un caractère interdit est trouvé, vider la liste des résultats
                results = []
                break
            if len(results) % 2 != 0:  # Changer les conditions si ce n'est pas le premier set.
                if len(results) > 4 and not (results[1][1] == 'AD' or results[4][1] == 'AD'):
                    results = []
                if len(results) < 4:
                    results = []
            elif len(results) == 6:
                if not (results[0][1] == 'MEDVEDEV' and
                        results[1][1] in ['0', '1', '2', '3', '4', '5', '6'] and
                        results[2][1] in ['0', '15', '30', '40'] and
                        results[3][1] == 'DJOKOVIC' and
                        results[4][1] in ['0', '1', '2', '3', '4', '5', '6'] and
                        results[5][1] in ['0', '15', '30', '40']):
                    results = []

        # Extraire le texte reconnu
        recognized_texts = [text for (_, text, _) in results]
        final_string = " ".join(recognized_texts)

        # Sauvegarder le texte dans un fichier
        output_txt_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + ".txt")
        with open(output_txt_path, 'w') as f:
            f.write(final_string)

        # Afficher le texte détecté dans le terminal
        print(f"\n{img_name} - Texte extrait : {final_string}")

def main():
    image_folder = "cerv/sb_frames1set"
    output_folder = "cerv/extracted_texts1set"
    extract_text_from_images(image_folder, output_folder)

if __name__ == "__main__":
    main()
