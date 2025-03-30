from PIL import Image
import numpy as np
import easyocr


# Charger l'image
image_path = "/Users/ethanbareille/Desktop/TennisProject/nouvelle_image.jpg"
image = Image.open(image_path)

# Convertir l'image en noir et blanc (mode 'L' pour niveaux de gris)
image_bw = image.convert('L')

# Convertir l'image en tableau NumPy
image_np = np.array(image_bw)

# Seuil à appliquer
threshold_value = 128

# Appliquer le thresholding : tout ce qui est supérieur au seuil devient blanc, le reste devient noir
image_thresholded = np.where(image_np > threshold_value, 255, 0)

# Convertir le tableau en uint8 (pour qu'il soit compatible avec EasyOCR)
image_thresholded = image_thresholded.astype(np.uint8)

# Initialiser le lecteur EasyOCR
reader = easyocr.Reader(['en'])

# Lire le texte dans l'image
results = reader.readtext(image_thresholded)

recognized_texts = [text for (_, text, _) in results]
final_string = " ".join(recognized_texts)
print(final_string)
 