import os
import cv2

def extract_and_save_rectangles(image_folder, output_folder):
    # Crée le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Coordonnées fixes pour la boîte de délimitation (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = 109, 896, 576, 978
    
    # Parcourt toutes les images dans le dossier
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])
    
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        
        # Lire l'image
        img = cv2.imread(img_path)
        
        # Extraire la région d'intérêt (ROI) de l'image
        roi = img[ymin:ymax, xmin:xmax]
        
        # Sauvegarder l'image du rectangle dans sb_frames
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, roi)
        print(f"Rectangle extrait et sauvegardé pour {img_name} dans {output_path}")

def main():
    image_folder = "frames"  # Dossier contenant les images
    output_folder = "sb_frames"  # Dossier où les images extraites seront sauvegardées
    
    extract_and_save_rectangles(image_folder, output_folder)

if __name__ == "__main__":
    main()
