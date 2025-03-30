import cv2
from PIL import Image
import numpy as np
import easyocr
import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Définir les chemins
VIDEO_PATH = "/Users/ethanbareille/Desktop/TennisProject/input_videos/cerv.mp4"  # Remplace par le chemin de ta vidéo
FRAMES_DIR = "frames"
MODEL_PATH = "/Users/ethanbareille/Desktop/TennisProject/models/fasterrcnn_scoreboard.pth"

# Créer le dossier pour stocker les frames si nécessaire
os.makedirs(FRAMES_DIR, exist_ok=True)

# Charger le modèle entraîné
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features

num_classes = 2  # 1 classe + 1 pour l'arrière-plan
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transformation pour les images
transform = T.Compose([T.ToTensor()])

# Détection et extraction du scoreboard
def extract_scoreboard(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    scores = predictions[0]['scores'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()
    
    if len(scores) > 0:
        max_idx = np.argmax(scores)
        if scores[max_idx] > 0.6:  # Seuil de confiance
            x_min, y_min, x_max, y_max = map(int, boxes[max_idx])
            scoreboard = image.crop((x_min-50, y_min-50, x_max+50, y_max+50))

            scoreboard.show()
            scoreboard.save("nouvelle_image.jpg")
            return scoreboard
    return None

# Extraire les frames de la vidéo toutes les 5 secondes
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * 5)  # 1 frame toutes les 5 secondes
    frame_id = 1  # Commence la numérotation à 1
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
        count += 1
    cap.release()

# Appliquer la détection et l'OCR sur toutes les frames
def process_frames(frames_dir):
    reader = easyocr.Reader(['en'])
    for frame_name in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_name)
        image = Image.open(frame_path).convert("RGB")
        scoreboard = extract_scoreboard(image)
        
        if scoreboard:
            scoreboard_np = np.array(scoreboard)
            results = reader.readtext(scoreboard_np)
            recognized_texts = [text for (_, text, _) in results]
            final_string = " ".join(recognized_texts)
            print(f"{frame_name} - Texte extrait :", final_string)
        else:
            print(f"{frame_name} - Aucun scoreboard détecté.")

# Exécution
def main():
    extract_frames(VIDEO_PATH, FRAMES_DIR)
    process_frames(FRAMES_DIR)

if __name__ == "__main__":
    main()
