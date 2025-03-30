from tracknet import BallTrackerNet  # Modèle TrackNet pour le suivi de balle
import torch  # Bibliothèque pour le calcul tensoriel et le deep learning
import cv2  # Bibliothèque pour la vision par ordinateur
import numpy as np  # Bibliothèque pour les calculs numériques
from scipy.spatial import distance  # Module pour calculer les distances entre points
from tqdm import tqdm  # Module pour afficher une barre de progression

class BallDetector:
    def __init__(self, path_model=None, device='cuda'):
        """
        Initialisation de la classe BallDetector.
        :params
            path_model: chemin vers le fichier du modèle pré-entraîné (str ou None)
            device: périphérique pour exécuter le modèle (str, 'cuda' ou 'cpu')
        """
        self.model = BallTrackerNet(input_channels=9, out_channels=256)  # torch.nn.Module
        self.device = device  # str

        if path_model:
            # Charger les poids du modèle à partir du chemin spécifié
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)  # torch.nn.Module
            self.model.eval()

        self.width = 640  # int
        self.height = 360  # int

    def infer_model(self, frames):
        """
        Effectue une inférence sur une liste de frames consécutives.
        :params
            frames: liste de frames vidéo consécutives (list[np.ndarray])
        :return
            ball_track: liste des coordonnées détectées de la balle (list[tuple[float, float]])
        """
        ball_track = [(None, None)] * 2  # list[tuple[None, None]]
        prev_pred = [None, None]  # list[None, None]

        for num in tqdm(range(2, len(frames))):  # num: int
            # Redimensionnement des trois frames nécessaires à l'inférence
            img = cv2.resize(frames[num], (self.width, self.height))  # np.ndarray
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))  # np.ndarray
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))  # np.ndarray

            # Création d'une entrée à 9 canaux (concaténation des trois frames)
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)  # np.ndarray
            imgs = imgs.astype(np.float32) / 255.0  # np.ndarray
            imgs = np.rollaxis(imgs, 2, 0)  # np.ndarray
            inp = np.expand_dims(imgs, axis=0)  # np.ndarray

            # Passage des données au modèle
            out = self.model(torch.from_numpy(inp).float().to(self.device))  # torch.Tensor
            output = out.argmax(dim=1).detach().cpu().numpy()  # np.ndarray

            # Post-traitement des prédictions
            x_pred, y_pred = self.postprocess(output, prev_pred)  # tuple[float, float]
            prev_pred = [x_pred, y_pred]  # list[float, float]
            ball_track.append((x_pred, y_pred))  # list[tuple[float, float]]

        return ball_track  # list[tuple[float, float]]

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        Post-traitement des prédictions pour obtenir les coordonnées de la balle.
        :params
            feature_map: carte des caractéristiques avec la forme (1, 360, 640) (np.ndarray)
            prev_pred: coordonnées [x, y] de la prédiction précédente (list[float, float])
            scale: facteur d'échelle pour ajuster les coordonnées à la résolution d'origine (int)
            max_dist: distance maximale autorisée pour éviter les fausses détections (int) a modifier éventuellement 
        :return
            x, y: coordonnées de la balle (float, float)
        """
        feature_map *= 255  # np.ndarray
        feature_map = feature_map.reshape((self.height, self.width)).astype(np.uint8)  # np.ndarray
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)  # ret: float, heatmap: np.ndarray
        # Détecter les cercles potentiels représentant la balle
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7
        )  # np.ndarray ou None

        x, y = None, None  # float, float
        if circles is not None:  # Si des cercles sont détectés
            if prev_pred[0]:  # Si une prédiction précédente existe
                for i in range(len(circles[0])):  
                    x_temp = circles[0][i][0] * scale  # float
                    y_temp = circles[0][i][1] * scale  # float
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)  # float
                    if dist < max_dist:  # Vérifier si la distance est acceptable
                        x, y = x_temp, y_temp  # float, float
                        break
            else:  # Si aucune prédiction précédente n'existe
                x = circles[0][0][0] * scale  # float
                y = circles[0][0][1] * scale  # float

        return x, y  # float, float
