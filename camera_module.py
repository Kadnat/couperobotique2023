import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

# Chargement des paramètres de calibration de la caméra
with open('calibration\camera_cal.npy', 'rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)

# Définition de la taille du marqueur ArUco en mètres
markerLengths = {
    47:0.03,
    13:0.03,
    22:0.1,
    21:0.1,
    20:0.1,
    23:0.1,
    4:0.1,
    7:0.1,
}

# Normalisation de la taille des marqueurs
max_marker_length = max(markerLengths.values())
markerLengths = {id: size / max_marker_length for id, size in markerLengths.items()}

# Création du dictionnaire ArUco et des paramètres de détection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# Lecture de l'image
frame = cv2.imread('photos\image2592lumiere_0003.jpg')

# Détection des marqueurs ArUco dans l'image
corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

# Définition des positions de référence des marqueurs dans le monde réel
reference_markers = {
    20: np.array([500, 750, 0]),
    21: np.array([500, -750, 0]),
    22: np.array([-500, 750, 0]),
    23: np.array([-500, -750, 0])
}

# Si au moins un marqueur a été détecté
if len(corners) > 0:
    # Initialisation des listes pour stocker les positions des marqueurs dans le monde réel et dans le système de coordonnées de la caméra
    world_positions = []
    camera_positions = []
    
    # Pour chaque marqueur détecté
    for i in range(len(ids)):
        markerLength = markerLengths[ids[i][0]]

        # Estimation de la pose des marqueurs
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, camera_distortion)
        # Si le marqueur est un marqueur de référence
        if ids[i][0] in reference_markers:
            # Ajout de la position du marqueur dans le monde réel et dans le système de coordonnées de la caméra aux listes correspondantes
            world_positions.append(reference_markers[ids[i][0]])
            camera_positions.append(tvecs[i][0])
    
    # Conversion des listes en tableaux numpy pour le calcul ultérieur
    world_positions = np.array(world_positions)
    camera_positions = np.array(camera_positions)
    
    # Calcul de la transformation affine entre le système de coordonnées de la caméra et le système de coordonnées du monde réel
    transformation = np.linalg.lstsq(camera_positions, world_positions, rcond=None)[0]
    
    # Pour chaque marqueur détecté
    for i in range(len(ids)):
        # Calcul de la position du centre du marqueur dans l'image
        c = corners[i][0]
        x = int((c[0, 0] + c[2, 0]) / 2)
        y = int((c[0, 1] + c[2, 1]) / 2)
        
        # Calcul de la position du marqueur dans le monde réel en utilisant la transformation
        real_world_position = np.dot(tvecs[i][0], transformation)
        
        # Affichage de l'ID du marqueur et de sa position dans le monde réel sur l'image
        cv2.putText(frame, "ID: {} Pos: ({:.2f}, {:.2f}, {:.2f})".format(ids[i][0], real_world_position[0], real_world_position[1], real_world_position[2]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        
         # Affichage de l'ID du marqueur et de sa position dans le monde réel dans la console
        print("ID: {} Real World Position: ({:.2f}, {:.2f}, {:.2f})".format(ids[i][0], real_world_position[0], real_world_position[1], real_world_position[2]))
        
        # Dessin des marqueurs détectés sur l'image
        aruco.drawDetectedMarkers(frame, corners)

# Affichage de l'image
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 1920, 1080)
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
