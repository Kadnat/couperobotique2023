import numpy as np
import cv2
import cv2.aruco as aruco
import os

# Créer le dictionnaire ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

# Créer la planche Charuco
square_length = 1.5 # Longueur du carré en unités arbitraires, par exemple en centimètres
marker_length = 1.2 # Longueur du marqueur en unités arbitraires, par exemple en centimètres
squares_x = 11 # Nombre de carrés en x
squares_y = 8  # Nombre de carrés en y
charuco_board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

# Utiliser le dossier actuel pour les images
image_dir = '.'

# Tableaux pour stocker les points d'objets et les points d'image de toutes les images.
all_charuco_corners = []
all_charuco_ids = []

# Parcourir toutes les images dans le dossier spécifié
for filename in os.listdir(image_dir):
    # Lire chaque image
    frame = cv2.imread(os.path.join(image_dir, filename))
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les marqueurs ArUco dans l'image
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    # Si au moins un marqueur a été détecté
    if len(corners) > 0:
        # Interpoler les coins Charuco
        res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and res2[2].size > 3:
            all_charuco_corners.append(res2[1])
            all_charuco_ids.append(res2[2])

    # Afficher l'image en cours de traitement
    cv2.imshow('Image en cours de traitement', frame)
    # Si 'q' est pressé sur le clavier, arrêter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer toutes les fenêtres OpenCV
cv2.destroyAllWindows()

# Calibration de la caméra
imsize = gray.shape
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, charuco_board, imsize, None, None)

# Enregistrement des coefficients de calibration dans un fichier npz
np.savez('calibration_charuco.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
