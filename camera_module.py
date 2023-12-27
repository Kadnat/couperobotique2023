import cv2
import numpy as np

# Charger le dictionnaire ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

# Initialiser le détecteur de paramètres ArUco
parameters =  cv2.aruco.DetectorParameters_create()

# Taille des marqueurs ArUco de référence (en mètres)
marker_size = 0.1

# Charger les paramètres intrinsèques de la caméra et les coefficients de distorsion à partir du fichier .npz
with np.load('calibration_charuco.npz') as X:
    camera_matrix, dist_coeffs, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

# Positions des marqueurs ArUco de référence dans le référentiel de l'arène de jeu
reference_markers = {
    20: np.array([500, 750, 0]),
    21: np.array([500, -750, 0]),
    22: np.array([-500, 750, 0]),
    23: np.array([-500, -750, 0])
}

def detect_aruco(frame):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les marqueurs ArUco
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Estimer la pose des marqueurs ArUco
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

    # Calculer la pose des marqueurs dans le référentiel de l'arène de jeu
    marker_poses = {}
    M_camera_to_arena = np.eye(4)  # Initialiser à la matrice d'identité
    if ids is not None:
        for i in range(len(ids)):
            # Obtenir la transformation du référentiel de la caméra au référentiel du marqueur
            R_marker, _ = cv2.Rodrigues(rvecs[i])
            T_marker = tvecs[i]
            M_marker = np.block([[R_marker, T_marker.T], [0, 0, 0, 1]])

            if int(ids[i][0]) in reference_markers:
                # Obtenir la transformation du référentiel de l'arène de jeu au référentiel du marqueur
                T_arena = reference_markers[int(ids[i][0])]
                R_arena = np.eye(3)  # suppose que le marqueur n'est pas orienté
                M_arena = np.block([[R_arena, np.reshape(T_arena, (3, 1))], [0, 0, 0, 1]])

                # Calculer la transformation du référentiel de la caméra au référentiel de l'arène de jeu
                M_camera_to_arena = np.linalg.inv(M_marker) @ M_arena
            else:
                # Calculer la transformation du référentiel du marqueur au référentiel de l'arène de jeu
                M_marker_to_arena = M_camera_to_arena @ M_marker

                # La pose du marqueur dans le référentiel de l'arène de jeu est la translation de cette transformation
                marker_pose = M_marker_to_arena[:3, 3]
                marker_poses[int(ids[i][0])] = marker_pose

                # Calculer et afficher la distance entre le marqueur détecté et chaque marqueur de référence
                for ref_id, ref_pose in reference_markers.items():
                    distance = np.linalg.norm(marker_pose - ref_pose)
                    print(f"Distance from marker {int(ids[i][0])} to reference marker {ref_id}: {distance} mm")
                    cv2.putText(frame, f"Distance from marker {int(ids[i][0])} to reference marker {ref_id}: {distance} mm", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Dessiner une ligne entre le marqueur détecté et le marqueur de référence
                    if ref_id in ids:
                        ref_index = np.where(ids == ref_id)[0][0]
                        cv2.line(frame, tuple(corners[i][0][0].astype(int)), tuple(corners[ref_index][0][0].astype(int)), (0, 255, 0), 2)

    return marker_poses

# Créer une fenêtre
cv2.namedWindow('Distances', cv2.WINDOW_NORMAL)

# Ouvrir le flux vidéo de la caméra
cap = cv2.VideoCapture(0)

while True:
    # Lire une image du flux vidéo
    ret, frame = cap.read()

    # Détecter les marqueurs ArUco
    marker_poses = detect_aruco(frame)

    # Redimensionner la fenêtre à la taille de l'écran
    cv2.resizeWindow('Distances', 600, 600)

    # Afficher l'image avec les distances
    cv2.imshow('Distances', frame)

    # Si 'q' est pressé sur le clavier, arrêter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer le flux vidéo
cap.release()

# Fermer toutes les fenêtres OpenCV
cv2.destroyAllWindows()
