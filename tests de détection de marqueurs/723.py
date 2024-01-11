import cv2
import cv2.aruco as aruco
import numpy as np

with open('../calibration/camera_cal.npy', 'rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)

markerLength = 0.03  # Largeur du marqueur ArUco en mètres

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

frame = cv2.imread('image32cm_0005.jpg')

corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

if len(corners) > 0:
    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, camera_distortion)
    
    # Création d'un dictionnaire pour stocker les positions des marqueurs
    marker_positions = {}
    
    for i in range(len(ids)):
        c = corners[i][0]
        x = int((c[0, 0] + c[2, 0]) / 2)
        y = int((c[0, 1] + c[2, 1]) / 2)
        
        cv2.putText(frame, "ID: {} Position: ({}, {})".format(ids[i][0], x, y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        
        # Stockage de la position du marqueur dans le dictionnaire
        marker_positions[ids[i][0]] = tvecs[i]
        
        aruco.drawDetectedMarkers(frame, corners)
        
    # Calcul de la distance entre les marqueurs 7 et 23
    if 7 in marker_positions and 23 in marker_positions:
        distance = np.linalg.norm(marker_positions[7] - marker_positions[23])
        print("La distance entre les marqueurs 7 et 23 est de {:.2f} cm".format(distance * 100))  # Conversion de mètres en centimètres

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 1920, 1080)
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
