import numpy as np
import cv2
import glob

# Critères de terminaison : précision jusqu'à 0.1 mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparation des points d'objets, comme (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Tableaux pour stocker les points d'objets et les points d'image de toutes les images.
objpoints = [] # points 3d dans l'espace réel
imgpoints = [] # points 2d dans le plan de l'image.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Trouver les coins du damier
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    # Si on les trouve, ajouter des points d'objets, des points d'image (après avoir affiné les coins)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Dessiner et afficher les coins
        img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration de la caméra
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Enregistrement des coefficients de calibration
np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
