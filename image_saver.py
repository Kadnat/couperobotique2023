import cv2
import time

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(0)

# Vérifier si la caméra a été correctement ouverte
if not cap.isOpened():
    raise IOError("Impossible d'ouvrir la caméra")

compteur = 0
temps_precedent = time.time()

while True:
    # Lire une image du flux
    ret, frame = cap.read()

    # Si l'image a été correctement lue
    if ret:
        # Vérifier si une seconde s'est écoulée depuis la dernière image enregistrée
        if time.time() - temps_precedent >= 1.0:
            # Enregistrer l'image en tant que fichier .jpg
            cv2.imwrite('image_{:04d}.jpg'.format(compteur), frame)
            compteur += 1
            temps_precedent = time.time()

    # Afficher l'image dans une fenêtre
    cv2.imshow('Input', frame)

    # Si 'q' est pressé sur le clavier, arrêter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo
cap.release()
# Fermer toutes les fenêtres OpenCV
cv2.destroyAllWindows()
