# 1-Importer les packages necessaires
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


# Initialisation de la variable
capture = cv2.VideoCapture(0)

# 2-Initialiser mediapipe
mpHands = mp.solutions.hands # Creation d'un objet qui nous permettre de reconnaitre la main
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7) # le nbre de main, taux de confidence de main
mpDraw = mp.solutions.drawing_utils # Pour tracer les lignes de la main
# 3-Initialiser modele de reconnaissance des gestes de la main
model = load_model('mp_hand_gesture')
# 4-Charger les noms des classes
file = open('gesture.names', 'r')
classnames = file.read().split('\n')
file.close()
print(classnames)
# 5-Inialiser lecture images par webcam
while True:
    success, img = capture.read()
    x, y, z = img.shape
    img = cv2.flip(img, 1)
    imgcolor = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get hand landmark prediction
    result = hands.process(imgcolor)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)
                #print(id, lm)
# 7-Reconnaitre les signes
            prediction = model.predict([landmarks])
            #print(prediction)
            classId = np.argmax(prediction)
            className = classnames[classId]
            #print(className)
        cv2.putText(img, className, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Afficher la sortie final
    cv2.imshow("Sortie", img)
    if cv2.waitKey(1) == ord("n"):
        break
# Active le webcam et arreter windows activity
capture.release()
cv2.destroyAllWindows()
# 6-Detectez les lignes de la main


