import cv2
import numpy as np


def nothing(x):
    pass

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

#Convertendo o padrao de cores para hsv
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    #Definindo os limites de cada cor

    # Red color
    low_red1 = np.array([0, 200, 80])
    high_red1 = np.array([8, 255, 255])
    low_red2 = np.array([170, 175, 80])
    high_red2 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv, low_red1, high_red1)
    red_mask2 = cv2.inRange(hsv, low_red2, high_red2)
    red_mask = red_mask1 + red_mask2
    

    # Blue color
    low_blue = np.array([100, 150, 0])
    high_blue = np.array([144, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    


    mask = blue_mask + red_mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    res = cv2.bitwise_and(frame,frame, mask= mask)


    # Detectando os contornos de objetos
    count = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Aproximando para diminuir ruído

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # Definindo uma área com numero de pixels grande para evitar falhas

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            count = count + 1

            #Definindo as formas geometricas
            
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                if ar < 1.1 and ar > 0.9:
                    cv2.putText(frame, "Quadrado", (x, y), font, 1, (0, 0, 0))
                else:
                    cv2.putText(frame, "Retangulo", (x, y), font, 1, (0, 0, 0))

            elif 6 < len(approx) < 20:
                cv2.putText(frame, "Círculo", (x, y), font, 1, (0, 0, 0))

    

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    print ("O numero de objetos e: " + str(count))
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()