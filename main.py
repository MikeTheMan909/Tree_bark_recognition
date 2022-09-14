import cv2
import numpy as np

#Image path
path_to_foto = 'fotos/Boom_6_A.jpg'


img = cv2.imread(path_to_foto)              #For opening image
imS = cv2.resize(img, (640, 480))           #Resize to smaller size for easy screen
hsv = cv2.cvtColor(imS, cv2.COLOR_BGR2HSV)  #HSV filter

cv2.imshow("img",imS)                       #Output original
cv2.imshow("HSV",hsv)                       #Output HSV

#Destroy window
cv2.waitKey(0)
cv2.destroyAllWindows()
#commrny