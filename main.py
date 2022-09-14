import cv2
import numpy as np

#Image path
path_to_foto = 'fotos/Boom_7_A.jpg'


img = cv2.imread(path_to_foto)              #For opening image
#imS = cv2.resize(img, (640, 480))           #Resize to smaller size for easy screen
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #HSV filter
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

cv2.imshow('color',color)
cv2.imshow("img",img)                       #Output original
cv2.imshow("HSV",hsv)                       #Output HSV






#Destroy window
cv2.waitKey(0)
cv2.destroyAllWindows()

