import cv2
import numpy as np
import sys




if __name__ == '__main__':

    print("Tree Bark Recognition")
    #Image path
    lenght = len(sys.argv)
    if(lenght >= 2):
        path_to_foto = 'fotos/' + sys.argv[1]
        print(path_to_foto)
    else:
        path_to_foto = 'fotos/Boom_1_A.jpg'

    imS = cv2.imread(path_to_foto)              #For opening image
    img = cv2.resize(imS, (640, 480))           #Resize to smaller size for easy screen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #HSV filter
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow('color',color)
    cv2.imshow("img",img)                       #Output original
    cv2.imshow("HSV",hsv)                       #Output HSV






    #Destroy window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

