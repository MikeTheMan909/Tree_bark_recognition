import cv2
import math
import numpy as np


pathB = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\fotos'
pathB = pathB + '\japanse_venijnboom_3.jpg'
output = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\houghlines\img.png'
imgs = cv2.imread(pathB)

def houghlines():
    horizontal = 0
    vertical = 0
    img = imgs
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 120)
    linesP = cv2.HoughLinesP(edges, 20, np.pi / 360, 10, None, 60, 40)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            #here l contains x1,y1,x2,y2  of your line
            #so you can compute the orientation of the line
            p1 = np.array([l[0],l[1]])
            p2 = np.array([l[2],l[3]])

            p0 = np.subtract( p1,p1 ) #not used
            p3 = np.subtract( p2,p1 ) #translate p2 by p1

            angle_radiants = math.atan2(p3[1],p3[0])
            angle_degree = angle_radiants * 180 / math.pi

            #print("line degree", angle_degree)

            if 90 < angle_degree < 115 or 90 > angle_degree > 65 or -65 > angle_degree > -90 or -115 < angle_degree < -90:
                cv2.line(img,  (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
                horizontal = horizontal+1
            if 0 <= angle_degree < 30 or 0 >= angle_degree > -15:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv2.LINE_AA)
                vertical = vertical + 1

    cv2.imwrite(output, img)
    return (horizontal/(horizontal + vertical)), (vertical/(horizontal + vertical)), ((horizontal + vertical))


print(houghlines())

