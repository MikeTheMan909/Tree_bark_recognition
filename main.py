import sys
import random
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import numpy as np
import math

output = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\houghlines\img.png'

def Average(lst):
    return sum(lst) / len(lst)

def avg_color(image): #contour
    channels = cv2.mean(imS) #, mask
    h = channels[0]
    s = channels[1]
    v = channels[2]
    h = int(round(h,0))
    s = int(round(s,0))
    v = int(round(v,0))
    return h, s, v

def houghlines(img):
    horizontal = 0
    vertical = 0
    img = cv2.resize(img, [640,480])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #blur = cv2.GaussianBlur(blur, (5, 5), 0)
    #blur = cv2.GaussianBlur(blur, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 120)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, None, 40, 10)
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
                vertical = vertical + 1
            if 0 <= angle_degree < 30 or 0 >= angle_degree > -15:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv2.LINE_AA)
                horizontal = horizontal + 1

    cv2.imwrite(output, img)
    return (horizontal/(horizontal + vertical)), (vertical/(horizontal + vertical)), ((horizontal + vertical))

def image_testing():
    img = cv2.resize(imS, (640, 480))  # Resize to smaller size for easy screen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV filter
    # blur = cv2.fastNlMeansDenoisingColored(img, None, 90, 10, 7, 21)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)
    color = cv2.applyColorMap(invert, cv2.COLORMAP_JET)
    cv2.imshow('noise', blur)
    cv2.imshow('color', color)
    cv2.imshow("img", img)  # Output original
    cv2.imshow("HSV", hsv)  # Output HSV


def GCLM_calc(imSA, show):
    pathA = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\fotos\Berteris_spec_3.jpg'
    pathB = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\fotos\Berteris_spec_1.jpg'

    PATCH_SIZE = 100
    sample_amount = 100

    imgA = cv2.resize(imSA, (640, 480))  # Resize to smaller size for easy screen
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)

    barkALocation = []
    for x in range(sample_amount):
        x = random.randint(0, 480 - PATCH_SIZE)
        y = random.randint(0, 640 - PATCH_SIZE)
        a = x, y
        barkALocation.append(a)

    barkALoc = []
    for loc in barkALocation:
        barkALoc.append(grayA[loc[0]:loc[0] + PATCH_SIZE,
                        loc[1]:loc[1] + PATCH_SIZE])
    xs = []
    ys = []
    # compute some GLCM properties each patch
    for patch in barkALoc:
        glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])

    boomA = Average(xs[:len(barkALocation)])

    if show:
        fig = plt.figure(figsize=(5, 5))

        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(grayA, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in barkALocation:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
        ax.set_xlabel('Original Image A')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')

        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(barkALocation)], ys[:len(barkALocation)], 'go',
                label='Boom A')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()

        fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()

    return boomA

if __name__ == '__main__':

    print("Tree Bark Recognition")
    # Image path
    lenght = len(sys.argv)
    if lenght >= 2:
        path_to_foto = sys.argv[1]
        print(path_to_foto)
    else:
        path_to_foto = 'watercipres_2.jpg'

    imS = cv2.imread(path_to_foto)  # For opening image
    print(houghlines(imS))
    boomA =GCLM_calc(imS, 0)
    print("Average of dissimilarity tree A:" + str(boomA))
    a=avg_color(imS)
    print(a)

    # Destroy window
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
