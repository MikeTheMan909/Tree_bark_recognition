
import sys
import random
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

def Average(lst):
    return sum(lst) / len(lst)

def GCLM_calc():
    pathA = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\fotos\Berteris_spec_3.jpg'
    pathB = r'C:\Users\mike0\OneDrive - Stichting Hogeschool Utrecht\Documenten\Hogeschool Utrecht\ElektroTechniek\Jaar 4\Beeldherkenning\werkmap\code\fotos\Berteris_spec_1.jpg'

    PATCH_SIZE = 100
    sample_amount = 100
    imSA = cv2.imread(pathA)  # For opening image
    imgA = cv2.resize(imSA, (640, 480))  # Resize to smaller size for easy screen
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=grayA, threshold1=150, threshold2=250)
    cv2.imshow("edges", edges)  # Output HSV
    imSB = cv2.imread(pathB)  # For opening image
    imgB = cv2.resize(imSB, (640, 480))  # Resize to smaller size for easy screen
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    # select some patches from bark areas of the image
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

    barkBLocation = []

    for x in range(sample_amount):
        x = random.randint(0, 480 - PATCH_SIZE)
        y = random.randint(60, 580 - PATCH_SIZE)
        a = x, y

        barkBLocation.append(a)
    barkBLoc = []

    for loc in barkBLocation:
        barkBLoc.append(grayB[loc[0]:loc[0] + PATCH_SIZE,
                        loc[1]:loc[1] + PATCH_SIZE])
    xs = []
    ys = []
    # compute some GLCM properties each patch
    for patch in (barkALoc + barkBLoc):
        glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])

    boomA = Average(xs[:len(barkALocation)])
    boomB = Average(xs[len(barkALocation):])
    print("Average of dissimilarity tree A:" + str(boomA))
    print("Average of dissimilarity tree B:" + str(boomB))
    fig = plt.figure(figsize=(8, 8))

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

    ax = fig.add_subplot(3, 2, 3)
    ax.imshow(grayB, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in barkBLocation:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image B')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(barkALocation)], ys[:len(barkALocation)], 'go',
            label='Boom A')
    ax.plot(xs[len(barkALocation):], ys[len(barkALocation):], 'bo',
            label='Boom B')
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
        path_to_foto = 'fotos/oude fotos/Boom_4_C.jpg'

    imS = cv2.imread(path_to_foto)  # For opening image
    img = cv2.resize(imS, (640, 480))  # Resize to smaller size for easy screen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV filter
    blur = cv2.fastNlMeansDenoisingColored(img, None, 15, 10, 7, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)
    color = cv2.applyColorMap(invert, cv2.COLORMAP_JET)
    cv2.imshow('noise', blur)
    cv2.imshow('color', color)
    cv2.imshow("img", img)  # Output original
    cv2.imshow("HSV", hsv)  # Output HSV
    GCLM_calc()
    # Destroy window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
