# IF I HAVE TIME:
#   add round ends to getLine

from PIL import ImageDraw, Image
from random import randint
from image_functions import *


# color over eyebrows using skin color from forehead
def blurEyebrows(img, landmarks):
        # get current image and eyebrow landmarks
        marked = ImageDraw.Draw(img)
        pntNum = list(range(17, 27))+list(range(85, 95))
        pts = [landmarks[pt] for pt in pntNum]
        bList = []
        # color over for each space between points
        minX, minY = getPerp((pts[0], pts[10]), (pts[9], pts[19]), 5)
        for p in range(9):
            # get angle values for line perpendicular to eyebrows, scaled for distance from center
            maxX, maxY = getPerp((pts[0], pts[10]), (pts[9], pts[19]), 15-2*abs(4-p))
            # split each line into 4
            for s in [0, 0.25, 0.5, 0.75]:
                pt = ((pts[p]*(1-s))+(pts[p+1]*s), (pts[p+10]*(1-s))+(pts[p+11]*s))
                # sample 10 random points from a box above the segment
                colSamp = getLine((pt[0]-minX, pt[1]-minY), (pt[0]-maxX, pt[1]-maxY), 9)
                # get average color of sampled pixels
                color = pixelAverage([colSamp[randint(0, len(colSamp)-1)] for i in range(10)], img)
                # draw line in new color over eyebrow segment
                marked.line([pt[0], pt[1]+3, (pts[p]*(0.75-s))+(pts[p+1]*(s+0.25)),
                             ((pts[p+10]*(0.75-s))+(pts[p+11]*(s+0.25)))+3], fill=color, width=14)
            bList.extend(getLine((pts[p], pts[p+10]), (pts[p+1], pts[p+11]), 20))
        bList = list(set(bList))
        blur(bList, img, 2)


# color over eyes using skin color from forehead and cheeks
def blurEyes(img, lms):
    marked = ImageDraw.Draw(img)
    ebpts = [(lms[lm], lms[lm+68]) for lm in range(17, 27)]
    epts = [(lms[lm], lms[lm+68]) for lm in range(36, 48)]
    lY = getHypotenuse(ebpts[3], epts[2]) / getHypotenuse(epts[2], epts[4]) - 0.5
    lX = getHypotenuse(ebpts[0], ebpts[4]) / getHypotenuse(epts[0], epts[3]) - 1
    lpts = scalePolygon(epts[:6], lX, lY, epts[0], epts[3])
    rY = getHypotenuse(ebpts[6], epts[7]) / getHypotenuse(epts[7], epts[11]) - 0.5
    rX = getHypotenuse(ebpts[5], ebpts[9]) / getHypotenuse(epts[6], epts[9]) - 1
    rpts = scalePolygon(epts[6:], rX, rY, epts[6], epts[9])
    lTris = [*getTri(lpts[0],lpts[1],lpts[5]), *getTri(lpts[1],lpts[4],lpts[5]),
             *getTri(lpts[1],lpts[2],lpts[4]), *getTri(lpts[2],lpts[3],lpts[4])]
    rTris = [*getTri(rpts[0],rpts[1],rpts[5]), *getTri(rpts[1],rpts[4],rpts[5]),
             *getTri(rpts[1],rpts[2],rpts[4]), *getTri(rpts[2],rpts[3],rpts[4])]
    for pixel in lTris:
        img.putpixel(pixel, (0, 255, 0))
    for pixel in rTris:
        img.putpixel(pixel, (255, 0, 0))


# color over nose using skin color from nose and cheeks
def blurNose(img, landmarks):
    # get nose landmarks from landmarks list
    ptsX = [lm for lm in landmarks[27:36]]
    ptsY = [lm for lm in landmarks[95:104]]
    # get all pixels in nose area using getTri
    l = (min(ptsX)-15, ptsY[ptsX.index(min(ptsX))])
    r = (max(ptsX)+15, ptsY[ptsX.index(max(ptsX))])
    tl = (ptsX[ptsY.index(min(ptsY))]-15, min(ptsY)-5)
    tr = (tl[0]+30, tl[1])
    b = (ptsX[ptsY.index(max(ptsY))], max(ptsY)+4)
    pixels = getTri(l, tl, b)
    pixels.extend(getTri(tl, tr, b))
    pixels.extend(getTri(tr, r, b))
    pixels = list(set(pixels))
    # get upper nose color samples from bridge of nose
    colBox1 = getLine((ptsX[2], ptsY[2]), (ptsX[0], ptsY[0]-5), 5)
    col1 = pixelAverage(colBox1, img)
    # get lower nose color samples from cheeks
    colBox2 = []
    for x in range(10, 31, 2):
        for y in range(-5, 6, 2):
            colBox2.append((ptsX[4]-x, (ptsY[4]+landmarks[107])//2+y))
            colBox2.append((ptsX[8]+x, (ptsY[8]+landmarks[114])//2+y))
    col2 = pixelAverage(colBox2, img)
    # color each pixel in nose area
    for pixel in pixels:
        if pixel[1] <= ptsY[2]:
            img.putpixel(pixel, col1)
        else:
            img.putpixel(pixel, col2)
    # get wider area around nose and blur
    pixels.extend([*getLine((tl[0]-1,tl[1]),(l[0]-3,l[1]),6), *getLine((l[0]-5,l[1]),(b[0],b[1]+3),6),
                   *getLine((b[0],b[1]+3),(r[0]+5,r[1]),6), *getLine((r[0]+3,r[1]),(tr[0]+1,tr[1]),6)])
    pixels = list(set(pixels))
    blur(pixels, img, 3)
