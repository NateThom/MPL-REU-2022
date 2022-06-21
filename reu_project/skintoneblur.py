# IF I HAVE TIME:
#   adjust eyebrow sampling for angled faces
#   add round ends to getLine

from PIL import ImageDraw, Image
from random import randint
from math import floor, ceil, sqrt


# color over eyebrows using skin color from forehead
def blurEyebrows(img, landmarks):
        # get current image and eyebrow landmarks
        marked = ImageDraw.Draw(img)
        pntNum = list(range(17, 27))+list(range(85, 95))
        pts = [landmarks[pt] for pt in pntNum]
        bList = []
        # color over for each space between points
        for p in range(9):
            # split each line into 4
            for s in [0, 0.25, 0.5, 0.75]:
                # sample 10 random points from a box above the segment. box is smaller at edge
                colSamp = [((pts[p]*(1-s))+(pts[p+1]*s)+randint(-4,5),
                           (pts[p+10]*(1-s))+(pts[p+11]*s)-randint(4,15-2*abs(4-p)))
                           for c in range(10)]
                # get average color of sampled pixels
                color = pixelAverage(colSamp, img)
                # draw line in new color over eyebrow segment
                marked.line([(pts[p]*(1-s))+(pts[p+1]*s), ((pts[p+10]*(1-s))+(pts[p+11]*s)+3),
                             (pts[p]*(0.75-s))+(pts[p+1]*(s+0.25)), ((pts[p+10]*(0.75-s))+(pts[p+11]*(s+0.25)))+3],
                            fill=color, width=14)
            bList.extend(getLine((pts[p], pts[p+10]), (pts[p+1], pts[p+11]), 20))
        bList = list(set(bList))
        blur(bList, img, 2)


# color over eyes using skin color from forehead and cheeks
def blurEyes(img, lms):
    lX = ((lms[21]-lms[17]) / (lms[39]-lms[36]+0.0001)) - 0.8
    lY = (lms[106]-lms[88]) / (lms[108]-lms[106]+0.0001)
    lpts = scalePolygon([(lms[lm], lms[lm+68]) for lm in range(36, 42)], lX, lY)
    rX = ((lms[26]-lms[22]) / (lms[45]-lms[42]+0.0001)) - 0.9
    rY = (lms[111]-lms[91]) / (lms[115]-lms[111]+0.0001)
    rpts = scalePolygon([(lms[lm], lms[lm+68]) for lm in range(42, 48)], rX, rY)
    for pts in [lpts, rpts]:
        for pt in range(6):
            if pts[pt][1] > max(pts[4][1], pts[5][1]):
                pts[pt] = (pts[pt][0], max(pts[4][1], pts[5][1]))
            elif pts[pt][1] < min(pts[1][1], pts[2][1]):
                pts[pt] = (pts[pt][0], min(pts[1][1], pts[2][1]))
            if pts[pt][0] > pts[3][0]:
                pts[pt] = (pts[3][0], pts[pt][1])
            elif pts[pt][0] < pts[0][0]:
                pts[pt] = (pts[0][0], pts[pt][1])
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


# make sure a list of pixels is within image boundaries. accepts a list of 2-tuples
def inBounds(pixels):
    # make x and y values of every pixel at least 0 and at most 223
    for p in range(len(pixels)):
        pixels[p] = (min(pixels[p][0], 223), min(pixels[p][1], 223))
        pixels[p] = (max(pixels[p][0], 0), max(pixels[p][1], 0))
    return pixels


# scale a set of points away from their center of mass
def scalePolygon(pts, x, y):
    # find the center point of the cluster
    xcent = sum(pt[0] for pt in pts) // len(pts)
    ycent = sum(pt[1] for pt in pts) // len(pts)
    # move each point the given percentage away from the center
    for pt in range(len(pts)):
        pts[pt] = (pts[pt][0]+((pts[pt][0]-xcent)*x), pts[pt][1]+((pts[pt][1]-ycent)*y))
    return pts


# determine whether a point is above or below a line
def getLineSide(a, b, pt):
    dist = (a[0]-b[0])*(b[1]-pt[1])-(a[1]-b[1])*(b[0]-pt[0])
    # return +1 or -1 depending on which side the point is on
    if dist >= 0:
        return 1
    else:
        return -1


# get all pixels inside a quadrilateral. accepts 4 2-tuples
def getTri(a, b, c):
    # get width and triangle orientation
    width = range(floor(min(a[0], b[0], c[0])), ceil(max(a[0], b[0], c[0])))
    height = range(floor(min(a[1], b[1], c[1])), ceil(max(a[1], b[1], c[1])))
    aSide = getLineSide(b, c, a)
    bSide = getLineSide(c, a, b)
    cSide = getLineSide(a, b, c)
    # add every pixel within the triangle bounds to a list of pixels
    pixels = []
    for x in width:
        for y in height:
            abSide = getLineSide(a, b, (x, y))
            acSide = getLineSide(c, a, (x, y))
            bcSide = getLineSide(b, c, (x, y))
            if abSide == cSide and acSide == bSide and bcSide == aSide:
                pixels.append((x, y))
    # ensure that pixels are within image bounds and return
    return inBounds(pixels)


# get all pixels on a straight line between two points. accepts two 2-tuples and an int
def getLine(ptA, ptB, width):
    # find x, y, and hypotenuse lengths
    xlen, ylen = (ptA[0]-ptB[0], ptA[1]-ptB[1])
    hlen = sqrt(xlen**2 + ylen**2)/(width/2)+0.00000001
    # calculate corners of the line
    x, y = [(0-lens)/hlen for lens in [ylen, xlen]]
    corners = [(ptA[0]+x,ptA[1]-y), (ptA[0]-x,ptA[1]+y),
               (ptB[0]+x,ptB[1]-y), (ptB[0]-x,ptB[1]+y)]
    # get pixels using getTri
    pixels = getTri(corners[0], corners[2], corners[3])
    pixels.extend(getTri(corners[0], corners[1], corners[3]))
    return list(set(pixels))


# blurs pixels in a square. accepts a list of 2-tuples
def blur(pixels, img, rad):
    # for every pixel, get surrounding pixels and blur
    for p in pixels:
        bPixels = []
        # grab every pixel in a box around current pixel

        for xrad in range(-rad, rad+1):
            for yRad in range(-rad, rad+1):
                bPixels.append((p[0]+xrad, p[1]+yRad))
        # take average color of 5x5 box and apply to current pixel
        bCol = pixelAverage(bPixels, img)
        img.putpixel(p, bCol)


# take the average of a set of sample pixels. accepts a list of 2-tuples
def pixelAverage(pixels, img):
    # use inBounds function to make sure every sample pixel is in image bounds
    pixels = inBounds(pixels)
    # get colors from each sample pixel
    cols = [img.getpixel(pixel) for pixel in pixels]
    # add mean value from every sample color to new color
    newCol = [0, 0, 0]
    for col in cols:
        for val in range(3):
            newCol[val] += (col[val] / len(cols))
    # turn new color into a 3-tuple of ints and return it
    newCol = (floor(newCol[0]), floor(newCol[1]), floor(newCol[2]))
    return newCol
