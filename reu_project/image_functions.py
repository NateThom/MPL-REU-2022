# various functions used by the functions in skintoneblur.

from math import floor, ceil, sqrt, atan, sin, cos

# find the distance between two points using the hypotenuse they form
def getHyp(a, b):
    # get the x and y distances between the two points
    xlen = b[0] - a[0]
    ylen = b[1] - a[1]
    # good ol' Pythagoras
    hlen = sqrt(xlen**2 + ylen**2)
    return hlen


# find rise and run for line of given length, perpendicular to given line. accepts two 2-tuples and an int.
def getPerp(a, b, leng):
    # find hypotenuse of reference line
    hlen = getHyp(a, b)
    # find scale factor from reference line to desired line
    if leng == 0:
        factor = 0
    else:
        factor = 1/ (hlen / leng)
    # reverse rise and run for perpendicularity and scale to desired line length
    rise = (b[0] - a[0]) * factor
    run = 0 - (b[1] - a[1]) * factor
    # return rise and run as two ints
    return run, rise


# make sure a list of pixels is within image boundaries. accepts a list of 2-tuples
def inBounds(pixels):
    # make x and y values of every pixel at least 0 and at most 223
    for p in range(len(pixels)):
        # for images of different resolution, change the 223 and 0 in the next two lines to your image bounds
        pixels[p] = (min(pixels[p][0], 223), min(pixels[p][1], 223))
        pixels[p] = (max(pixels[p][0], 0), max(pixels[p][1], 0))
    # return a list of 2-tuples
    return pixels


# scale a set of points away from their midpoint
def transformPolygon(pts, xS, yS, xT,yT, a, b):
    # pts = the polygon to be scaled
    # xS, yS = how far to scale horizontally and vertically (percentage as a float)
    # xT, yT = the number of pixels by which to translate the polygon (integer)
    # a, b = the endpoints of the polygon's local x-axis

    # find the midpoint of the local x-axis
    midPt = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    # find the sine and cosine of the angle of the cluster with respect to global x-axis
    rads = atan((b[1]-a[1]) / (b[0]-a[0]))
    rads_cos = cos(rads)
    rads_sin = sin(rads)

    # for each point, rotate around the midpoint so that local x and y match global x and y,
    # scale point away from the midpoint by the percentages given as parameters,
    # and rotate back to the original angle
    for pt in range(len(pts)):
        # rotation around a point formula from Lyle Scott
        # https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
        xDiff = pts[pt][0] - midPt[0]
        yDiff = pts[pt][1] - midPt[1]
        rX = midPt[0] + rads_cos * xDiff + rads_sin * yDiff
        rY = midPt[1] - rads_sin * xDiff + rads_cos * yDiff
        # move point away from midpoint the given percentage
        rX += ((rX - midPt[0]) * xS) + xT
        rY += ((rY - midPt[1]) * yS) + yT
        # use new distance to the midpoint and reverse equation to rotate back
        xDiff = rX - midPt[0]
        yDiff = rY - midPt[1]
        rX = midPt[0] + rads_cos * xDiff - rads_sin * yDiff
        rY = midPt[1] + rads_sin * xDiff + rads_cos * yDiff + 3
        pts[pt] = (rX, rY)

    # return the modified list of points
    return pts


# fint the positive or negative distance from a point to a line
def getLineSide(a, b, pt):
    dist = (a[0]-b[0])*(b[1]-pt[1])-(a[1]-b[1])*(b[0]-pt[0])
    # return distance
    if dist >= 0:
        return 1
    else:
        return -1


# get all pixels inside a triangle of points. accepts 3 2-tuples
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
    # ensure that pixels are within image bounds and return as list of 2-tuples
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
    # return a non-repeated list of pixels as 2-tuples
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
        bCol = pxMean(bPixels, img)
        img.putpixel(p, bCol)


# take the average of a set of sample pixels. accepts a list of 2-tuples
def pxMean(pixels, img):
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
