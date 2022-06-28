from random import randint
from image_functions import *


# each of these functions accepts as parameters:
# img: an Image variable from PIL.Image()
# lms: a list of integer facial region landmarks corresponding to the image; from landmarks.csv

# color over eyebrows using skin color from forehead
def blurEyebrows(img, lms):
        # get current image and eyebrow landmarks
        pntNum = list(range(17, 27))+list(range(85, 95))
        pts = [lms[pt] for pt in pntNum]
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
                color = pxMean([colSamp[randint(0, len(colSamp)-1)] for i in range(10)], img)
                # draw line in new color over eyebrow segment
                pixels = getLine((pt[0], pt[1]+3), ((pts[p]*(0.75-s))+(pts[p+1]*(s+0.25)),
                             ((pts[p+10]*(0.75-s))+(pts[p+11]*(s+0.25)))+3), 14)
                for pixel in pixels:
                    img.putpixel(pixel, color)
            bList.extend(getLine((pts[p], pts[p+10]), (pts[p+1], pts[p+11]), 20))
        bList = list(set(bList))
        blur(bList, img, 2)


# color over eyes using skin color from nose and cheeks
def blurEyes(img, lms):
    # FOR THE LOVE OF ALL THAT'S HOLY, CLEAN UP THE GARBAGE FIRE THAT IS THIS FUNCTION!
    # get eyebrow and eye points from landmarks list
    ebpts = [(lms[lm], lms[lm+68]) for lm in range(17, 27)]
    epts = [(lms[lm], lms[lm+68]) for lm in range(36, 48)]
    # left and right eye regions must be scaled to fill the orbital
    # use the eyebrows as a measure for orbital size and find the percentage by which to scale in x and y
    lY = getHyp(ebpts[3], epts[2]) / getHyp(epts[2], epts[4]) + 0.4
    lX = getHyp(ebpts[0], ebpts[4]) / getHyp(epts[0], epts[3]) - 0.8
    rY = getHyp(ebpts[6], epts[7]) / getHyp(epts[7], epts[11]) + 0.4
    rX = getHyp(ebpts[5], ebpts[9]) / getHyp(epts[6], epts[9]) -0.8
    # scale and translate eye regions to fill the orbital
    lpts = transformPolygon(epts[:6], xS=lX, yS=lY, xT=-4, yT=2, a=epts[0], b=epts[3])
    rpts = transformPolygon(epts[6:], xS=rX, yS=rY, xT=4, yT=2, a=epts[6], b=epts[9])
    # get each pixel in each eye region divided into four triangular subregions
    lTris = {0:getTri(lpts[0],lpts[1],lpts[5]), 1:getTri(lpts[1],lpts[4],lpts[5]),
             2:getTri(lpts[1],lpts[2],lpts[4]), 3:getTri(lpts[2],lpts[3],lpts[4])}
    rTris = {0:getTri(rpts[2],rpts[3],rpts[4]), 2:getTri(rpts[2],rpts[4],rpts[5]),
             1:getTri(rpts[1],rpts[2],rpts[5]), 3:getTri(rpts[0],rpts[1],rpts[5])}
    # for each triangular subregion, get the average color of a nearby patch of skin
    # the color is stored as the last item in the subregion list. send help
    lTris[3].append(pxMean(getLine((lms[27]-4,lms[95]), (lms[28]-4,lms[96]), 4), img))
    rTris[3].append(pxMean(getLine((lms[27]+4,lms[95]), (lms[28]+4,lms[96]), 4), img))
    lTris[2].append(pxMean(getLine((lpts[4][0],lpts[4][1]+10), (lpts[4][0],lpts[4][1]+20), 4), img))
    rTris[2].append(pxMean(getLine((rpts[5][0],rpts[5][1]+10), (rpts[5][0],rpts[5][1]+20), 4), img))
    lTris[1].append(pxMean(getLine((lpts[5][0]+10,lpts[5][1]+10), (lpts[5][0]+15,lpts[5][1]+20), 4), img))
    rTris[1].append(pxMean(getLine((rpts[4][0]-10,rpts[4][1]+10), (rpts[4][0]-15,rpts[4][1]+20), 4), img))
    lTris[0].append(pxMean(getLine((lpts[0][0]+10,lpts[5][1]+10), (lpts[0][0]+15,lpts[5][1]+20), 4), img))
    rTris[0].append(pxMean(getLine((rpts[3][0]-10,rpts[4][1]+10), (rpts[3][0]-15,rpts[4][1]+20), 4), img))
    # color each triangular subregion the appropriate color
    for tri in [*[lTris[i] for i in range(4)], *[rTris[i] for i in range(4)]]:
        for px in tri[:-1]:
            img.putpixel(px, tri[-1])
    # once every pixel is colored, blur both eye regions
    for tri in [*[lTris[i] for i in range(4)], *[rTris[i] for i in range(4)]]:
        blur(tri[:-1], img, 2)


# color over nose using skin color from nose and cheeks
def blurNose(img, lms):
    # get nose landmarks from landmarks list
    ptsX = [lm for lm in lms[27:36]]
    ptsY = [lm for lm in lms[95:104]]
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
    col1 = pxMean(colBox1, img)
    # get lower nose color samples from cheeks
    colBox2 = []
    for x in range(10, 31, 2):
        for y in range(-5, 6, 2):
            colBox2.append((ptsX[4]-x, (ptsY[4]+lms[107])//2+y))
            colBox2.append((ptsX[8]+x, (ptsY[8]+lms[114])//2+y))
    col2 = pxMean(colBox2, img)
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
    blur(pixels, img, 2)


# color over mouth using skin color from chin and cheeks
def blurMouth(img, lms):
    # get mouth landmarks
    mpts = [(lms[i], lms[i+68]) for i in [48,49,50,52,53,54,55,56,58,59,61,67,63,65]]
    # get local x axis of mouth
    endpt1 = ((mpts[-4][0]+mpts[-3][0])/2, (mpts[-4][1]+mpts[-3][1])/2)
    endpt2 = ((mpts[-2][0]+mpts[-1][0])/2, (mpts[-2][1]+mpts[-1][1])/2)
    # scale mouth and translate up slightly
    mpts = transformPolygon(mpts, xS=0.3, yS=0.4, xT=0, yT=-4, a=endpt1, b=endpt2)
    # get reference points for sampling color
    spts = [*transformPolygon([(lms[31],lms[99]), (lms[35],lms[103])], xS=0.2),
            *transformPolygon([(lms[6],lms[74]), (lms[10],lms[78])], xS=-0.2, yT=-10)]
    # get eight triangular subregions of the mouth
    mtris = {0:getTri(mpts[0], mpts[1], mpts[9]), 1:getTri(mpts[1], mpts[2], mpts[9]),
             2:getTri(mpts[2], mpts[8], mpts[9]), 3:getTri(mpts[3], mpts[2], mpts[8]),
             4:getTri(mpts[3], mpts[7], mpts[8]), 5:getTri(mpts[3], mpts[4], mpts[7]),
             6:getTri(mpts[7], mpts[4], mpts[6]), 7:getTri(mpts[4], mpts[5], mpts[6])}
    # get color samples for each triangular subregion
    cols = {0:pxMean(getLine(mpts[0], spts[0],3), img),
            1:pxMean([*getLine(mpts[1], spts[0],3), *getLine(mpts[8], spts[2],3)], img),
            2: pxMean([*getLine(mpts[2], spts[0], 3), *getLine(mpts[7], spts[3], 3)], img),
            3: pxMean([*getLine(mpts[3], spts[1], 3), *getLine(mpts[6], spts[3], 3)], img),
            4:pxMean(getLine(mpts[5], spts[1],3), img)}
    # for each tri, fill with corresponding color from cols dictionary
    for tri in range(8):
        for px in mtris[tri][:-1]:
            img.putpixel(px, cols[ceil(tri/2)])
    # once all tris are colored, blur each tri
    for tri in range(8):
        blur(mtris[tri][:-1], img, 2)


# color over chin using skin color from chin
def blurChin(img, lms):
    # get relevant chin and mouth points from landmarks list
    cpts = [(lms[i], lms[i+68]) for i in range(5, 12)]
    mpts = {0:(lms[56], lms[124]), 1:(lms[57], lms[125]), 2:(lms[58], lms[126])}
    # add a central point just below the mouth
    mdist = tuple(getPerp(mpts[0], mpts[2], -10))
    cpts.append((mpts[1][0]+mdist[0], mpts[1][1]+mdist[1]))
    # scale the chin region and move it down slightly
    cpts = transformPolygon(cpts, 0.2, 0.1, 0, -6, cpts[0], cpts[-2])
    # get six triangular subregions of the chin region
    ctris = [getTri(*[cpts[p] for p in [0, 1, -1]]), getTri(*[cpts[p] for p in [1, 2, -1]]),
             getTri(*[cpts[p] for p in [2, 3, -1]]), getTri(*[cpts[p] for p in [3, 4, -1]]),
             getTri(*[cpts[p] for p in [4, 5, -1]]), getTri(*[cpts[p] for p in [5, 6, -1]])]
    # dictionary correlating subregion to reference point for sampling color
    samplePts = {0:0, 1:1, 2:2, 3:4, 4:5, 5:6}
    # for each triangular subregion, get the average color of a line along it, and fill
    for t in range(6):
        # get the point to sample from for this subregion, and take the average color on a line
        p = samplePts[t]
        color = pxMean(getLine(((cpts[p][0]+cpts[-1][0])//2,(cpts[p][1]+cpts[-1][1])//2), cpts[-1], 4), img)
        # fill subregion with sampled color
        for pt in ctris[t]:
            img.putpixel(pt, color)
    # once all subregions are filled, blur each subregion
    for t in range(6):
        blur(ctris[t][:-1], img, 2)
