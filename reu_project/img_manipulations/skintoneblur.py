from random import randint
from image_functions import *
from PIL import ImageDraw


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
    # DEAR FUTURE GABE, FOR THE LOVE OF ALL THAT'S HOLY, CLEAN UP THE GARBAGE FIRE THAT
    # IS THIS FUNCTION! WHAT WERE YOU THINKING? - Sincerely, past Gabe
    # get eyebrow and eye points from landmarks list
    ebpts = [(lms[lm], lms[lm+68]) for lm in range(17, 27)]
    epts = [(lms[lm], lms[lm+68]) for lm in range(36, 48)]
    # left and right eye regions must be scaled to fill the orbital
    # use the eyebrows as a measure for orbital size and find the percentage by which to scale in x and y
    lY = getHyp(ebpts[3], epts[2]) / max(getHyp(epts[2], epts[4]), 1) + 0.4
    lX = getHyp(ebpts[0], ebpts[4]) / max(getHyp(epts[0], epts[3]), 1) - 0.8
    rY = getHyp(ebpts[6], epts[7]) / max(getHyp(epts[7], epts[11]), 1) + 0.4
    rX = getHyp(ebpts[5], ebpts[9]) / max(getHyp(epts[6], epts[9]), 1) - 0.8
    # scale and translate eye regions to fill the orbital
    lpts = transformPolygon(epts[:6], xS=lX, yS=lY, xT=-4, yT=2, a=epts[0], b=epts[3])
    rpts = transformPolygon(epts[6:], xS=rX, yS=rY, xT=4, yT=2, a=epts[6], b=epts[9])
    # get the boundary points of eight triangular subregions
    trpl = [(lpts[0],lpts[1],lpts[5]), (lpts[1],lpts[4],lpts[5]), (lpts[1],lpts[2],lpts[4]),
            (lpts[2],lpts[3],lpts[4]), (rpts[2],rpts[3],rpts[4]), (rpts[2],rpts[4],rpts[5]),
            (rpts[1],rpts[2],rpts[5]), (rpts[0],rpts[1],rpts[5])]
    # get each pixel in each eye region divided into four triangular subregions
    lTris = {0:getTri(*trpl[0]), 1:getTri(*trpl[1]), 2:getTri(*trpl[2]), 3:getTri(*trpl[3])}
    rTris = {0:getTri(*trpl[4]), 1:getTri(*trpl[5]), 2:getTri(*trpl[6]), 3:getTri(*trpl[7])}
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
    blist = []
    for t in trpl:
        blist.extend(getEdges(t, 3))
    blur(list(set(blist)), img, 3)


# color over nose using skin color from nose and cheeks
def blurNose(img, lms):
    marked = ImageDraw.Draw(img)
    # get nose landmarks from landmarks list
    ptsX = [lm for lm in lms[27:36]]
    ptsY = [lm for lm in lms[95:104]]
    # get all pixels in nose area using getTri
    l = (min(ptsX)-10, ptsY[ptsX.index(min(ptsX))])
    r = (max(ptsX)+10, ptsY[ptsX.index(max(ptsX))])
    tl = (ptsX[ptsY.index(min(ptsY))]-15, min(ptsY)-5)
    tr = (tl[0]+30, tl[1])
    b = (ptsX[ptsY.index(max(ptsY))], max(ptsY)+4)
    tris = [getTri(l, tl, b), getTri(tl, tr, b), getTri(tr, r, b)]
    # get upper nose color samples for nose
    colBox = [getLine((ptsX[2]-10, ptsY[2]), (lms[40]-7, lms[108]+15), 5)]
    colBox.append(getLine((ptsX[3], ptsY[3]), (ptsX[1], ptsY[1]), 5))
    colBox.append(getLine((ptsX[2]+10, ptsY[2]), (lms[47]+7, lms[115]+15), 5))
    for t in range(3):
        tris[t].append(pxMean(colBox[t], img))
    # color each pixel in nose area
    for tri in tris:
        for px in tri[:-1]:
            img.putpixel(px, tri[-1])
    # get wider area around nose and blur
    blist = [*getEdges((l,tl,b), 3), *getEdges((tl,tr,b), 5), *getEdges((tr,r,b), 3)]
    blur(list(set(blist)), img, 2)


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
    trpl = ((mpts[0], mpts[1], mpts[9]), (mpts[1], mpts[2], mpts[9]), (mpts[2], mpts[8],mpts[9]),
            (mpts[3], mpts[2], mpts[8]), (mpts[3], mpts[7], mpts[8]), (mpts[3], mpts[4], mpts[7]),
            (mpts[7], mpts[4], mpts[6]), (mpts[4], mpts[5], mpts[6]))
    mtris = {0: getTri(*trpl[0]), 1: getTri(*trpl[1]), 2: getTri(*trpl[2]), 3: getTri(*trpl[3]),
             4: getTri(*trpl[4]), 5: getTri(*trpl[5]), 6: getTri(*trpl[6]), 7: getTri(*trpl[7])}
    # get color samples for each triangular subregion
    cols = {0: pxMean(getLine(mpts[0], spts[0],3), img),
            1: pxMean([*getLine(mpts[1], spts[0],3), *getLine(mpts[8], spts[2],3)], img),
            2: pxMean([*getLine(mpts[2], spts[0], 3), *getLine(mpts[7], spts[3], 3)], img),
            3: pxMean([*getLine(mpts[3], spts[1], 3), *getLine(mpts[6], spts[3], 3)], img),
            4: pxMean(getLine(mpts[5], spts[1],3), img)}
    # for each tri, fill with corresponding color from cols dictionary
    for tri in range(8):
        for px in mtris[tri][:-1]:
            img.putpixel(px, cols[ceil(tri/2)])
    # once all tris are colored, blur each tri
    blist = []
    for t in trpl:
       blist.extend(getEdges(t, 3))
    blur(list(set(blist)), img, 2)


# color over chin using skin color from chin
def blurChin(img, lms):
    # get relevant chin and mouth points from landmarks list
    cpts = [(lms[i], lms[i+68]) for i in range(5, 12)]
    mpts = {0:(lms[56], lms[124]), 1:(lms[57], lms[125]), 2:(lms[58], lms[126])}
    # add a central point just below the mouth
    mdist = tuple(getPerp(mpts[0], mpts[2], -10))
    rpt = (mpts[1][0]+mdist[0], mpts[1][1]+mdist[1])
    # scale the chin region and move it down slightly
    cpts = transformPolygon(cpts, 0.2, 0.1, 0, -6, cpts[0], cpts[-1])
    # get six triangular subregions of the chin region
    trpl = [(cpts[n], cpts[n+1], rpt) for n in range(6)]
    ctris = [getTri(*trpl[t]) for t in range(6)]
    # dictionary correlating subregion to reference point for sampling color
    samplePts = {0:0, 1:1, 2:2, 3:4, 4:5, 5:6}
    # for each triangular subregion, get the average color of a line along it, and fill
    for t in range(6):
        # get the point to sample from for this subregion, and take the average color on a line
        p = samplePts[t]
        color = pxMean(getLine(((cpts[p][0]+rpt[0])//2,(cpts[p][1]+rpt[1])//2), rpt, 4), img)
        # fill subregion with sampled color
        for pt in ctris[t]:
            img.putpixel(pt, color)
    # once all subregions are filled, blur each subregion
    blist = []
    for t in trpl:
        blist.extend(getEdges(t, 3))
    blur(list(set(blist)), img, 2)
