from random import randint
from image_functions import *


# each of these functions accepts as parameters:
# img: an Image variable from PIL.Image()
# lms: a list of integer facial region landmarks corresponding to the image; from landmarks.csv

# color over eyebrows using skin color from forehead
def blur_eyebrows(img, lms):
    # get current image and eyebrow landmarks
    pnt_num = list(range(17, 27))+list(range(85, 95))
    pts = [lms[pt] for pt in pnt_num]
    b_list = []
    # color over for each space between points
    min_x, min_y = get_perp((pts[0], pts[10]), (pts[9], pts[19]), 5)
    for p in range(9):
        # get angle values for line perpendicular to eyebrows, scaled for distance from center
        max_x, max_y = get_perp((pts[0], pts[10]), (pts[9], pts[19]), 15-2*abs(4-p))
        # split each line into 4
        for s in [0, 0.25, 0.5, 0.75]:
            pt = ((pts[p]*(1-s))+(pts[p+1]*s), (pts[p+10]*(1-s))+(pts[p+11]*s))
            # sample 10 random points from a box above the segment
            col_samp = get_line((pt[0]-min_x, pt[1]-min_y), (pt[0]-max_x, pt[1]-max_y), 9)
            # get average color of sampled pixels
            color = px_mean([col_samp[randint(0, len(col_samp)-1)] for i in range(10)], img)
            # draw line in new color over eyebrow segment
            pixels = get_line((pt[0], pt[1]+3), ((pts[p]*(0.75-s))+(pts[p+1]*(s+0.25)),
                             ((pts[p+10]*(0.75-s))+(pts[p+11]*(s+0.25)))+3), 14)
            for pixel in pixels:
                img.putpixel(pixel, color)
        b_list.extend(get_line((pts[p], pts[p+10]), (pts[p+1], pts[p+11]), 20))
    b_list = list(set(b_list))
    blur(b_list, img, 2)


# color over eyes using skin color from nose and cheeks
def blur_eyes(img, lms):
    # DEAR FUTURE GABE, FOR THE LOVE OF ALL THAT'S HOlY, CLEAN UP THE GARBAGE FIRE THAT
    # IS THIS FUNCTION! WHAT WERE YOU THINKING? - Sincerely, past Gabe
    # get eyebrow and eye points from landmarks list
    ebpts = [(lms[lm], lms[lm+68]) for lm in range(17, 27)]
    epts = [(lms[lm], lms[lm+68]) for lm in range(36, 48)]
    # left and right eye regions must be scaled to fill the orbital
    # use the eyebrows as a measure for orbital size and find the percentage by which to scale in x and y
    ly = get_hyp(ebpts[3], epts[2]) / max(get_hyp(epts[2], epts[4]), 1) + 0.4
    lx = get_hyp(ebpts[0], ebpts[4]) / max(get_hyp(epts[0], epts[3]), 1) - 0.8
    ry = get_hyp(ebpts[6], epts[7]) / max(get_hyp(epts[7], epts[11]), 1) + 0.4
    rx = get_hyp(ebpts[5], ebpts[9]) / max(get_hyp(epts[6], epts[9]), 1) - 0.8
    # scale and translate eye regions to fill the orbital
    lpts = transform_polygon(epts[:6], x_s=lx, y_s=ly, x_t=-4, y_t=2, a=epts[0], b=epts[3])
    rpts = transform_polygon(epts[6:], x_s=rx, y_s=ry, x_t=4, y_t=2, a=epts[6], b=epts[9])
    # get the boundary points of eight triangular subregions
    trpl = [(lpts[0], lpts[1], lpts[5]), (lpts[1], lpts[4], lpts[5]),
            (lpts[1], lpts[2], lpts[4]), (lpts[2], lpts[3], lpts[4]),
            (rpts[2], rpts[3], rpts[4]), (rpts[2], rpts[4], rpts[5]),
            (rpts[1], rpts[2], rpts[5]), (rpts[0], rpts[1], rpts[5])]
    # get each pixel in each eye region divided into four triangular subregions
    l_tris = {0: get_tri(*trpl[0]), 1: get_tri(*trpl[1]), 2: get_tri(*trpl[2]), 3: get_tri(*trpl[3])}
    r_tris = {0: get_tri(*trpl[4]), 1: get_tri(*trpl[5]), 2: get_tri(*trpl[6]), 3: get_tri(*trpl[7])}
    # for each triangular subregion, get the average color of a nearby patch of skin
    # the color is stored as the last item in the subregion list. send help
    l_tris[3].append(px_mean(get_line((lms[27]-4, lms[95]), (lms[28]-4, lms[96]), 4), img))
    r_tris[3].append(px_mean(get_line((lms[27]+4, lms[95]), (lms[28]+4, lms[96]), 4), img))
    l_tris[2].append(px_mean(get_line((lpts[4][0], lpts[4][1]+10), (lpts[4][0], lpts[4][1]+20), 4), img))
    r_tris[2].append(px_mean(get_line((rpts[5][0], rpts[5][1]+10), (rpts[5][0], rpts[5][1]+20), 4), img))
    l_tris[1].append(px_mean(get_line((lpts[5][0]+10, lpts[5][1]+10), (lpts[5][0]+15, lpts[5][1]+20), 4), img))
    r_tris[1].append(px_mean(get_line((rpts[4][0]-10, rpts[4][1]+10), (rpts[4][0]-15, rpts[4][1]+20), 4), img))
    l_tris[0].append(px_mean(get_line((lpts[0][0]+10, lpts[5][1]+10), (lpts[0][0]+15, lpts[5][1]+20), 4), img))
    r_tris[0].append(px_mean(get_line((rpts[3][0]-10, rpts[4][1]+10), (rpts[3][0]-15, rpts[4][1]+20), 4), img))
    # color each triangular subregion the appropriate color
    for tri in [*[l_tris[i] for i in range(4)], *[r_tris[i] for i in range(4)]]:
        for px in tri[:-1]:
            img.putpixel(px, tri[-1])
    # once every pixel is colored, blur both eye regions
    b_list = []
    for t in trpl:
        b_list.extend(get_edges(t, 3))
    blur(list(set(b_list)), img, 3)


# color over nose using skin color from nose and cheeks
def blur_nose(img, lms):
    # get nose landmarks from landmarks list
    pts_x = [lm for lm in lms[27:36]]
    pts_y = [lm for lm in lms[95:104]]
    # get all pixels in nose area using get_tri
    l = (min(pts_x)-10, pts_y[pts_x.index(min(pts_x))])
    r = (max(pts_x)+10, pts_y[pts_x.index(max(pts_x))])
    tl = (pts_x[pts_y.index(min(pts_y))]-15, min(pts_y)-5)
    tr = (tl[0]+30, tl[1])
    b = (pts_x[pts_y.index(max(pts_y))], max(pts_y)+4)
    tris = [get_tri(l, tl, b), get_tri(tl, tr, b), get_tri(tr, r, b)]
    # get upper nose color samples for nose
    col_box = [get_line((pts_x[2]-10, pts_y[2]), (lms[40]-7, lms[108]+15), 5)]
    col_box.append(get_line((pts_x[3], pts_y[3]), (pts_x[1], pts_y[1]), 5))
    col_box.append(get_line((pts_x[2]+10, pts_y[2]), (lms[47]+7, lms[115]+15), 5))
    for t in range(3):
        tris[t].append(px_mean(col_box[t], img))
    # color each pixel in nose area
    for tri in tris:
        for px in tri[:-1]:
            img.putpixel(px, tri[-1])
    # get wider area around nose and blur
    b_list = [*get_edges((l, tl, b), 3), *get_edges((tl, tr, b), 5), *get_edges((tr, r, b), 3)]
    blur(list(set(b_list)), img, 2)


# color over mouth using skin color from chin and cheeks
def blur_mouth(img, lms):
    # get mouth landmarks
    mpts = [(lms[i], lms[i+68]) for i in [*range(48, 51), *range(52, 57), 58, 59, 61, 67, 63, 65]]
    # get local x-axis of mouth
    endpt1 = ((mpts[-4][0]+mpts[-3][0])/2, (mpts[-4][1]+mpts[-3][1])/2)
    endpt2 = ((mpts[-2][0]+mpts[-1][0])/2, (mpts[-2][1]+mpts[-1][1])/2)
    # scale mouth and translate up slightly
    mpts = transform_polygon(mpts, x_s=0.3, y_s=0.4, x_t=0, y_t=-4, a=endpt1, b=endpt2)
    # get reference points for sampling color
    spts = [*transform_polygon([(lms[31], lms[99]), (lms[35], lms[103])], x_s=0.2),
            *transform_polygon([(lms[6], lms[74]), (lms[10], lms[78])], x_s=-0.2, y_t=-10)]
    # get eight triangular subregions of the mouth
    trpl = ((mpts[0], mpts[1], mpts[9]), (mpts[1], mpts[2], mpts[9]), (mpts[2], mpts[8], mpts[9]),
            (mpts[3], mpts[2], mpts[8]), (mpts[3], mpts[7], mpts[8]), (mpts[3], mpts[4], mpts[7]),
            (mpts[7], mpts[4], mpts[6]), (mpts[4], mpts[5], mpts[6]))
    mtris = {0: get_tri(*trpl[0]), 1: get_tri(*trpl[1]), 2: get_tri(*trpl[2]), 3: get_tri(*trpl[3]),
             4: get_tri(*trpl[4]), 5: get_tri(*trpl[5]), 6: get_tri(*trpl[6]), 7: get_tri(*trpl[7])}
    # get color samples for each triangular subregion
    cols = {0: px_mean(get_line(mpts[0], spts[0], 3), img),
            1: px_mean([*get_line(mpts[1], spts[0], 3), *get_line(mpts[8], spts[2], 3)], img),
            2: px_mean([*get_line(mpts[2], spts[0], 3), *get_line(mpts[7], spts[3], 3)], img),
            3: px_mean([*get_line(mpts[3], spts[1], 3), *get_line(mpts[6], spts[3], 3)], img),
            4: px_mean(get_line(mpts[5], spts[1], 3), img)}
    # for each tri, fill with corresponding color from cols dictionary
    for tri in range(8):
        for px in mtris[tri][:-1]:
            img.putpixel(px, cols[ceil(tri/2)])
    # once all tris are colored, blur each tri
    b_list = []
    for t in trpl:
        b_list.extend(get_edges(t, 3))
    blur(list(set(b_list)), img, 2)


# color over chin using skin color from chin
def blur_chin(img, lms):
    # get relevant chin and mouth points from landmarks list
    cpts = [(lms[i], lms[i+68]) for i in range(5, 12)]
    mpts = {0: (lms[56], lms[124]), 1: (lms[57], lms[125]), 2: (lms[58], lms[126])}
    # add a central point just below the mouth
    mdist = tuple(get_perp(mpts[0], mpts[2], -10))
    rpt = (mpts[1][0]+mdist[0], mpts[1][1]+mdist[1])
    # scale the chin region and move it down slightly
    cpts = transform_polygon(cpts, 0.2, 0.1, 0, -6, cpts[0], cpts[-1])
    # get six triangular subregions of the chin region
    trpl = [(cpts[n], cpts[n+1], rpt) for n in range(6)]
    ctris = [get_tri(*trpl[t]) for t in range(6)]
    # dictionary correlating subregion to reference point for sampling color
    sample_pts = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    # for each triangular subregion, get the average color of a line along it, and fill
    for t in range(6):
        # get the point to sample from for this subregion, and take the average color on a line
        p = sample_pts[t]
        color = px_mean(get_line(((cpts[p][0]+rpt[0])//2, (cpts[p][1]+rpt[1])//2), rpt, 4), img)
        # fill subregion with sampled color
        for pt in ctris[t]:
            img.putpixel(pt, color)
    # once all subregions are filled, blur each subregion
    b_list = []
    for t in trpl:
        b_list.extend(get_edges(t, 3))
    blur(list(set(b_list)), img, 2)
