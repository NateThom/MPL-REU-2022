from math import floor, ceil, sqrt, atan, sin, cos


# various functions used by the functions in skintoneblur.

# find the distance between two points using the hypotenuse they form
def get_hyp(a, b):
    # get the x and y distances between the two points
    xlen = b[0] - a[0]
    ylen = b[1] - a[1]
    # good ol' Pythagoras
    hlen = sqrt(xlen**2 + ylen**2)
    return hlen


# find rise and run for line of given length, perpendicular to given line. accepts two 2-tuples and an int.
def get_perp(a, b, leng):
    # find hypotenuse of reference line
    hlen = get_hyp(a, b)
    # find scale factor from reference line to desired line
    if leng == 0 or hlen == 0:
        factor = 0
    else:
        factor = 1 / (hlen / leng)
    # reverse rise and run for perpendicularity and scale to desired line length
    rise = (b[0] - a[0]) * factor
    run = 0 - (b[1] - a[1]) * factor
    # return rise and run as two ints
    return run, rise


# make sure a list of pixels is within image boundaries. accepts a list of 2-tuples
def in_bounds(pixels):
    # make x and y values of every pixel at least 0 and at most 223
    for p in range(len(pixels)):
        # for images of different resolution, change the 223 and 0 in the next two lines to your image bounds
        pixels[p] = (min(pixels[p][0], 223), min(pixels[p][1], 223))
        pixels[p] = (max(pixels[p][0], 0), max(pixels[p][1], 0))
    # return a list of 2-tuples
    return pixels


# scale a set of points away from their midpoint
def transform_polygon(pts, x_s=0, y_s=0, x_t=0, y_t=0, a=None, b=None):
    # pts = the polygon to be scaled
    # xS, yS = how far to scale horizontally and vertically (percentage as a float)
    # xT, yT = the number of pixels by which to translate the polygon (integer)
    # a, b = the endpoints of the polygon's local x-axis

    # if a and b are not specified, use the first and last points in pts
    if a is None and b is None:
        a = pts[0]
        b = pts[-1]

    # find the midpoint of the local x-axis
    mid_pt = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    # find the sine and cosine of the angle of the cluster with respect to global x-axis
    if b[0] - a[0] != 0:
        rads = atan((b[1]-a[1]) / (b[0]-a[0]))
    else:
        rads = 1.57079632679
    rads_cos = cos(rads)
    rads_sin = sin(rads)

    # for each point, rotate around the midpoint so that local x and y match global x and y,
    # scale point away from the midpoint by the percentages given as parameters,
    # and rotate back to the original angle
    for pt in range(len(pts)):
        # rotation around a point formula from Lyle Scott
        # https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
        x_diff = pts[pt][0] - mid_pt[0]
        y_diff = pts[pt][1] - mid_pt[1]
        rx = mid_pt[0] + rads_cos * x_diff + rads_sin * y_diff
        ry = mid_pt[1] - rads_sin * x_diff + rads_cos * y_diff
        # move point away from midpoint the given percentage
        rx += ((rx - mid_pt[0]) * x_s) + x_t
        ry += ((ry - mid_pt[1]) * y_s) + y_t
        # use new distance to the midpoint and reverse equation to rotate back
        x_diff = rx - mid_pt[0]
        y_diff = ry - mid_pt[1]
        rx = mid_pt[0] + rads_cos * x_diff - rads_sin * y_diff
        ry = mid_pt[1] + rads_sin * x_diff + rads_cos * y_diff + 3
        pts[pt] = (rx, ry)

    # return the modified list of points
    return pts


# find the positive or negative distance from a point to a line
def get_line_side(a, b, pt):
    dist = (a[0]-b[0])*(b[1]-pt[1])-(a[1]-b[1])*(b[0]-pt[0])
    # return distance
    if dist >= 0:
        return 1
    else:
        return -1


# get all pixels inside a triangle of points. accepts 3 2-tuples
def get_tri(a, b, c):
    # get width and triangle orientation
    width = range(floor(min(a[0], b[0], c[0])), ceil(max(a[0], b[0], c[0])))
    height = range(floor(min(a[1], b[1], c[1])), ceil(max(a[1], b[1], c[1])))
    a_side = get_line_side(b, c, a)
    b_side = get_line_side(c, a, b)
    c_side = get_line_side(a, b, c)
    # add every pixel within the triangle bounds to a list of pixels
    pixels = []
    for x in width:
        for y in height:
            ab_side = get_line_side(a, b, (x, y))
            ac_side = get_line_side(c, a, (x, y))
            bc_side = get_line_side(b, c, (x, y))
            if ab_side == c_side and ac_side == b_side and bc_side == a_side:
                pixels.append((x, y))
    # ensure that pixels are within image bounds and return as list of 2-tuples
    return in_bounds(pixels)


# get all pixels on lines between a set of points
def get_edges(pts, width):
    pxs = []
    for pt in range(len(pts)):
        for pt2 in range(len(pts[pt:])):
            pxs.extend(get_line(pts[pt], pts[pt2], width))
    return pxs


# get all pixels on a straight line between two points. accepts two 2-tuples and an int
def get_line(pt_a, pt_b, width):
    # find x, y, and hypotenuse lengths
    xlen, ylen = (pt_a[0]-pt_b[0], pt_a[1]-pt_b[1])
    hlen = sqrt(xlen**2 + ylen**2)/(width/2)
    # calculate corners of the line
    if hlen == 0:
        x, y = 0, 0
    else:
        x, y = [(0-lens)/hlen for lens in [ylen, xlen]]
    corners = [(pt_a[0]+x, pt_a[1]-y), (pt_a[0]-x, pt_a[1]+y),
               (pt_b[0]+x, pt_b[1]-y), (pt_b[0]-x, pt_b[1]+y)]
    # get pixels using get_tri
    pixels = get_tri(corners[0], corners[2], corners[3])
    pixels.extend(get_tri(corners[0], corners[1], corners[3]))
    # return a non-repeated list of pixels as 2-tuples
    return list(set(pixels))


# blurs pixels in a square. accepts a list of 2-tuples
def blur(pixels, img, rad):
    # for every pixel, get surrounding pixels and blur
    for p in pixels:
        b_pixels = []
        # grab every pixel in a box around current pixel
        for xrad in range(-rad, rad+1):
            for yrad in range(-rad, rad+1):
                if sqrt(xrad**2 + yrad**2) <= rad:
                    b_pixels.append((p[0]+xrad, p[1]+yrad))
        # take average color of 5x5 box and apply to current pixel
        b_col = px_mean(b_pixels, img)
        img.putpixel(p, b_col)


# take the average of a set of sample pixels. accepts a list of 2-tuples
def px_mean(pixels, img):
    # use in_bounds function to make sure every sample pixel is in image bounds
    pixels = in_bounds(pixels)
    # get colors from each sample pixel
    cols = [img.getpixel(pixel) for pixel in pixels]
    # add mean value from every sample color to new color
    new_col = [0, 0, 0]
    for col in cols:
        for val in range(3):
            new_col[val] += (col[val] / len(cols))
    # turn new color into a 3-tuple of ints and return it
    new_col = (floor(new_col[0]), floor(new_col[1]), floor(new_col[2]))
    return new_col
