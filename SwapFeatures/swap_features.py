import cv2
import csv
import numpy as np


def plot_landmarks(curr_row):
    points = []
    for i in range(1, 69):
        points.append((int(curr_row[i]), int(curr_row[68 + i])))
        # cv2.circle(image, (int(curr_row[i]), int(curr_row[68 + i])), 1, (255, 0, 0), -1)

    return points


def delaunay_triangulation(points):
    # convex hull around landmarks
    points_array = np.asanyarray(points)
    hull = cv2.convexHull(points_array)

    # bound the face in a rectangle and use rect to subdivide the plane
    bounding_rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(bounding_rect)

    for p in points:
        subdiv.insert(p)

    # get the triangles
    triangles = subdiv.getTriangleList()
    return np.array(triangles, dtype=np.int32)


def draw_delaunay(triangles, image):
    for t in triangles:
        t_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        cv2.polylines(image, np.array([t_points]), True, (255, 255, 255), 1)


# 1 to 17 is chin
# 18 to 27 is eyebrows
# 28 to 36 is nose
# 37 to 42 is left eye
# 43 to 48 is right eye
# 49 to 68 is mouth
def isolate_feature(triangles, points):
    isolated_triangles = []

    for t in triangles:
        if (t[0], t[1]) in points or (t[2], t[3]) in points or (t[4], t[5]) in points:
            isolated_triangles.append(t)

    return isolated_triangles


def delaunay_copy(points, points_to_copy, triangles):
    dest_triangles = np.zeros((1, 6), dtype=np.int32)

    for t in triangles:
        curr_points = []
        t_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

        for tp in t_points:
            index = points_to_copy.index(tp)
            curr_points.append(points[index])

        dest_points = np.resize(np.array([curr_points]), (6,))
        dest_triangles = np.vstack([dest_triangles, dest_points])

    dest_triangles = np.delete(dest_triangles, 0, axis=0)
    return dest_triangles


def fill_skin_color_background(points, image):
    outside_points = np.array(points[0:33] + points[33:38] + points[42:47])

    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.fillConvexPoly(mask, cv2.convexHull(outside_points), (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(image, mask=mask)[:3]
    mask_inv = cv2.bitwise_not(mask)

    roi = np.zeros(image.shape, np.uint8)
    roi[:] = mean

    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(image, image, mask=mask)
    new_image = cv2.add(roi_bg, img_fg)

    return new_image


def cut_triangle(points, isolated_points, k):
    num = 0
    check_points = [0, 1, 2]
    points_on_feature = []
    new_points = []
    bottom_points = []

    for i in range(0, 3):
        if tuple(points[i]) in isolated_points:
            num += 1
            points_on_feature.append(i)

    # case when all points on the feature we want
    if num == 3:
        return points

    # case when points contain two points of the feature we want
    if num == 2:
        new_points.append(points[points_on_feature[0]])
        new_points.append(points[points_on_feature[1]])
        top_point = next(iter(set(check_points) - set(points_on_feature)))
        midpoint_1 = (int(points[points_on_feature[0]][0] + k * (points[top_point][0] - points[points_on_feature[0]][0]))),\
                     int(points[points_on_feature[0]][1] + k * (points[top_point][1] - points[points_on_feature[0]][1]))
        midpoint_2 = (int(points[points_on_feature[1]][0] + k * (points[top_point][0] - points[points_on_feature[1]][0]))), \
                     int(points[points_on_feature[1]][1] + k * (points[top_point][1] - points[points_on_feature[1]][1]))
        # convert midpoint floats to ints
        new_points.append(midpoint_1)
        new_points.append(midpoint_2)

    # case when points contain one points of the feature we want
    if num == 1:
        new_points.append(points[points_on_feature[0]])
        for s in (set(check_points) - set(points_on_feature)):
            bottom_points.append(s)

        midpoint_1 = (int(points[points_on_feature[0]][0] + k * (points[bottom_points[0]][0] - points[points_on_feature[0]][0]))),\
                     int(points[points_on_feature[0]][1] + k * (points[bottom_points[0]][1] - points[points_on_feature[0]][1]))
        midpoint_2 = (int(points[points_on_feature[0]][0] + k * (points[bottom_points[1]][0] - points[points_on_feature[0]][0]))), \
                     int(points[points_on_feature[0]][1] + k * (points[bottom_points[1]][1] - points[points_on_feature[0]][1]))
        new_points.append(midpoint_1)
        new_points.append(midpoint_2)

    return np.asarray(new_points).astype(int)


def average_or(im1, im2):
    return_image = cv2.bitwise_or(im1, im2)
    temp_image = np.bitwise_and(im1, im2)
    indices = np.nonzero(temp_image)
    if indices[0].size and indices[1].size:
        indices = np.vstack([indices[0], indices[1]])
        indices = np.swapaxes(np.unique(indices, axis=1), 0, 1)
        for i in indices:
            return_image[i[0]][i[1]] = np.maximum(im1[i[0]][i[1]], im2[i[0]][i[1]])

    return return_image


def copy_triangles(t_source, t_dest, im_source, im_dest, isolated_points):
    im_dest_copy = im_dest.copy()
    feature_image_isolate = np.zeros(im_dest.shape, np.uint8)
    seamless_clone_destination = im_dest.copy()

    for t, d in zip(t_source, t_dest):
        # crop rectangle around triangle
        source_pts = np.resize(t, (3, 2))
        x, y, w, h = cv2.boundingRect(source_pts)
        cropped = im_source[y:y+h, x:x+w].copy()

        # adjust source points as necessary
        source_pts = source_pts - source_pts.min(axis=0)

        # create a mask using the triangle, use the mask to create the triangle slice
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [source_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        triangle_slice = cv2.bitwise_and(cropped, cropped, mask=mask)

        # take destination points
        dest_pts = np.resize(d, (3, 2))
        x, y, w, h = cv2.boundingRect(dest_pts)
        x_new = x
        y_new = y

        # take image, affine transform into destination triangle slice
        m = cv2.getAffineTransform(np.float32(source_pts), np.float32(dest_pts - dest_pts.min(axis=0)))
        triangle_slice = cv2.warpAffine(triangle_slice, m, (w, h))

        # find the points that outline the triangle piece after cutting down
        # the area around the feature by some k
        k = 1 / 2
        dest_pts = cut_triangle(dest_pts, isolated_points, k)
        x, y, w, h = cv2.boundingRect(dest_pts)

        # adjust destination points as necessary
        dest_pts = dest_pts - dest_pts.min(axis=0)

        # create mask for pasting warped triangle
        mask = np.zeros((h, w, 3), np.uint8)
        cv2.drawContours(mask, [dest_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # cut down triangle slice based on the mask
        x_new = x - x_new
        y_new = y - y_new
        triangle_slice = triangle_slice[y_new:y_new+h, x_new:x_new+w]

        # check dimensions based on destination image
        if x < 0 or y < 0:
            x = y = 0

        roi = im_dest_copy[y:y+h, x:x+w]
        if roi.shape[:2] != mask.shape[:2] or roi.shape[:2] != triangle_slice.shape[:2]:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            triangle_slice = cv2.resize(triangle_slice, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

        # invert mask, get background for image
        mask_inv = cv2.bitwise_not(mask)
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # get foreground for image
        triangle_fg = cv2.bitwise_and(triangle_slice, triangle_slice, mask=mask)

        # paste warped triangle to the face mask with the other warped triangles
        im_dest_copy[y:y + h, x:x + w] = average_or(roi_bg, triangle_fg)

        # get background for isolating feature
        roi = feature_image_isolate[y:y+h, x:x+w]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # paste warped triangle to the black background with other warped triangles
        # to isolate the feature
        feature_image_isolate[y:y+h, x:x+w] = cv2.bitwise_or(roi_bg, triangle_fg)

    # create a mask for the feature using the isolated feature image
    mask = cv2.cvtColor(feature_image_isolate, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # find the center of the mask using contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]

    center = (int((left + right) / 2), int((top + bottom) / 2))

    # seamlessly clone the images together
    mask_inv = cv2.bitwise_not(mask)
    clear_out_feature = cv2.bitwise_and(seamless_clone_destination, seamless_clone_destination, mask=mask_inv)
    seamless_clone_source = cv2.add(im_dest_copy, clear_out_feature)
    final_image = cv2.seamlessClone(seamless_clone_source, seamless_clone_destination, mask, center, cv2.NORMAL_CLONE)

    return final_image


file = open('landmarks_highres_fixed.csv')
csvreader = csv.reader(file)
next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

for r_source in rows:
    for r_dest in rows:
        if r_source != r_dest:
            image_source = cv2.imread('img_high_res/' + r_source[0])
            points_source = plot_landmarks(r_source)

            image_dest = cv2.imread('img_high_res/' + r_dest[0])
            points_dest = plot_landmarks(r_dest)

            # find the triangulation for the destination image (to match in the source image)
            triangles_dest = delaunay_triangulation(points_dest)

            # 1 to 17 is chin
            # 18 to 27 is eyebrows
            # 28 to 36 is nose
            # 36 to 42 is left eye
            # 43 to 47 is right eye
            # 48 to 68 is mouth
            points_feature = points_dest[36:48]
            triangles_dest = isolate_feature(triangles_dest, points_feature)
            # draw_delaunay(triangles_dest, image_dest)

            # copy triangulation to the source image
            triangles_source = delaunay_copy(points_source, points_dest, triangles_dest)
            # draw_delaunay(triangles_source, image_source)

            # fill background of source image with skin color
            new_image_source = fill_skin_color_background(points_source, image_source)

            # copy attributes from source to destination
            transformed_image = copy_triangles(triangles_source, triangles_dest, new_image_source, image_dest, points_feature)

            # save image
            source_number = r_source[0].split('.')
            filename = 'swapped_attributes/eyes/' + str(source_number[0]) + '_' + str(r_dest[0])
            cv2.imwrite(filename, transformed_image)

cv2.destroyAllWindows()
