import random
import time
import cv2
import csv
import numpy as np
from multiprocessing import Pool


def plot_landmarks(curr_row):
    points = []
    for i in range(1, 69):
        points.append((int(curr_row[i]), int(curr_row[68 + i])))

    return points


def delaunay_triangulation(points):
    # convex hull around landmarks
    points_array = np.asanyarray(points)
    hull = cv2.convexHull(points_array)

    # bound the face in a rectangle and use rect to subdivide the plane
    bounding_rect = cv2.boundingRect(hull)
    sub_div = cv2.Subdiv2D(bounding_rect)

    for p in points:
        sub_div.insert(p)

    # get the triangles
    triangles = sub_div.getTriangleList()
    return np.array(triangles, dtype=np.int32)


def draw_delaunay(triangles, image):
    for t in triangles:
        t_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        cv2.polylines(image, np.array([t_points]), True, (255, 255, 255), 1)


def isolate_feature(triangles, points):
    isolated_triangles = []

    for t in triangles:
        if (t[0], t[1]) in points or (t[2], t[3]) in points or (t[4], t[5]) in points:
            isolated_triangles.append(t)

    return np.array(isolated_triangles)


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
        # return points
        return points

    # case when points contain two points of the feature we want
    if num == 2:
        new_points.append(points[points_on_feature[0]])
        new_points.append(points[points_on_feature[1]])
        top_point = next(iter(set(check_points) - set(points_on_feature)))
        midpoint_1 = (int(
            points[points_on_feature[0]][0] + k * (points[top_point][0] - points[points_on_feature[0]][0]))), \
            int(points[points_on_feature[0]][1] + k * (points[top_point][1] - points[points_on_feature[0]][1]))
        midpoint_2 = (int(
            points[points_on_feature[1]][0] + k * (points[top_point][0] - points[points_on_feature[1]][0]))), \
            int(points[points_on_feature[1]][1] + k * (points[top_point][1] - points[points_on_feature[1]][1]))
        # convert midpoint floats to ints
        new_points.append(midpoint_2)
        new_points.append(midpoint_1)

    # case when points contain one points of the feature we want
    if num == 1:
        new_points.append(points[points_on_feature[0]])
        for s in (set(check_points) - set(points_on_feature)):
            bottom_points.append(s)

        midpoint_1 = (int(
            points[points_on_feature[0]][0] + k * (points[bottom_points[0]][0] - points[points_on_feature[0]][0]))), \
            int(points[points_on_feature[0]][1] + k * (points[bottom_points[0]][1] - points[points_on_feature[0]][1]))
        midpoint_2 = (int(
            points[points_on_feature[0]][0] + k * (points[bottom_points[1]][0] - points[points_on_feature[0]][0]))), \
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

        # check that triangle has a width or height (eg. not cut off by the edges of the image)
        if h > 1 and w > 1:
            cropped = im_source[y:y + h, x:x + w].copy()

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

            if dest_pts.size != 0:
                # check that we want to copy this triangle
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
                triangle_slice = triangle_slice[y_new:y_new + h, x_new:x_new + w]

                # check dimensions based on destination image
                roi = im_dest_copy[y:y + h, x:x + w]
                if roi.shape[:2] != mask.shape[:2] or roi.shape[:2] != triangle_slice.shape[:2]:
                    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
                    triangle_slice = cv2.resize(triangle_slice, (roi.shape[1], roi.shape[0]),
                                                interpolation=cv2.INTER_AREA)

                # invert mask, get background for image
                mask_inv = cv2.bitwise_not(mask)
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # get foreground for image
                triangle_fg = cv2.bitwise_and(triangle_slice, triangle_slice, mask=mask)

                # paste warped triangle to the face mask with the other warped triangles
                im_dest_copy[y:y + h, x:x + w] = average_or(roi_bg, triangle_fg)

                # get background for isolating feature
                roi = feature_image_isolate[y:y + h, x:x + w]
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # paste warped triangle to the black background with other warped triangles
                # to isolate the feature
                feature_image_isolate[y:y + h, x:x + w] = cv2.bitwise_or(roi_bg, triangle_fg)

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


def choose_attribute(points, attribute):
    attribute_dict = {0: 'chin', 1: 'jaw', 2: 'eyebrows', 3: 'nose', 4: 'eyes', 5: 'mouth', 6: 'cheek'}
    curr_points = []

    for a in attribute:
        match attribute_dict[a]:
            case 'chin':
                curr_points = curr_points + points[6:11]
            case 'jaw':
                curr_points = curr_points + points[:17]
            case 'eyebrows':
                curr_points = curr_points + points[17:26]
            case 'nose':
                curr_points = curr_points + points[27:37]
            case 'eyes':
                curr_points = curr_points + points[36:48]
            case 'mouth':
                curr_points = curr_points + points[48:]
            case 'cheek':
                curr_points = curr_points + [points[48], points[54]]

    return curr_points


def swap_attributes(image_pair):
    source, dest = image_pair
    image_source = cv2.imread(image_source_path + source[0])
    points_source = plot_landmarks(source)
    image_dest = cv2.imread(image_dest_path + dest[0])
    points_dest = plot_landmarks(dest)
    attribute_type = [0, 6]
    # {0: 'chin', 1: 'jaw', 2: 'eyebrows', 3: 'nose', 4: 'eyes', 5: 'mouth', 6: 'cheek'}

    # find the triangulation for the destination image (to match in the source image)
    triangles_dest = delaunay_triangulation(points_dest)
    points_feature = choose_attribute(points_dest, attribute_type)
    triangles_dest = isolate_feature(triangles_dest, points_feature)

    # copy triangulation to the source image
    triangles_source = delaunay_copy(points_source, points_dest, triangles_dest)

    # fill background of source image with skin color
    new_image_source = fill_skin_color_background(points_source, image_source)

    # copy attributes from source to destination
    transformed_image = copy_triangles(triangles_source, triangles_dest, new_image_source, image_dest,
                                       points_feature)
    # save image
    source_number = source[0].split('.')
    filename = final_images_path + str(source_number[0]) + '_' + str(dest[0])
    cv2.imwrite(filename, transformed_image)


def find_image_pairs(source, dest):
    """Documentation here"""
    pairs = []
    for d in dest:
        source_images = random.sample(source, 1)
        for s in source_images:
            pairs.append([s, d])

    return pairs


# Modify the lines below to fit your system and file organization
file_source = open('landmarks_male.csv')
file_dest = open('landmarks_female.csv')
image_source_path = '/home/guest/MPL-REU-2022/male/'
image_dest_path = '/home/guest/MPL-REU-2022/female/'
final_images_path = '/home/guest/MPL-REU-2022/swapped_attributes/double2/chin_cheek/male->female/'

# Dataset generation begins here, open landmark files
csvreader = csv.reader(file_source)
rows_source = list(csvreader)
csvreader = csv.reader(file_dest)
rows_dest = list(csvreader)

# image_pair = (rows_source[70469], rows_dest[32162])
# 167867.jpg // 055435.jpg: mouth, mixed clone, male->female
# swap_attributes(image_pair)

# randomly generate image pairs
image_pairs = find_image_pairs(rows_source, rows_dest)
start_time = time.time()

# with 18 CPUs, run the main swap_attributes function
with Pool(18) as pool:
    pool.map(swap_attributes, image_pairs)

print(time.time() - start_time)
