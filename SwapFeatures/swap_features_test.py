import cv2
import numpy as np
import dlib
import time
import csv


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def plot_landmarks(curr_row):
    points = []
    for i in range(1, 69):
        points.append((int(curr_row[i]), int(curr_row[68 + i])))

    return points


# 1 to 17 is chin
# 18 to 27 is eyebrows
# 28 to 36 is nose
# 37 to 42 is left eye
# 43 to 48 is right eye
# 49 to 68 is mouth
def isolate_feature():
    return 0


file = open('landmarks_highres_cnn.csv')
csvreader = csv.reader(file)
next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

ij = 0

for r1 in rows:
    for r2 in rows:
        if r1 != r2:
            img = cv2.imread('img_high_res/' + r1[0])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            landmarks_points = plot_landmarks(r1)
            isolate_landmark_points = landmarks_points[36:48]
            mask = np.zeros_like(img_gray)

            img2 = cv2.imread('img_high_res/' + r2[0])
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            landmarks_points2 = plot_landmarks(r2)

            height, width, channels = img2.shape
            img2_new_face = np.zeros((height, width, channels), np.uint8)

            # face 1
            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, convexhull, 255)

            face_image_1 = cv2.bitwise_and(img, img, mask=mask)

            # Delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = extract_index_nparray(index_pt1)

                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = extract_index_nparray(index_pt2)

                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = extract_index_nparray(index_pt3)

                if pt1 in isolate_landmark_points or pt2 in isolate_landmark_points or pt3 in isolate_landmark_points:
                    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                        triangle = [index_pt1, index_pt2, index_pt3]
                        indexes_triangles.append(triangle)

            # Face 2
            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)

            lines_space_mask = np.zeros_like(img_gray)
            lines_space_new_face = np.zeros_like(img2)
            # Triangulation of both faces
            for triangle_index in indexes_triangles:
                # Triangulation of the first face
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)

                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                # Lines space
                cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
                cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
                cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
                lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

                # Triangulation of second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

                if warped_triangle.shape[0] != mask_triangles_designed.shape[0] or warped_triangle.shape[1] != mask_triangles_designed.shape[1]:
                    warped_triangle = cv2.resize(warped_triangle, (mask_triangles_designed.shape[1], mask_triangles_designed.shape[0]), interpolation=cv2.INTER_AREA)

                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

            border_crop_image = img2_new_face.copy()

            # Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(img2_gray)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)

            img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, img2_new_face)

            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

            border_crop_grey = cv2.cvtColor(border_crop_image, cv2.COLOR_BGR2GRAY)
            bc_put, bc_thresh = cv2.threshold(border_crop_grey, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bc_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x2, y2, w2, h2 = cv2.boundingRect(cnt)
            bc_crop = border_crop_image[y2:y2 + h2, x2:x2 + w2]

            # cv2.imshow('crop', bc_crop)
            # cv2.waitKey(0)

            # print(center_face2[0] + w2)
            # print(center_face2[0] - w2)
            # print(center_face2[1] + h2)
            # print(center_face2[1] - h2)
            #
            # print(result.shape)
            # print(img2.shape)
            # print(img2_head_mask.shape)
            # print(center_face2)

            # img2 = cv2.circle(img2, center_face2, 0, (0, 0, 255), 3)
            # cv2.imshow('image', img2)

            try:
                seamless_clone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
                print(str(ij))
            except:
                print('error ' + str(ij) + '---------------------------------------------------------------')

            # cv2.imshow("seamless clone", seamless_clone)
            # cv2.waitKey(0)
            #
            # cv2.destroyAllWindows()

            ij += 1