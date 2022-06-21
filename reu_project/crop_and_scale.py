import csv
from math import floor


def cropScale():
    print('\n\nCropping and Scaling')
    # get crop points
    with open('../Data_Augmentation/imagepoints/crop_points.csv') as f:
        cpts = list(csv.reader(f))
    with open('../Data_Augmentation/imagepoints/landmarks_68.csv') as f:
        lms = list(csv.reader(f))
    # crop and scale each landmark
    newFile = open('../Data_Augmentation/imagepoints/landmarks_scaled.csv', 'w', newline='')
    writer = csv.writer(newFile)
    writer.writerow(lms[0])
    for i in range(1, 202600):
        # get crop points
        minX = int(cpts[i][1])
        maxX = int(cpts[i][2])
        minY = int(cpts[i][3])
        maxY = int(cpts[i][4])
        width = maxX - minX
        height = maxY - minY
        # crop and scale landmarks
        for lm in range(1, 137):
            if lm < 69:
                lms[i][lm] = str(int(lms[i][lm]) - minX)
                lms[i][lm] = str(floor(int(lms[i][lm]) * (224/width)))
            else:
                lms[i][lm] = str(int(lms[i][lm]) - minY)
                lms[i][lm] = str(floor(int(lms[i][lm]) * (224/height)))
            lms[i][lm] = str(max(0, int(lms[i][lm])))
            lms[i][lm] = str(min(int(lms[i][lm]), 224))
        writer.writerow(lms[i])
    newFile.close()

cropScale()


