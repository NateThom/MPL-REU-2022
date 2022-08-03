import time
from multiprocessing import Pool
import csv
import cv2


def sort_image(row):
    image = cv2.imread('/home/guest/MPL-REU-2022/male/' + row[0])
    h, w, ch = image.shape

    for i in range(1, 69):
        if int(row[i]) > w:
            row[i] = w

    for i in range(69, 137):
        if int(row[i]) > h:
            row[i] = h

    # add row to file
    file_zero = open('landmarks_male_upper.csv', 'a')
    writer = csv.writer(file_zero)
    writer.writerow(row)
    file_zero.close()


file = open('landmarks_male.csv', 'r')
csvreader = csv.reader(file)
rows = list(csvreader)

# clear files
f = open('landmarks_male_upper.csv', 'w')
f.close()

start_time = time.time()
with Pool(18) as p:
    p.map(sort_image, rows)

file.close()

print(time.time() - start_time)

