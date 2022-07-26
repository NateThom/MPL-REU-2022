# male - 21
import time
from multiprocessing import Pool
import csv


def sort_image(zipped_line):
    line, row = zipped_line
    attributes = line.split()
    if int(attributes[21]) == -1:
        # female
        file_female = open('landmarks_female.csv', 'a')
        writer = csv.writer(file_female)
        writer.writerow(row)
        file_female.close()

    if int(attributes[21]) == 1:
        # male
        file_male = open('landmarks_male.csv', 'a')
        writer = csv.writer(file_male)
        writer.writerow(row)
        file_male.close()


file = open('landmarks_highres_fixed.csv')
csvreader = csv.reader(file)
next(csvreader)
rows = list(csvreader)

file = open('list_attr_celeba.txt', 'r')
lines = file.readlines()[2:]

# clear files
f = open('landmarks_female.csv', 'w')
f.close()
f = open('landmarks_male.csv', 'w')
f.close()

start_time = time.time()
with Pool(18) as p:
    p.map(sort_image, zip(lines, rows))

print(time.time() - start_time)

