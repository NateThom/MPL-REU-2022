import os
import shutil
import csv

file_path = '../../../LFW/'

with open(file_path + 'female_names.txt') as female:
    femlist = list(female)
femlist = [name.rstrip() for name in femlist]

with open(file_path + 'male_names.txt') as male:
    malelist = list(male)
malelist = [name.rstrip() for name in malelist]

files = os.listdir(file_path + 'IMG_orig/')

genderlist = [['image_name', 'gender']]
for file in range(len(files)):
    if files[file] in femlist:
        genderlist.append([str(file + 1).zfill(6) + '.jpg', '0'])
    if files[file] in malelist:
        genderlist.append([str(file + 1).zfill(6) + '.jpg', '1'])
    shutil.copy(file_path + 'IMG_orig/' + files[file],
                file_path + 'IMG_numbered/' + str(file + 1).zfill(6) + '.jpg')

with open(file_path + 'gender_attr.csv', 'w+') as genderFile:
    writer = csv.writer(genderFile)
    for line in genderlist:
        writer.writerow(line)
