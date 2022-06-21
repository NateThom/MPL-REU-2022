import csv

with open('../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
    hres = list(csv.reader(f))
with open('../Data_Augmentation/imagepoints/landmarks_scaled.csv') as f:
    scld = list(csv.reader(f))
with open('../errors.txt') as f:
    errs = list(csv.reader(f))

new = [hres[0]]
for i in range(1, 202600):
    if [str(i).zfill(6)+'.jpg'] in errs:
        new.append(scld[i])
    else:
        new.append(hres[i])

newFile = open('../Data_Augmentation/imagepoints/landmarks_fixed.csv', 'w', newline='')
writer = csv.writer(newFile)
for line in new:
    writer.writerow(line)
newFile.close()