import csv

with open('../../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
    hres = list(csv.reader(f))
with open('../../Data_Augmentation/imagepoints/landmarks_errors_hogfirst_cnnsecond.csv') as f:
    fixed = list(csv.reader(f))
#with open('../../errors.txt') as f:
#    errs = [*i for i in list(csv.reader(f))]

errs = []
fixed_dict = {}
for i in fixed[1:]:
    if i[1] != 'x_0':
        fixed_dict[int(i[0])] = int(fixed.index(i))
        i[0] = i[0].rstrip().zfill(6) + '.jpg'
        errs.append(i[0])

print(errs)
new = [hres[0]]
for i in range(1, 202600):
    if str(i).zfill(6)+'.jpg' in errs:
        new.append(fixed[fixed_dict[i]])
        print('yo')
    else:
        new.append(hres[i])

newFile = open('../../Data_Augmentation/imagepoints/landmarks_fixed.csv', 'w', newline='')
writer = csv.writer(newFile)
for line in new:
    writer.writerow(line)
newFile.close()