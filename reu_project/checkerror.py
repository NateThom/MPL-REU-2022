import csv

def checkErrors():
    with open('../Data_Augmentation/imagepoints/landmarks_scaled.csv') as f:
        orig = list(csv.reader(f))
    with open('../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
        landmarks = list(csv.reader(f))

    errors = []

    for line in range(len(landmarks)-1):
        if landmarks[line+1][1:] == landmarks[line][1:]:
            errors.append(landmarks[line+1][0])

    for i in range(1, len(landmarks)):
        error = False
        for pt in range(1, len(landmarks[i])):
            if abs(int(landmarks[i][pt]) - int(orig[i][pt])) > 20:
                error = True
        if error == True:
            errors.append(landmarks[i][0])

    errors = list(set(errors))
    errors.sort()
    with open('../errors.txt', 'w') as f:
        for e in errors:
            f.write(e+'\n')

checkErrors()