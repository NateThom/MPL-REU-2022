from PIL import Image
import csv
import skintoneblur
import draw_landmarks

# ask for number of images
numImg = (input("number of images requested: "))
if numImg == 'all':
    numImg = 202599
else:
    numImg = int(numImg)

# import landmarks
with open('../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
    reader = csv.reader(f)
    landmarks = list(reader)

def image(imgNum):
    # open image - edit file path to grab other images
    img = Image.open('../Data_Augmentation/IMG_HiRes/' + str(imgNum).zfill(6) + '.jpg')

    # get correct row of landmarks file and convert to ints
    lms = [int(i) for i in landmarks[imgNum][1:]]

    # manimpulate image - change functions to perform different manipulations
#    skintoneblur.blurEyebrows(img, lms)
#    skintoneblur.blurNose(img, lms)
#    skintoneblur.blurEyes(img, lms)
    draw_landmarks.drawLms(img, lms)

    # save and close image - edit file path to save elsewhere
    img.save('../Data_Augmentation/IMG_landmarks/' + str(imgNum).zfill(6)+'.jpg')
    img.close()


for imgNum in range(1, numImg+1):
    image(imgNum)
