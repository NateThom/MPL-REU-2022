from PIL import Image, ImageDraw
import csv


# draw landmarks
def drawLms(img, landmarks):
    # get current image and landmarks
    marked = ImageDraw.Draw(img)
    # draw each landmark
    for i in range(27, 36):
        # choose color based on feature
        if i in range(0, 17):
            color = (0, 255, 0)
        elif i in range(17, 27):
            color = (255, 0, 0)
        elif i in range(27, 36):
            color = (0, 0, 255)
        elif i in range(36, 48):
            color = (255, 255, 0)
        else:
            color = (255, 0, 255)
        color = (0, 255, 0)
        # draw landmark as ellipse
        lmX = landmarks[i]
        lmY = landmarks[i+68]
        marked.ellipse([lmX-2, lmY-2, lmX+2, lmY+2], fill=color)


# import landmarks
with open('../../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
    reader = csv.reader(f)
    global landmarks
    landmarks = list(reader)
    del reader

# open image and begin list of reference images
for imgNum in [22]:
    with Image.open('../../Data_Augmentation/IMG_HiRes/' + str(imgNum).zfill(6) + '.jpg') as img:
        lms = [int(landmarks[imgNum][i]) for i in range(1 , len(landmarks[imgNum]))]
        drawLms(img, lms)
        img.save('../../Data_Augmentation/samples_etc/IMG_landmarks/'
                 + str(imgNum).zfill(6) + '.png')
