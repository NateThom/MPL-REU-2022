from PIL import Image
import csv
import skintoneblur
import time
from multiprocessing import Pool


def image(imgNum):
    # open image and load into numpy array
    pimg = Image.open(opath + str(imgNum).zfill(6) + '.jpg')

    # get correct row of landmarks file and convert to ints
    lms = [int(i) for i in landmarks[imgNum][1:]]

    # loop through binary for each feature
    for eb in range(2):
        for e in range(2):
            for n in range(2):
                for m in range(2):
                    for c in range(2):
                        # make a clean copy of original image
                        img = pimg.copy()
                        # perform each required manipulation
                        if eb == 1:
                            skintoneblur.blurEyebrows(img, lms)
                        if e == 1:
                            skintoneblur.blurEyes(img, lms)
                        if n == 1:
                            skintoneblur.blurNose(img, lms)
                        if m == 1:
                            skintoneblur.blurMouth(img, lms)
                        if c == 1:
                            skintoneblur.blurChin(img, lms)
                        # save image in correct folder
                        folder = str(eb) + str(e) + str(n) + str(m) + str(c) + '/'
                        img.save(spath+folder+str(imgNum).zfill(6)+'.jpg')

    pimg.close()


def main():
    # ask for number of images - change line 43 for a different sized dataset
    numImg = (input("number of images requested: "))
    if numImg == 'all':
        numImg = 202599
    else:
        numImg = int(numImg)

    # import landmarks
    with open('../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
        reader = csv.reader(f)
        global landmarks
        landmarks = list(reader)
        del reader

    # file paths - change lines 54 and 55 for different open and save paths
    global opath, spath
    opath = '../Data_Augmentation/IMG_HiRes/'
    spath = '../Data_Augmentation/augmented_data/'

    # start timer
    startTime = time.time()

    # multiprocessing loop through images
    with Pool(16) as p:
        p.map(image, range(1, numImg+1))

    # end timer
    totalTime = time.time() - startTime
    print("Elapsed Time: ", totalTime)


if __name__ == "__main__":
    main()
#    image(101)