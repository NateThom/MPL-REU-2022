from PIL import Image
import csv
import skintoneblur
import time
from multiprocessing import Pool


def compareImgs(imgs, img):
    for cimg in imgs:
        diffs = 5
        for i in range(5):
            if cimg[i] == img[i]:
                diffs -= 1
            if diffs == 1:
                return imgs.index(cimg)


def image(imgNum):
    # open image and begin list of reference images
    with Image.open(opath + str(imgNum).zfill(6) + '.jpg') as fimg:
        rimgs = [fimg.copy()]
        rnums = [(0, 0, 0, 0, 0)]

    # get correct row of landmarks file and convert to ints
    lms = [int(i) for i in landmarks[imgNum][1:]]

    # loop through binary for each feature
    try:
        for c in range(2):
            for m in range(2):
                for n in range(2):
                    for e in range(2):
                        for eb in range(2):
                            # get a clean copy of the closest reference image
                            img = rimgs[compareImgs(rnums, (c, m, n, e, eb))].copy()
                            # perform the earliest manipulation
                            if c == 1:
                                skintoneblur.blurEyebrows(img, lms)
                            elif m == 1:
                                skintoneblur.blurEyes(img, lms)
                            elif n == 1:
                                skintoneblur.blurNose(img, lms)
                            elif e == 1:
                                skintoneblur.blurMouth(img, lms)
                            elif eb == 1:
                                skintoneblur.blurChin(img, lms)
                            # save image to reference image list
                            if c == 0:
                                rimgs.append(img)
                                rnums.append((c, m, n, e, eb))
                            # save image in correct folder
                            folder = str(c) + str(m) + str(n) + str(e) + str(eb) + '/'
                            img.save(spath+folder+str(imgNum).zfill(6)+'.jpg')
    except:
        print('help')
        with open('./errors.txt', 'a') as errors:
            errors.append(str(imgNum).zfill(6))


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
    with Pool(15) as p:
        p.map(image, range(50850, numImg+1))

    # end timer
    totalTime = time.time() - startTime
    print("Elapsed Time: ", totalTime)


if __name__ == "__main__":
    main()