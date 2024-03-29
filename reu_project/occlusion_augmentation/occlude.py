from PIL import Image
import csv
import skintone_occlusions
import time
from multiprocessing import Pool


def compare_imgs(imgs, img):
    for cimg in imgs:
        diffs = 5
        for i in range(5):
            if cimg[i] == img[i]:
                diffs -= 1
            if diffs == 1:
                return imgs.index(cimg)


def image(img_num):
    try:
        # open image and begin list of reference images
        with Image.open(opath + str(img_num).zfill(6) + '.jpg') as fimg:
            rimgs = [fimg.copy()]
            rnums = [(0, 0, 0, 0, 0)]

        # get correct row of landmarks file and convert to ints
        lms = [int(i) for i in landmarks[img_num][1:]]

        # loop through binary for each feature
        for c in range(2):
            for m in range(2):
                for n in range(2):
                    for e in range(2):
                        for eb in range(2):
                            # get a clean copy of the closest reference image
                            img = rimgs[compare_imgs(rnums, (c, m, n, e, eb))].copy()
                            # perform the earliest manipulation
                            if c == 1:
                                skintone_occlusions.blur_eyebrows(img, lms)
                            elif m == 1:
                                skintone_occlusions.blur_eyes(img, lms)
                            elif n == 1:
                                skintone_occlusions.blur_nose(img, lms)
                            elif e == 1:
                                skintone_occlusions.blur_mouth(img, lms)
                            elif eb == 1:
                                skintone_occlusions.blur_chin(img, lms)
                            # save image to reference image list
                            if c == 0:
                                rimgs.append(img)
                                rnums.append((c, m, n, e, eb))
                            # save image in correct folder
                            folder = str(c) + str(m) + str(n) + str(e) + str(eb) + '/'
                            img.save(spath+folder+str(img_num).zfill(6)+'.jpg')
    except Exception as e:
        print(e)
        print('help ', img_num)


def main():
    # ask for number of images - change line 62 for a different sized dataset
    num_img = (input("number of images requested: "))
    if num_img == 'all':
        num_img = 202599
    else:
        num_img = int(num_img)

    # import landmarks
    with open('../../Data_Augmentation/imagepoints/landmarks_highres.csv') as f:
        reader = csv.reader(f)
        global landmarks
        landmarks = list(reader)
        del reader

    # file paths - change lines 75 and 76 for different open and save paths
    global opath, spath
    opath = '../../Data_Augmentation/IMG_HiRes/'
    spath = '../../Data_Augmentation/augmented_data/'

    #  uncomment this section if you have a list of specific images to process.
    #  add    if i in imglist    to list comprehension in line 92
    # with open('../../imglist.txt', 'r') as f:
    #     imglist = {}
    #     for line in f:
    #         l = int(line.rstrip())
    #         imglist[l] = l

    # start timer
    start_time = time.time()

    # multiprocessing loop through images
    with Pool(12) as p:
        p.map(image, [i for i in range(1, num_img+1)]) #if i in imglist])

    # end timer
    total_time = time.time() - start_time
    print("Elapsed Time: ", total_time)


if __name__ == "__main__":
    main()
