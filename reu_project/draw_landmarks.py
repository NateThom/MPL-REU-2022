from PIL import ImageDraw


# draw landmarks
def drawLms(img, landmarks):
    # get current image and landmarks
    marked = ImageDraw.Draw(img)
    # draw each landmark
    for i in range(68):
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
            color = (255, 0 ,255)
        # draw landmark as ellipse
        lmX = landmarks[i]
        lmY = landmarks[i+68]
        marked.ellipse([lmX-2, lmY-2, lmX+2, lmY+2], fill=color)
