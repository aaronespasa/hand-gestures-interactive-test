import cv2
import numpy as np

path = '../resources/gallery-view-video.mp4'
cap = cv2.VideoCapture(0)


def bgSubMasking(frame):
    """Create a foreground (hand) mask
    @param frame: The video frame
    @return: A masked frame
    """
    fgmask = bgSubtractor.apply(frame, learningRate=0)

    kernel = np.ones((4, 4), np.uint8)

    # The effect is to remove the noise in the background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)  # To close the holes in the objects
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply the mask on the frame and return
    return cv2.bitwise_and(frame, frame, mask=fgmask)

bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=90, detectShadows=False)

if cap.isOpened() == False:
    print("Error opening video  file")

while cap.isOpened():
    ret, frame = cap.read()
    cv2.rectangle(frame, (125, 125), (300, 300), (255, 0, 0), 4)
    crop_image = frame[125:300, 125:300]
    mask = bgSubMasking(crop_image)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

#blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #lowerb = np.array([0, 0, 0])
    #upperb = np.array([12, 255, 255])
    #mask = cv2.inRange(hsv, lowerb, upperb)

    #kernel = np.ones((5, 5))
    #dilation = cv2.dilate(mask, kernel, iterations=1)
    #erosion = cv2.erode(dilation, kernel, iterations=1)
    #filtered = cv2.GaussianBlur(erosion, (3, 3), 0)

    #threshold = [200, 255]
    #ret, thresh = cv2.threshold(filtered, threshold[0], threshold[1], 0)
