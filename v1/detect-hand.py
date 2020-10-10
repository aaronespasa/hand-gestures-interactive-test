import cv2
import numpy as np
import math
import time
import sys

vid_path = '../resources/hand-first-video.mp4'
# cap = cv2.VideoCapture(vid_path)
# To use your webcam comment the line 8
# and uncomment the line 11
cap = cv2.VideoCapture(0)

initial_time = time.time()
gesture_delay = time.time() + 3
gestures_counter = 0
text = "No gestures"

output = {
    'one-finger': [],
    'two-fingers': [],
    'three-fingers': [],
    'ok': []
}

while cap.isOpened():
    ret, frame = cap.read()

    # Get hand data from a rectangle sub window
    pt1, pt2 = (50, 50), (300, 300)
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 0)

    try:
        crop_image = frame[100:300, 100:300]
    except:
        print(output)
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors
    # and rest is black
    lowerb = np.array([0, 0, 0])
    upperb = np.array([12, 255, 255])
    mask2 = cv2.inRange(hsv, lowerb, upperb)

    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 155, 255, 0)

    cv2.imshow("Thresholded image", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w , h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (255, 0, 255), 0)

        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find the angle of the far point from the start
        # and end point
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            # cv2.line(crop_image, (start[0], start[1]), (far[0], far[1]), [0, 50, 120], 4)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # cv2.line(crop_image, (far[0], far[1]), (end[0], end[1]), [0, 50, 120], 4)

            cv2.circle(crop_image, far, 4, [255, 255, 255], -1)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / math.pi

            if angle <= 100:
                count_defects += 1
                cv2.circle(crop_image, far, 4, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [255, 255, 0], 2)

        delay = 1 # sec
        cond_delay = time.time() > gesture_delay
        # first_three_sec = time.time() < 3
        # Print the number of fingers applying a delay between gesture
        if cond_delay: # or first_three_sec
            if count_defects == 0:
                text = "ONE"
                output['one-finger'].append(round(time.time() - initial_time, 2))
                gesture_delay = time.time() + delay
                gestures_counter += 1
            elif count_defects == 1:
                text = "TWO"
                output['two-fingers'].append(round(time.time() - initial_time, 2))
                gesture_delay = time.time() + delay
                gestures_counter += 1
            elif count_defects == 2:
                text = "THREE"
                output['three-fingers'].append(round(time.time() - initial_time, 2))
                gesture_delay = time.time() + delay
                gestures_counter += 1
            elif count_defects in range(3, 6):
                text = "OK"
                output['ok'].append(round(time.time() - initial_time, 2))
                gesture_delay = time.time() + delay
                gestures_counter += 1

        cv2.putText(frame, str(gestures_counter)+ ": " + text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('p'):
        cv2.waitKey(-1)

print(output,
      '\n\n')
cap.release()
cv2.destroyAllWindows()
