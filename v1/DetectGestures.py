import cv2
import numpy as np
import math
import time
import sys
import operator
from itertools import repeat
import json


class DetectGestures:
    def __init__(self, path=0, initial_delay=3):  # use webcam by default
        """
        :param path: Video path (String)
        :param delay_gestures: Time to wait before analyzing another gesture
        :param initial_delay: Delay before starting to analyze the gestures
        """
        self.path = path
        self.initial_time = time.time()
        self.initial_delay = initial_delay
        self.gesture_delay = [0, 0, 0]
        self.gestures_counter = [0, 0, 0, 0]
        self.text = "No gestures"
        self.output = {
            "0": {
                "one-finger": [],
                "two-fingers": [],
                "three-fingers": []
            },
            "1": {
                "one-finger": [],
                "two-fingers": [],
                "three-fingers": []
            },
            "2": {
                "one-finger": [],
                "two-fingers": [],
                "three-fingers": []
            }
        }
        self.threshold = [155, 200]
        self.text_position = [(50, 50), (640, 50), (0, 600)]
        self.analyze_video()

    def analyze_video(self):
        """Detect the gestures of the person and add them to output

        Press 'p' to pause the analysis
        Press 'q' to quit the analysis (output will be modified anyway)
        """
        cap = cv2.VideoCapture(self.path)

        #out = cv2.VideoWriter('finalVideoKeypoints.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1280, 720))

        if cap.isOpened() == False:
            print("Error opening video  file")

        while cap.isOpened():
            ret, frame = cap.read()

            # Get hand data from a rectangle sub window
            vid_points = {
                0: {
                    'x': (150, 300),
                    'y': (100, 350)
                },
                1: {
                    'x': (300, 600),
                    'y': (50, 300)
                },
                2: {
                    'x': (50, 250),
                    'y': (100, 300)
                }
            }
            pt1 = [
                (vid_points[0]['x'][0], vid_points[0]['y'][0]),
                (vid_points[1]['x'][0], vid_points[1]['y'][0]),
                (vid_points[2]['x'][0], vid_points[2]['y'][0])
            ]
            pt2 = [
                (vid_points[0]['x'][1], vid_points[0]['y'][1]),
                (vid_points[1]['x'][1], vid_points[1]['y'][1]),
                (vid_points[2]['x'][1], vid_points[2]['y'][1])
            ]

            try:
                # First video: (0, 0), (640, 360)
                # Second video: (640, 0), (1280, 360)
                # Third video: (0', 360), (640, 720)
                first_video = frame[0:360, 0:640]
                second_video = frame[0:360, 640:1280]
                third_video = frame[360:720, 0:640]

                cv2.rectangle(first_video, pt1[0], pt2[0], (0, 255, 255), 3)
                cv2.rectangle(second_video, pt1[1], pt2[1], (0, 255, 255), 3)
                cv2.rectangle(third_video, pt1[2], pt2[2], (0, 255, 255), 3)

                crop_image_1 = first_video[pt1[0][1]:pt2[0][1], pt1[0][0]:pt2[0][0]]
                crop_image_2 = second_video[pt1[1][1]:pt2[1][1], pt1[1][0]:pt2[1][0]]
                crop_image_3 = third_video[pt1[2][1]:pt2[2][1], pt1[2][0]:pt2[2][0]]
                crop_image = [crop_image_1, crop_image_2, crop_image_3]
            except:
                print(self.output)
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

            blur_1 = cv2.GaussianBlur(crop_image_1, (3, 3), 0)
            blur_2 = cv2.GaussianBlur(crop_image_2, (3, 3), 0)
            blur_3 = cv2.GaussianBlur(crop_image_3, (3, 3), 0)

            hsv_1 = cv2.cvtColor(blur_1, cv2.COLOR_BGR2HSV)
            hsv_2 = cv2.cvtColor(blur_2, cv2.COLOR_BGR2HSV)
            hsv_3 = cv2.cvtColor(blur_3, cv2.COLOR_BGR2HSV)

            # Create a binary image with where white will be skin colors
            # and rest is black
            lowerb = np.array([0, 0, 0])
            upperb = np.array([12, 255, 255])
            mask_1 = cv2.inRange(hsv_1, lowerb, upperb)
            mask_2 = cv2.inRange(hsv_2, lowerb, upperb)
            mask_3 = cv2.inRange(hsv_3, lowerb, upperb)

            kernel = np.ones((5, 5))

            # Apply morphological transformations to filter out the background noise
            dilation_1 = cv2.dilate(mask_1, kernel, iterations=1)
            dilation_2 = cv2.dilate(mask_2, kernel, iterations=1)
            dilation_3 = cv2.dilate(mask_3, kernel, iterations=1)
            erosion_1 = cv2.erode(dilation_1, kernel, iterations=1)
            erosion_2 = cv2.erode(dilation_2, kernel, iterations=1)
            erosion_3 = cv2.erode(dilation_3, kernel, iterations=1)

            filtered_1 = cv2.GaussianBlur(erosion_1, (3, 3), 0)
            filtered_2 = cv2.GaussianBlur(erosion_2, (3, 3), 0)
            filtered_3 = cv2.GaussianBlur(erosion_3, (3, 3), 0)
            ret_1, thresh_1 = cv2.threshold(filtered_1, self.threshold[0], self.threshold[1], 0)
            ret_2, thresh_2 = cv2.threshold(filtered_2, self.threshold[0], self.threshold[1], 0)
            ret_3, thresh_3 = cv2.threshold(filtered_3, self.threshold[0], self.threshold[1], 0)

            cv2.imshow("Thresholded 1", thresh_1)
            cv2.imshow("Thresholded 2", thresh_2)
            cv2.imshow("Thresholded 3", thresh_3)

            # Find contours
            contours_1, hierarchy_1 = cv2.findContours(thresh_1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_2, hierarchy_2 = cv2.findContours(thresh_2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_3, hierarchy_3 = cv2.findContours(thresh_3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            try:
                # Find contour with maximum area
                contour_1 = max(contours_1, key=lambda x: cv2.contourArea(x))
                contour_2 = max(contours_2, key=lambda x: cv2.contourArea(x))
                contour_3 = max(contours_3, key=lambda x: cv2.contourArea(x))
                contour = [contour_1, contour_2, contour_3]

                # Create bounding rectangle around the contour
                x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
                x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)
                x_3, y_3, w_3, h_3 = cv2.boundingRect(contour_3)
                cv2.rectangle(crop_image_1, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (255, 0, 255), 0)
                cv2.rectangle(crop_image_2, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (255, 0, 255), 0)
                cv2.rectangle(crop_image_3, (x_3, y_3), (x_3 + w_3, y_3 + h_3), (255, 0, 255), 0)

                hull_1 = cv2.convexHull(contour_1)
                hull_2 = cv2.convexHull(contour_2)
                hull_3 = cv2.convexHull(contour_3)

                # Draw contour
                drawing_1 = np.zeros(crop_image_1.shape, np.uint8)
                drawing_2 = np.zeros(crop_image_2.shape, np.uint8)
                drawing_3 = np.zeros(crop_image_3.shape, np.uint8)
                cv2.drawContours(drawing_1, [contour_1], -1, (0, 255, 0), 0)
                cv2.drawContours(drawing_2, [contour_2], -1, (0, 255, 0), 0)
                cv2.drawContours(drawing_3, [contour_3], -1, (0, 255, 0), 0)
                cv2.drawContours(drawing_1, [hull_1], -1, (0, 0, 255), 0)
                cv2.drawContours(drawing_2, [hull_2], -1, (0, 0, 255), 0)
                cv2.drawContours(drawing_2, [hull_3], -1, (0, 0, 255), 0)

                # Find convexity defects
                hull_1 = cv2.convexHull(contour_1, returnPoints=False)
                hull_2 = cv2.convexHull(contour_2, returnPoints=False)
                hull_3 = cv2.convexHull(contour_3, returnPoints=False)
                defects_1 = cv2.convexityDefects(contour_1, hull_1)
                defects_2 = cv2.convexityDefects(contour_2, hull_2)
                defects_3 = cv2.convexityDefects(contour_3, hull_3)
                defects = [defects_1, defects_2, defects_3]

                # Use cosine rule to find the angle of the far point from the start
                # and end point
                count_defects = 0

                for j in range(len(defects)):
                    for i in range(defects[j].shape[0]):
                        contour_temp = contour[j]
                        crop_image_temp = crop_image[j]
                        s, e, f, d = defects[j][i, 0]
                        start = tuple(contour_temp[s][0])
                        end = tuple(contour_temp[e][0])
                        far = tuple(contour_temp[f][0])

                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        cv2.line(crop_image_temp, (start[0], start[1]), (far[0], far[1]), [0, 50, 120], 4)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        cv2.line(crop_image_temp, (far[0], far[1]), (end[0], end[1]), [0, 50, 120], 4)

                        cv2.circle(crop_image_temp, far, 4, [255, 255, 255], -1)

                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / math.pi

                        if angle <= 90:
                            count_defects += 1
                            cv2.circle(crop_image_temp, far, 4, [0, 0, 255], -1)


                        cv2.line(crop_image_temp, start, end, [255, 255, 0], 2)

                current_time = time.time() - self.initial_time
                first_three_sec = current_time > 3

                for j in range(len(defects)):

                    cond_delay = current_time > self.gesture_delay[j]

                    # Print the number of fingers applying a delay between gesture
                    if cond_delay and first_three_sec:  # or first_three_sec
                        if count_defects == 0:
                            self.text = "ONE"
                            self.output[str(j)]["one-finger"].append(round(time.time() - self.initial_time, 2))
                            self.gesture_delay[j] = (time.time()-self.initial_time) + 5
                            self.gestures_counter[j] += 1
                        elif count_defects == 1:
                            self.text = "TWO"
                            self.output[str(j)]["two-fingers"].append(round(time.time() - self.initial_time, 2))
                            self.gesture_delay[j] = (time.time()-self.initial_time) + 5
                            self.gestures_counter[j] += 1
                        elif count_defects == 2:
                            self.text = "THREE"
                            self.output[str(j)]["three-fingers"].append(round(time.time() - self.initial_time, 2))
                            self.gesture_delay[j] = (time.time()-self.initial_time) + 5
                            self.gestures_counter[j] += 1

                    #cv2.putText(frame, str(self.gestures_counter[j]) + ": " + self.text, self.text_position[j], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            except:
                pass

            # Show required images
            cv2.imshow("Gesture", frame)
            out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('p'):
                cv2.waitKey(-1)

        cap.release()
        cv2.destroyAllWindows()

        self.return_values()

    def return_values(self):
        """Return a dictionary with the time when the gesture was done
        and the duration of the video in seconds: (dic, duration)"""
        #for i in range(len(self.output)): # 57 / 24
        #    for j in range(len(self.output[str(i)])):
        #        self.output[str(i)][j] = map(operator.mul, self.output[str(i)][j], repeat((57 / 24)))
        output_json = json.dumps(self.output)
        print(output_json)
        return output_json
