#Images will be supplied in PNG format

import sys
from typing import Any
import cv2 as cv2
import numpy as np


def main(argv):
    default_file = 'images/death-star2.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Load image
    image = cv2.imread(cv2.samples.findFile(filename))

    # Check if image is successfully loaded
    if image is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')

    averages = average_values(image)
    # stars = star_detect(image)
    circles = circle_detect(image)

    print("Average value on edges: " + str(averages))
    print("Number of circles: " + str(circles))


def star_detect(image: cv2.Mat | np.ndarray[Any, np.dtype]):
    ...


def circle_detect(image: cv2.Mat | np.ndarray[Any, np.dtype]) -> int:

    h, w, _ = image.shape
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    # idx = 0
    # while idx >= 0:
    #     center, radius = cv2.minEnclosingCircle(contours[idx])
    #     center = (int(center[0]), int(center[1]))
    #
    #     cv2.circle(image, center, int(radius), (0, 255, 255), 2)
    #
    #     idx = hierarchy[0][idx][0]
    #
    #     cv2.imshow("image with circles", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    params = cv2.SimpleBlobDetector.Params()
    params.minCircularity = 0.9
    params.minArea = 3.14 * (w/6) ** 2
    params.maxArea = 3.14 * w ** 2

    blob_detector = cv2.SimpleBlobDetector.create(params)
    blob_detector.setParams(params)

    keypoints = blob_detector.detect(image)

    number_of_blobs = len(keypoints)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return number_of_blobs


def average_values(image: cv2.Mat | np.ndarray[Any, np.dtype]) -> int:
    h, w, _ = image.shape
    # Define the number of rows and columns
    rows, cols = 4, 4

    # Computer height and width of each rectangle
    rect_height = h // rows
    rect_width = w // cols

    average_colors = []

    for row in range(rows):
        for col in range(cols):
            start_x = col * rect_width
            start_y = row * rect_height
            end_x = start_x + rect_width
            end_y = start_y + rect_height

            rect = image[start_y:end_y, start_x:end_x]

            average_color = np.mean(rect, axis=(0, 1))

            average_colors.append(average_color)

    average_values = np.array([int(sum(i) / len(i)) for i in average_colors])
    average_values = average_values.reshape(rows, cols)
    # print(average_values)

    average = sum([average_values[i][j] for i in range(rows) for j in range(cols) if bool(set([0,3]) & set([i,j]))]) // ((rows * cols) - ((rows - 2) * (cols - 2)))

    # print(average)

    return average


if __name__ == "__main__":
    main(sys.argv[1:])
