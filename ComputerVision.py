#Images will be supplied in PNG format

import sys
from typing import Any
import cv2 as cv2
import numpy as np
from PIL import Image
import imagehash
import os
import glob
from pathlib import Path

#TODO: Store reference perceptual hashes instead of having to bring over reference images
#TODO: Write red circle detection

def main(argv):
    default_file = 'references/deathstar23.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Load image
    image = cv2.imread(cv2.samples.findFile(filename))

    # Check if image is successfully loaded
    if image is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')

    red_present = red_pixels_present(image)
    print(red_present)
    # perceptual_hash(filename)
    # averages = average_values(image)
    # stars = star_detect(image)
    # circles = circle_detect(image)


def perceptual_hash(filename: str) -> int:
    img = Image.open(filename)
    image_hash = imagehash.whash(img)

    current_directory = sys.argv[1]
    folder_path = os.path.join(current_directory, 'references')
    files = glob.glob(os.path.join(folder_path, '*'))
    min_similarity = 1000

    for file_path in files:
        with Image.open(file_path) as file:
            reference_hash = imagehash.whash(file)
            similarity = abs(image_hash - reference_hash)
            min_similarity = similarity if similarity < min_similarity else min_similarity

    return min_similarity


def circle_detect(image: cv2.Mat | np.ndarray[Any, np.dtype]) -> int:

    h, w, _ = image.shape

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


def red_pixels_present(image: cv2.Mat | np.ndarray[Any, np.dtype]) -> bool:
    h, w, _ = image.shape
    for col in range(w):
        for row in range(h):
            r, g, b = image[row][col]
            if r > 100:
                if (g+b)/r < 0.5:
                    return True
    return False


# Code taken from https://github.com/ChristophRahn/red-circle-detection/blob/master/red-circle-detection.py
def red_circle_detect(image: cv2.Mat | np.ndarray[Any, np.dtype]) -> set[list[int]]:
    image_copy = image.copy()
    
    # Convert image to BGR so we can convert to lab
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    image_bgr = cv2.medianBlur(image, 3)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Threshold the lab image to keep only red pixels
    # Possible yellow threshold: [20, 110, 170][255, 140, 215]
    # Possible blue threshold: [20, 115, 70][255, 145, 120]
    image_lab_red = cv2.inRange(image_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
    # Another blur to reduce noise
    image_lab_red = cv2.GaussianBlur(image_lab_red, (5, 5), 2, 2)
    circles = cv2.HoughCircles(image_lab_red, cv2.HOUGH_GRADIENT, 1, image_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=500)

    if circles is not None:
        circles = np.round(circles[0, :].astype("int"))
        cv2.circle(image_copy, center=(circles[0,0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)

    cv2.imshow('image', image_copy)
    if cv2.waitKey(1) & 0xFF == ('q'):
        return circles


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

    average = sum([average_values[i][j] for i in range(rows) for j in range(cols) if bool(set([0,3]) & set([i,j]))]) // ((rows * cols) - ((rows - 2) * (cols - 2)))

    return average


if __name__ == "__main__":
    main(sys.argv[1:])
