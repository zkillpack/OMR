import argparse
import cv2
import numpy as np
from score import *


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to score image")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])

	score = Score(image)

	cv2.waitKey(0)
