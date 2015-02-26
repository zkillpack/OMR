import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
import abc
import score


class RotationFinder:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_rotation_angle(self, imagescore):
        """Calculates the ideal angle of rotation."""
        return

    def rotate(self):
        """Returns the rotated image."""
        return


class HoughRotationFinder(RotationFinder):

    def __init__(self, imagescore):
        self.ImageScore = imagescore
        self.image = self.ImageScore.get_image()
        self.img_height, self.img_width = self.ImageScore.img_height, self.ImageScore.img_width
        self.lines = None
        get_hough_lines(self)

    def get_hough_lines(self):
        """
         Performs line detection at a changing threshold until a proper numer of
         lines is detected Lines must be image width / threshold long in order
         to register. Since the threshold is quite large, this should only
         detect horizontal lines. The number can be relatively inaccurate, since
         the theta parameter of each line is a "vote" for how rotated the image
         is.

         Very slow. Does not work for large rotations. Used as a last resort.
        """
        hough_threshold = 1.5
        self.lines = cv2.HoughLines(
            img, 1, np.pi / 720, int(self.img_width / hough_threshold))
        # THESE LIMITS SHOULD DEPEND ON IMAGE SIZE
        print "Performing Hough iteration:"
        hough_counter = 1
        print hough_counter
        while ((self.lines is None or self.lines.size < 10 or self.lines.size >= 550) and hough_counter < 15):
            print hough_counter
            hough_counter += 1
            hough_threshold += .1
            self.lines = cv2.HoughLines(
                img, 1, np.pi / 720, int(self.img_width / hough_threshold))

    def warp_by_hough_lines(self):
        """ Given the lines detected in an image, return a rotated copy so that
        as many lines as possible are horizontal.
        """
        average_theta = np.median(self.lines[0, :, 1])
        theta_offset = average_theta - (np.pi / 2)
        theta_offset_degrees = theta_offset * 360 / (2 * np.pi)
        rotation_matrix = cv2.getRotationMatrix2D(
            (self.img_width / 2, self.img_height / 2), theta_offset_degrees, 1)
        rotated_img = cv2.warpAffine(
            self.image, rotation_matrix, (self.img_width, self.img_height), flags=cv2.INTER_NEAREST)
        return rotated_img

    def display_hough_results(self):
        """Given an image and an array lines (resulting from a call to cv2.HoughLines),
        add the lines to the image for visualization purposes.
        """
        color_image = self.ImageScore.getO
        for rho, theta in self.lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + img_width * (-b))
            y1 = int(y0 + img_width * (a))
            x2 = int(x0 - img_width * (-b))
            y2 = int(y0 - img_width * (a))
            cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imShow("Hough transform lines", self.image)

    def rotate(self):
        return warp_by_hough_lines(get_hough_lines(self.image), self.image)


class EntropyRotationFinder(RotationFinder):

    def __init__(self, imagescore):
        self.ImageScore = imagescore
        self.image = self.ImageScore.get_image()
        self.img_height, self.img_width = self.ImageScore.img_height, self.ImageScore.img_width

    @staticmethod
    def get_entropy(img):
        """Takes an image and finds the Shannon entropy H
        of the normalized horizontal projection h.
        cf. Cui et al 2010:
        An adaptive staff line removal in music score images
        """
        # Get normalized projection
        horizontal_projection = score.ImageScore.get_horizontal_projection_array(
            img).astype(float)
        horizontal_projection_normalized = horizontal_projection / \
            sum(horizontal_projection)

        # Calculate entropy
        print horizontal_projection_normalized
        log_projection = np.log2(horizontal_projection_normalized)
        log_projection[np.where(log_projection < -10 ** 6)] = 0
        log_projection = -log_projection
        entropy = np.sum(
            np.multiply(horizontal_projection_normalized, log_projection))

        return entropy

    def get_rotation_angle(self):
        """Takes an image and finds the angle theta that 
        minimizes entropy i.e. the rotation angle that makes
        the most distinct peaks in the horizontal projection
        """
        return minimize_scalar(lambda x: self.get_entropy(self._rotate_by_angle(self.image, x)), bounds=(-5, 5)).x

    def _rotate_by_angle(self, img, theta):
        """
        Rotates img by angle theta with nearest-neighbor interpolation
        """
        rotation_matrix = cv2.getRotationMatrix2D(
            (self.img_width / 2, self.img_height / 2), theta, 1)
        return cv2.warpAffine(self.image, rotation_matrix, (self.img_width, self.img_height), flags=cv2.INTER_NEAREST)

    def rotate(self):
        rotation_matrix = cv2.getRotationMatrix2D(
            (self.img_width / 2, self.img_height / 2), self.get_rotation_angle(), 1)
        return cv2.warpAffine(self.image, rotation_matrix, (self.img_width, self.img_height), flags=cv2.INTER_NEAREST)
