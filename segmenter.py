import abc
import cv2
import numpy as np
import score


class Segmenter:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_segments(self, image):
        return


class Segment:
    """ put stuff here at some point """
    def __init__(self, put_stuff_here_eventually):
        pass

class SIFTSegmenter(Segmenter):
    """ Finds SIFT features on a score without lines """
    def __init__(self, imagescore):
        self.ImageScore = imagescore

    def get_segments(self):
        sift = cv2.SIFT()
        keypoints = sift.detect(self.ImageScore.removed_vertical_segments, None)
        sift_image = cv2.drawKeypoints(np.copy(self.ImageScore.removed_vertical_segments),keypoints)
        cv2.imshow("SIFT", sift_image)

        return keypoints

class VerticalSegmenter(Segmenter):
    """ Finds vertical segments """

    def __init__(self, imagescore):
        self.ImageScore = imagescore

    def get_segments(self):
        """Takes an image with the staff lines removed
        (eventually an object from which the staff_lines_removed will be used)
        and identifies segments in the remaining foreground objects
        """
        structuring_element_height = int(
            1.75 * self.ImageScore.staves.average_inter_line_distance)
        structuring_element_width = 1
        kernel_size = structuring_element_height + \
            ((structuring_element_height - 1) % 2)
        kernel = np.ones(
            (structuring_element_height, structuring_element_width), np.uint8)

        # Calculate opening with a structuring element of 1.5x staff spacing
        # This first gets the erosion (set of locations you can place this vertical line
        # at while having them be completely contained within the image)
        # This turns the image into very thin vertical lines
        # Next, get the dilation of this erosion by the same structuring element
        # (this is the set of locations you can place it at with any overlap at all,
        # which has the effect of plumping things up again, so to speak)

        opening = cv2.bitwise_not(
            cv2.morphologyEx(self.ImageScore.image_inv, cv2.MORPH_OPEN, kernel))
        cv2.imshow("opening", opening)

        staff_start = 344
        staff_end = 415
        staff_slice = self.ImageScore.image[staff_start:staff_end, :]

        cv2.imshow("example slice", staff_slice)
        score.ImageScore.get_vertical_projection(staff_slice, show=True)

        # Run a scanline at 1.5x staff width to detect Vertical Mass?
        # Sadly, not robust! Beams and dynamics and lyrics can add (m)Assloads of Mass
        scanline = staff_slice[
            structuring_element_height:structuring_element_height + 1, :]
        cv2.imshow("Scanline slice", scanline)

        # This should really return segments...but those don't exist yet.
        return opening
