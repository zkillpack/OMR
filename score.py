import cv2
import numpy as np
from rotationfinder import EntropyRotationFinder
from stafflineremover import ScanLineRemover
from segmenter import VerticalSegmenter, SIFTSegmenter


class Score:

    """Represents a piece of music, comprising (possibly) multiple images
    and a symbolic representation of the music.
    """

    def __init__(self, image):

        self.image = ImageScore(image)
        self.music = MusicScore()


class ImageScore:

    def __init__(self, image):
        """ Tentative pipeline:
        Preprocess image (binarize, rotate [eventually,
                apply homography to transform camera photos], get projections)

        Identify and remove staff lines
        Object segmentation, first verticals, then musical objects(template matching/SVM)
        Pitch/rhythm detection
        Verification w/ semantic model of music
        Output to MusicScore object(which will have midi? musicxml?)
        """
        self.original_image = np.copy(image)
        try:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print "Image format not recognized."
            exit()
        self.img_height, self.img_width = self.image.shape
        self.image = EntropyRotationFinder(self).rotate()
        self.image = self.binarize(self.image)
        self.image_inv = self.binarize(self.image, inverted=True)

        self.music = MusicScore()

        self.horizontal_projection = self.get_horizontal_projection(self.image)
        self.vertical_projection = self.get_vertical_projection(self.image)
        self.horizontal_projection_array = self.get_horizontal_projection_array(
            self.image)

        self.staves = Staves()
        self.staff_lines_removed = ScanLineRemover(self).remove_staff_lines()

        self.vertical_segments = VerticalSegmenter(self).get_segments()
        self.removed_vertical_segments = cv2.bitwise_not(cv2.bitwise_xor(self.staff_lines_removed, self.vertical_segments))
        # Demo the removed objects cause why not
        cv2.imshow("removed verticals", self.removed_vertical_segments)

        self.sift_features = SIFTSegmenter(self).get_segments()

    def get_image(self):
        """Returns a defensive copy of the image for destructive processing elsewhere."""
        return np.copy(self.image)

    def get_original_image(self):
        """Returns a defensive copy of the original (i.e. unrotated color) image for visualizing various things."""
        return np.copy(self.original_image)

    @staticmethod
    def binarize(img, inverted=False):
        """Use Otsu's global thresholding to binarize the intensity values after
        first converting to grayscale.
        Otsu's method assumes a bimodal image histogram (i.e. foreground/background)
        and minimizes within-class variance when thresholding. Blurring is applplied
        beforehand to remove outliers introduced by scan noise.
        """
        # Blur and threshold
        img = cv2.GaussianBlur(img, (5, 5), 0)
        if inverted:
            _, thresh = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return np.copy(thresh)

    @staticmethod
    def get_horizontal_projection(img, show=False):
        """Projects a 2D image onto a 1D array i.e. just the vertical axis
        and then converts it back to a histogram image
        """
        # Move down rows, collapsing each row into a count of white pixels
        histogram = np.sum(img, axis=1) / 255
        # Convert count to black pixels
        histogram = np.max(histogram) - histogram
        histogram_image = np.ones_like(img) * 255

        for i, row in enumerate(histogram):
            for j in range(row):
                histogram_image[i][j] = 0
        if show:
            cv2.imshow("horizontal histogram", histogram_image)
       
        return histogram_image

    @staticmethod
    def get_horizontal_projection_array(img):
        """
        Projects a 2D image onto a 1D array i.e. just the vertical axis
        """
        # Move down rows, collapsing each row into a count of white pixels
        histogram = np.sum(img, axis=1) / 255
        # Convert count to black pixels
        histogram = np.max(histogram) - histogram
        
        return histogram

    @staticmethod
    def get_vertical_projection(img, show=False):
        # Move down columns, collapsing each column into a count of white
        # pixels
        histogram = np.sum(img, axis=0) / 255
        # Convert count to black pixels
        histogram = np.max(histogram) - histogram
        histogram_image = np.ones_like(img) * 255

        for i, col in enumerate(histogram):
            for j in range(col):
                histogram_image[j][i] = 0

        if show:
            cv2.imshow("vertical histogram", histogram_image)

        return histogram_image


class Staves:

    """ Holds information about the staves contained in a ScoreImage,
    including various statistics used for later calculations."""

    def __init__(self):
        self.number_of_staves = 0

        self.staff_line_lengths = []
        self.inter_line_lengths = []
        self.inter_staff_lengths = []

        self.line_starts = []
        self.line_centers = []
        self.line_ends = []

        self.average_inter_line_distance = 0
        self.average_staff_line_thickness = 0
        self.average_inter_staff_distance = 0

        self.staves = []


class Staff:

    """A group of five StaffLines that belongs to a Staves"""

    def __init__(self, PUT_THINGS_HERE):
        self.index = 0
        self.top = 0
        self.bottom = 0
        self.distance_to_next = None
        self.stafflines = []


class StaffLine:

    """Not sure how to implement...depends on what I discover while 
    doing future staff line removal algorithms.
    Just storing top/bottom/width only works for a scanline approach 
    where you assume it's of uniform thickness,
    which, while robust rotation detection helps (and I've got that!), it's 
    really not something you can assume...I'll figure it out when I do stable paths!"""

    def __init__(self, PUT_THINGS_HERE):
        self.index = 0
        self.top = 0
        self.bottom = 0
        self.distance_to_next = None


class MusicScore:

    """Will hold MIDI or MusicXML (or MEI?)...eventually..."""
    pass
