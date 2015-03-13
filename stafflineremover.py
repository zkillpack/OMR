import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
import abc
import score

class StaffLineRemover(object):
    """Once additional algorithms are in place, this will be complemented by a
    StaffLinePreprocessor which will populate the Staves object with information
    without removing staff lines. The information will then be fed to other algorithms
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def remove_staff_lines(self):
        """Removes staff lines from an ImageScore.
        Populates the Staves object of an ImageScore with information obtained in the process."""
        return


class ScanLineRemover(StaffLineRemover):

    def __init__(self, imagescore):
        self.ImageScore = imagescore
        self.staff_lines_removed = self.ImageScore.get_image()

    def remove_staff_lines(self):
        """
        Examines a vertical scanline in the horizontal projection to look for staff lines.

        The first half of this might be abstracted out if I end up doing multiple variations on a scanline approach,
        otherwise the statistics calculated with the scanline might be useless for other removal algorithms...

        """
        H_prime = self.ImageScore.horizontal_projection[:, int(3 * self.ImageScore.img_width / 5.0)]
       
        try:
            binarized_candidates = (H_prime == 0)
        except:
            # What exactly should we do if there's nothing there?
            # The RotationFinder shouldn't really even give us such an image
            print "No staff lines detected!"
            return None

        candidate_pixels = np.nonzero(binarized_candidates)[0]
        staves = []

        # Find staff line lengths and spacings
        line_boundaries = np.where(np.diff(binarized_candidates) == 1)[0] + 1
        num_staff_lines = len(line_boundaries) / 2
        segment_lengths = np.diff(line_boundaries)

        for i, length in enumerate(segment_lengths):
            if ((i + 1) % 10) == 0:
                self.ImageScore.staves.inter_staff_lengths.append(length)
            elif ((i + 1) % 2) == 0:
                self.ImageScore.staves.inter_line_lengths.append(length)
            else:
                self.ImageScore.staves.staff_line_lengths.append(length)
                for j in range(length):
                    if j == 0:
                        self.ImageScore.staves.line_starts.append(
                            line_boundaries[i])
                    elif j == (length - 1):
                        self.ImageScore.staves.line_ends.append(
                            line_boundaries[i] + j)
                    else:
                        self.ImageScore.staves.line_centers.append(
                            line_boundaries[i] + j)

        self.ImageScore.staves.average_inter_line_distance = np.average(
            self.ImageScore.staves.inter_line_lengths)
        self.ImageScore.staves.average_staff_line_thickness = np.average(
            self.ImageScore.staves.staff_line_lengths)
        self.ImageScore.staves.average_inter_staff_distance = np.average(
            self.ImageScore.staves.inter_staff_lengths)
        self.ImageScore.staves.number_of_staves = num_staff_lines

        print "\n%d staff lines detected" % num_staff_lines
        print "%d staves detected\n" % self.ImageScore.staves.number_of_staves

        print "Average interstaff distance: %.1f" % self.ImageScore.staves.average_inter_staff_distance
        print "Average interline distance:  %.1f" % self.ImageScore.staves.average_inter_line_distance
        print "Average line thickness:      %.1f" % self.ImageScore.staves.average_staff_line_thickness

        """And now, we delete the staff lines! 

        ~HORRIBLY inefficient and altogether mediocre placeholder algorithm~
        ~soon this will be replaced by the stable paths algorithm~

        For a given pixel, if its y coordinate is the top or bottom of a line,
        check n pixels above or below where n depends on how skewed the lines are
        (for now, n = 3). For example, if three pixels above a and below a line is clear,
        delete the whole swath (to remove uncertainty introduced by rotation 
        and detecting lines by a tiny slice). If it's not clear,
        then we're probably in the middle of an object, so remove nothing.
        If it's in a center row, delete if clear."""

        line_threshold = 5

        



        for row_to_remove in self.ImageScore.staves.line_starts:
            for col in range(self.ImageScore.img_width):
                if self.ImageScore.image[row_to_remove - line_threshold][col] == 255:
                    for offset in range(line_threshold + 1):
                        self.staff_lines_removed[
                            row_to_remove - offset][col] = 255

        for row_to_remove in self.ImageScore.staves.line_ends:
            for col in range(self.ImageScore.img_width):
                if self.ImageScore.image[row_to_remove + line_threshold][col] == 255:
                    for offset in range(line_threshold + 1):
                        self.staff_lines_removed[
                            row_to_remove + offset][col] = 255

        for row_to_remove in self.ImageScore.staves.line_centers:
            for col in range(self.ImageScore.img_width):
                if (self.ImageScore.image[row_to_remove + line_threshold][col] == 255
                        and self.ImageScore.image[row_to_remove - line_threshold][col] == 255):
                    self.staff_lines_removed[row_to_remove][col] = 255

        cv2.imshow("Staff lines removed", self.staff_lines_removed)
        return self.staff_lines_removed

