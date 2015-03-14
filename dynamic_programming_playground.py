import numpy as np
import itertools
from skimage import viewer
from functools import partial
from profilehooks import profile

# These are a result of the preprocessing from the main application
# and are specific to the example image

interstaff_distance = 122
interline_distance = 17.4
line_thickness = 3.0

# Available at https://github.com/dakotalk/OMR/blob/master/test_image_array.npy
image = np.load("test_image_array.npy")

# The zone to search for staff lines is predetermined in the main application
# This is constant for the purposes of this experimental file

image = image[75:205, 35:2445]

# OpenCV's image format uses reversed x/y
img_height, img_width = image.shape


def graph_weight(pixel_1, pixel_2, vertical_map, black_pixels, vertical_threshold=4):
    """
    Helper function for building graph.
    Calculates weight for an edge between pixel 1 and pixel 2
    in our image, given the following rules:
    Prefer black pixels.
    If a given black pixel has a high number of black pixels above and below it,
    it's likely a non-staffline musical object, so increase cost.
    Left-to-right edges have lower cost than right-to-left.
    Straight edges have lower cost than diagonal.
    """
    weight = 2

    # Favor left-right edge over anything else
    if pixel_1[1] >= pixel_2[1]:
        weight += 4

    # Additional cost for diagonal
    if pixel_1[0] != pixel_2[0]:
        weight += 2

    # Additional cost for white
    if pixel_1 not in black_pixels:
        weight += 1

    if pixel_2 not in black_pixels:
        weight += 1

    if vertical_map[pixel_1] >= vertical_threshold:
        weight += 2

    if vertical_map[pixel_2] >= vertical_threshold:
        weight += 2

    return weight


def dynamic_weight(black_pixels, pixel):
    """
    Calculates importance for a given pixel for use in dynamic programming.
    Assumes only left-to-right motion.
    Energy is a combination of pixels that might lead into a given pixel
    as well as the current pixel's value.
    The vertical_map is not currently used: we may find paths of black pixels
    through vertical objects, only leaving them in when the time comes to delete them.
    """
    weight = 10

    if pixel in black_pixels:
        weight -= 3

    # Looks at the three pixels neighboring to the left, assuming L-R paths.
    # We might want to experiment with making diagonals more likely to be removed,
    # since many lines have warts: -----^-----

    for candidate_pixel in itertools.product(
            range(pixel[0] - 1, pixel[0] + 2), [pixel[1] - 1]):
        if candidate_pixel in black_pixels:
            weight -= 1

    return weight


def get_vertical_map(black_pixels, image, line_threshold=5):
    """
    Identify how many pixels above and below each black pixel are also black,
    for use in weight calculation.
    If a large number of pixels above and below a black pixel are black,
    that pixel likely belongs to a larger object.
    """
    black_map = (image == 0)

    vertical_map = np.zeros((img_height, img_width), dtype=np.uint8)

    for y, x in black_pixels:
        try:
            vertical_map[y][
                x] += np.sum(black_map[y - line_threshold][y + line_threshold + 1])
        except IndexError:
            # We can keep the edge values zero.
            # Staff lines should not be on on the edges of the image.
            pass
    return vertical_map


def find_black_pixels(image):
    """
    Returns set of which pixels in the image are black.
    """
    black_y, black_x = (image == 0).nonzero()
    return {(y, x) for (y, x) in zip(black_y, black_x)}


def get_neighbors(origin_y, origin_x):
    """
    Returns a list of tuples containing coordinates of neighbors
    for a given y,x with the constraint that they stay within the image bounds.
    """
    return [
        (neighbor_y, neighbor_x)
        for (neighbor_y, neighbor_x)
        in itertools.product(
            range(
                origin_y - 1, origin_y + 2), range(origin_x - 1, origin_x + 2)
        )

        if (
            (neighbor_y, neighbor_x) != (origin_y, origin_x)
            and 0 <= neighbor_y < img_height
            and 0 <= neighbor_x < img_width
        )
    ]


def build_graph(image):
    """
    Represent image as a graph via adjacency list:
    map each pixel to a dict of adjacent pixels with edge weights.
    """

    black_pixels = find_black_pixels(image)
    vertical_map = get_vertical_map(black_pixels, image)

    def get_weighted_neighbors(origin_y, origin_x):

        return {
            (neighbor_y, neighbor_x):
            graph_weight((origin_y, origin_x), (neighbor_y, neighbor_x),
                         vertical_map, black_pixels)
            for (neighbor_y, neighbor_x) in get_neighbors(origin_y, origin_x)
        }

    return {
        (y, x): get_weighted_neighbors(y, x)
        for (y, x)
        in itertools.product(range(img_height), range(img_width))
    }


@profile
def main():
    # Just a test that things are ready and working for dynamic programming

    # Visualize vertical map

    black_pixels = find_black_pixels(image)
    vertical_map = get_vertical_map(black_pixels, image)

    v = viewer.ImageViewer(vertical_map * 20)
    v.show()

    weight_function = partial(dynamic_weight, black_pixels)
    weights = np.zeros((img_height, img_width), dtype=np.uint8)

    # Visualize weights
    for (y, x) in itertools.product(range(img_height), range(img_width)):
        weights[y, x] = weight_function((y, x))

    v = viewer.ImageViewer(weights * 20)
    v.show()

if __name__ == '__main__':
    main()
