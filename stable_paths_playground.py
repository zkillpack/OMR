import numpy as np
import itertools
import pickle


# These are a result of the preprocessing from StaffLineRemover
# and are specific to the example image
interstaff_distance = 122
interline_distance = 17.4
line_thickness = 3.0

image = np.load("test_image_array.npy")
# The zone to search for staff lines is predetermined; this is fixed for testing
image = image[75:205, 35:2445]

# OpenCV's image format uses reversed x/y
img_height, img_width = image.shape


def weight(p1, p2, vertical_map, black_pixels):
    """
    Calculate weight for an edge between pixel 1 and pixel 2
    in our image, given the following rules:
    Strongly prefer black pixels
    If a given black pixel has a high number of black pixels above and below it,
    it's likely a non-staffline musical object: greatly increase cost
    Left-to-right edges have lower cost than right-to-left
    Straight edges have lower cost than diagonal
    """
    weight = 2

    # Favor left-right edge over anything else
    if p1[1] >= p2[1]:
        weight += 4

    # Additional cost for diagonal
    if p1[0] != p2[0]:
        weight += 2

    # Additional cost for white
    if p1 not in black_pixels:
        weight += 1

    if p2 not in black_pixels:
        weight += 1

    vertical_threshold = 4

    if vertical_map[p1] >= vertical_threshold:
        weight += 2

    if vertical_map[p2] >= vertical_threshold:
        weight += 2

    return weight


def get_vertical_map(black_pixels, image):
    """
    Identify how many pixels above and below each black pixel are also black,
    for use in weight calculation.
    """
    line_threshold = 5

    black_map = (image == 0)

    vertical_map = np.zeros((img_height, img_width), dtype=np.uint8)

    for y, x in black_pixels:
        try:
            vertical_map[y][x] += np.sum(black_map[y - line_threshold][y + line_threshold + 1])
        except IndexError:
            # We can keep the edge values zero.
            # Staff lines should not be on on the edges of the image.
            pass
    return vertical_map


def find_black_pixels(image):
    black_y, black_x = (image == 0).nonzero()
    return {(y, x) for (y, x) in zip(black_y, black_x)}


def build_graph(image):
    """
    Represent graph as adjacency list: map each pixel
    to a dict of adjacent pixels with edge weights
     """

    black_pixels = find_black_pixels(image)
    vertical_map = get_vertical_map(black_pixels, image)

    def get_weighted_neighbors(origin_y, origin_x):

        return {
            (neighbor_y, neighbor_x):
            weight((origin_y, origin_x), (neighbor_y, neighbor_x),
                   vertical_map, black_pixels)
            for (neighbor_y, neighbor_x)
            in itertools.product(
                range(origin_y - 1, origin_y + 2), range(origin_x - 1, origin_x + 2))
            if (
                (neighbor_y, neighbor_x) != (origin_y, origin_x)
                and 0 < neighbor_y < img_height
                and 0 < neighbor_x < img_width
            )
        }
    return {
            (y, x): get_weighted_neighbors(y, x)
            for (y, x) 
            in itertools.product(range(img_height), range(img_width))
            }

graph = build_graph(image)
pickle.dump(graph, open("graph.pkl", "wb"))
