import cv2
import numpy as np
import os
import sys
import time
import argparse
from math import exp, pow
from FordFulkerson import FordFulkerson
from PushRelabel import PushRelabel
import networkx as nx
import copy
from compute_score import compute_score

Algorithms = {"Push-Relabel": PushRelabel, "Ford-Fulkerson": FordFulkerson}
source, sink = -2, -1
scaling_factor = 10
sigma = 30
object_color, background_color = (255, 225, 255), (0, 255, 50)
object_num, background_num = 1, 2
object_, background = "object", "background"


def parseArgs():
    """Parse arguments"""

    def algorithm(string):
        if string in Algorithms:
            return string
        raise argparse.ArgumentTypeError("algorithm should be one :", Algorithms.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument(
        "--imagesize", "-s", default=20, type=int, help="Defaults to 20x20"
    )
    parser.add_argument("--cutalgo", "-a", default="Push-Relabel", type=algorithm)
    return parser.parse_args()


def show_image(image):
    """Show image on which the segmentation will be performed

    Args:
        image (array): image on which the segmentation will be performed
    """
    windowname = "MaxFlow"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def connect_sink_source(image, imagefile):
    """Connect the sink and source to pixels from the image

    Args:
        image (array): image to be segmented
        imagefile (string): image file name
    """

    def draw_seeds(x, y, pixelType):
        if pixelType == object_:
            color, code = object_color, object_num
        else:
            color, code = background_color, background_num
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(
            initialisation,
            (x // scaling_factor, y // scaling_factor),
            radius // scaling_factor,
            code,
            thickness,
        )

    def mouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            draw_seeds(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            draw_seeds(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def show_connected_pixels(pixelType):
        print("connecting", pixelType, "pixels")
        global drawing
        drawing = False
        windowname = "connecting " + pixelType + "pixels"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, mouse, pixelType)
        while 1:
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    initialisation = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)

    radius = 3
    thickness = -1
    global drawing
    drawing = False
    # if not(os.path.isfile('tests/init_' + imagefile)):
    show_connected_pixels(object_)
    show_connected_pixels(background)
    # else:
    #    initialisation = cv2.imread('tests/init_' + imagefile, cv2.IMREAD_GRAYSCALE)
    return initialisation, image


def build_graph(image, imagefile):
    """Build the graph from the image

    Args:
        image (array):image from which the graph will be built
        imagefile (string): image file name

    Returns:
        nx graph, array: the created graph, the image with connections ti image and seed
    """
    nodes = image.size + 2
    graph_adj = np.zeros((nodes, nodes), dtype="int32")
    source_flow = pixel_similarity(graph_adj, image)
    initialisation, initialised_Image = connect_sink_source(image, imagefile)
    unary_similarity(graph_adj, initialisation, source_flow)

    dt = [("capacity", int)]
    A = np.array(graph_adj, dtype=dt)
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    flow = 0
    nx.set_edge_attributes(G, flow, "flow")
    return G, initialised_Image


def similarity(ip, iq):
    """Measure pixel similarity

    Args:
        ip (float): pixel intensity
        iq (float): pixel intensity

    Returns:
        int: similarity measure
    """
    s = 100 * exp(-pow(int(ip) - int(iq), 2) / (2 * pow(sigma, 2)))
    return s


def pixel_similarity(graph, image):
    """Set pixel cost

    Args:
        graph (array): graph in array form
        image (array): the image to be segmented
    Returns:
        int: the flow that will be used to link the source and sink
    """
    source_flow = -float("inf")
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            x = i * W + j
            if i + 1 < H:  # pixel below
                y = (i + 1) * W + j
                pixel_sim = similarity(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = pixel_sim
                source_flow = max(source_flow, pixel_sim)
            if j + 1 < W:  # pixel to the right
                y = i * W + j + 1
                pixel_sim = similarity(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = pixel_sim
                source_flow = max(source_flow, pixel_sim)
            if i - 1 > -1:
                y = (i - 1) * W + j  # pixel above
                pixel_sim = similarity(image[i][j], image[i - 1][j])
                graph[x][y] = graph[y][x] = pixel_sim
                source_flow = max(source_flow, pixel_sim)
            if j - 1 > -1:  # pixel to the left
                y = i * W + j - 1
                pixel_sim = similarity(image[i][j], image[i][j - 1])
                graph[x][y] = graph[y][x] = pixel_sim
                source_flow = max(source_flow, pixel_sim)
    return source_flow


def unary_similarity(graph, initialisation, source_flow):
    """Set unary cost

    Args:
        graph (array): graph in array form
        initialisation (array): the links to source and sink set manually
        source_flow (int): the flow that will be used to link the source and sink
    """
    H, W = initialisation.shape
    for i in range(H):
        for j in range(W):
            x = i * W + j
            if initialisation[i][j] == object_num:
                graph[source][x] = source_flow
            elif initialisation[i][j] == background_num:
                graph[x][sink] = source_flow


def displaySegmentation(components, image):
    """Display Generated Segmentation

    Args:
        components (set): all strongly connected componenets
        image (array):the image to be segmented

    Returns:
       array: image with segmentation applied
    """
    segmented = np.zeros_like(image)
    H, W = image.shape
    max_len = 0
    max_componenet = {}
    for component in components:
        if len(component) > max_len:
            max_len = len(component)
            max_componenet = component
    for pixel in max_componenet:
        if pixel != sink and pixel != source:
            segmented[pixel // H, pixel % H] = 255
    return segmented


def Segmentation(imagefile, size=(30, 30), algo="Ford-Fulkerson"):
    """Perform segmentation

    Args:
        imagefile (string): image file
        size (tuple, optional): downscaled image size . Defaults to (30, 30).
        algo (str, optional): min-cut algorithm to use. Defaults to "Ford-Fulkerson".
    """
    if not (os.path.isdir("tests")):
        os.mkdir("tests")
    subdir = os.path.join("tests", f"{algo}")
    if not (os.path.isdir(subdir)):
        os.mkdir(subdir)
    subsubdir = os.path.join(subdir, f"{size[0]}_{size[0]}")
    if not (os.path.isdir(subsubdir)):
        os.mkdir(subsubdir)

    image = cv2.imread("data/" + imagefile, cv2.IMREAD_GRAYSCALE)
    global scaling_factor
    scaling_factor = image.shape[0] // size[0]
    print("scaling_factor", scaling_factor)
    image = cv2.resize(image, size)
    print("image.shape", image.shape)
    graph, initialised_Image = build_graph(image, imagefile)
    print("number of nodes", len(graph))
    print("number of edges", len(graph.edges()))
    cv2.imwrite("tests/init_" + imagefile, initialised_Image)

    global source, sink
    source += len(graph)
    sink += len(graph)
    s = time.time()
    if algo == "Ford-Fulkerson":
        print("Ford-Fulkerson ... ")
        FF = FordFulkerson(graph)
        FF.min_cut(source, sink)
        capacities = nx.get_edge_attributes(FF.frozen_graph, "capacity")
        flows = nx.get_edge_attributes(FF.frozen_graph, "flow")
        cuts = []
        for edge in capacities.keys():
            if capacities[edge] == flows[edge]:
                cuts.append(edge)
                try:
                    FF.dummy_graph.remove_edge(edge[1], edge[0])
                except:
                    pass
        components = list(nx.strongly_connected_components(FF.dummy_graph))
        image = displaySegmentation(components, image)

    if algo == "Push-Relabel":
        print("Push Relabel ... ")
        PR = PushRelabel(graph, source, sink)
        PR.min_cut()
        capacities = nx.get_edge_attributes(PR.graph, "capacity")
        flows = nx.get_edge_attributes(PR.graph, "flow")
        cuts = []
        for edge in capacities.keys():
            if capacities[edge] == flows[edge]:
                cuts.append(edge)
                try:
                    PR.graph.remove_edge(edge[0], edge[1])
                except:
                    pass
                try:
                    PR.graph.remove_edge(edge[1], edge[0])
                except:
                    pass
        components = list(nx.strongly_connected_components(PR.graph))
        image = displaySegmentation(components, image)
    print(f"execution time: {time.time() - s:.4f}")
    image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)
    show_image(image)
    savename = os.path.join(subsubdir, "cut_" + imagefile)
    cv2.imwrite(savename, image)
    print("saved segmented image as", savename)
    compute_score(imagefile.split(".")[0], image, subsubdir)


if __name__ == "__main__":

    args = parseArgs()
    Segmentation(args.imagefile, (args.imagesize, args.imagesize), args.cutalgo)
