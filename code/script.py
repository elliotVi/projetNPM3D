#
#
#      0===================0
#      |    Project    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#       Based on Detection, segmentation and classification of 3D urban objects using mathematical morphology and
#       supervised learning - Andrés Serna, Beatriz Marcotegui
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Antoine GUEDON, Elliot VINCENT - 31/03/2020
#
# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import matplotlib.pyplot as plt
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_zmin(points, pw=0.2):
    pxmin, pxmax = np.min(points[:, 0]), np.max(points[:, 0])
    pymin, pymax = np.min(points[:, 1]), np.max(points[:, 1])
    pzmin, pzmax = np.min(points[:, 2]), np.max(points[:, 2])
    xmax = pxmax - pxmin
    ymax = pymax - pymin
    points = points - np.array([pxmin, pymin, pzmin])
    h = int(xmax / pw) + 1
    w = int(ymax / pw) + 1
    max_elevation_img = np.zeros((h, w))
    min_elevation_img = np.ones((h, w)) * (pzmax - pzmin)
    accumulation_img = np.zeros((h, w))
    for p in points:
        i, j = int(p[0] / pw), int(p[1] / pw)
        max_elevation_img[i, j] = max(max_elevation_img[i, j], p[2])
        min_elevation_img[i, j] = min(min_elevation_img[i, j], p[2])
        accumulation_img[i, j] += 1
    height_diff_img = max_elevation_img - min_elevation_img
    return max_elevation_img, min_elevation_img, height_diff_img, accumulation_img


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/MiniLille1.ply'
    #file_path = '../data/MiniLille2.ply'
    #file_path = '../data/MiniParis1.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    N = len(points)

    res = compute_zmin(points)
    plt.imshow(res[2])
    plt.show()
    '''
    write_ply('../best_planes.ply',
              [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
    write_ply('../remaining_points_.ply',
              [points[remaining_inds], colors[remaining_inds]],
              ['x', 'y', 'z', 'red', 'green', 'blue'])

    print('Done')
    '''
