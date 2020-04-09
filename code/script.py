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

# Import the hand-made partitioning class
from class_partitioning import Partition


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def elevation_images(points, pw=0.2):
    pxmin, pxmax = np.min(points[:,0]), np.max(points[:,0])
    pymin, pymax = np.min(points[:,1]), np.max(points[:,1])
    pzmin, pzmax = np.min(points[:,2]), np.max(points[:,2])
    points -= np.array([pxmin, pymin, pzmin])

    h, w = int( (pxmax-pxmin)/pw ) + 1, int( (pymax-pymin)/pw ) + 1
    max_elevation, min_elevation = np.zeros((h,w)), (pzmax-pzmin)*np.ones((h,w))
    accumulation = np.zeros((h,w))

    for p in points:
        x, y = int(p[0]/pw), int(p[1]/pw)
        max_elevation[x,y] = max(p[2], max_elevation[x,y])
        min_elevation[x,y] = min(p[2], min_elevation[x,y])
        accumulation[x,y] += 1

    for i in range(len(min_elevation)):
        for j in range(len(min_elevation[0])):
            if min_elevation[i, j] == pzmax - pzmin:
                min_elevation[i, j] = 0

    height_difference = max_elevation - min_elevation
    return max_elevation, min_elevation, height_difference, accumulation


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

def compute_lambda_flat_zones(image, accumulation, lambd):
    partition = Partition(accumulation>0)
    neighbours_shift = [(1,0), (0,1), (-1,0), (0,-1)]
    nrow, ncol = image.shape[0], image.shape[1]

    for i in range(nrow):
        for j in range(ncol):
            if(partition.image[i,j]):
                for shift in neighbours_shift:
                    x, y = i + shift[0], j+shift[1]
                    if(x>=0 and y>=0 and x<nrow and y<ncol and partition.image[x,y] and np.abs(image[i,j]-image[x,y])<=lambd):
                        partition.merge_classes((i,j), (x,y))
    return partition.get_all_classes()

def segment_ground(image, accumulation):
    zones = compute_lambda_flat_zones(image, accumulation, 0.2)
    lengths = [len(zones[i]) for i in range(len(zones))]
    args = np.argsort(lengths)
    return zones[args[-1]]

def show_ground(image, accumulation):
    result0 = np.zeros(image.shape)
    ground0 = segment_ground(image, accumulation)
    for pixel in ground0:
        result0[pixel] = image[pixel]
    return result0


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

    t0 = time.time()
    max_elevation, min_elevation, height_difference, accumulation = elevation_images(points)
    t1 = time.time()
    print("Elevation images computed in ", t1-t0 ," seconds.")
    plt.imshow(max_elevation)
    plt.show()
    t0 = time.time()
    ground0 = show_ground(max_elevation, accumulation)
    t1 = time.time()
    print("Ground segmented in ", t1-t0, " seconds.")
    plt.imshow(ground0)
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
