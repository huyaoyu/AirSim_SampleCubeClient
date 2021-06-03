
import cv2
import math
import numpy as np
import numba
from numba import cuda
import os
import time

def get_filename_parts(fn):
    '''Return the dirname, stem, and the extension of a filename. '''
    s0 = os.path.split(fn)
    if ( s0[0] == '' ):
        s0[0] = '.'

    s1 = os.path.splitext(s0[1])

    return s0[0], s1[0], s1[1]

def read_compressed_float(fn):
    assert( os.path.isfile(fn) ), \
        '%s does not exist. ' % (fn)

    return np.squeeze( 
        cv2.imread(fn, cv2.IMREAD_UNCHANGED).view('<f4'), 
        axis=-1 )

def float_as_grey(img, limits):
    img = (img - limits[0]) / (limits[1] - limits[0])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def write_float_as_grey(fn, img, limits):
    cv2.imwrite(fn, float_as_grey(img, limits))

def meshgrid_from_img(img):
    '''Create coordinate meshgrid. '''
    # Get the shape of the image.
    H, W = img.shape[:2]

    # Coordinate ranges.
    x = np.arange(W, dtype=np.int32)
    y = np.arange(H, dtype=np.int32)

    # Meshgrid.
    xx, yy = np.meshgrid(x, y)

    return xx, yy

@numba.jit(nopython=True)
def plane_angles(lon, lat):
    '''Compute the angles with respect to the yz and yx planes. '''
    a_yz = np.arctan2( np.sin(lat) * np.abs(np.cos(lon)), np.abs(np.cos(lat)) )
    a_yx = np.arctan2( np.sin(lat) * np.abs(np.sin(lon)), np.abs(np.cos(lat)) )
    return a_yz, a_yx

@numba.jit(nopython=True)
def depth_2_distance(pn_depth, xx, yy, pn_dist):
    '''pn_depth (Array): The panorama depth.
    xx (Array): The x-coordinates of the meshgrid. 
    yy (Array): The y-coordinates of the meshgrid.
    pn_dist (Array): The output distance array.'''

    # Get the shape of the input panorama depth image.
    H, W = pn_depth.shape[:2]

    # Map the meshgrid to radian angle.
    # Arrays are automatically converted into float type.
    xx = xx / (W-1) * (2 * np.pi) # Longitute.
    yy = yy / (H-1) * np.pi # Latitute.

    # Some constants.
    one_fourth_pi    = np.pi / 4
    three_fourths_pi = 3 * one_fourth_pi
    five_fourths_pi  = 5 * one_fourth_pi
    seven_fourths_pi = 7 * one_fourth_pi
    two_pi           = 2 * np.pi

    # Gaint loop.
    res = True
    for i in range(H):
        for j in range(W):
            lon = xx[i, j]
            lat = yy[i, j]

            # Plane angles.
            a_yz, a_yx = plane_angles(lon, lat)

            if ( a_yz <= one_fourth_pi and a_yx <= one_fourth_pi ):
                # Top or bottom face.
                pn_dist[i, j] = pn_depth[i, j] / np.abs(np.cos(lat))
                continue

            # The other four faces.
            if ( 0 <= lon < one_fourth_pi or seven_fourths_pi <= lon <= two_pi ):
                pn_dist[i, j] = pn_depth[i, j] / np.sin(lat) / np.cos(lon)
            elif ( one_fourth_pi <= lon < three_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / np.sin(lat) / np.sin(lon)
            elif ( three_fourths_pi <= lon < five_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / np.sin(lat) / -np.cos(lon)
            elif ( five_fourths_pi <= lon < seven_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / np.sin(lat) / -np.sin(lon)
            else:
                print('Should never happen. [', i, ', ', j, ']: lon = ', lon, ', lat = ', lat)
                res = False
                break
        if ( not res ):
            break
    
    return res

@cuda.jit()
def k_depth_2_distance(pn_depth, pn_dist, res):
    '''pn_depth (device array): The input depth array. 
    pn_dist (device array): The output distance array.
    res (Array): The output flag. Zero for success. '''
    # Cuda indices and strides.
    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    stride_x = cuda.blockDim.x * cuda.gridDim.x
    stride_y = cuda.blockDim.y * cuda.gridDim.y

    # Constants.
    one_fourth_pi    = np.pi / 4
    three_fourths_pi = 3 * one_fourth_pi
    five_fourths_pi  = 5 * one_fourth_pi
    seven_fourths_pi = 7 * one_fourth_pi
    two_pi           = 2 * np.pi

    H, W = pn_depth.shape[:2]

    # Loop.
    flag = True
    for i in range( idx_y, pn_depth.shape[0], stride_y ):
        lat = np.pi * i / (H-1)
        for j in range( idx_x, pn_depth.shape[1], stride_x ):
            lon = two_pi * j / (W-1)

            # Plane angles.
            a_yz = math.atan2( math.sin(lat) * math.fabs(math.cos(lon)), math.fabs(math.cos(lat)) )
            a_yx = math.atan2( math.sin(lat) * math.fabs(math.sin(lon)), math.fabs(math.cos(lat)) )

            if ( a_yz <= one_fourth_pi and a_yx <= one_fourth_pi ):
                # Top or bottom face.
                pn_dist[i, j] = pn_depth[i, j] / math.fabs(math.cos(lat))
                continue

            # The other four faces.
            if ( 0 <= lon < one_fourth_pi or seven_fourths_pi <= lon <= two_pi ):
                pn_dist[i, j] = pn_depth[i, j] / math.sin(lat) / math.cos(lon)
            elif ( one_fourth_pi <= lon < three_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / math.sin(lat) / math.sin(lon)
            elif ( three_fourths_pi <= lon < five_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / math.sin(lat) / -math.cos(lon)
            elif ( five_fourths_pi <= lon < seven_fourths_pi ):
                pn_dist[i, j] = pn_depth[i, j] / math.sin(lat) / -math.sin(lon)
            else:
                res[0] += 1
                flag = False
                break
        if ( not flag ):
            break

    cuda.syncthreads()

def cuda_depth_2_distance(pn_depth, pn_dist):
    '''pn_depth (Array): The input depth image.
    pn_dist (Array): The output distance image. '''
    # The return value for the kernel function.
    res = np.array([0,], dtype=np.int32)
    
    # Transfer data to the GPU.
    d_pn_depth = cuda.to_device(pn_depth)
    d_pn_dist  = cuda.to_device(pn_dist)

    # Call the CUDA kernel.
    cuda.synchronize()
    k_depth_2_distance[[16, 16, 1], [16, 16, 1]](d_pn_depth, d_pn_dist, res)
    cuda.synchronize()

    # Check the return value.
    if ( res[0] != 0 ):
        print('k_depth_2_distance() failed')
        return False
    else:
        d_pn_dist.copy_to_host(pn_dist)
        return True

class Timing(object):
    def __init__(self, name):
        super(Timing, self).__init__()
        self.name  = name
        self.start = None
        self.end   = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time.time()
        print('%s in %fs. ' % (self.name, self.end - self.start))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert a panorama depth image into distance image. ')
    parser.add_argument('indepth', type=str, help='The input depth image. ')
    args = parser.parse_args()

    # Read the depth image.
    pn_depth = read_compressed_float(args.indepth)

    # Check the dimension.
    assert(2*pn_depth.shape[0] == pn_depth.shape[1]), \
        'Wrong panorama shape: {}'.format(pn_depth.shape)

    # Create the meshgrid.
    xx, yy = meshgrid_from_img(pn_depth)

    # Convert.
    pn_dist = np.zeros_like(pn_depth)
    with Timing('depth_2_distance'):
        depth_2_distance(pn_depth, xx, yy, pn_dist)

    # Save the distance image as grey visualization image.
    parts = get_filename_parts(args.indepth)
    out_fn = '%s_dist.png' % (parts[1])
    write_float_as_grey(out_fn, pn_dist, [0, 75])

    # Convert by CUDA.
    pn_dist = np.zeros_like(pn_depth)
    with Timing('cuda_depth_2_distance'):
        cuda_depth_2_distance(pn_depth, pn_dist)
    
    # Save the distance image as grey visualization image.
    out_fn = '%s_dist_cuda.png' % (parts[1])
    write_float_as_grey(out_fn, pn_dist, [0, 75])