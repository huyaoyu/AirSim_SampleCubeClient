
# Remember to enable the computer vision mode in the settings.json file.

# AirSim packages.
import setup_path 
import airsim

# System pakcages.
import cv2
import numpy as np
import os
import pprint
import tempfile
import time

# Local module.
from depth_2_distance import (meshgrid_from_img, depth_2_distance)

def scale_float_array(array, maxA=50):
    '''array (NumPy): The float image.
    maxA (float): Upper limit of array for clipping.'''
    minA = array.min()
    assert(minA < maxA)
    return (array - minA) / ( maxA - minA )

pp = pprint.PrettyPrinter(indent=4)

# Pre parthe output directory.
# out_dir = './blocks_512'
# out_dir = './oldtown_512'
# out_dir = './acienttowns'
# out_dir = './blocks_512_nearest'
# out_dir = './blocks_1024_nearest'
# out_dir = './acient_town_1024_nearest'
# out_dir = './american_diner_1024_nearest'
# out_dir = './antiquity_3d_1024_nearest'
# out_dir = './castle_fortress_1024_nearest'
# out_dir = './construction_site_1024_nearest'
# out_dir = './country_house_1024_nearest'
# out_dir = './cyber_punk_downtown_1024_nearest'
# out_dir = './fantasy_1024_nearest'
# out_dir = './HQ_western_saloon_1024_nearest'
# out_dir = './industrial_hangar_1024_nearest'
# out_dir = './modular_neighborhood_1024_nearest'
# out_dir = './nordic_harbor_1024_nearest'
# out_dir = './old_brick_house_1024_nearest'
# out_dir = './old_scandinavia_1024_nearest'
# out_dir = './old_industrial_city_1024_nearest'
# out_dir = './polar_scifi_facility_1024_nearest'
# out_dir = './retro_office_1024_nearest'
# out_dir = './ruins_1024_nearest'
# out_dir = './sewerage_1024_nearest'
# out_dir = './shore_cave_1024_nearest'
# out_dir = './urban_construction_1024_nearest'
# out_dir = './victorian_street_1024_nearest'
# out_dir = './20211116_nearest'
# out_dir = './20220327_nearest'
# out_dir = './20220703_nearest'
# out_dir = './20220703_nearest_oldtown'
out_dir = './20220717_oldtown'

if ( not os.path.isdir(out_dir) ):
    os.makedirs(out_dir)

# The AirSim client.
client = airsim.VehicleClient()

# The request list.
# request = [
#     airsim.ImageRequest('2', airsim.ImageType.Scene, pixels_as_float=False, compress=True),
#     airsim.ImageRequest('2', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
#     airsim.ImageRequest('2', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),
#     airsim.ImageRequest('1', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
#     airsim.ImageRequest('1', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False)]

request = [
    airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    airsim.ImageRequest('0', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    airsim.ImageRequest('0', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False) ]

# Capture two sets of images from the left (2) and right (1) cameras.
for x in range(2):
    print('Before request. x = %d' % (x))
    time_start = time.time()
    responses = client.simGetImages(request)
    time_end = time.time()
    print('After request. x = %d, time span = %f' % (x, time_end - time_start))

    # Write the response to the filesystem.
    for i, response in enumerate(responses):
        if response.pixels_as_float:
            print( "Type %d, size %d, pos \n%s" % \
                ( response.image_type, 
                  len(response.image_data_float), 
                  pprint.pformat(response.camera_position) ) )
            
            # Get the raw floating-point data.
            pfm_array = airsim.get_pfm_array(response).reshape((response.height, response.width, 1))

            # Create the meshgrid.
            xx, yy = meshgrid_from_img(pfm_array)

            # Convert the depth values to distance.
            dist_array = np.zeros_like(pfm_array)
            if ( not depth_2_distance(pfm_array, xx, yy, dist_array) ):
                raise Exception('Failed to conver the depth to distance. ')

            # Save the floating-point data as compressed PNG file.
            img_array = dist_array.view('<u1')
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%03d_%02d_depth.png' % (x, i))), img_array)

            # Visualize the depth.
            scaled = scale_float_array(dist_array, maxA=50)
            scaled_grey = (np.clip( scaled, 0, 1 ) * 255).astype(np.uint8)
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%03d_%02d_blackwhite.png' % (x, i))), scaled_grey)
        else:
            print("Type %d, size %d, pos \n%s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            
            # Decode the image directly from the bytes.
            decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)

            # Write the image as a PNG file.
            # The incoming data has 4 channels. The alpha channel is all zero.
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%03d_%02d.png' % (x, i))), decoded[:, :, :3])
