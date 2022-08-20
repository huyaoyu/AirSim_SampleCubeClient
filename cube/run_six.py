
# Remember to enable the computer vision mode in the settings.json file.

# AirSim packages.
import setup_path 
import airsim
from airsim.types import Pose, Vector3r, Quaternionr

# System pakcages.
import cv2
import numpy as np
import os
import pprint
from pyquaternion import Quaternion
import time

# Local module.
from depth_2_distance import (meshgrid_from_img, depth_2_distance_planar)

def scale_float_array(array, maxA=50):
    '''array (NumPy): The float image.
    maxA (float): Upper limit of array for clipping.'''
    minA = array.min()
    if minA < maxA:
        return (array - minA) / ( maxA - minA )
    else:
        print(f'scale_float_array: minA = {minA}, maxA = {maxA}')
        return np.ones_like(array)

def get_six_posese(position):
    '''
    position (array): xyz coordinates.
    '''
    
    pos_vec = Vector3r( position[0], position[1], position[2] )
    
    posese = []
    pose_names = []
    
    # Front.
    q = Quaternion(axis=(0.0, 0.0, 1.0), degrees=0.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('front')

    # Right.
    q = Quaternion(axis=(0.0, 0.0, 1.0), degrees=90.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('right')
    
    # Back.
    q = Quaternion(axis=(0.0, 0.0, 1.0), degrees=180.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('back')
    
    # Left.
    q = Quaternion(axis=(0.0, 0.0, 1.0), degrees=-90.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('left')
    
    # Top.
    q = Quaternion(axis=(0.0, 1.0, 0.0), degrees=90.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('top')
    
    # Bottom.
    q = Quaternion(axis=(0.0, 1.0, 0.0), degrees=-90.0)
    airsim_q = Quaternionr(q[1], q[2], q[3], q[0])
    posese.append( Pose(pos_vec, airsim_q) )
    pose_names.append('bottom')
    
    return posese, pose_names

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    # Pre parthe output directory.
    # out_dir = './six'
    # out_dir = './six_640'
    # out_dir = './six_768'
    # out_dir = './six_896'
    # out_dir = './six_1024'
    # out_dir = './six_distance_896'
    out_dir = './six_distance_896_20220703_oldtown'

    if ( not os.path.isdir(out_dir) ):
        os.makedirs(out_dir)

    # The AirSim client.
    client = airsim.VehicleClient()

    # The request list.
    request = [
        airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest('0', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True)]

    # Capture one set of images for warm up.
    print('Warm up... ')
    client.simGetImages(request)
    print('Warm up done. ')

    # New pinhole only request.
    request = [
        airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest('0', airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False) ]

    # Six posese.
    ori_pos = np.zeros((3), dtype=float)
    # ori_pos = np.array([80, 37, -2.5], dtype=float)
    airsim_posese, pose_names = get_six_posese(ori_pos)

    # Capture six images.
    time_start = time.time()
    for p, name in zip( airsim_posese, pose_names ):
        print(f'Pose {name}')
        
        # Set the pose.
        client.simSetVehiclePose(p, ignore_collision=True)
        time.sleep(0.02)
        
        # Pause.
        client.simPause(True)
        
        # Get the images.
        responses = client.simGetImages(request)
        # responses = client.simGetImages(request)
        client.simPause(False)
        
        # Write the response to the filesystem.
        for _, response in enumerate(responses):
            if response.pixels_as_float:
                print( "Type %d, size %d, pos \n%s" % \
                    ( response.image_type, 
                    len(response.image_data_float), 
                    pprint.pformat(response.camera_position) ) )
                
                # Get the raw floating-point data.
                pfm_array = airsim.get_pfm_array(response).reshape((response.height, response.width, 1))

                # Create the meshgrid.
                xx, yy = meshgrid_from_img(pfm_array)

                # Figure out the focal length and principle point.
                f  = ( pfm_array.shape[1] - 1 ) / 2
                cx = ( pfm_array.shape[1] - 1 ) / 2
                cy = ( pfm_array.shape[0] - 1 ) / 2
                
                # Convert the depth values to distance.
                dist_array = depth_2_distance_planar(np.squeeze( pfm_array), f, (cx, cy), xx, yy)

                # Save the floating-point data as compressed PNG file.
                img_array = np.expand_dims(dist_array, axis=-1).view('<u1')
                cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s_distance.png' % (name))), img_array)

                # Visualize the depth.
                scaled = scale_float_array(dist_array, maxA=50)
                scaled_grey = (np.clip( scaled, 0, 1 ) * 255).astype(np.uint8)
                cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s_distance_vis.png' % (name))), scaled_grey)
            else:
                print("%s: Type %d, size %d, pos \n%s" % (name, response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                
                # Decode the image directly from the bytes.
                decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)

                # Write the image as a PNG file.
                # The incoming data has 4 channels. The alpha channel is all zero.
                cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s.png' % (name))), decoded[:, :, :3])
                
    time_end = time.time()
    print(f'Six pinhole in {time_end - time_start}s. ')
    
    # ===========================================================
    
    name = 'cube'
    print(name)
    
    # New request for a cube image.
    request = [
        airsim.ImageRequest('0', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True)]
    
    # Set the pose.
    client.simSetVehiclePose(airsim_posese[0], ignore_collision=True)
    time.sleep(0.02)
    
    # Pause.
    client.simPause(True)
    
    # Get the images.
    responses = client.simGetImages(request)
    # responses = client.simGetImages(request)
    client.simPause(False)
    
    # Write the response to the filesystem.
    for _, response in enumerate(responses):
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
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s_distance.png' % (name))), img_array)

            # Visualize the depth.
            scaled = scale_float_array(dist_array, maxA=50)
            scaled_grey = (np.clip( scaled, 0, 1 ) * 255).astype(np.uint8)
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s_depth_vis.png' % (name))), scaled_grey)
        else:
            print("%s: Type %d, size %d, pos \n%s" % (name, response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            
            # Decode the image directly from the bytes.
            decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)

            # Write the image as a PNG file.
            # The incoming data has 4 channels. The alpha channel is all zero.
            cv2.imwrite(os.path.normpath(os.path.join(out_dir, '%s.png' % (name))), decoded[:, :, :3])