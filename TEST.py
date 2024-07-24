import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
from pyquaternion import Quaternion
from PIL import Image

def load_nuscenes_data(version, dataroot):
    return NuScenes(version=version, dataroot=dataroot, verbose=True)

def get_transformed_image(nusc, sample_data, cam_intrinsic, ego_pose, sensor_pose):
    cam_path = os.path.join(nusc.dataroot, sample_data['filename'])
    cam_img = Image.open(cam_path)

    # Transform the image to BEV perspective (simple placeholder transformation)
    img_width, img_height = cam_img.size
    transformed_img = cam_img.transform((img_width, img_height), Image.QUAD,
                                        (0, 0, img_width, 0, img_width, img_height, 0, img_height))
    return np.array(transformed_img)

def stitch_images(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    stitched_image = np.zeros((800, 800, 3), dtype=np.uint8)  # Placeholder size, adjust as needed

    for channel in camera_channels:
        cam_data = nusc.get('sample_data', sample['data'][channel])
        calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        cam_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

        transformed_img = get_transformed_image(nusc, cam_data, cam_intrinsic, ego_pose, calibrated_sensor)
        transformed_img_resized = Image.fromarray(transformed_img).resize((400, 400))  # Resize to fit the allocated space

        transformed_img_resized = np.array(transformed_img_resized)

        # Simple placeholder to position images in the stitched view
        # You need to compute correct positions based on sensor calibration
        if channel == 'CAM_FRONT':
            stitched_image[200:600, 200:600, :] = transformed_img_resized
        elif channel == 'CAM_FRONT_LEFT':
            stitched_image[0:400, 0:400, :] = transformed_img_resized
        elif channel == 'CAM_FRONT_RIGHT':
            stitched_image[0:400, 400:800, :] = transformed_img_resized
        elif channel == 'CAM_BACK':
            stitched_image[400:800, 200:600, :] = transformed_img_resized
        elif channel == 'CAM_BACK_LEFT':
            stitched_image[400:800, 0:400, :] = transformed_img_resized
        elif channel == 'CAM_BACK_RIGHT':
            stitched_image[400:800, 400:800, :] = transformed_img_resized

    return stitched_image

def visualize_bev(nusc, sample_token):
    stitched_image = stitch_images(nusc, sample_token)

    plt.figure(figsize=(10, 10))
    plt.imshow(stitched_image)
    plt.title('Stitched BEV View')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    dataroot = 'data/nuscenes'  # Ensure this path is correct
    version = 'v1.0-mini'
    sample_token = 'ca9a282c9e77460f8360f564131a8af5'  # Replace with a valid sample token

    nusc = load_nuscenes_data(version, dataroot)
    visualize_bev(nusc, sample_token)
