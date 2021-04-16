import numpy as np
import math
from .pyrender_wrapper import RenderApp
import pyrender
from scipy.spatial.transform import Rotation
from scipy.ndimage import laplace, sobel
from skimage import io
import matplotlib.pyplot as plt
import cv2

def get_rotation_matrix(angle, axis='y'):
    matrix = np.identity(4)
    matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    return matrix

def get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_distance=2):
    camera_transform = np.identity(4)
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform

# Camera transform from position and look direction
def get_camera_transform(position, look_direction):
    camera_forward = -look_direction / np.linalg.norm(look_direction)
    camera_right = np.cross(camera_forward, np.array((0, 0, -1)))
    if np.linalg.norm(camera_right) < 0.5:
        camera_right = np.array((0, 1, 0))
    camera_up = np.cross(camera_forward, camera_right)

    rotation = np.identity(4)
    rotation[:3, 0] = camera_right
    rotation[:3, 1] = camera_up
    rotation[:3, 2] = camera_forward

    translation = np.identity(4)
    translation[:3, 3] = position

    return np.matmul(translation, rotation)

'''
A virtual laser scan of an object from one point in space.
This renders a normal and depth buffer and reprojects it into a point cloud.
The resulting point cloud contains a point for every pixel in the buffer that hit the model.
'''
class Scan():
    def __init__(self, render_app, theta, camera, camera_transform, calculate_normals=True):
        self.camera_transform = camera_transform
        self.camera_position = np.matmul(self.camera_transform, np.array([0, 0, 0, 1]))[:3]
        self.theta = theta
        self.resolution = render_app.resolution

        self.projection_matrix = camera.get_projection_matrix()

        color, depth = render_app.render_normal_and_depth_buffers(self.camera_transform)
        #Wish we could output float here...
        color = color.astype(np.float32) / 255
        self.normal_buffer = color if calculate_normals else None
        self.edges = None
        self.edges_sharp = None
        self.edge_img = None

        #self.flip_normals = None
        self.depth_buffer = depth.copy()

        #plt.imshow(depth, cmap='gray')
        #plt.show()

        indices = np.argwhere(depth != 0)
        depth[depth == 0] = float('inf')

        # This reverts the processing that pyrender does and calculates the original depth buffer in clipping space
        self.depth = (camera.zfar + camera.znear - (2.0 * camera.znear * camera.zfar) / depth) / (camera.zfar - camera.znear)

        points = np.ones((indices.shape[0], 4))
        points[:, [1, 0]] = indices.astype(float) / (self.resolution - 1) * 2 - 1
        points[:, 1] *= -1
        points[:, 2] = self.depth[indices[:, 0], indices[:, 1]]

        self.points_camera = self.depth [indices[:, 0], indices[:, 1]]

        clipping_to_world = np.matmul(self.camera_transform, np.linalg.inv(self.projection_matrix))

        points = np.matmul(points, clipping_to_world.transpose())
        points /= points[:, 3][:, np.newaxis]
        self.points = points[:, :3]

        if calculate_normals:
            normals = color[indices[:, 0], indices[:, 1]] * 2 - 1
            camera_to_points = self.camera_position - self.points
            normal_orientation = np.einsum('ij,ij->i', camera_to_points, normals)
            normals[normal_orientation < 0] = -normals[normal_orientation < 0]
            self.normals = normals

            #Save the gradient/edges of the normal map so we can assign probabilities for sampling later
            #We also have to flip wrong normals in the image before we apply the sobel filter
            color_norm = color * 2 - 1
            color_norm[indices[:, 0][normal_orientation<0], indices[:, 1][normal_orientation<0],:] = \
                -color_norm[indices[:, 0][normal_orientation<0], indices[:, 1][normal_orientation<0],:]

            kernel = np.ones((7,7), np.uint8)
            maskBefore = np.any(color_norm != (1,1,1),axis=-1).astype(np.float)
            mask = cv2.erode(maskBefore, kernel, 3)
            #maskCombined = np.stack([mask, maskBefore, mask], axis = 2)
            #plt.imshow(maskCombined)
            #plt.show()

            #Use a smoother edge detector for weighting
            #sx = sobel(color_norm, axis=1, mode='constant')
            #sy = sobel(color_norm, axis=0, mode='constant')
            sx = cv2.Sobel(color_norm,cv2.CV_32F, 1,0, ksize=5)
            sy = cv2.Sobel(color_norm,cv2.CV_32F, 0,1, ksize=5)
            self.edge_img = np.hypot(sx, sy) * mask[...,None]

            #plt.imshow(color_norm)
            #plt.show()
            #plt.imshow(self.edge_img)
            #plt.show()
            self.edges = self.edge_img[indices[:, 0], indices[:, 1]]

            #And a sharp one for the actual edges to learn on
            sx = cv2.Sobel(color_norm,cv2.CV_32F, 1,0, ksize=1)
            sy = cv2.Sobel(color_norm,cv2.CV_32F, 0,1, ksize=1)
            edges_sharp = np.hypot(sx, sy) * mask[...,None]
            self.edges_sharp = edges_sharp[indices[:, 0], indices[:, 1]]

            #Weight flip normal suggestions based on the position of the camera
            #Some types of models tend to have more holes on the bottom
            #Not 100%  if always good...
            #normal_value = np.expand_dims(normal_orientation,1)
            #self.flip_normals = np.sign(normal_value)  *  1 - np.clip(self.theta / (math.pi/2), 0, 1)
        else:
            self.normals = None

    def convert_world_space_to_viewport(self, points):
        half_viewport_size = 0.5 * self.resolution
        clipping_to_viewport = np.array([
            [half_viewport_size, 0.0, 0.0, half_viewport_size],
            [0.0, -half_viewport_size, 0.0, half_viewport_size],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0.0, 1.0]
        ])

        world_to_clipping = np.matmul(self.projection_matrix, np.linalg.inv(self.camera_transform))
        world_to_viewport = np.matmul(clipping_to_viewport, world_to_clipping)
        
        world_space_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        viewport_points = np.matmul(world_space_points, world_to_viewport.transpose())
        viewport_points /= viewport_points[:, 3][:, np.newaxis]
        return viewport_points

    def is_visible(self, points):
        viewport_points = self.convert_world_space_to_viewport(points)
        pixels = viewport_points[:, :2].astype(int)

        # This only has an effect if the camera is inside the model
        in_viewport = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < self.resolution) & (pixels[:, 1] < self.resolution) & (viewport_points[:, 2] > -1)

        result = np.zeros(points.shape[0], dtype=bool)
        result[in_viewport] = viewport_points[in_viewport, 2] < self.depth[pixels[in_viewport, 1], pixels[in_viewport, 0]]

        return result

    def show(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    def save(self, filename_depth, filename_normals=None):
        if filename_normals is None and self.normal_buffer is not None:
            items = filename_depth.split('.')
            filename_normals = '.'.join(items[:-1]) + "_normals." + items[-1]
        
        depth = self.depth_buffer / np.max(self.depth_buffer) * 255

        io.imsave(filename_depth, depth.astype(np.uint8))
        if self.normal_buffer is not None:
            io.imsave(filename_normals, self.normal_buffer.astype(np.uint8))