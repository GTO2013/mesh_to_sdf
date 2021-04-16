from .scan import Scan, get_camera_transform_looking_at_origin

import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import math
from .pyrender_wrapper import RenderApp
import pyrender
from .utils import sample_uniform_points_in_unit_sphere
from tqdm import tqdm

class BadMeshException(Exception):
    pass

class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, edges = None, edges_sharp = None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.edges = edges
        self.edges_sharp = edges_sharp
        self.scans = scans

        self.kd_tree = KDTree(points)

    def get_random_surface_points(self, count, use_scans=True):
        if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            return self.mesh.sample(count)

    def get_sdf(self, query_points, use_depth_buffer=False, sample_count=11, max_distance = 0.01):
        if use_depth_buffer:
            distances, _ = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1) * -1
            distances[self.is_outside(query_points)] *= -1
            return distances
        else:
            distances, indices = self.kd_tree.query(query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_to_surface = query_points[:, np.newaxis, :] - closest_points
            inside = np.einsum('ijk,ijk->ij', direction_to_surface, self.normals[indices]) < 0

            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1
            return distances

    def get_sdf_in_batches(self, query_points, use_depth_buffer=False, sample_count=11, batch_size=1000000):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points, use_depth_buffer=use_depth_buffer, sample_count=sample_count)
        
        result = np.zeros(query_points.shape[0])
        for i in range(int(math.ceil(query_points.shape[0] / batch_size))):
            start = int(i * batch_size)
            end = int(min(result.shape[0], (i + 1) * batch_size))
            result[start:end] = self.get_sdf(query_points[start:end, :], use_depth_buffer=use_depth_buffer, sample_count=sample_count, max_distance = 0.01)
        return result

    def get_voxels(self, voxel_resolution, use_depth_buffer=False, sample_count=11, pad=False, check_result=False):
        from mesh_to_sdf.utils import get_raster_points, check_voxels
        
        sdf = self.get_sdf_in_batches(get_raster_points(voxel_resolution), use_depth_buffer, sample_count)
        voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

        if check_result and not check_voxels(voxels):
            raise BadMeshException()

        if pad:
            voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

        return voxels

    def createProbabiltyDistribution(self, pointsHull, minChanceOffset = 0.1, clipValue = 4, distanceMult = 16, normalDiffMult = 1):
        #Map point height to 0-1
        bottom = (self.points[:,1] - np.min(self.points[:,1])) / (np.max(self.points[:,1]) - np.min(self.points[:,1]))
        #width = (self.points[:, 0] - np.min(self.points[:, 0])) / (np.max(self.points[:, 0]) - np.min(self.points[:, 0]))
        normalWeight = self.normals[:, 1].copy()

        threshold = -0.95
        #Weight normals that point down way less, but only up to half of the height
        normalWeight[normalWeight > threshold] = 1
        normalWeight[normalWeight <= threshold] = np.clip(np.sqrt(bottom[normalWeight <= threshold]), 0, 1)

        #Reapply some weights to the bottom of the tires
        normalWeight[bottom < 0.05] = 1
        #Reapply weights after half the cars height (we only want to catch the bottom)
        normalWeight[bottom > 0.5] = 1

        normalDiffWeight = np.zeros((self.points.shape[0]),dtype=np.float32)

        if normalDiffMult > 0:
            #axisList = [np.array([0,-1,-1]), np.array([0,1,1]),
            #            np.array([1,1,0]), np.array([-1,-1,0]),
            #            np.array([1,0,1]), np.array([-1,0,-1])]

            axisList = [np.array([0,-1,-1]), np.array([0,1,1]),
                        np.array([1,0,1]), np.array([-1,0,-1])]

            #axisList = [np.array([1,0,1])]

            thresholdDistance = 0.02
            thresholdAngle = np.array([1.2,1.2,1.2])

            for axis in axisList:
                if np.max(axis) == 1:
                    maskPos = np.all(self.normals < axis, axis=1)
                    maskNeg = np.all(self.normals > -axis, axis=1)
                else:
                    maskPos = np.all(self.normals > axis, axis=1)
                    maskNeg = np.all(self.normals < -axis, axis=1)

                normalPointsPos = self.points[maskPos]
                normalPointsNeg = self.points[maskNeg]

                tree_normal_diff = KDTree(normalPointsNeg)
                distancesNormalDiff, indicies = tree_normal_diff.query(normalPointsPos)
                indicies = indicies[:,0]
                distancesNormalDiff = distancesNormalDiff[:, 0]

                fixedIndices = np.arange(self.normals.shape[0])[maskNeg][indicies]
                values, indexU, counts = np.unique(fixedIndices, return_inverse=True, return_counts=True)
                count_fixed = counts[indexU]

                normalDistanceWeight = np.zeros((distancesNormalDiff.shape[0]),dtype=np.float32)
                normalDistanceWeight[distancesNormalDiff > thresholdDistance] = 0
                normalDistanceWeight[distancesNormalDiff <= thresholdDistance] = np.any(np.abs(self.normals[maskPos]- self.normals[fixedIndices]) > thresholdAngle, axis=1)[distancesNormalDiff <= thresholdDistance]
                normalDistanceWeight[count_fixed > 100] = 0

                normalDiffWeight[maskPos] = normalDiffWeight[maskPos] + normalDistanceWeight
                normalDiffWeight[fixedIndices] = normalDiffWeight[fixedIndices] + normalDistanceWeight


        #colorsHull = np.tile([[255,0,0]], pointsHull.shape[0])
        #colorsPoints = np.tile([[0, 255, 0]], self.points.shape[0])
        #colorsCombined = np.concatenate([colorsHull, colorsPoints], axis=1)
        #colorsCombined = np.reshape(colorsCombined, (-1,3))
        #combined = np.concatenate([pointsHull, self.points], axis=0)
        #p = trimesh.points.PointCloud(combined, colors=colorsCombined)
        #p.show()

        #Calculate distance to the visual hull
        tree_hull = KDTree(pointsHull)
        distances,_ = tree_hull.query(self.points)
        distances = distances[:,0]

        #distances = np.clip(distances,0.0,0.1)*10
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        distances = distances  * distanceMult

        edges = np.clip(self.edges, minChanceOffset, clipValue)
        edges = edges.sum(1)
        normalWeightDistance = normalWeight.copy()
        normalWeightDistance[normalWeightDistance < 0.9]  = normalWeightDistance[normalWeightDistance < 0.9] * 0.05

        weights = edges * normalWeight + distances * normalWeightDistance + normalDiffWeight * normalDiffMult
        return weights / weights.sum(0)

    def sample_sdf_near_surface(self, pointsHull, number_of_points=500000, use_scans=True, sign_method='normal', normal_sample_count=11, min_size=0, sigma = 0.005):
        query_points = []
        number_of_points = min(number_of_points, len(self.points))
        surface_sample_count = int(0.95 * number_of_points)
        indices = np.random.choice(self.points.shape[0], surface_sample_count, p = self.createProbabiltyDistribution(pointsHull))
        surface_points =  self.points[indices, :]
        moved_points = surface_points + np.random.normal(scale=sigma, size=(surface_sample_count, 3))
        normals = self.normals[indices,:]
        edges = self.edges_sharp[indices,:]

        #length = [1,1,1]
        length = (np.max(surface_points, axis=0)-np.min(surface_points, axis=0)) + [0.2,0.2,0.2]
        random_sample_count = number_of_points - surface_sample_count
        random_points = np.random.rand(random_sample_count, 3) * length - length/2

        query_points = np.concatenate([moved_points, random_points], 0)
        np.random.shuffle(query_points)

        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=True)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))

        #Fix SDF samples that are outside of bounding box
        minPoints = query_points < np.min(surface_points, axis=0)
        maxPoints = query_points > np.max(surface_points, axis=0)
        minPoints = np.any(minPoints, axis=1)
        maxPoints = np.any(maxPoints, axis=1)
        sdf[minPoints] = np.abs(sdf)[minPoints]
        sdf[maxPoints] = np.abs(sdf)[maxPoints]

        if min_size > 0:
            model_size = np.count_nonzero(sdf[-random_sample_count:] < 0) / random_sample_count
            if model_size < min_size:
                raise BadMeshException()

        return query_points, sdf, normals, edges, surface_points

    def show(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
        
    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result

def get_equidistant_camera_angles(count, maxTheta = 2):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + maxTheta * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta

def create_from_scans(mesh, bounding_radius=1, scan_count=100, scan_resolution=400, calculate_normals=True):
    scans = []
    camera = pyrender.PerspectiveCamera(yfov=1.0472, aspectRatio=1.0, znear=bounding_radius * 1, zfar=bounding_radius * 3)
    renderApp = RenderApp(scan_resolution, mesh, camera)

    for phi, theta in get_equidistant_camera_angles(scan_count, 2):
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, camera_distance=1.5 * bounding_radius)
        scans.append(Scan(render_app=renderApp, camera=camera,
            theta = theta,
            camera_transform=camera_transform,
            calculate_normals=calculate_normals
        ))

    return SurfacePointCloud(mesh,
        points = np.concatenate([scan.points for scan in scans], axis=0),
        normals=np.concatenate([scan.normals for scan in scans], axis=0) if calculate_normals else None,
        edges=np.concatenate([scan.edges for scan in scans], axis=0) if calculate_normals else None,
        edges_sharp=np.concatenate([scan.edges_sharp for scan in scans],axis=0) if calculate_normals else None,
        scans=scans
    )

def sample_from_mesh(mesh, sample_point_count=10000000, calculate_normals=True):
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count, return_index=False)

    return SurfacePointCloud(mesh, 
        points=points,
        normals=normals if calculate_normals else None,
        scans=None
    )
