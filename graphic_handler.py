import numpy as np
import pyvista as pv
from PVGeo.filters import VoxelizePoints
import open3d as o3d

class GrphicHandler:
    def __init__(self, points: np.ndarray):
        self.points = points

    # Define some helpers - ignore these and use your own data!
    def generate_pointcloud(self) -> pv.PolyData:
        """A helper to make a 3D NumPy array of points (n_points by 3)

        Returns
        -------
        pyvista.PolyData
            point cloud
        """
        point_cloud = pv.PolyData(self.points)
        return point_cloud

    def convert_point2voxel(self, voxel_size: np.ndarray) -> pv.UnstructuredGrid:
        """Convert points to voxels

        Parameters
        ----------
        self : np.ndarray
            3d points in numpy array format
            shape is (3,)

        Returns
        -------

        """
        point_cloud = self.generate_pointcloud()
        voxelizer = VoxelizePoints()
        voxelizer.set_deltas(*voxel_size)
        voxelizer.set_estimate_grid(False)
        grid = voxelizer.apply(point_cloud)
        return grid

    def convert_point2mesh_pv(self, alpha=0.) -> pv.PolyData:
        point_cloud = self.generate_pointcloud()
        mesh = point_cloud.delaunay_3d(alpha=alpha, progress_bar=True)
        mesh = mesh.extract_geometry().triangulate()
        return mesh

    # def convert_point2mesh_o3d(self):

    @staticmethod
    def triangulate_voxel(voxels: pv.UnstructuredGrid) -> pv.PolyData:
        mesh = voxels.extract_geometry().triangulate()
        return mesh

    @staticmethod
    def count_voxel(voxel: pv.UnstructuredGrid) -> int:
        return voxel.n_cells

    @staticmethod
    def save_object(mesh: pv.PolyData, output_name: str):
        mesh.save(mesh, output_name)
