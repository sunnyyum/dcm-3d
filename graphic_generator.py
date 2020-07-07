import numpy as np
import pyvista as pv
from PVGeo.filters import VoxelizePoints
import vtk
import mesh_reconstructor


class GraphicGenerator:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.mr = mesh_reconstructor.MeshReconstructor()

    def generate_pointcloud(self) -> pv.PolyData:
        """A helper to make a 3D NumPy array of points (n_points by 3)

        Returns
        -------
        pyvista.PolyData
            point cloud
        """
        point_cloud = pv.PolyData(self.points)
        return point_cloud

    def generate_voxel(self, voxel_size: np.ndarray) -> pv.UnstructuredGrid:
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
        voxels = voxelizer.apply(point_cloud)
        return voxels

    def convert_voxel2mesh(self, voxels: pv.UnstructuredGrid, voxel_size: np.ndarray, smooth=False) -> pv.PolyData:
        voxel_polydata = self.voxel2polydata(voxels)
        volume = self.mr.convert_poly2volume(voxel_polydata, voxel_size, smooth=smooth)
        mesh = self.mr.discreteMarchingCubes(volume)
        return mesh

    def convert_pcd2mesh(self, point_cloud: pv.PolyData) -> pv.PolyData:
        mesh = self.mr.convert_pcd2mesh(point_cloud)
        return mesh

    @staticmethod
    def voxel2polydata(voxels: pv.UnstructuredGrid) -> pv.PolyData:
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(voxels)
        geo_filter.Update()
        polydata = geo_filter.GetOutput()
        better = pv.wrap(polydata)
        return better

    @staticmethod
    def count_voxel(voxel: pv.UnstructuredGrid) -> int:
        return voxel.n_cells

    @staticmethod
    def smooth_mesh_laplacian(mesh: pv.PolyData, n_iter=20) -> pv.PolyData:
        mesh = mesh.smooth(n_iter=n_iter)
        return mesh

    @staticmethod
    def smooth_mesh(mesh, n_iter=15, pass_band=0.001, feature_angle=120.0):
        # mesh = vtk.vtkPolyData(mesh)
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(n_iter)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        mesh = smoother.GetOutput()
        mesh = pv.wrap(mesh)
        return mesh

    @staticmethod
    def save_object(mesh: pv.PolyData, output_name: str):
        mesh.save(mesh, output_name)
