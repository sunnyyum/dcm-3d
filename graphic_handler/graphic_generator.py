from PVGeo.filters import VoxelizePoints
from graphic_handler.mesh_reconstructor import *
from graphic_handler.imagedata_generator import *


def generate_pointcloud(points: np.ndarray) -> pv.PolyData:
    """A helper to make a 3D NumPy array of points (n_points by 3)

    Parameters
    ----------
    points : np.ndarray
        3D points for point cloud

    Returns
    -------
    pyvista.PolyData
        point cloud
    """
    point_cloud = pv.PolyData(points)
    return point_cloud


def generate_voxel(points: np.ndarray, voxel_size: np.ndarray) -> pv.UnstructuredGrid:
    """Convert points to voxels

    Parameters
    ----------
    points     : np.ndarray
        points that represent the center of voxel
    voxel_size : np.ndarray
        3d points in numpy array format
        shape is (3,)

    Returns
    -------
    pyvista.UnstructuredGrid
        voxel object
    """
    point_cloud = generate_pointcloud(points)
    voxelizer = VoxelizePoints()
    voxelizer.set_deltas(*voxel_size)
    voxelizer.set_estimate_grid(False)
    voxels = voxelizer.apply(point_cloud)
    return voxels


def convert_pcd2mesh(point_cloud: pv.PolyData) -> pv.PolyData:
    mesh = convert_pcd2mesh(point_cloud)
    return mesh


def convert_voxel2mesh(voxels: pv.UnstructuredGrid, voxel_size: np.ndarray, algo: str, combine_img=False, imgs=None,
                       smooth=False) -> pv.PolyData:
    algorithm = get_vol_algo_dict()

    if type(voxels).__name__ == 'UnstructuredGrid':
        polydata = convert_voxel2polydata(voxels)
    elif type(voxels).__name__ == 'PolyData':
        polydata = voxels
    else:
        assert f'{type(voxels)} has to be either UnstructuredGrid or PolyData'

    if combine_img:
        image_data = create_imagedata(imgs, voxel_size)
    else:
        image_data = create_imagedata(polydata, voxel_size)

    if smooth:
        image_data = smooth_image_gauss(image_data)

    volume = combine_img_poly(image_data, polydata)
    volume_pad = pad_imagedata(volume)
    mesh = algorithm[algo](volume_pad)
    return mesh


def convert_img2mesh(imgs: np.ndarray, bounds, size, algo: str, smooth=False) -> pv.PolyData:
    data = (imgs, bounds)
    algorithm = get_vol_algo_dict()
    volume = create_imagedata(data, size)
    if smooth:
        volume = smooth_image_gauss(volume)
    volume_pad = pad_imagedata(volume)
    mesh = algorithm[algo](volume_pad)
    return mesh


def convert_voxel2polydata(voxels: pv.UnstructuredGrid) -> pv.PolyData:
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(voxels)
    geo_filter.Update()
    polydata = geo_filter.GetOutput()
    polydata_pv = pv.wrap(polydata)
    return polydata_pv


def get_vol_algo_dict():
    algorithm = {
        'marching cubes': lambda vol: marchingCubes(vol),
        'discrete marching cubes': lambda vol: discreteMarchingCubes(vol),
        'synchronized templates 3D': lambda vol: synchronizedTemplates3D(vol),
        'flying edges': lambda vol: flyingEdges(vol)
    }
    return algorithm
