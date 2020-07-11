from typing import Union

import os.path as path
import pyvista as pv
import vtk


def show_obj(file_path, turning=False, output_path=''):
    print(f'Reading {file_path} file...', end=' ', flush=True)
    obj = pv.read(file_path)
    print('Done')

    plotter = pv.Plotter()
    plotter.add_mesh(obj)

    if not turning:
        plotter.show_grid()
        plotter.show()
        plotter.close()
    else:
        _, ext = path.splitext(output_path)
        if ext != '.gif':
            assert f'{output_path} is not valid extension for gif'

        plotter.show(auto_close=False)
        orbit_path = plotter.generate_orbital_path(n_points=36, shift=obj.length)
        plotter.open_gif(output_path)
        plotter.orbit_on_path(orbit_path, write_frames=True)
        plotter.close()


def count_voxel(voxel: pv.UnstructuredGrid) -> int:
    return voxel.n_cells


def smooth_mesh_laplacian(mesh: pv.PolyData, n_iter=20) -> pv.PolyData:
    mesh = mesh.smooth(n_iter=n_iter)
    return mesh


def smooth_mesh(mesh: Union[vtk.vtkPolyData, pv.PolyData], n_iter=15, pass_band=0.001, feature_angle=120.0):
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


def save_object(mesh, output_name: str):
    mesh.save(output_name)
