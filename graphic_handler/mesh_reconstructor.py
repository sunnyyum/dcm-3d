import vtk
import pyvista as pv


def convert_pcd2mesh(point_cloud: pv.PolyData, alpha=0.) -> pv.PolyData:
    mesh = point_cloud.delaunay_3d(alpha=alpha, progress_bar=True)
    mesh = mesh.extract_geometry().triangulate()
    return mesh


def marchingCubes(volume) -> pv.PolyData:
    # use marching cube algorithm
    cf = vtk.vtkMarchingCubes()
    cf.SetInputData(volume)
    cf.SetValue(0, 1)
    cf.Update()

    # reverse the normal
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(cf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()

    reverse.Update()

    mesh = reverse.GetOutput()
    mesh = pv.wrap(mesh)
    return mesh


def flyingEdges(volume) -> pv.PolyData:
    # use flying edges algorithm
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(volume)
    fe.SetValue(0, 1)
    fe.ComputeNormalsOn()
    fe.Update()

    mesh = fe.GetOutput()
    mesh = pv.wrap(mesh)
    return mesh


def discreteMarchingCubes(volume) -> pv.PolyData:
    # use discrete marching cube algorithm
    dm = vtk.vtkDiscreteMarchingCubes()
    dm.SetInputData(volume)
    dm.ComputeNormalsOn()
    dm.GenerateValues(1, 0, 255)
    dm.Update()

    mesh = dm.GetOutput()
    mesh = pv.wrap(mesh)
    return mesh


def synchronizedTemplates3D(volume) -> pv.PolyData:
    # use SynchronizedTemplates3D algorithm
    st = vtk.vtkSynchronizedTemplates3D()
    st.SetInputData(volume)
    st.SetValue(0, 1)
    st.ComputeNormalsOn()
    st.Update()

    mesh = st.GetOutput()
    mesh = pv.wrap(mesh)
    return mesh
