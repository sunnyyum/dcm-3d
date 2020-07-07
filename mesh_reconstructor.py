import vtk
import pyvista as pv
import math


class MeshReconstructor:
    """Inspired by the code in the website
        https://vtk.org/pipermail/vtkusers/2017-March/098281.html
    """

    @staticmethod
    def convert_pcd2mesh(point_cloud: pv.PolyData, alpha=0.) -> pv.PolyData:
        mesh = point_cloud.delaunay_3d(alpha=alpha, progress_bar=True)
        mesh = mesh.extract_geometry().triangulate()
        return mesh

    @staticmethod
    def convert_poly2volume(polydata, size, smooth=False):
        whiteImage = vtk.vtkImageData()
        # getting bounds
        bounds = polydata.GetBounds()
        # setting volume size (same with pixel spacing)
        spacing = size

        whiteImage.SetSpacing(spacing)

        # calculating the dimension size (image size)
        # formula = ceil(total length of an axis / spacing) + 1
        dim = [0] * 3
        for i in range(3):
            dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i])) + 1
            if dim[i] < 1:
                dim[i] = 1
        whiteImage.SetDimensions(dim)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

        # setting origin which is starting points in x,y,z axis from boundary
        origin = [0] * 3
        origin[0] = bounds[0]
        origin[1] = bounds[2]
        origin[2] = bounds[4]
        whiteImage.SetOrigin(origin)

        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # marking 3d points in vtkimagedata
        inval = 255
        # outval will be used as a background image value
        outval = 0

        for i in range(whiteImage.GetNumberOfPoints()):
            whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

        if smooth:
            # smooth with deviation 8
            gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
            gaussianSmoothFilter.SetInputData(whiteImage)
            gaussianSmoothFilter.SetStandardDeviation(8.)
            gaussianSmoothFilter.Update()
            whiteImage = gaussianSmoothFilter.GetOutput()

        # from polydata to image volume
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)

        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        return imgstenc

    @staticmethod
    def marchingCubes(volume) -> pv.PolyData:
        # use marching cube algorithm
        cf = vtk.vtkMarchingCubes()
        cf.SetInputData(volume.GetOutput())
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

    @staticmethod
    def flyingEdges(volume) -> pv.PolyData:
        # use flying edges algorithm
        fe = vtk.vtkFlyingEdges3D()
        fe.SetInputData(volume.GetOutput())
        fe.SetValue(0, 1)
        fe.ComputeNormalsOn()
        fe.Update()

        mesh = fe.GetOutput()
        mesh = pv.wrap(mesh)
        return mesh

    @staticmethod
    def discreteMarchingCubes(volume) -> pv.PolyData:
        # use discrete marching cube algorithm
        dm = vtk.vtkDiscreteMarchingCubes()
        dm.SetInputData(volume.GetOutput())
        dm.ComputeNormalsOn()
        dm.GenerateValues(1, 0, 255)
        dm.Update()

        mesh = dm.GetOutput()
        # mesh = pv.wrap(mesh)
        return mesh

    @staticmethod
    def synchronizedTemplates3D(volume) -> pv.PolyData:
        # use SynchronizedTemplates3D algorithm
        st = vtk.vtkSynchronizedTemplates3D()
        st.SetInputData(volume.GetOutput())
        st.SetValue(0, 1)
        st.ComputeNormalsOn()
        st.Update()

        mesh = st.GetOutput()
        mesh = pv.wrap(mesh)
        return mesh
