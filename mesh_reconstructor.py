import vtk
from vtk.util import numpy_support
import pyvista as pv
import math
import numpy as np


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
    def convert_np2volume(imgs: np.ndarray, bounds, size, smooth=False):
        vtk_data_arr = numpy_support.numpy_to_vtk(
            num_array=imgs.ravel(order='F'),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )

        dims = imgs.shape
        whiteImage = vtk.vtkImageData()
        whiteImage.SetDimensions(dims)
        whiteImage.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        whiteImage.ComputeBounds()

        whiteImage.SetSpacing(size)

        origin = [0] * 3
        origin[0] = bounds[0]
        origin[1] = bounds[2]
        origin[2] = bounds[4]
        whiteImage.SetOrigin(origin)

        whiteImage.GetPointData().SetScalars(vtk_data_arr)

        if smooth:
            # smooth with deviation 8
            gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
            gaussianSmoothFilter.SetInputData(whiteImage)
            gaussianSmoothFilter.SetStandardDeviation(8.)
            gaussianSmoothFilter.Update()
            whiteImage = gaussianSmoothFilter.GetOutput()

        return whiteImage

    @staticmethod
    def pad_imagedata(orig_image_data: vtk.vtkImageData):
        image_data = vtk.vtkImageData()
        image_data.DeepCopy(orig_image_data)

        dim = image_data.GetDimensions()
        point_data = image_data.GetPointData()
        # Ensure that only one array exists within the 'vtkPointData' object
        assert (point_data.GetNumberOfArrays() == 1)
        # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
        point_arr = point_data.GetArray(0)

        # Convert the `vtkArray` to a NumPy array
        image_arr = numpy_support.vtk_to_numpy(point_arr)
        # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
        image_arr = image_arr.reshape(dim, order='F')

        zero_pad = np.zeros((dim[0], dim[1], 1))
        image_arr = np.concatenate((zero_pad, image_arr, zero_pad), axis=2)
        dim_pad = image_arr.shape

        image_data.SetDimensions(dim_pad)
        image_data.SetExtent(0, dim_pad[0] - 1, 0, dim_pad[1] - 1, 0, dim_pad[2] - 1)

        origin = image_data.GetOrigin()
        spacing = image_data.GetSpacing()

        origin_update = [origin[0], origin[1], origin[2]]
        origin_update[2] -= spacing[2]
        image_data.SetOrigin(origin_update)
        image_data.ComputeBounds()

        vtk_data_arr = numpy_support.numpy_to_vtk(
            num_array=image_arr.ravel(order='F'),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        image_data.GetPointData().SetScalars(vtk_data_arr)
        return image_data

    @staticmethod
    def combine_img_poly(image_data: vtk.vtkImageData, polydata):
        origin = image_data.GetOrigin()
        spacing = image_data.GetSpacing()
        outval = 0

        # from polydata to image volume
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)

        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(image_data.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(image_data)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        return imgstenc

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

        whiteImage.AllocateScalars(vtk.VTK_FLOAT, 1)

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

    @staticmethod
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

    @staticmethod
    def discreteMarchingCubes(volume) -> pv.PolyData:
        # use discrete marching cube algorithm
        dm = vtk.vtkDiscreteMarchingCubes()
        dm.SetInputData(volume)
        dm.ComputeNormalsOn()
        # dm.GenerateValues(1, 0, 255)
        dm.Update()

        mesh = dm.GetOutput()
        mesh = pv.wrap(mesh)
        return mesh

    @staticmethod
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
