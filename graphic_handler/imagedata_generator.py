from vtk.util import numpy_support
import vtk
import pyvista as pv
import numpy as np
import math
from typing import Union


def set_arr2imagedata(image_data: vtk.vtkImageData, imgs: np.ndarray, vtk_type='f'):
    if vtk_type == 'f':
        arr_type = vtk.VTK_FLOAT
    elif vtk_type == 'uc':
        arr_type = vtk.VTK_UNSIGNED_CHAR
    else:
        assert "With given choices, arr type has to be either float ('f') or unsigned char ('uc')"

    vtk_data_arr = numpy_support.numpy_to_vtk(
        num_array=imgs.ravel(order='F'),
        deep=True,
        array_type=arr_type
    )

    image_data.GetPointData().SetScalars(vtk_data_arr)


def setup_imagedata(image_data: vtk.vtkImageData, spacing: Union[list, np.ndarray], bound: Union[list, np.ndarray],
                    dim: Union[list, np.ndarray, tuple]):
    origin = [0] * 3
    if len(bound) == 6:
        origin[0] = bound[0]
        origin[1] = bound[2]
        origin[2] = bound[4]
    elif len(bound) == 3:
        origin = bound
    else:
        assert f'the size {len(bound)} of bound is not correct'

    image_data.SetOrigin(origin)
    image_data.SetSpacing(spacing)
    image_data.SetDimensions(dim)

    image_data.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)


def create_imagedata(data: Union[tuple, vtk.vtkPolyData, pv.PolyData], size: Union[list, np.ndarray]):
    """Inspired by the code in the website
        https://vtk.org/pipermail/vtkusers/2017-March/098281.html

    Parameters
    ----------
    data
    size

    Returns
    -------

    """
    white_image = vtk.vtkImageData()
    if type(data).__name__ == 'PolyData':
        bounds = data.GetBounds()
        # calculating the dimension size (image size)
        # formula = ceil(total length of an axis / spacing) + 1
        dim = [0] * 3
        for i in range(3):
            dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / size[i])) + 1
            if dim[i] < 1:
                dim[i] = 1

        setup_imagedata(white_image, size, bounds, dim)

        white_image.AllocateScalars(vtk.VTK_FLOAT, 1)

        # marking 3d points in vtkimagedata
        inval = 255

        for i in range(white_image.GetNumberOfPoints()):
            white_image.GetPointData().GetScalars().SetTuple1(i, inval)

    elif isinstance(data, tuple) and isinstance(data[0], np.ndarray) and (
            isinstance(data[1], np.ndarray) or isinstance(data[1], list)):
        imgs = data[0]
        bounds = data[1]
        dim = imgs.shape

        setup_imagedata(white_image, size, bounds, dim)
        set_arr2imagedata(white_image, imgs)

    else:
        assert 'For input data, it has to be either numpy array or PolyData'

    return white_image


def smooth_image_gauss(image_data: vtk.vtkImageData, deviation=8.):
    gaussianSmoothFilter = vtk.vtkImageGaussianSmooth()
    gaussianSmoothFilter.SetInputData(image_data)
    gaussianSmoothFilter.SetStandardDeviation(deviation)
    gaussianSmoothFilter.Update()
    return gaussianSmoothFilter.GetOutput()


def combine_img_poly(image_data: vtk.vtkImageData, polydata: Union[pv.PolyData, vtk.vtkPolyData]) -> vtk.vtkImageData:
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

    return imgstenc.GetOutput()


def pad_imagedata(orig_image_data: vtk.vtkImageData) -> vtk.vtkImageData:
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

    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    origin_update = [origin[0], origin[1], origin[2]]
    origin_update[2] -= spacing[2]

    setup_imagedata(image_data, spacing, origin, dim_pad)
    image_data.ComputeBounds()

    set_arr2imagedata(image_data, image_arr)
    return image_data
