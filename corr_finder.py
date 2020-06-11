import numpy as np
import pydicom


class Correspondence:

    def __init__(self, slice_info: pydicom.dataset.FileDataset) -> None:
        self.slice_info = slice_info
        self.affine_mat = None

    def generate_matrix(self) -> np.ndarray:
        """Generate Affine Formula based on a slice metadata

        Returns
        -------
        numpy.ndarray
            Affine Formula Matrix
        """
        row_space, col_space = [self.slice_info.PixelSpacing[i] for i in (0, 1)]
        Xx, Xy, Xz, Yx, Yy, Yz = [self.slice_info.ImageOrientationPatient[i] for i in (0, 1, 2, 3, 4, 5)]
        Sx, Sy, Sz = [self.slice_info.ImagePositionPatient[i] for i in (0, 1, 2)]

        self.affine_mat = np.asarray([[Xx * col_space, Yx * row_space, 0, Sx],
                                      [Xy * col_space, Yy * row_space, 0, Sy],
                                      [Xz * col_space, Yz * row_space, 0, Sz],
                                      [0, 0, 0, 1]])

    def convert_2d_3d(self, coor_2d: np.ndarray) -> np.ndarray:
        """Convert coordinate in a dicom image to 3d coordinate

        Parameters
        ----------
        coor_2d : Numpy.ndarray
            a coordinate in a dicom image
            shape should be (4,) (ex. [row_corr, col_coor, 0, 1])

        Returns
        -------
        Numpy.ndarray
            3d coordinate
        """
        return np.matmul(self.affine_mat, coor_2d)
