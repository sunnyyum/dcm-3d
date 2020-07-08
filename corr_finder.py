import numpy as np
import pydicom


class Correspondence:
    """Convert a 2D coordinate to a corresponded 3D coordinate
    Every dicom file has information to create an affine formula
    The Affine formula can transform a 2D coordinate in a dicom image to a 3D coordinate
    """

    def __init__(self, dcm_info: pydicom.dataset.FileDataset) -> None:
        """Initialize the class

        Parameters
        ----------
        dcm_info : pydicom.dataset.FileDataset
            information in a dicom file
        """
        self.dcm_info = dcm_info
        self.affine_mat = None

    def generate_matrix(self, index: int) -> None:
        """Generate Affine Formula based on a dicom metadata

        Parameters
        ----------
        index : int
            the index of dicom metadata set
        """
        row_space, col_space = [self.dcm_info[index].PixelSpacing[i] for i in (0, 1)]
        Xx, Xy, Xz, Yx, Yy, Yz = [self.dcm_info[index].ImageOrientationPatient[i] for i in (0, 1, 2, 3, 4, 5)]
        Sx, Sy, Sz = [self.dcm_info[index].ImagePositionPatient[i] for i in (0, 1, 2)]

        self.affine_mat = np.asarray([[Xx * col_space, Yx * row_space, 0, Sx],
                                      [Xy * col_space, Yy * row_space, 0, Sy],
                                      [Xz * col_space, Yz * row_space, 0, Sz],
                                      [0, 0, 0, 1]])

    def convert_2d_3d(self, indicies: np.ndarray) -> np.ndarray:
        """Convert coordinate in a dicom image to 3d coordinate

        Parameters
        ----------
        indicies : numpy.ndarray
            a coordinate in a dicom image
            shape of each coordinate should be (4,) (ex. [row_corr, col_coor, 0, 1])

        Returns
        -------
        numpy.ndarray
            3d coordinate
        """
        coor_3d = []
        for dcm_index, loc_li in indicies:
            self.generate_matrix(dcm_index)
            for coor in loc_li:
                coor_3d.append(np.delete(np.matmul(self.affine_mat, np.append(coor, [0, 1])), -1))

        return np.asarray(coor_3d, dtype=np.float64)

    def calculate_voxel_center(self, coor_3d: np.ndarray) -> (np.ndarray, np.ndarray):
        """Calculate the 3D coordinate for the center of a voxel point
        The input 3D coordinate (coor_3d) is on a dicom image.
        This function will transform this coordinate into a coordinate that is the center of a voxel

        Parameters
        ----------
        coor_3d : np.ndarray
            3D coordinate that is on a dicom image

        Returns
        -------
        (np.ndarray, np.ndarray)
            first element  : coordinates that each of them is the center of a voxel point
            second element : a size of voxel
        """
        if coor_3d.shape[1] != 3:
            raise ValueError('input size must be 3')

        voxel_size = self.calculate_voxel_size()

        # calculate the 3D coordinate for the center of a voxel point
        voxel_coor = []
        for point in coor_3d:
            voxel_coor.append(point + (voxel_size / 2))

        return np.asarray(voxel_coor, dtype=np.float64), voxel_size

    def calculate_voxel_size(self):
        # get the thickness in the dicom image set
        # assume that the thickness is uniform across the dicom image set
        try:
            thickness = np.float64(self.dcm_info[0].SliceThickness)
        except AssertionError:
            z1 = np.float64(self.dcm_info[0].ImagePositionPatient[2])
            z2 = np.float64(self.dcm_info[1].ImagePositionPatient[2])
            thickness = np.abs(z1 - z2)

        # get pixel space list
        # assume that the pixel sapceing is uniform across the dicom image set
        pixel_space = np.asarray(self.dcm_info[0].PixelSpacing, dtype=np.float64)

        voxel_size = np.append(pixel_space, thickness)
        return voxel_size

    def find_boundary(self, coor_3d: np.ndarray):
        coor_3d = coor_3d.transpose()
        bounds = np.zeros(6)
        for i in range(coor_3d.shape[0]):
            bounds[2 * i] = np.floor(np.min(coor_3d[i]))
            bounds[2 * i + 1] = np.ceil(np.max(coor_3d[i]))

        bounds = bounds.astype(np.int16)

        return bounds
