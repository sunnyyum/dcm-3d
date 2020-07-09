import pydicom
import numpy as np
import os
import copy as cp
import SimpleITK as sitk


class DicomDecoder:
    """Reasons for using two different libraries to decode a dicom file
    Here, I used pydicom library to read a dicom information and SimpleITK to decode a dicom image.
    This is because SimpleITK decodes a dicom image better than pydicom.
    Sometimes pydicom can't decode a dicom image due to a different decoding style.
    However, pydicom reads information in a dicom file better than SimpleITK.
    """

    def __init__(self, dir_path: str) -> None:
        """Initialize the class

        Parameters
        ----------
        dir_path : string
            directory path that contains labeled dicom files
        """
        if not len(dir_path):
            raise ValueError('please give the directory path')

        self.dir_path = dir_path
        self.info = None
        self.files = None
        self.imgs = None
        self.is_hounsfield = False

    def read_info(self) -> None:
        """Read information of dicom files
        """
        # collect files in the given directory path stored
        files = [self.dir_path + '/' + s for s in os.listdir(self.dir_path)]

        # read information in dicom files
        info = [pydicom.read_file(s) for s in files]

        # get sorted index based on InstanceNumber of a dicom file
        sorted_ind = sorted(range(len(info)), key=lambda x: int(info[x].InstanceNumber))

        # sort dicom information list and file list based on  InstanceNumber of a dicom file
        info = [info[ind] for ind in sorted_ind]
        files = [files[ind] for ind in sorted_ind]

        self.info = info
        self.files = files

    def convert_dcm2img(self) -> None:
        """Decode each dicom file to convert into image
        """
        # read each dicom file to convert into an image
        self.imgs = [sitk.ReadImage(f) for f in self.files]
        # convert images into array and stack them
        # shape = (dicom image row, dicom image column, the number of dicom images)
        self.imgs = np.stack([np.squeeze(sitk.GetArrayFromImage(img)) for img in self.imgs], axis=2)

        # the dicom images can be encoded in different data type
        # By casting to int16, it can maintain the same data type
        self.imgs = self.imgs.astype(np.int16)

    def calculate_hounsfield(self) -> None:
        """Calculate Hounsfield unit (HU)
        """
        # calculate hounsfield units, once
        if self.is_hounsfield:
            print('Hounsfield unit is already calculated')
            return

        self.is_hounsfield = True

        # calculate hounsfield units(HU)
        intercept = np.float64(self.info[0].RescaleIntercept)
        slope = np.float64(self.info[0].RescaleSlope)

        # HU unit = slope * value + intercept
        self.imgs = slope * self.imgs.astype(np.float64)
        self.imgs += intercept

        # casting back to int16
        self.imgs = self.imgs.astype(np.int16)

    def collect_label_loc(self) -> list:
        """Collect labeled pixel location
        collect labeled pixel location, where the value is more than 0

        Returns
        -------
        list
            indices of labeled pixel in each dicom image
        """

        if not self.is_hounsfield:
            print('Not converted into Hounsfield unit, yet')
            return

        # not to compromise original images,
        # use the copy of original images
        imgs = self.get_imgs()
        imgs[imgs <= 0] = 0

        label_loc = []
        len_dcm = imgs.shape[2]
        for index in range(len_dcm):
            locs = np.nonzero(imgs[:, :, index])
            # collect non zero (labeled) indexes
            indexes = np.transpose((locs[0], locs[1]))
            label_loc.append((index, indexes))

        return label_loc

    def collect_edge(self):
        if not self.is_hounsfield:
            print('Not converted into Hounsfield unit, yet')
            return

        if not isinstance(self.imgs, np.ndarray) or not self.imgs.any():
            print('DICOM files are not loaded')
            return

        edge_point_loc = []
        row, col, _ = self.imgs.shape

        # 0,0  row+1,0  0,col+1  row+1,col+1
        for index in [0, -1]:
            indices = [[0, 0], [row + 1, 0], [0, col + 1], [row + 1, col + 1]]
            edge_point_loc.append((index, np.asarray(indices)))

        return edge_point_loc

    def get_files(self) -> list:
        """Return file list that is sorted by InstanceNumber of a dicom file

        Returns
        -------
        list
            deepcopy of file list
        """
        return cp.deepcopy(self.files)

    def get_info(self) -> list:
        """Return dicom information list that is sorted by InstanceNumber of a dicom file

        Returns
        -------
        list
            deepcopy of dicom information list
        """
        return cp.deepcopy(self.info)

    def get_imgs(self) -> np.ndarray:
        """Return stacked images that is sorted by InstanceNumber of a dicom file

        Returns
        -------
        numpy.ndarray
            deepcopy of dicom images
        """
        return cp.deepcopy(self.imgs)
