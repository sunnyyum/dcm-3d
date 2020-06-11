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
        self.imgs = np.stack([np.squeeze(sitk.GetArrayViewFromImage(img)) for img in self.imgs])

        # the dicom images can be encoded in different data type
        # By casting to int16, it can maintain the same data type
        self.imgs = self.imgs.astype(np.int16)

    def get_files(self) -> list:
        """Return file list that is sorted by InstanceNumber of a dicom file

        Returns
        -------
        list
            deepcopy of file list
        """
        return cp.deepcopy(self.files)

    def get_slices(self) -> list:
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
