import dicom_decoder
import sys
import argparse
import corr_finder
import graphic_handler
import numpy as np


def read_opt(args: list) -> argparse.Namespace:
    """Read the arguments and store into variable
    Maps the input from command line with variables, based on flag in the command line input.

    Parameters
    ----------
    args : list
        contains all the arguments after python file itself

    Returns
    -------
    argparse.Namespace
         an object holding attribute of input directory path
    """
    parse = argparse.ArgumentParser(description="convert labeled dicom images to 3D model")

    parse.add_argument('-i', required=True, dest='input_dir',
                       help='input directory path that has labeled dicom files')

    # parse.add_argument('-o', required=False, default='./output/report.csv', dest='output_path',
    #                    help='output file including file path')

    return parse.parse_args(args)


def print_files(file_li: list, interval: tuple, sep='\n') -> None:
    print(*file_li[interval[0]:interval[1]], sep=sep)


if __name__ == '__main__':
    # read arguments
    opts = read_opt(sys.argv[1::])
    dir_path = opts.input_dir

    # decode dicom files for converting a dicom file to a dicom image
    dcmDecoder = dicom_decoder.DicomDecoder(dir_path)
    dcmDecoder.read_info()
    dcmDecoder.convert_dcm2img()
    dcmDecoder.calculate_hounsfield()

    corr = corr_finder.Correspondence(dcmDecoder.get_info())
    print('Collecting label location...', end=' ', flush=True)
    indices = dcmDecoder.collect_label_loc()
    print('Done')

    print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
    points = corr.convert_2d_3d(indices)
    print('Done')

    print('Converting 3d point to 3d voxel center point...', end=' ', flush=True)
    voxel_points, voxel_size = corr.calculate_voxel_center(points)
    print('Done')

    gh = graphic_handler.GrphicHandler(voxel_points)

    # print('Creating voxels...', end=' ', flush=True)
    # grid = gh.convert_point2voxel(voxel_size)
    # print('Done')
    # grid.plot(show_grid=True)

    # print('Creating mesh...', end=' ', flush=True)
    # mesh = gh.convert_point2mesh()
    # print('Done')
    # mesh.plot()
