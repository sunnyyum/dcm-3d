import argparse
import sys
from graphic_handler.graphic_util import save_object
from graphic_handler.graphic_generator import *

from dicom_handler import dicom_decoder, corr_finder


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
    parser = argparse.ArgumentParser(description='convert labeled dicom images to point cloud represented object')
    parser.add_argument('-i', required=True, dest='input_dir',
                        help='input directory path that has labeled dicom files')
    parser.add_argument('-o', required=True, default='./output.vtk', dest='output_path',
                        help='output file including file path')

    return parser.parse_args(args)


if __name__ == '__main__':
    opts = read_opt(sys.argv[1::])

    # decode dicom files for converting a dicom file to a dicom image
    decoder = dicom_decoder.DicomDecoder(opts.input_dir)

    decoder.read_info()

    print('Snippet of imported files:')
    print(*decoder.get_files()[0:3], sep='\n')

    decoder.convert_dcm2img()
    decoder.calculate_hounsfield()

    corr = corr_finder.Correspondence(decoder.get_info())

    print('Collecting label location...', end=' ', flush=True)
    indices = decoder.collect_label_loc()
    print('Done')

    print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
    points = corr.convert_2d_3d(indices)
    print('Done')

    mesh = generate_pointcloud(points)

    print(f'Saving the output to {opts.output_path}...', end=' ', flush=True)
    save_object(mesh, opts.output_path)
    print('Done')
