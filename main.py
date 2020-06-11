import dicom_decoder
import sys
import argparse


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


if __name__ == '__main__':
    # read arguments
    opts = read_opt(sys.argv[1::])
    dir_path = opts.input_dir

    # decode dicom files for converting a dicom file to a dicom image
    dcmDecoder = dicom_decoder.DicomDecoder(dir_path)
    dcmDecoder.read_info()
    dcmDecoder.convert_dcm2img()

    imgs = dcmDecoder.get_imgs()
    slices = dcmDecoder.get_slices()
