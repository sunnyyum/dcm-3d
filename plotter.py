import argparse
import sys
import os.path as path
from graphic_handler.graphic_util import *


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


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
         an object holding attribute of input numpy paths
    """
    parse = argparse.ArgumentParser(description="visualize 3D object from the input object")

    parse.add_argument('-i', required=True, dest='input_obj',
                       help='input 3D object file')

    parse.add_argument('--ani', required=False, default=False, dest='animation', type=str_to_bool, nargs='?',
                       const=True, help='orbiting the object')

    parse.add_argument('-o', required=False, dest='output_path',
                       help='output file including file path')

    return parse.parse_args(args)


if __name__ == '__main__':
    opts = read_opt(sys.argv[1::])

    file_path = opts.input_obj
    output_path = opts.output_path if not opts.output_path else 'None'
    is_ani = opts.animation

    if is_ani and output_path:
        _, ext = path.splitext(output_path)
        if ext != '.gif':
            assert f'{output_path} is not valid extension for gif'

    show_obj(file_path, turning=is_ani, output_path=output_path)
