import argparse
import sys

from graphic_handler.graphic_util import show_obj


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
    parser = argparse.ArgumentParser(description="visualize 3D object from the input object")

    parser.add_argument('-i', required=True, dest='input_obj', help='input 3D object file')

    ani_group = parser.add_mutually_exclusive_group()
    ani_group.add_argument('--no-ani', default=False, dest='animation', action='store_false',
                           help='only display the input object')

    ani_group.add_argument('--ani', dest='output_path', help='save the orbiting object to output file')

    return parser.parse_args(args)


if __name__ == '__main__':
    opts = read_opt(sys.argv[1::])

    file_path = opts.input_obj
    output_path = opts.output_path
    is_ani = opts.animation if not output_path else True

    show_obj(file_path, turning=is_ani, output_path=output_path)
