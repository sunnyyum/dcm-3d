import argparse
import sys
import textwrap
from graphic_handler.graphic_util import save_object, smooth_mesh, smooth_mesh_laplacian
from graphic_handler.graphic_generator import *

from dicom_handler import dicom_decoder, corr_finder
import inquirer


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
    parser = argparse.ArgumentParser(description='convert labeled dicom images to mesh',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest='input_dir',
                        help='input directory path that has labeled dicom files')
    parser.add_argument('-s', required=True, choices=['pc', 'img', 'vox', 'iv'], type=str, dest='src',
                        help=textwrap.dedent("""\
                       source for creating mesh
                       img : creating from DICOM image
                       pc  : creating from point cloud (using delaunay 3d algorithm)
                       vox : creating from voxel object
                       iv  : creating mesh using DICOM image and voxel object information\
                       """)
                        )
    parser.add_argument('--smo', required=False, choices=['lap', 'win'], type=str, dest='smooth',
                        help=textwrap.dedent("""\
                        algorithm for smoothing
                        lap : Laplacian algorithm
                        win : vtk's WindowedSincPolyDataFilter\
                        """)
                        )
    parser.add_argument('-o', required=True, default='./output.vtk', dest='output_path',
                        help='output file including file path')

    return parser.parse_args(args)


if __name__ == '__main__':
    opts = read_opt(sys.argv[1::])
    if opts.src != 'pc':
        questions = [
            inquirer.List('algorithm',
                          message="What algorithm do you want to use?",
                          choices=['marching cubes', 'discrete marching cubes', 'synchronized templates 3D',
                                   'flying edges'],
                          ),
        ]
        answer = inquirer.prompt(questions)

    # decode dicom files for converting a dicom file to a dicom image
    decoder = dicom_decoder.DicomDecoder(opts.input_dir)

    decoder.read_info()

    print('Snippet of imported files:')
    print(*decoder.get_files()[0:3], sep='\n')

    decoder.convert_dcm2img()
    decoder.calculate_hounsfield()

    corr = corr_finder.Correspondence(decoder.get_info())

    if opts.src == 'img':
        print('Retrieving DICOM image...', end=' ', flush=True)
        imgs = decoder.get_imgs()
        print('Done')

        print('Calculating boundary and voxel size...', end=' ', flush=True)
        edge_indices = decoder.collect_edge()
        edge_points = corr.convert_2d_3d(edge_indices)
        bounds = corr.find_boundary(edge_points)
        voxel_size = corr.calculate_voxel_size()
        print('Done')

        print('Converting into mesh...', end=' ', flush=True)
        mesh = convert_img2mesh(imgs, bounds, voxel_size, smooth=False)
        print('Done')

    else:
        print('Collecting label location...', end=' ', flush=True)
        indices = decoder.collect_label_loc()
        print('Done')

        print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
        points = corr.convert_2d_3d(indices)
        print('Done')

        if opts.src == 'pc':
            mesh = generate_pointcloud(points)
            mesh = convert_pcd2mesh(mesh)
        else:
            print('Converting 3d point to 3d voxel center point...', end=' ', flush=True)
            voxel_points, voxel_size = corr.calculate_voxel_center(points)
            print('Done')

            mesh = generate_voxel(voxel_points, voxel_size)

            print('Converting into mesh...', end=' ', flush=True)
            if opts.src == 'vox':
                mesh = convert_voxel2mesh(mesh, voxel_size)
            else:
                imgs = decoder.get_imgs()
                mesh = convert_voxel2mesh(mesh, voxel_size, combine_img=True, imgs=imgs)
            print('Done')

    if opts.smooth is not None:
        if opts.smooth == 'lap':
            mesh = smooth_mesh_laplacian(mesh)
        else:
            mesh = smooth_mesh(mesh)

    print(f'Saving the output to {opts.output_path}...', end=' ', flush=True)
    save_object(mesh, opts.output_path)
    print('Done')
