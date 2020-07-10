import argparse
import sys
import textwrap

from dicom_handler import dicom_decoder, corr_finder
from graphic_handler.graphic_util import save_object
from graphic_handler.graphic_generator import *


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
    parse = argparse.ArgumentParser(description='convert labeled dicom images to 3D model',
                                    formatter_class=argparse.RawTextHelpFormatter)

    parse.add_argument('-i', required=True, dest='input_dir',
                       help='input directory path that has labeled dicom files')

    parse.add_argument('--pc', required=False, default=False, dest='is_pc', action='store_true',
                       help='flag for creating point cloud')

    parse.add_argument('--voxel', required=False, default=False, dest='is_voxel', action='store_true',
                       help='flag for creating voxel object')

    parse.add_argument('--mesh', required=False, default=False, dest='is_mesh', action='store_true',
                       help='flag for creating mesh')

    parse.add_argument('-s', required=False, default='img', dest='src',
                       help=textwrap.dedent("""\
                       source for creating mesh
                       img : creating from DICOM image
                       pc  : creating from point cloud (using delaunay 3d algorithm)
                       vox : creating from voxel object
                       iv  : creating mesh using DICOM image and voxel object information\
                       """))

    parse.add_argument('-a', required=False, default='d', dest='algo',
                       help=textwrap.dedent("""\
                       algorithms for creating mesh from volume data (voxel object or DICOM)
                       m  : marching cubes
                       dm : discrete marching cubes
                       st : synchronized templates 3D
                       f  : flying edges\
                       """))

    parse.add_argument('-o', required=True, default='./output.vtk', dest='output_path',
                       help='output file including file path')

    return parse.parse_args(args)


def print_files(file_li: list, interval: tuple, sep='\n') -> None:
    print(*file_li[interval[0]:interval[1]], sep=sep)


if __name__ == '__main__':
    # read arguments
    opts = read_opt(sys.argv[1::])
    dir_path = opts.input_dir

    # decode dicom files for converting a dicom file to a dicom image
    dcmDecoder = dicom_decoder.DicomDecoder(dir_path)

    print('Snippet of imported files')
    print_files(dcmDecoder.get_files(), (0, 5))

    dcmDecoder.read_info()
    dcmDecoder.convert_dcm2img()
    dcmDecoder.calculate_hounsfield()

    if not (opts.is_pc and opts.is_voxel and opts.is_mesh):
        assert 'Please choose 3D representation type'
    else:
        corr = corr_finder.Correspondence(dcmDecoder.get_info())
        mesh = None
        if opts.is_pc:
            print('Collecting label location...', end=' ', flush=True)
            indices = dcmDecoder.collect_label_loc()
            print('Done')

            print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
            points = corr.convert_2d_3d(indices)
            print('Done')

            mesh = generate_pointcloud(points)
        elif opts.is_voxel:
            print('Collecting label location...', end=' ', flush=True)
            indices = dcmDecoder.collect_label_loc()
            print('Done')

            print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
            points = corr.convert_2d_3d(indices)
            print('Done')

            print('Converting 3d point to 3d voxel center point...', end=' ', flush=True)
            voxel_points, voxel_size = corr.calculate_voxel_center(points)
            print('Done')
            mesh = generate_voxel(voxel_points, voxel_size)
        elif opts.is_mesh:
            if opts.src == 'img':
                imgs = dcmDecoder.get_imgs()
                edge_indices = dcmDecoder.collect_edge()

                edge_points = corr.convert_2d_3d(edge_indices)
                bounds = corr.find_boundary(edge_points)
                voxel_size = corr.calculate_voxel_size()

                mesh = convert_img2mesh(imgs, bounds, voxel_size, smooth=False)
            elif opts.src == 'pc':
                print('Collecting label location...', end=' ', flush=True)
                indices = dcmDecoder.collect_label_loc()
                print('Done')

                print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
                points = corr.convert_2d_3d(indices)
                print('Done')

                mesh = generate_pointcloud(points)
                mesh = convert_pcd2mesh(mesh)
            elif opts.src == 'vox':
                print('Collecting label location...', end=' ', flush=True)
                indices = dcmDecoder.collect_label_loc()
                print('Done')

                print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
                points = corr.convert_2d_3d(indices)
                print('Done')

                print('Converting 3d point to 3d voxel center point...', end=' ', flush=True)
                voxel_points, voxel_size = corr.calculate_voxel_center(points)
                print('Done')
                mesh = generate_voxel(voxel_points, voxel_size)
                mesh = convert_voxel2mesh(mesh, voxel_size)
            elif opts.src == 'iv':
                print('Collecting label location...', end=' ', flush=True)
                indices = dcmDecoder.collect_label_loc()
                print('Done')

                print('Converting 2d coordinate to 3d coordinate...', end=' ', flush=True)
                points = corr.convert_2d_3d(indices)
                print('Done')

                print('Converting 3d point to 3d voxel center point...', end=' ', flush=True)
                voxel_points, voxel_size = corr.calculate_voxel_center(points)
                print('Done')
                mesh = generate_voxel(voxel_points, voxel_size)
                imgs = dcmDecoder.get_imgs()
                mesh = convert_voxel2mesh(mesh, voxel_size, combine_img=True, imgs=imgs)

        if not mesh:
            print(f'Saving the output to {opts.output_path}...', end=' ', flush=True)
            save_object(mesh, opts.output_path)
            print('Done')
        else:
            assert 'Mesh is None'
