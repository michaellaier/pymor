from __future__ import absolute_import, division, print_function

import tempfile
import collections
import os
import sys

from pymor.domaindescriptions.basic import PolygonalDomain
from pymor.playground.grids.gmsh import GmshBoundaryInfo
from pymor.playground.grids.gmsh import GmshGrid


def discretize_Gmsh(domain_description=None, geo_file=None, geo_file_path=None, msh_file_path=None,
                    mesh_algorithm='meshadapt', clscale=1., clmin=0.1, clmax=0.2, options=''):
    """Discretize a |DomainDescription| of a |PolygonalDomain| or a already existing Gmsh GEO-file using the Gmsh
    Mesh module.

    Parameters
    ----------
    domain_description
        A |DomainDescription| of the |PolygonalDomain| to discretize. Has to be None when geo_file is not None.
    geo_file
        File handle of the Gmsh Geo-file to discretize. Has to be None when domain_description is not None.
    geo_file_path
        Path to the Gmsh GEO-file. When discretizing a |PolygonalDomain| and geo_file_path is None a temporary file will
        be created. If geo_file is specified this is ignored and the path to geo_file will be used.
    msh_file_path
        Path to the Gmsh GEO-file. If None a temporary file will be created.
    mesh_algorithm
        The algorithm used to mesh the domain (meshadapt, del2d, front2d, delquad, del3d, front3d, mmg3d, pack).
    clscale
        Mesh element size scaling factor.
    clmin
        Minimum mesh element size.
    clmax
        Maximum mesh element size.
    options
        Other options to control the meshing procedure of Gmsh. See
        http://geuz.org/gmsh/doc/texinfo/gmsh.html#Command_002dline-options for all available options.
    Returns
    -------
    grid
        The generated |GmshGrid|.
    boundary_info
        The generated |GmshBoundaryInfo|.
    """
    assert domain_description is None or geo_file is None

    try:
        # When a |PolygonalDomain| has to be discretized create a Gmsh GE0-file and write all data.
        if domain_description is not None:
            assert isinstance(domain_description, PolygonalDomain)
            # Create a temporary GEO-file if None is specified
            if geo_file_path is None:
                geo_file = tempfile.NamedTemporaryFile(delete=False, suffix='.geo')
                geo_file_path = geo_file.name
            else:
                geo_file = open(geo_file_path, 'w')

            # combine points and holes, since holes are points, too, and have to be stored as such.
            points = [domain_description.points]
            points.extend(domain_description.holes)
            # assign ids to all points and write them to the GEO-file.
            for id, p in enumerate([p for ps in points for p in ps]):
                assert len(p) == 2
                geo_file.write('Point('+str(id+1)+') = '+str(p+[0, 0]).replace('[', '{').replace(']', '}')+';\n')

            # store points and there ids
            point_ids = dict(zip([str(p) for ps in points for p in ps], range(1, len([p for ps in points for p in ps])+1)))
            # shift points 1 entry to the left.
            points_deque = [collections.deque(ps) for ps in points]
            for ps_d in points_deque:
                ps_d.rotate(-1)
            # create lines by connecting the points with shifted points, such that they form a polygonal chains.
            lines = [[point_ids[str(p0)], point_ids[str(p1)]] for ps, ps_d in zip(points, points_deque) for p0, p1 in zip(ps, ps_d)]
            # assign ids to all lines and write them to the GEO-file.
            for l_id, l in enumerate(lines):
                    geo_file.write('Line('+str(l_id+1)+')'+' = '+str(l).replace('[', '{').replace(']', '}')+';\n')

            # form line_loops (polygonal chains), create ids and write them to file.
            line_loops = [[point_ids[str(p)] for p in ps] for ps in points]
            line_loop_ids = range(len(lines)+1, len(lines)+len(line_loops)+1)
            for ll_id, ll in zip(line_loop_ids, line_loops):
                geo_file.write('Line Loop('+str(ll_id)+')'+' = '+str(ll).replace('[', '{').replace(']', '}')+';\n')

            #create the surface defined by line loops, starting with the exterior and then the holes.
            geo_file.write('Plane Surface('+str(line_loop_ids[0]+1)+')'+' = '+str(line_loop_ids).replace('[', '{').replace(']', '}')+';\n')
            geo_file.write('Physical Surface("boundary") = {'+str(line_loop_ids[0]+1)+'};\n')

            # write boundaries.
            for boundary_type, bs in domain_description.boundary_types.iteritems():
                geo_file.write('Physical Line'+'("'+str(boundary_type)+'")'+' = '+str([l_id for l_id in bs]).replace('[', '{').replace(']', '}')+';\n')

            geo_file.close()
        # When a GEO-File is provided just get the corresponding file path.
        else:
            geo_file_path = geo_file.name
        # Create a temporary MSH-file if no path is specified.
        if msh_file_path is None:
            msh_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msh')
            msh_file_path = msh_file.name
            msh_file.close()

        # Run Gmsh
        try:
            from subprocess import PIPE, Popen, CalledProcessError
            gmsh = Popen(['gmsh', geo_file_path, '-2', '-algo', mesh_algorithm, '-clscale', str(clscale), '-clmin',
                          str(clmin), '-clmax', str(clmax), options, '-o', msh_file_path], stdout=PIPE, stderr=PIPE)
            out, err = gmsh.communicate()
            print(out)
            if gmsh.returncode != 0:
                print(err)
        except (OSError, ValueError) as e:
            print('Gmsh encountered an error: {}'.format(e))
        except:
            print('Gmsh encountered an unexpected error: {}'.format(sys.exc_info()[0]))

        # Create |GmshGrid| and |GmshBoundaryInfo| form the just created MSH-file.
        grid = GmshGrid(open(msh_file_path))
        bi = GmshBoundaryInfo(grid, open(msh_file_path))
    finally:
        # delete tempfiles if they were created beforehand.
        if isinstance(geo_file, tempfile._TemporaryFileWrapper):
            os.remove(geo_file_path)
        if isinstance(msh_file, tempfile._TemporaryFileWrapper):
            os.remove(msh_file_path)

    return grid, bi
