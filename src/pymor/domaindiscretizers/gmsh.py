from __future__ import absolute_import, division, print_function

import tempfile
import collections
import os

from pymor.domaindescriptions.basic import PolygonalDomain
from pymor.playground.grids.gmsh import GmshBoundaryInfo
from pymor.playground.grids.gmsh import GmshGrid


def discretize_Gmsh(domain_description=None, geo_file=None, geo_file_path=None, msh_file_path=None,
                    mesh_algorithm='meshadapt', clscale=1., clmin=0.1, clmax=0.2, options=''):
    assert domain_description is None or geo_file is None

    try:
        if domain_description is not None:
            assert isinstance(domain_description, PolygonalDomain)
            if geo_file_path is None:
                f = tempfile.NamedTemporaryFile(delete=False, suffix='.geo')
                geo_file_path = f.name
            else:
                f = open(geo_file_path, 'w')

            points = [domain_description.points]
            points.extend(domain_description.holes)
            for id, p in enumerate([p for ps in points for p in ps]):
                assert len(p) == 2
                f.write('Point('+str(id+1)+') = '+str(p+[0, 0]).replace('[', '{').replace(']', '}')+';\n')

            point_ids = dict(zip([str(p) for ps in points for p in ps], range(1, len([p for ps in points for p in ps])+1)))
            points_deque = [collections.deque(ps) for ps in points]
            for ps_d in points_deque:
                ps_d.rotate(-1)
            lines = [[point_ids[str(p0)], point_ids[str(p1)]] for ps, ps_d in zip(points, points_deque) for p0, p1 in zip(ps, ps_d)]
            for l_id, l in enumerate(lines):
                    f.write('Line('+str(l_id+1)+')'+' = '+str(l).replace('[', '{').replace(']', '}')+';\n')

            line_loops = [[point_ids[str(p)] for p in ps] for ps in points]
            line_loop_ids = range(len(lines)+1, len(lines)+len(line_loops)+1)
            for ll_id, ll in zip(line_loop_ids, line_loops):
                f.write('Line Loop('+str(ll_id)+')'+' = '+str(ll).replace('[', '{').replace(']', '}')+';\n')

            line_loop_ids.reverse()
            f.write('Plane Surface('+str(line_loop_ids[0]+1)+')'+' = '+str(line_loop_ids).replace('[', '{').replace(']', '}')+';\n')
            f.write('Physical Surface("boundary") = {'+str(line_loop_ids[0]+1)+'};\n')

            for boundary_type, bs in domain_description.boundary_types.iteritems():
                f.write('Physical Line'+'("'+str(boundary_type)+'")'+' = '+str([l_id for l_id in bs]).replace('[', '{').replace(']', '}')+';\n')

            f.close()
        else:
            geo_file_path = geo_file
        if msh_file_path is None:
            msh_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msh')
            msh_file_path = msh_file.name
            msh_file.close()

        from subprocess import PIPE, Popen
        gmsh = Popen(['gmsh', geo_file_path, '-2', '-algo', mesh_algorithm, '-clscale', str(clscale), '-clmin', str(clmin),
                      '-clmax', str(clmax), options, '-o', msh_file_path], stdout=PIPE)
        gmsh.wait()
        print(gmsh.stdout.read())

        grid = GmshGrid(open(msh_file_path))
        bi = GmshBoundaryInfo(grid, open(msh_file_path))
    finally:
        if isinstance(f, tempfile._TemporaryFileWrapper):
            os.remove(geo_file_path)
        if isinstance(msh_file, tempfile._TemporaryFileWrapper):
            os.remove(msh_file_path)

    return grid, bi
