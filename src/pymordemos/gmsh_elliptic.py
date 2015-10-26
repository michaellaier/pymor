#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple demonstration of solving the Poisson equation in 2D on a non-convex domain using gmsh for meshing.

Usage:
    gmsh_elliptic.py [--fv] ANGLE

Arguments:
    ANGLE        The angle of the corner.

Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.
"""

from __future__ import absolute_import, division, print_function

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.domaindescriptions.basic import PieDomain
from pymor.domaindiscretizers.gmsh import discretize_Gmsh
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.vectorarrays.numpy import NumpyVectorArray


def gmsh_elliptic_demo(args):
    args['ANGLE'] = float(args['ANGLE'])

    domain = PieDomain(args['ANGLE'], num_points=50)

    rhs = ConstantFunction(np.array(0.), dim_domain=2, name='rhs')
    def dirichlet(X):
        r = np.sqrt(np.power(X[..., 0], 2)+np.power(X[..., 1], 2))
        phi = np.zeros(X.shape[:-1])
        phi[np.all([X[..., 1] >= 0, r > 0], axis=0)] = np.arccos((X[..., 0] / r)[np.all([X[..., 1] >= 0, r > 0], axis=0)])
        phi[[np.all([X[..., 1] < 0, r > 0], axis=0)]] = 2*np.pi - np.arccos((X[..., 0] / r)[np.all([X[..., 1] < 0, r > 0], axis=0)])
        return np.sin(phi*np.pi/args['ANGLE'])
    dirichlet_data = GenericFunction(dirichlet, dim_domain=2, name='dirichlet')

    print('Setup problem ...')
    problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet_data)

    print('Discretize ...')
    grid, bi = discretize_Gmsh(domain_description=domain, clmin=0.01, clmax=0.05)
    discretizer = discretize_elliptic_fv if args['--fv'] else discretize_elliptic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi)

    print('Solve ...')
    U = discretization.solve()

    print('Plot ...')
    discretization.visualize(U)

    print('Comparing with reference solution ...')
    def ref_sol(X):
        r = np.sqrt(np.power(X[..., 0], 2)+np.power(X[..., 1], 2))
        phi = np.zeros(X.shape[:-1])
        phi[np.all([X[..., 1] >= 0, r > 0], axis=0)] = np.arccos((X[..., 0] / r)[np.all([X[..., 1] >= 0, r > 0], axis=0)])
        phi[[np.all([X[..., 1] < 0, r > 0], axis=0)]] = 2*np.pi - np.arccos((X[..., 0] / r)[np.all([X[..., 1] < 0, r > 0], axis=0)])
        return np.power(r, np.pi/args['ANGLE']) * np.sin(phi*np.pi/args['ANGLE'])
    solution = GenericFunction(ref_sol, 2)
    U_ref = NumpyVectorArray(solution(grid.centers(2)))
    discretization.visualize((U, U_ref, U-U_ref), separate_colorbars=True)

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    gmsh_elliptic_demo(args)
