#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""Simple demonstration of solving the heat equation in 2D using pyMOR's builtin discretizations.

Usage:
    parabolic.py [-h] [--help] [--fv] [--rect] [--grid=NI] DIFF NT

Arguments:
    DIFF         The diffusion constant

    NT           The number of time-steps.

Options:
    -h, --help   Show this message.

    --fv         Use finite volume discretization instead of finite elements.

    --rect       Use RectGrid instead of TriaGrid.

    --grid=NI              Use grid with 2^NI elements [default: 7].
"""

from __future__ import absolute_import, division, print_function

import math as m
from docopt import docopt
import numpy as np

from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.discretizers.parabolic import discretize_parabolic_cg, discretize_parabolic_fv
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid


def parabolic_demo(args):
    args['DIFF'] = float(args['DIFF'])
    args['NT'] = int(args['NT'])
    args['--grid'] = int(args['--grid'])

    n = 2**args['--grid']
    grid_name = '{1}(({0},{0}))'.format(n, 'RectGrid' if args['--rect'] else 'TriaGrid')
    print('Solving on {0}'.format(grid_name))

    print('Setup problem ...')
    domain = RectDomain(left=BoundaryType('neumann'), right=BoundaryType('neumann'))
    rhs = ConstantFunction(value=0, dim_domain=2)
    diffusion_functional = GenericParameterFunctional(mapping=lambda mu: mu['diffusion'],
                                                      parameter_type={'diffusion': 0})
    dirichlet = GenericFunction(lambda X: np.sin(np.pi*X[..., 0]), dim_domain=2)
    neumann = ConstantFunction(value=0, dim_domain=2)
    initial = GenericFunction(lambda X: np.cos(np.pi*X[..., 0])*np.sin(np.pi*X[..., 1]), dim_domain=2)

    problem = ParabolicProblem(domain=domain, rhs=rhs, diffusion_functionals=[diffusion_functional],
                                   dirichlet_data=dirichlet, neumann_data=neumann, initial_data=initial)

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=m.sqrt(2) / n, grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / n, grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=args['NT'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    mu = {'diffusion': args['DIFF']}
    print('Solving for diffusion = {} ... '.format(mu['diffusion']))
    U = discretization.solve(mu)

    print('Plot ...')
    discretization.visualize(U, title=grid_name)

    print('')


if __name__ == '__main__':
    args = docopt(__doc__)
    parabolic_demo(args)
