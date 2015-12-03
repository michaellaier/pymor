#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""Parabolic problem with POD or greedy demo.

Usage:
  parabolic_mor.py [-hp] [--grid=NI] [--fv] [--rect] [--help] [--plot-err] [--plot-solutions] [--greedy]
                   [--pod-norm=NORM] [--test=COUNT] NT SNAPSHOTS RBSIZE


Arguments:
  NT         The number of time-steps.

  SNAPSHOTS  Number of snapshots for basis generation.

  RBSIZE     Size of the reduced basis


Options:
  --grid=NI              Use grid with 2^NI elements [default: 7].

  --fv                   Use finite volume discretization instead of finite elements.

  --rect                 Use RectGrid instead of TriaGrid.

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --greedy               Use greedy algorithm for basis generation instead of POD.

  --pod-norm=NORM        Norm (trivial, h1) w.r.t. which to calculate the POD
                         [default: h1].

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
"""

from __future__ import absolute_import, division, print_function

import sys
import time

import math as m
import numpy as np
from docopt import docopt

from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.functions.basic import ConstantFunction, GenericFunction
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.parameters.spaces import CubicParameterSpace
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.discretizers.parabolic import discretize_parabolic_cg, discretize_parabolic_fv
from pymor.algorithms.greedy import greedy
from pymor.algorithms.pod import pod
from pymor.reductors.basic import reduce_generic_rb


def parabolic_mor_demo(args):
    args['--grid'] = int(args['--grid'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--pod-norm'] = args['--pod-norm'].lower()
    assert args['--pod-norm'] in {'trivial', 'h1'}

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
                               dirichlet_data=dirichlet, neumann_data=neumann, initial_data=initial,
                               parameter_space=CubicParameterSpace({'diffusion': 0}, minimum=0, maximum=1))

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=m.sqrt(2) / n, grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / n, grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=int(args['NT']))

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        discretization.visualize(Us, legend=legend, title='Detailed Solutions for different parameters', block=True)

    print('RB generation ...')

    tic = time.time()

    if args['--greedy']:
        greedy_data = greedy(discretization, reduce_generic_rb,
                             discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']), use_estimator=False,
                             error_norm=discretization.h1_norm if args['--pod-norm'] == 'h1' else None,
                             max_extensions=args['RBSIZE'])

        rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']
    else:
        print('Solving on training set ...')
        S_train = list(discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']))
        snapshots = discretization.operator.source.empty(reserve=len(S_train))
        for mu in S_train:
            snapshots.append(discretization.solve(mu))

        print('Performing POD ...')
        pod_product = discretization.h1_0_semi_product if args['--pod-norm'] == 'h1' else None
        rb = pod(snapshots, modes=args['RBSIZE'], product=pod_product)[0]

        print('Reducing ...')
        reductor = reduce_generic_rb
        rb_discretization, reconstructor, _ = reductor(discretization, rb)

    toc = time.time()
    t_offline = toc - tic

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    h1_err_max = -1
    cond_max = -1
    for mu in discretization.parameter_space.sample_randomly(args['--test']):
        print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
        URB = reconstructor.reconstruct(rb_discretization.solve(mu))
        U = discretization.solve(mu)
        h1_err = discretization.h1_0_semi_norm(U - URB)[0]
        cond = np.linalg.cond(rb_discretization.operator.assemble(mu)._matrix)
        if h1_err > h1_err_max:
            h1_err_max = h1_err
            mumax = mu
        if cond > cond_max:
            cond_max = cond
            cond_max_mu = mu
        print('H1-error = {}, condition = {}'.format(h1_err, cond))
    toc = time.time()
    t_est = toc - tic
    real_rb_size = len(greedy_data['basis']) if args['--greedy'] else len(rb)

    print('''
    *** RESULTS ***

    Problem:
       grid:                               {grid_name}
       h:                                  sqrt(2)/{n}

    POD/Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}
       pod/error norm:                     {args[--pod-norm]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {t_offline}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-err']:
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    parabolic_mor_demo(args)
