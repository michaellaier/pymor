#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""Parabolic problem with greedy basis generation demo.

Usage:
  parabolic_mor.py [-hp] [--help] [--grid=NI] [--nt=COUNT] [--fv] [--rect] [--estimator-norm=NORM] [--without-estimator]
                   [--test=COUNT] [--plot-err] [--plot-solutions] [--plot-error-sequence] SNAPSHOTS RBSIZE


Arguments:
  SNAPSHOTS  Number of snapshots for basis generation.

  RBSIZE     Size of the reduced basis


Options:
  -h, --help             Show this message.

  --grid=NI              Use grid with 2^NI elements [default: 7].

  --nt=COUNT             Number of time steps [default: 100].

  --fv                   Use finite volume discretization instead of finite elements.

  --rect                 Use RectGrid instead of TriaGrid.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: h1].

  --without-estimator    Do not use error estimator for basis generation.

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --plot-error-sequence  Plot reduction error vs. basis size.
"""

from __future__ import absolute_import, division, print_function

import sys
import time
from functools import partial

import math as m
import numpy as np
import matplotlib.pyplot as plt
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
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.linear import reduce_instationary_affine_linear
from pymor.algorithms.basisextension import pod_basis_extension
from pymor.algorithms.greedy import greedy
from pymor.reductors.basic import reduce_to_subbasis


def parabolic_mor_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--nt'] = int(args['--nt'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--test'] = int(args['--test'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])

    n = 2**args['--grid']
    grid_name = '{1}(({0},{0}))'.format(n, 'RectGrid' if args['--rect'] else 'TriaGrid')
    print('Solving on {0}'.format(grid_name))

    print('Setup problem ...')
    domain = RectDomain(top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))
    rhs = ConstantFunction(value=0, dim_domain=2)
    diffusion_functional = GenericParameterFunctional(mapping=lambda mu: mu['diffusion'],
                                                      parameter_type={'diffusion': 0})
    dirichlet = GenericFunction(lambda X: np.cos(np.pi*X[..., 0])*np.sin(np.pi*X[..., 1]), dim_domain=2)
    neumann = ConstantFunction(value=1, dim_domain=2)
    initial = GenericFunction(lambda X: np.cos(np.pi*X[..., 0])*np.sin(np.pi*X[..., 1]), dim_domain=2)

    problem = ParabolicProblem(domain=domain, rhs=rhs, diffusion_functionals=[diffusion_functional],
                               dirichlet_data=dirichlet, neumann_data=neumann, initial_data=initial,
                               parameter_space=CubicParameterSpace({'diffusion': 0}, minimum=0.1, maximum=1))

    print('Discretize ...')
    if args['--rect']:
        grid, bi = discretize_domain_default(problem.domain, diameter=m.sqrt(2) / n, grid_type=RectGrid)
    else:
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / n, grid_type=TriaGrid)
    discretizer = discretize_parabolic_fv if args['--fv'] else discretize_parabolic_cg
    discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi, nt=args['--nt'])

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

    error_product = discretization.h1_0_semi_product if args['--estimator-norm'] == 'h1' else None
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', discretization.parameter_type)
    reductor = partial(reduce_instationary_affine_linear, error_product=error_product,
                       coercivity_estimator=coercivity_estimator)
    extension_algorithm = pod_basis_extension
    initial_basis = None # discretization.initial_data.as_vector()

    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         initial_basis=initial_basis, use_estimator=not args['--without-estimator'],
                         error_norm=lambda U: np.max(discretization.h1_0_semi_norm(U)),
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])

    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    print('\nSearching for maximum error on random snapshots ...')

    def error_analysis(d, rd, rc, mus):
        print('N = {}: '.format(rd.operator.source.dim), end='')
        h1_err_max = -1
        h1_est_max = -1
        cond_max = -1
        for mu in mus:
            print('.', end='')
            sys.stdout.flush()
            u = rd.solve(mu)
            URB = rc.reconstruct(u)
            U = d.solve(mu)
            h1_err = np.max(d.h1_0_semi_norm(U - URB))
            h1_est = rd.estimate(u, mu=mu)
            cond = np.linalg.cond(rd.operator.assemble(mu)._matrix)
            if h1_err > h1_err_max:
                h1_err_max = h1_err
                mumax = mu
            if h1_est > h1_est_max:
                h1_est_max = h1_est
                mu_est_max = mu
            if cond > cond_max:
                cond_max = cond
                cond_max_mu = mu
        print()
        return h1_err_max, mumax, h1_est_max, mu_est_max, cond_max, cond_max_mu

    tic = time.time()

    real_rb_size = len(greedy_data['basis'])
    if args['--plot-error-sequence']:
        N_count = min(real_rb_size - 1, 25)
        Ns = np.linspace(1, real_rb_size, N_count).astype(np.int)
    else:
        Ns = np.array([real_rb_size])
    rd_rcs = [reduce_to_subbasis(rb_discretization, N, reconstructor)[:2] for N in Ns]
    mus = list(discretization.parameter_space.sample_randomly(args['--test']))

    errs, err_mus, ests, est_mus, conds, cond_mus = zip(*(error_analysis(discretization, rd, rc, mus)
                                                        for rd, rc in rd_rcs))
    h1_err_max = errs[-1]
    mumax = err_mus[-1]
    cond_max = conds[-1]
    cond_max_mu = cond_mus[-1]
    toc = time.time()
    t_est = toc - tic

    print('''
    *** RESULTS ***

    Problem:
       grid:                               {grid_name}
       h:                                  sqrt(2)/{n}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}
       estimator disabled:                 {args[--without-estimator]}
       estimator norm:                     {args[--estimator-norm]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()

    if args['--plot-error-sequence']:
        plt.semilogy(Ns, errs, Ns, ests)
        plt.legend(('error', 'estimator'))
        plt.show()
    if args['--plot-err']:
        U = discretization.solve(mumax)
        URB = reconstructor.reconstruct(rb_discretization.solve(mumax))
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True, block=True)



if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    parabolic_mor_demo(args)
