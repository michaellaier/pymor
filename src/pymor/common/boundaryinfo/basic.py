from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .interfaces import BoundaryInfoInterface
from pymor.common import BoundaryType


class FromIndicators(BoundaryInfoInterface):

    def __init__(self, grid, indicators):
        self.grid = grid
        self.boundary_types = indicators.keys()
        self._masks = {boundary_type:[np.zeros(grid.size(codim), dtype='bool') for codim in xrange(1, grid.dim + 1)]
                       for boundary_type in self.boundary_types}
        for boundary_type, codims in self._masks.iteritems():
            for c, mask in enumerate(codims):
                mask[grid.boundaries(c + 1)] = indicators[boundary_type](grid.centers(c + 1)[grid.boundaries(c + 1)])

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        return self._masks[boundary_type][codim - 1]


class AllDirichlet(BoundaryInfoInterface):

    def __init__(self, grid):
        self.grid = grid
        self.boundary_types = set((BoundaryType('dirichlet'),))

    def mask(self, boundary_type, codim):
        assert boundary_type == BoundaryType('dirichlet'), ValueError('Has no boundary_type "{}"'.format(boundary_type))
        assert 1 <= codim <= self.grid.dim
        return np.ones(self.grid.size(codim), dtype='bool') * self.grid.boundary_mask(codim)
