"""Explicit assembly dynamics and conventional diagnostic controls.

See docs/LEARNING_RULE.md for every update and docs/MATHEMATICAL_AUDIT.md
for the exact signed-coordinate representation. No autograd library is used.
"""
from itertools import combinations_with_replacement
import sys

import numpy as np


def catalogue(d=8):
    pairs = [(0, 0)] + [(0, i) for i in range(1, d)]
    pairs += list(combinations_with_replacement(range(1, d), 2))
    indices = np.array(pairs, dtype=np.int64)
    incidence = np.zeros((d, len(pairs)))
    incidence[0, 0] = 1
    for a, (i, j) in enumerate(pairs[1:], start=1):
        incidence[j, a] += 1
        if i != 0:
            incidence[i, a] += 1
    return indices, incidence


def object_bytes(obj, seen=None):
    """Owned Python state including array buffers; no duplicate references counted."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        # Owned arrays' getsizeof includes their data. Count a view's base once.
        return size + (object_bytes(obj.base, seen) if obj.base is not None else 0)
    if isinstance(obj, dict):
        return size + sum(object_bytes(k, seen) + object_bytes(v, seen)
                          for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return size + sum(object_bytes(x, seen) for x in obj)
    if hasattr(obj, '__dict__'):
        return size + object_bytes(vars(obj), seen)
    return size


class FeatureState:
    def __init__(self, eta=0.03, capacity=20, fixed=False):
        self.indices, self.N = catalogue()
        self.d, self.F = self.N.shape
        self.eta, self.capacity, self.fixed = float(eta), capacity, fixed
        self.active = np.zeros(self.F, dtype=bool)
        self.active[:self.d] = True
        if fixed:
            self.active[np.linspace(self.d, self.F - 1,
                                    capacity - self.d, dtype=int)] = True
        self.scores = np.zeros(self.F)
        self.steps = 0
        self.births = 0
        self.evictions = 0

    def phi(self, z):
        return z[self.indices[:, 0]] * z[self.indices[:, 1]]

    def propose(self, phi, residual):
        self.steps += 1
        self.scores *= 0.97
        self.scores += 0.03 * np.clip(residual, -2, 2) * phi
        if self.fixed or self.steps % 16:
            return None
        inactive = np.flatnonzero(~self.active)
        if inactive.size == 0:
            return None
        j = int(inactive[np.argmax(np.abs(self.scores[inactive]))])
        if abs(self.scores[j]) <= 0.005:
            return None
        old = None
        if np.count_nonzero(self.active) >= self.capacity:
            removable = np.flatnonzero(self.active & (np.arange(self.F) >= self.d))
            if removable.size == 0:
                return None
            old = int(removable[np.argmin(np.abs(self.scores[removable]))])
            if abs(self.scores[j]) <= 1.1 * abs(self.scores[old]) + 0.005:
                return None
            self.active[old] = False
            self.evictions += 1
        self.active[j] = True
        self.births += 1
        return old

    def state_bytes(self):
        # Four streaming trace scalars plus previous response are shared encoder state.
        return object_bytes(self) + 5 * 8


class Assembly(FeatureState):
    def __init__(self, eta=0.03, capacity=20, fixed=False, pool=True):
        super().__init__(eta, capacity, fixed)
        self.pool = pool
        self.M = np.full(self.d, 4.0)
        self.c = self.M.copy()
        self.plus = np.zeros(self.F)
        self.minus = np.zeros(self.F)

    def predict(self, z):
        return float((self.plus - self.minus) @ self.phi(z))

    def update(self, z, y):
        phi = self.phi(z)
        residual = float(y - (self.plus - self.minus) @ phi)
        old = self.propose(phi, residual)
        if old is not None:
            if self.pool:
                self.c += self.N[:, old] * (self.plus[old] + self.minus[old])
            self.plus[old] = self.minus[old] = 0.0
        h = self.eta * np.clip(residual, -2, 2) * phi * self.active
        magnitude = np.abs(h)
        dissolve = np.minimum(magnitude, 1.0) * np.where(h >= 0, self.minus, self.plus)
        self.minus -= np.where(h >= 0, dissolve, 0.0)
        self.plus -= np.where(h < 0, dissolve, 0.0)
        if self.pool:
            self.c += self.N @ dissolve
            # Only tiny round-off negatives are admissible; do not hide depletion bugs.
            if np.min(self.c) < -1e-10:
                raise ArithmeticError('negative free component amount')
            fractions = np.where(self.c / self.M > 1e-12, self.c / self.M, 0.0)
            availability = np.prod(fractions[:, None] ** self.N, axis=0)
        else:
            availability = np.ones(self.F)
        growth = magnitude * availability
        demand = self.N @ growth
        alpha = 1.0
        if self.pool and np.any(demand > 0):
            alpha = min(1.0, float(np.min(self.c[demand > 0] / demand[demand > 0])))
            if alpha < 1.0:
                alpha = max(0.0, alpha) * (1 - 1e-14)
        growth *= alpha
        self.plus += np.where(h >= 0, growth, 0.0)
        self.minus += np.where(h < 0, growth, 0.0)
        if self.pool:
            self.c -= self.N @ growth
        return residual

    def invariants(self):
        if not self.pool:
            return {'conserved': False}
        return {
            'conserved': True,
            'max_mass_error': float(np.max(np.abs(self.c + self.N @ (self.plus + self.minus) - self.M))),
            'min_amount': float(min(np.min(self.c), np.min(self.plus), np.min(self.minus))),
            'active': int(np.count_nonzero(self.active)),
            'births': self.births, 'evictions': self.evictions,
        }


class SignedAssembly(FeatureState):
    """Independent coefficient/occupancy implementation of the exact reduction.

The same feature-admission policy is intentionally shared; the concentration
update is not. This is a correctness witness, not a predictive comparator.
"""
    def __init__(self, eta=0.03, capacity=20):
        super().__init__(eta, capacity)
        self.M = np.full(self.d, 4.0)
        self.w = np.zeros(self.F)
        self.m = np.zeros(self.F)

    def predict(self, z):
        return float(self.w @ self.phi(z))

    def update(self, z, y):
        phi = self.phi(z)
        residual = float(y - self.w @ phi)
        old = self.propose(phi, residual)
        if old is not None:
            self.w[old] = self.m[old] = 0.0
        h = self.eta * np.clip(residual, -2, 2) * phi * self.active
        direction = np.sign(h)
        removed = np.minimum(np.abs(h), 1) * (self.m - direction * self.w) / 2
        self.w += direction * removed
        self.m -= removed
        free = self.M - self.N @ self.m
        fractions = np.where(free / self.M > 1e-12, free / self.M, 0.0)
        g = np.prod(fractions[:, None] ** self.N, axis=0)
        added = np.abs(h) * g
        demand = self.N @ added
        nonzero = demand > 0
        alpha = min(1.0, float(np.min(free[nonzero] / demand[nonzero]))) if np.any(nonzero) else 1.0
        if alpha < 1:
            alpha = max(0.0, alpha) * (1 - 1e-14)
        self.w += direction * alpha * added
        self.m += alpha * added
        return residual


class LMS(FeatureState):
    def __init__(self, eta=0.03, kind='quadratic'):
        super().__init__(eta, fixed=kind != 'adaptive')
        self.kind = kind
        self.w = np.zeros(self.F)
        if kind == 'quadratic':
            self.active[:] = True
        elif kind == 'history':
            self.active[:] = False
            self.active[:self.d] = True
        elif kind == 'current':
            self.active[:] = False
            self.active[:3] = True

    def predict(self, z):
        return float(self.w @ self.phi(z))

    def update(self, z, y):
        phi = self.phi(z)
        residual = float(y - self.w @ phi)
        old = self.propose(phi, residual)
        if old is not None:
            self.w[old] = 0.0
        phi *= self.active
        self.w += self.eta * np.clip(residual, -2, 2) * phi / (1e-6 + phi @ phi)
        return residual


def make_model(name, eta):
    if name in ('assembly', 'assembly_shuffled', 'assembly_untrained'):
        return Assembly(eta)
    if name == 'assembly_fixed':
        return Assembly(eta, fixed=True)
    if name == 'assembly_no_pool':
        return Assembly(eta, pool=False)
    return LMS(eta, kind=name.removesuffix('_lms'))
