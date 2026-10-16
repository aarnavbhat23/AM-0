import unittest

import numpy as np

from assembly_memory.data import dataset, encode, data_hash
from assembly_memory.models import Assembly, SignedAssembly, make_model


class AssemblyTests(unittest.TestCase):
    def test_exact_reduction_through_births_and_evictions(self):
        rng = np.random.default_rng(202)
        reaction = Assembly(eta=0.1, capacity=10)
        coefficient = SignedAssembly(eta=0.1, capacity=10)
        worst = 0.0
        for t in range(1200):
            z = np.r_[1., rng.uniform(-1, 1, 7)]
            y = float(np.sin(t / 31) + 0.7 * z[1] * z[3])
            worst = max(worst, abs(reaction.predict(z) - coefficient.predict(z)))
            reaction.update(z, y)
            coefficient.update(z, y)
            np.testing.assert_array_equal(reaction.active, coefficient.active)
            np.testing.assert_allclose(reaction.plus - reaction.minus, coefficient.w, atol=1e-11)
            np.testing.assert_allclose(reaction.plus + reaction.minus, coefficient.m, atol=1e-11)
        self.assertLess(worst, 1e-11)
        self.assertGreater(reaction.evictions, 1)

    def test_conservation_under_large_error_and_steps(self):
        rng = np.random.default_rng(203)
        for rate in (0.01, 0.1, 10.0):
            model = Assembly(eta=rate, capacity=11)
            for _ in range(600):
                model.update(np.r_[1., rng.uniform(-1, 1, 7)], rng.normal() * 10)
                invariant = model.invariants()
                self.assertLess(invariant['max_mass_error'], 1e-10)
                self.assertGreaterEqual(invariant['min_amount'], -1e-12)
                self.assertLessEqual(invariant['active'], 11)
                self.assertTrue(np.all(model.plus[~model.active] == 0))
                self.assertTrue(np.all(model.minus[~model.active] == 0))

    def test_finite_step_matches_full_signed_formula_when_capped(self):
        rng = np.random.default_rng(221)
        a, b = Assembly(eta=10), SignedAssembly(eta=10)
        for _ in range(80):
            x, y = np.r_[1., rng.uniform(-1, 1, 7)], rng.normal() * 20
            a.update(x, y)
            b.update(x, y)
            np.testing.assert_allclose(a.plus - a.minus, b.w, atol=1e-10)
            np.testing.assert_allclose(a.plus + a.minus, b.m, atol=1e-10)

    def test_prediction_does_not_write_memory(self):
        a = Assembly()
        x = np.r_[1., np.full(7, 0.4)]
        a.update(x, 0.7)
        snapshot = (a.plus.copy(), a.minus.copy(), a.c.copy(), a.steps)
        for _ in range(10):
            a.predict(x)
        np.testing.assert_array_equal(a.plus, snapshot[0])
        np.testing.assert_array_equal(a.minus, snapshot[1])
        np.testing.assert_array_equal(a.c, snapshot[2])
        self.assertEqual(a.steps, snapshot[3])

    def test_learns_simple_unseen_mapping(self):
        rng = np.random.default_rng(204)
        a = Assembly(eta=0.03)
        x = np.c_[np.ones(1000), rng.uniform(-1, 1, (1000, 7))]
        y = 0.6 * x[:, 1] - 0.2 * x[:, 3]
        for z, target in zip(x, y):
            a.update(z, target)
        test = np.c_[np.ones(100), rng.uniform(-1, 1, (100, 7))]
        target = 0.6 * test[:, 1] - 0.2 * test[:, 3]
        err = np.mean((np.array([a.predict(z) for z in test]) - target) ** 2)
        self.assertLess(err, np.mean(target ** 2) * 0.2)

    def test_update_direction_without_eviction(self):
        a = Assembly(eta=0.03, fixed=True)
        rng = np.random.default_rng(205)
        for _ in range(200):
            x, y = np.r_[1., rng.uniform(-1, 1, 7)], rng.normal()
            before = a.predict(x)
            a.update(x, y)
            self.assertGreaterEqual((y - before) * (a.predict(x) - before), -1e-12)

    def test_footprint_ceiling(self):
        for name in ('assembly', 'assembly_fixed', 'assembly_no_pool', 'adaptive_lms',
                     'quadratic_lms', 'history_lms', 'current_lms'):
            m = make_model(name, 0.03)
            self.assertLess(m.state_bytes(), 32768)


class DataTests(unittest.TestCase):
    def test_no_future_response_in_encoder(self):
        rng = np.random.default_rng(206)
        u, y = rng.random((20, 2)), rng.uniform(-1, 1, 20)
        before = encode(u, y)
        changed = y.copy()
        changed[10:] += 0.5
        after = encode(u, changed)
        np.testing.assert_array_equal(before[:11], after[:11])
        self.assertFalse(np.array_equal(before[11:], after[11:]))

    def test_training_exclusion_and_distinct_splits(self):
        for family in ('adaptive', 'resource', 'delayed'):
            train = dataset(family, 207, 'train', 4)
            test = dataset(family, 207, 'test', 4)
            self.assertFalse(train['withheld'].any())
            self.assertNotEqual(data_hash(train), data_hash(test))
            np.testing.assert_array_equal(train['coefficients'], test['coefficients'])
            np.testing.assert_array_equal(train['x'], dataset(family, 207, 'train', 4)['x'])
            self.assertEqual(train['x'].shape, (4 * 44, 8))


if __name__ == '__main__':
    unittest.main()
