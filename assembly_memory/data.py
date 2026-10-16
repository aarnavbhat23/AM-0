"""Three abstract, benign history-dependent response generators.

No biological parameters or sequences are used. These equations are chosen
for a diagnostic and cannot validate prediction on real cellular data.
"""
import hashlib
import numpy as np

FAMILIES = ('adaptive', 'resource', 'delayed')


def system_coefficients(family, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family), 812]))
    return rng.uniform(0.45, 1.3, 3)


def sequence(family, coefficients, rng, length, restricted):
    k = coefficients
    inputs = np.zeros((length, 2))
    response = np.zeros(length)
    adapt, resource, prev = 0.0, 1.0, 0.0
    remaining, u = 0, np.zeros(2)
    for t in range(length):
        if remaining == 0:
            u = rng.uniform(0, 1, 2)
            while restricted and np.all(u > 0.65):
                u = rng.uniform(0, 1, 2)
            remaining = int(rng.integers(3, 9))
        remaining -= 1
        inputs[t] = u
        if family == 'adaptive':
            adapt = 0.92 * adapt + 0.08 * u[0]
            drive = np.tanh(k[0] * u[0] / (0.25 + 1.5 * adapt)
                            - k[1] * u[1] + k[2] * (u[0] - 0.5) * (u[1] - 0.5))
            current = 0.65 * prev + 0.35 * drive
        elif family == 'resource':
            resource += 0.06 * (1 - resource) - 0.12 * resource * u.sum()
            drive = np.tanh(2 * k[0] * resource * u[0] - 1.4 * k[1] * resource * u[1]
                            + 0.2 * k[2] * (u[0] - u[1]))
            current = 0.7 * prev + 0.3 * drive
        elif family == 'delayed':
            lag = response[t - 3] if t >= 3 else 0.0
            drive = np.tanh(k[0] * (2 * u[0] - 1) - k[1] * (2 * u[1] - 1) - 0.9 * k[2] * lag)
            current = 0.6 * prev + 0.4 * drive
        else:
            raise ValueError(family)
        response[t], prev = current, current
    return inputs, response


def encode(inputs, response):
    fast, slow = np.zeros(2), np.zeros(2)
    features = np.zeros((len(inputs), 8))
    for t, u in enumerate(inputs):
        fast = 0.5 * fast + 0.5 * u
        slow = 0.9 * slow + 0.1 * u
        previous = response[t - 1] if t > 0 else 0.0
        features[t] = np.r_[1.0, 2 * u - 1, 2 * fast - 1, 2 * slow - 1,
                            np.clip(previous, -1, 1)]
    return features


def dataset(family, seed, split, count, length=48, warmup=4):
    split_id = {'train': 0, 'validation': 1, 'test': 2}[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family), split_id, 942]))
    coefficients = system_coefficients(family, seed)
    xs, ys, ids, held, raw_inputs, raw_outputs = [], [], [], [], [], []
    for i in range(count):
        u, y = sequence(family, coefficients, rng, length, restricted=split != 'test')
        xs.append(encode(u, y)[warmup:])
        ys.append(y[warmup:])
        ids.extend([i] * (length - warmup))
        held.extend(np.all(u[warmup:] > 0.65, axis=1))
        raw_inputs.append(u)
        raw_outputs.append(y)
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), sequence=np.array(ids),
                withheld=np.array(held), inputs=np.array(raw_inputs), outputs=np.array(raw_outputs),
                coefficients=coefficients)


def data_hash(data):
    digest = hashlib.sha256()
    for name in sorted(data):
        array = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
