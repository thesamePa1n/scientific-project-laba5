import time
import numpy as np

def get_variant(number):
    if number == 1:
        def function(x, y):
            return (
                np.exp(-0.5 * (x**2 + y**2))
                * np.cos(3 * x)
                * np.sin(2 * y)
            )

        def mask_condition(u):
            return u > 0.25

        bounds = (-2.0, 2.0, -2.0, 2.0)

        return function, bounds, mask_condition

    raise ValueError(f"Вариант {number} не поддерживается")

def make_coordinates(bounds, nx, ny):
    xmin, xmax, ymin, ymax = bounds

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    x = xmin + (np.arange(nx) + 0.5) * dx
    y = ymin + (np.arange(ny) + 0.5) * dy

    return x, y, dx, dy

def compute_loop(function, x, y):
    ny = len(y)
    nx = len(x)

    U = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            U[i, j] = function(x[j], y[i])

    return U

def compute_numpy(function, x, y):
    x = x[None, :]
    y = y[:, None]

    U = function(x, y)

    return U

def analyze_field(U, mask, dx, dy, x, y):
    min_index = np.unravel_index(np.argmin(U), U.shape)
    max_index = np.unravel_index(np.argmax(U), U.shape)

    mask_fraction = np.mean(mask)
    mask_area = np.sum(mask) * dx * dy

    return {
        "minimum": float(np.min(U)),
        "maximum": float(np.max(U)),
        "mean": float(np.mean(U)),
        "std": float(np.std(U)),
        "mask_fraction": float(mask_fraction),
        "mask_area": float(mask_area),
        "min_x": float(x[min_index[1]]),
        "min_y": float(y[min_index[0]]),
        "max_x": float(x[max_index[1]]),
        "max_y": float(y[max_index[0]])
    }

def benchmark(function, bounds, mask_condition, sizes, repeats=3):
    results = []

    for nx, ny in sizes:
        x, y, dx, dy = make_coordinates(bounds, nx, ny)

        loop_times = []
        numpy_times = []

        for _ in range(repeats):
            start = time.perf_counter()
            U_loop = compute_loop(function, x, y)
            loop_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            U_numpy = compute_numpy(function, x, y)
            numpy_times.append(time.perf_counter() - start)

            if not np.allclose(U_loop, U_numpy, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Результаты не совпадают для сетки {nx} x {ny}")

            mask = mask_condition(U_numpy)

            if mask.dtype != bool or mask.shape != U_numpy.shape:
                raise ValueError("Некорректная маска")

        loop_median = float(np.median(loop_times))
        numpy_median = float(np.median(numpy_times))

        results.append({
            "nx": nx,
            "ny": ny,
            "loop_time": loop_median,
            "numpy_time": numpy_median,
            "speedup": loop_median / numpy_median
        })

    return results