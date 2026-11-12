import csv
import json
from pathlib import Path
import numpy as np
from src.calculations import (
    get_variant,
    make_coordinates,
    compute_loop,
    compute_numpy,
    analyze_field,
    benchmark
)
from src.visualization import save_field_plot

def main():
    variant_number = 1
    nx = 600
    ny = 480

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    function, bounds, mask_condition = get_variant(variant_number)

    x, y, dx, dy = make_coordinates(bounds, nx, ny)

    print("Вычисление поля циклами Python:")
    U_loop = compute_loop(function, x, y)

    print("Вычисление поля средствами NumPy:")
    U_numpy = compute_numpy(function, x, y)

    print("\nПроверка результатов:")

    expected_shape = (ny, nx)

    if U_loop.shape != expected_shape:
        raise ValueError("Неверная форма массива U_loop")

    if U_numpy.shape != expected_shape:
        raise ValueError("Неверная форма массива U_numpy")

    if not np.isfinite(U_loop).all():
        raise ValueError("Массив U_loop содержит NaN или inf")

    if not np.isfinite(U_numpy).all():
        raise ValueError("Массив U_numpy содержит NaN или inf")

    if not np.allclose(U_loop, U_numpy, rtol=1e-10, atol=1e-12):
        raise ValueError("Результаты двух реализаций не совпадают")

    print("Формы массивов:", U_loop.shape)
    print("Все значения конечны.")
    print("Результаты двух реализаций совпадают.")

    mask = mask_condition(U_numpy)

    if mask.dtype != bool:
        raise ValueError("Маска должна иметь тип bool")

    if mask.shape != U_numpy.shape:
        raise ValueError("Форма маски не совпадает с формой поля")

    statistics = analyze_field(U_numpy, mask, dx, dy, x, y)
    domain_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])

    if not 0 <= statistics["mask_area"] <= domain_area:
        raise ValueError("Площадь маски находится вне допустимых границ")

    print("\nХарактеристики поля:")

    for name, value in statistics.items():
        print(f"{name}: {value:.10f}")

    np.savez(results_dir / "field.npz", x=x, y=y, U=U_numpy, mask=mask)

    summary = {
        "variant": variant_number,
        "bounds": {
            "xmin": bounds[0],
            "xmax": bounds[1],
            "ymin": bounds[2],
            "ymax": bounds[3]
        },
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "arrays_match": True,
        "statistics": statistics
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\nСравнение производительности:")

    sizes = [
        (150, 120),
        (300, 240),
        (600, 480)
    ]

    timing_results = benchmark(function, bounds, mask_condition, sizes, repeats=3)

    with open(results_dir / "timings.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=["nx", "ny", "loop_time", "numpy_time", "speedup"])
        writer.writeheader()
        writer.writerows(timing_results)

    for result in timing_results:
        print(
            f"Размер {result['nx']} x {result['ny']}: "
            f"циклы = {result['loop_time']:.6f} с, "
            f"NumPy = {result['numpy_time']:.6f} с, "
            f"ускорение = {result['speedup']:.2f}"
        )

    save_field_plot(U_numpy, x, y, mask, results_dir / "field.png")

    print("\nВсе результаты сохранены в папку results.")

if __name__ == "__main__":
    main()