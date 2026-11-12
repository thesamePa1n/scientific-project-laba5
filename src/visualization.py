import matplotlib.pyplot as plt

def save_field_plot(U, x, y, mask, path):
    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.pcolormesh(x, y, U, shading="auto", cmap="viridis")

    fig.colorbar(image, ax=ax, label="u(x, y)")

    ax.contour(x, y, mask.astype(float), levels=[0.5], colors="red", linewidths=1.2)

    ax.set_title("Поле u(x, y), вариант 1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)