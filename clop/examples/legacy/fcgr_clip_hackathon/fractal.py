import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os


def normalize_image(image):
    """Normalize the image intensities to a [0, 1] range."""
    min_val, max_val = np.min(image), np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return np.zeros_like(image)


def calculate_voxel_density(image, scale, max_intensity_level):
    """Calculate the density of occupied voxels for a specific scale."""
    voxel_set = set()
    scaled_intensity = image * max_intensity_level
    is_uniform = np.all(image == image[0, 0])

    for x in range(0, image.shape[0], scale):
        for y in range(0, image.shape[1], scale):
            if is_uniform:
                voxel_set.add(
                    (x // scale, y // scale, int(scaled_intensity[0, 0]))
                )
            else:
                block = scaled_intensity[x : x + scale, y : y + scale]
                if block.size:
                    min_intensity, max_intensity = (
                        np.floor(block.min()),
                        np.ceil(block.max()),
                    )
                    for z in range(int(min_intensity), int(max_intensity) + 1):
                        voxel_set.add((x // scale, y // scale, z))
    return len(voxel_set)


def compute_fractal_dimension(
    image, max_box_size=32, min_box_size=2, step=2, max_intensity_level=32
):
    """Compute the fractal dimension of a grayscale image, treating intensity as depth, using parallel processing."""
    normalized_image = normalize_image(image)

    with ProcessPoolExecutor() as executor:
        scales = range(max_box_size, min_box_size - 1, -step)
        futures = [
            executor.submit(
                calculate_voxel_density,
                normalized_image,
                scale,
                max_intensity_level,
            )
            for scale in scales
        ]
        counts = [future.result() for future in futures]

    if not any(counts):
        return np.nan

    log_scales = np.log(np.array(list(scales)))
    log_counts = np.log(np.array(counts))
    fractal_dimension, _ = np.polyfit(log_scales, log_counts, 1)
    return fractal_dimension


def generate_sierpinski_triangle(size, iterations):
    """Generate a Sierpinski triangle using the chaos game method."""
    image = np.zeros((size, size))
    points = [(0, 0), (size - 1, 0), (size // 2, size - 1)]
    x, y = points[0]

    for i in range(size**2 * iterations):
        target = points[np.random.randint(3)]
        x, y = (x + target[0]) // 2, (y + target[1]) // 2
        image[y, x] = 1

    return image


def test_uniform_array():
    """Test the fractal dimension of a uniform array."""
    size = 256
    uniform_array = np.ones((size, size))
    fd = compute_fractal_dimension(uniform_array)
    print(f"Fractal dimension of a uniform array: {fd}")
    assert np.isclose(fd, 0, atol=0.1), (
        "Fractal dimension of a uniform array should be close to 0."
    )


def test_sierpinski_triangle():
    """Test the fractal dimension of a generated Sierpinski triangle."""
    size = 256
    triangle = generate_sierpinski_triangle(size, 10)
    fd = compute_fractal_dimension(triangle)
    print(f"Fractal dimension of the Sierpinski triangle: {fd}")
    theoretical_fd = np.log(3) / np.log(2)
    assert np.isclose(fd, theoretical_fd, atol=0.2), (
        f"Expected fractal dimension close to {theoretical_fd}, got {fd}"
    )


def test():
    """Run a series of tests to validate fractal dimension computation."""
    print("Testing with a uniform array...")
    test_uniform_array()

    print("\nTesting with a Sierpinski triangle...")
    test_sierpinski_triangle()

    # Additional tests can be added here as needed


if __name__ == "__main__":
    test()
