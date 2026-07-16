"""
Synthetic strong-lens image generator for GravLens-Net Phase 1.

Physics: lens galaxy modeled as a Sersic profile. A background source
(Gaussian blob) is ray-traced through a Singular Isothermal Sphere (SIS)
lens equation, so "lens" images show physically-motivated arcs / partial
Einstein rings rather than hand-drawn shapes. "Non-lens" images use three
realistic confuser types (isolated galaxy, merger/pair, spiral) so the
negative class isn't trivially separable.
"""
import numpy as np

IMG_SIZE = 64
HALF = IMG_SIZE / 2.0


def _grid():
    x = np.linspace(-HALF, HALF, IMG_SIZE)
    y = np.linspace(-HALF, HALF, IMG_SIZE)
    return np.meshgrid(x, y)


def sersic(x, y, amplitude, r_eff, n=2.0, x0=0.0, y0=0.0, ellip=0.0, theta=0.0):
    dx, dy = x - x0, y - y0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    q = 1.0 - ellip
    r = np.sqrt(xr ** 2 + (yr / max(q, 1e-3)) ** 2)
    bn = 1.9992 * n - 0.3271
    return amplitude * np.exp(-bn * ((r / r_eff) ** (1.0 / n) - 1.0))


def gaussian_blob(x, y, amplitude, sigma, x0=0.0, y0=0.0):
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def sis_deflect(x, y, theta_E, x0=0.0, y0=0.0):
    dx, dy = x - x0, y - y0
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_safe = np.maximum(r, 1e-6)
    alpha_x = theta_E * dx / r_safe
    alpha_y = theta_E * dy / r_safe
    return dx - alpha_x, dy - alpha_y  # beta_x, beta_y (relative to lens center)


def add_noise(img, sky_level=0.02, read_noise=0.015, rng=None):
    rng = rng or np.random.default_rng()
    img = np.clip(img, 0, None)
    shot = rng.normal(0, np.sqrt(img + sky_level) * 0.05, img.shape)
    read = rng.normal(0, read_noise, img.shape)
    return img + sky_level + shot + read


def make_lens_image(rng):
    x, y = _grid()
    lens_amp = rng.uniform(0.6, 1.0)
    lens_reff = rng.uniform(4.0, 8.0)
    lens_n = rng.uniform(1.5, 4.0)
    lens_ellip = rng.uniform(0.0, 0.3)
    lens_theta = rng.uniform(0, np.pi)

    img = sersic(x, y, lens_amp, lens_reff, lens_n, 0, 0, lens_ellip, lens_theta)

    theta_E = rng.uniform(6.0, 12.0)          # Einstein radius (pixels)
    src_offset = rng.uniform(0.0, 0.5) * theta_E   # small offset -> ring-like; larger -> arcs
    src_angle = rng.uniform(0, 2 * np.pi)
    src_x0 = src_offset * np.cos(src_angle)
    src_y0 = src_offset * np.sin(src_angle)
    src_amp = rng.uniform(0.5, 1.0)
    src_sigma = rng.uniform(1.2, 2.2)

    beta_x, beta_y = sis_deflect(x, y, theta_E)
    lensed = gaussian_blob(beta_x, beta_y, src_amp, src_sigma, src_x0, src_y0)
    img = img + lensed
    return add_noise(img, rng=rng)


def make_isolated_galaxy(rng):
    x, y = _grid()
    amp = rng.uniform(0.6, 1.0)
    reff = rng.uniform(4.0, 9.0)
    n = rng.uniform(0.8, 4.0)
    ellip = rng.uniform(0.0, 0.5)
    theta = rng.uniform(0, np.pi)
    img = sersic(x, y, amp, reff, n, 0, 0, ellip, theta)
    return add_noise(img, rng=rng)


def make_merger_pair(rng):
    x, y = _grid()
    img = sersic(x, y, rng.uniform(0.6, 1.0), rng.uniform(4, 7), rng.uniform(1, 3),
                 0, 0, rng.uniform(0, 0.4), rng.uniform(0, np.pi))
    sep = rng.uniform(6, 16)
    ang = rng.uniform(0, 2 * np.pi)
    x0, y0 = sep * np.cos(ang), sep * np.sin(ang)
    img = img + sersic(x, y, rng.uniform(0.3, 0.7), rng.uniform(2, 5), rng.uniform(1, 3),
                        x0, y0, rng.uniform(0, 0.4), rng.uniform(0, np.pi))
    return add_noise(img, rng=rng)


def make_spiral(rng):
    x, y = _grid()
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    n_arms = 2
    pitch = rng.uniform(0.25, 0.45)
    arm_pattern = 0.5 + 0.5 * np.cos(n_arms * (phi - r * pitch))
    disk = sersic(x, y, rng.uniform(0.5, 0.9), rng.uniform(6, 10), 1.0, 0, 0)
    img = disk * (0.5 + 0.5 * arm_pattern)
    return add_noise(img, rng=rng)


def generate_dataset(n_neg=2500, n_pos=50, seed=42):
    rng = np.random.default_rng(seed)
    images, labels, subtypes = [], [], []

    neg_makers = [make_isolated_galaxy, make_merger_pair, make_spiral]
    neg_weights = [0.6, 0.2, 0.2]
    for _ in range(n_neg):
        maker = rng.choice(neg_makers, p=neg_weights)
        images.append(maker(rng))
        labels.append(0)
        subtypes.append(maker.__name__)

    for _ in range(n_pos):
        images.append(make_lens_image(rng))
        labels.append(1)
        subtypes.append("lens")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    subtypes = np.array(subtypes)

    perm = rng.permutation(len(images))
    return images[perm], labels[perm], subtypes[perm]


if __name__ == "__main__":
    imgs, labels, subtypes = generate_dataset()
    print("dataset shape:", imgs.shape, "positives:", labels.sum(), "/", len(labels))
    print("subtype counts:", {s: int((subtypes == s).sum()) for s in np.unique(subtypes)})
