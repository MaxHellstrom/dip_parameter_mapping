import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import DnCNN
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from pathlib import Path

DIR_DATA = Path(r"C:\Users\mahe0239\git\paper3-private\data\data_parsed\T1_VFA")
idx_test = 0  # Patient to exclude from training

name_datasets = [
    "T1_VFA__Pat_04_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_05_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_06_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_18_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_20_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_38_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_41_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_42_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_43_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_44_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_45_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_46_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_47_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_48_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_49_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_50_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_51_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_52_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_53_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
    "T1_VFA__Pat_54_shape_222_185_48_fa_2_4_11_13_15_noise_0.02",
]

PATCH_SIZE = 64
PATCHES_PER_SLICE = 10
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MASK_MODE = "soft"
NOISE_STD = 0.02

test_name = name_datasets[idx_test][:14]
SAVE_PATH = f"n2n_excl_{test_name}.pth"
PLOT_PATH = f"n2n_training_loss_excl_{test_name}.png"


def simulate_complex_noise(image, noise_std):
    """Adds complex noise to 2D or 3D image"""
    img = image.astype(np.complex64)
    if img.ndim == 2:
        nx, ny = img.shape
        r1 = np.random.randn(nx, ny)
        r2 = np.random.randn(nx, ny)
        noisy = img + noise_std * (r1 + 1j * r2)
        return np.abs(noisy).astype(np.float32)
    elif img.ndim == 3:
        nc, nx, ny = img.shape
        res = np.zeros((nc, nx, ny), dtype=np.float32)
        r1 = np.random.randn(nc, nx, ny)
        r2 = np.random.randn(nc, nx, ny)
        for c in range(nc):
            tmp = img[c]
            noisy = tmp + noise_std * (r1[c] + 1j * r2[c])
            res[c] = np.abs(noisy)
        return res
    else:
        raise ValueError()


class Noise2NoiseDataset(Dataset):
    def __init__(
        self,
        y_ref,
        mask=None,
        patch_size=64,
        patches_per_slice=10,
        mask_mode="soft",
        noise_std=0.02,
    ):
        self.patch_size = patch_size
        self.noise_std = noise_std
        nFA, nz, nx, ny = y_ref.shape

        if mask is not None:
            assert mask.shape == (nz, nx, ny)
            if mask_mode == "strict":
                valid_mask = mask
            elif mask_mode == "soft":
                valid_mask = binary_erosion(mask, iterations=3)
            else:
                valid_mask = np.ones_like(mask, dtype=bool)
        else:
            valid_mask = np.ones((nz, nx, ny), dtype=bool)

        self.clean_patches = []
        for fa in range(nFA):
            for idx_z in range(nz):
                ref_slice = y_ref[fa, idx_z, ...]
                mask_slice = valid_mask[idx_z, ...]
                found = 0
                attempts = 0
                while found < patches_per_slice and attempts < patches_per_slice * 10:
                    x = np.random.randint(0, nx - patch_size)
                    y = np.random.randint(0, ny - patch_size)
                    patch_mask = mask_slice[x : x + patch_size, y : y + patch_size]
                    if (
                        mask_mode == "none"
                        or (mask_mode == "strict" and patch_mask.all())
                        or (mask_mode == "soft" and patch_mask.mean() >= 0.8)
                    ):
                        clean = ref_slice[
                            x : x + patch_size, y : y + patch_size
                        ].astype(np.float32)
                        self.clean_patches.append(clean)
                        found += 1
                    attempts += 1

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean = self.clean_patches[idx]
        noisy1 = simulate_complex_noise(clean, self.noise_std)
        noisy2 = simulate_complex_noise(clean, self.noise_std)
        t1 = torch.from_numpy(noisy1).unsqueeze(0)
        t2 = torch.from_numpy(noisy2).unsqueeze(0)
        return t1, t2


def train():
    print(f"Training all pats except: {name_datasets[idx_test]}")
    all_ref = []
    all_mask = []

    for i, name in enumerate(name_datasets):
        if i == idx_test:
            continue
        base = DIR_DATA / name / "source"
        print(f"→ Laddar referens: {name}")
        all_ref.append(np.load(base / "y_ref.npy"))
        all_mask.append(np.load(base / "mask.npy"))

    y_ref_all = np.concatenate(all_ref, axis=1)
    mask_all = np.concatenate(all_mask, axis=0)

    dataset = Noise2NoiseDataset(
        y_ref_all,
        mask=mask_all,
        patch_size=PATCH_SIZE,
        patches_per_slice=PATCHES_PER_SLICE,
        mask_mode=MASK_MODE,
        noise_std=NOISE_STD,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"num_patches: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN(channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for noisy1, noisy2 in loader:
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            optimizer.zero_grad()
            output = model(noisy1)
            loss = criterion(output, noisy2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"[{epoch}/{EPOCHS}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Modell saved to {SAVE_PATH}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss (N2N)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()


if __name__ == "__main__":
    train()
