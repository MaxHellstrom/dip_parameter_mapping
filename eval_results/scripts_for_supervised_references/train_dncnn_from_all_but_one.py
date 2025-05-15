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
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MASK_MODE = "soft"

test_name = name_datasets[idx_test][:14]
SAVE_PATH = f"dncnn_excl_{test_name}.pth"
PLOT_PATH = f"training_loss_excl_{test_name}.png"


class PatchDatasetFromStacks(Dataset):
    def __init__(
        self,
        y_noisy,
        y_clean,
        mask=None,
        patch_size=50,
        patches_per_slice=10,
        mask_mode="none",
    ):
        assert y_noisy.shape == y_clean.shape
        self.patches = []
        nFA, nz, nx, ny = y_noisy.shape

        if mask is not None:
            assert mask.shape == (nz, nx, ny)
            if mask_mode == "strict":
                valid_mask = mask
            elif mask_mode == "soft":
                valid_mask = binary_erosion(mask, iterations=3)
            elif mask_mode == "none":
                valid_mask = np.ones((nz, nx, ny), dtype=bool)
            else:
                raise ValueError()
        else:
            valid_mask = np.ones((nz, nx, ny), dtype=bool)

        for fa in range(nFA):
            for idx_z in range(nz):
                noisy_slice = y_noisy[fa, idx_z, ...]
                clean_slice = y_clean[fa, idx_z, ...]
                mask_slice = valid_mask[idx_z, ...]

                patches_found = 0
                attempts = 0
                max_attempts = patches_per_slice * 10

                while patches_found < patches_per_slice and attempts < max_attempts:
                    x = np.random.randint(0, nx - patch_size)
                    y = np.random.randint(0, ny - patch_size)
                    mask_patch = mask_slice[x : x + patch_size, y : y + patch_size]
                    mask_ratio = np.sum(mask_patch) / (patch_size * patch_size)

                    if (
                        mask_mode == "none"
                        or (mask_mode == "strict" and mask_ratio == 1.0)
                        or (mask_mode == "soft" and mask_ratio >= 0.8)
                    ):
                        noisy_patch = noisy_slice[
                            x : x + patch_size, y : y + patch_size
                        ]
                        clean_patch = clean_slice[
                            x : x + patch_size, y : y + patch_size
                        ]
                        self.patches.append(
                            (
                                noisy_patch.astype(np.float32),
                                clean_patch.astype(np.float32),
                            )
                        )
                        patches_found += 1
                    attempts += 1

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        noisy, clean = self.patches[idx]
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(
            0
        )


def train():
    print(f"Training all pats except: {name_datasets[idx_test]}")
    all_y, all_yref, all_mask = [], [], []

    for i, name in enumerate(name_datasets):
        if i == idx_test:
            continue
        path = DIR_DATA / name / "source"
        print(f"Loading: {name}")
        all_y.append(np.load(path / "y.npy"))
        all_yref.append(np.load(path / "y_ref.npy"))
        all_mask.append(np.load(path / "mask.npy"))

    y_all = np.concatenate(all_y, axis=1)
    y_ref_all = np.concatenate(all_yref, axis=1)
    mask_all = np.concatenate(all_mask, axis=0)

    dataset = PatchDatasetFromStacks(
        y_all,
        y_ref_all,
        mask_all,
        patch_size=PATCH_SIZE,
        patches_per_slice=PATCHES_PER_SLICE,
        mask_mode=MASK_MODE,
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

        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"[{epoch}/{EPOCHS}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Modell saved to {SAVE_PATH}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"DnCNN Training Loss ({test_name} excluded)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()


if __name__ == "__main__":
    train()
