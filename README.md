# BiCueCC

## 1. Install

Use Python 3.10 or newer. Install a compatible PyTorch and torchvision pair for
your CPU or CUDA environment, then install the dependencies:

```bash
git clone https://github.com/ChunxiaoLe/BiCueCC.git
cd BiCueCC
pip install -r requirements.txt
```

## 2. Prepare the data

For Gehler-Shi evaluation, place the preprocessed image arrays and color-chart
masks under a data directory. Images must be linear, black-level-corrected
three-channel HWC arrays. The default NumPy channel order is BGR.
Use the same preprocessing as the benchmark; raw camera files and rendered JPEGs
are not interchangeable with the preprocessed arrays.

The fold-specific test lists are in `splits/gs/`. Their image and mask paths
are relative to `--data-root`. To supply your own list, use this CSV format:

```csv
image_path,illum_r,illum_g,illum_b,mask_path
IMG_0753.npy,0.3406443894,0.7381058931,0.5823753476,IMG_0753_mask.npy
```

`mask_path` is optional for images whose color chart has already been masked.
Mask arrays must align with the image and use 0 for excluded pixels and 1 for
valid pixels. Ground-truth illuminants are RGB vectors.

## 3. Select the weights

The `weights/` directory contains `gs_fold0.pth`, `gs_fold1.pth`, and
`gs_fold2.pth`. Use each checkpoint with its matching fold test list.

## 4. Run evaluation

```bash
python test.py --checkpoint weights/gs_fold0.pth --csv splits/gs/fold0.csv --data-root /path/to/CC_full --output-csv results/fold0.csv
python test.py --checkpoint weights/gs_fold1.pth --csv splits/gs/fold1.csv --data-root /path/to/CC_full --output-csv results/fold1.csv
python test.py --checkpoint weights/gs_fold2.pth --csv splits/gs/fold2.csv --data-root /path/to/CC_full --output-csv results/fold2.csv
```

Use `--device cpu` for CPU evaluation, `--batch-size` to adjust memory use,
and `--npy-channel-order rgb` if your NumPy arrays are stored in RGB order.
Evaluation uses per-image maximum scaling, gamma 1/2.2, deterministic
512×512 resizing, and ImageNet normalization.

Each command prints mean, median, trimean, best-25%, worst-25%, and 95th-percentile
angular error in degrees, and writes per-image RGB predictions and errors.
To compute the three-fold result, pool the per-image errors from the three
output CSV files using:

```bash
python summarize.py results/fold0.csv results/fold1.csv results/fold2.csv
```
