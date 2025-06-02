import argparse
import os
import torch
import torchvision
from torchvision.transforms import v2
import numpy as np
from scipy.special import softmax
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Evaluate model on test data',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Path to the directory containing the Schmetterlinge '
                    'dataset folder')
parser.add_argument('--model', type=str, default="maxvit_t",
                    help='Model type to be used')
parser.add_argument('--model-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Path to the saved model to be tested')
parser.add_argument('--results-dir', type=str, default=".",
                    help='Directory to save the results')
args = parser.parse_args()


# Provide number of classes as constant
n_classes = 163

# Inference perhaps also on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import base model
model_fn = getattr(torchvision.models, args.model)
model = model_fn(weights=None, num_classes=n_classes)

model.to(device)

# Load model weights
state_dict = torch.load(args.model_dir, map_location=device)
model.load_state_dict(state_dict)

# transformation for validation and test data (only resizes and crops images and normalizes)
test_transform = v2.Compose([
    v2.Resize(size = 224),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = torchvision.datasets.ImageFolder(root = os.path.join(
    args.data_dir,"Schmetterlinge_sym"),
    transform=test_transform)

# Load indices of testdata
test_idx = np.load(os.path.join(args.data_dir, "preparation", "test_idx.npy"))

# data loader
batch_size = 64
num_workers = 8
num_samples = len(test_idx)
test_data = torch.utils.data.Subset(data, test_idx)
test_dl = torch.utils.data.DataLoader(test_data,
                                    batch_size = batch_size,
                                    num_workers = num_workers)

# Collect all predictions, the true labels, and softmax of outputs
pred_all = []
true_all = []
softmax_all = []

model.eval()
with torch.no_grad():
    # Loop over batches with progress bar
    for x, y in tqdm(test_dl, desc="Evaluating", unit="batch"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        y_pred = out.argmax(1).cpu().numpy()
        y_true = y.cpu().numpy()
        pred_all.extend(y_pred)
        true_all.extend(y_true)
        out_np = out.cpu().numpy()
        softmax_batch = softmax(out_np, axis=1)
        softmax_all.append(softmax_batch)

# Concatenate for homogeneous array
softmax_all = np.concatenate(softmax_all, axis=0)

os.makedirs(args.results_dir, exist_ok=True)
pred_true = {"prediction": pred_all, "true_label": true_all}
np.save(os.path.join(args.results_dir, "predictions_test.npy"), pred_true)
np.save(os.path.join(args.results_dir, "softmax_test.npy"), softmax_all)

