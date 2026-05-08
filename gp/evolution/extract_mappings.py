"""Extract LayerNorm input-output mappings from a pretrained ViT for GP fitness evaluation.

Runs a single forward pass through a pretrained timm model and uses forward hooks
to record the scalar input (x) and pre-affine output (y) of every LayerNorm layer.
The resulting (x, y) pairs are saved as a compressed .npz file used by the GP
fitness function to evaluate candidate replacement functions.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils import str2bool

def get_args():
    """Build the argument parser for the LN mapping extraction script.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser("Extract LayerNorm Mappings")
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    p.add_argument('--model_name', default='vit_base_patch16_224', type=str)
    p.add_argument('--imagenet_root', default='/path/to/imagenet/val', type=str)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--points_per_forward', default=10_000, type=int, help='Number of (x, y) pairs sampled per layer per forward pass.')
    p.add_argument('--output_file', default='ln_mappings.npz', type=str)
    return p.parse_args()

def collect_data_for_layer(model, layer, loader, device, points_per_forward, batch_size):
    """Collect (x, y) pairs from a single LayerNorm layer via a forward hook.

    Runs one forward pass and captures flattened input scalars (x) and
    pre-affine output scalars (y = (LN_out - bias) / (weight + eps)) for
    the given layer. A random subset of up to points_per_forward pairs is
    returned.

    Args:
        model: Pretrained model containing the layer.
        layer: The LayerNorm module to hook.
        loader: DataLoader supplying real ImageNet batches, or None to use
            synthetic random inputs.
        device: Torch device to run inference on.
        points_per_forward: Maximum number of (x, y) pairs to sample.
        batch_size: Batch size used when generating synthetic inputs.

    Returns:
        Tuple of (x_np, y_np): numpy arrays of sampled input and
        pre-affine output scalars.
    """
    store = {}
    def hook_fn(module, inp, out):
        x = inp[0].detach()
        y_post = out.detach()
        w = module.weight.detach()
        b = module.bias.detach()
        y_pre = (y_post - b) / (w + 1e-12)

        x1, y1 = x.reshape(-1), y_pre.reshape(-1)
        n = x1.numel()
        m = min(points_per_forward, n)
        idx = torch.randint(0, n, (m,), device=x1.device)

        store["x"] = x1[idx].float().cpu().numpy()
        store["y"] = y1[idx].float().cpu().numpy()

    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        if loader is not None:
            imgs, _labels = next(iter(loader))
            imgs = imgs.to(device)
        else:
            imgs = torch.randn(batch_size, 3, 224, 224, device=device)
        _ = model(imgs)  # forward pass triggers the hook

    handle.remove()
    return store["x"], store["y"]

def main():
    """Run LN mapping extraction across all LayerNorm layers in the model.

    Loads a pretrained timm model, attempts to build an ImageNet DataLoader,
    then iterates over every LayerNorm layer and collects (x, y) pairs via
    collect_data_for_layer. Results are saved to a compressed .npz file.
    """
    args = get_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True).to(device).eval()
    
    loader = None
    try:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(args.imagenet_root, transform=trans)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Failed to load ImageNet, using synthetic data: {e}")

    ln_layers = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.LayerNorm)}
    print(f"Found {len(ln_layers)} LayerNorm layers. Extracting...")

    all_layer_data = {}
    for layer_name, ln_layer in ln_layers.items():
        x_np, y_np = collect_data_for_layer(
            model, ln_layer, loader, device,
            points_per_forward=args.points_per_forward,
            batch_size=args.batch_size,
        )
        all_layer_data[f"{layer_name}_x"] = x_np
        all_layer_data[f"{layer_name}_y"] = y_np

    np.savez_compressed(args.output_file, **all_layer_data)
    print(f"Successfully saved all mappings to {args.output_file}")

if __name__ == "__main__":
    main()