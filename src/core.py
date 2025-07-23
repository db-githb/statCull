import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
import re
import types
from datetime import datetime


PARAMS = {"means", "opacities", "scales", "quats", "features_dc", "features_rest"}
PATTERN = re.compile(r'(?:^|\.)(means|opacities|scales|quats|features_dc|features_rest)$')

class GaussParamStore(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        for n, t in tensors.items():
            self.register_buffer(n, tensors[n])

def load_config(load_path: Path, device="cuda"):
    ckpt = torch.load(load_path, map_location="cpu", weights_only=False)# state_dict
    state = ckpt.get("pipeline", ckpt) 
    out = {}
    for k, v in state.items():
        m = PATTERN.search(k)
        if m:
            out[m.group(1)] = v.to(device)
    # sanity
    missing = [p for p in PARAMS if p not in out]
    if missing:
        raise KeyError(f"Missing tensors: {missing}\nFound: {list(out.keys())}")
    return out

def setup_write_ply(model):
    """Works with MiniModel that only exposes the six tensors above."""
    map_to_tensors = OrderedDict()

    # ---- positions ----
    positions = model.means.detach().cpu().numpy()
    n = positions.shape[0]
    map_to_tensors["x"] = positions[:, 0]
    map_to_tensors["y"] = positions[:, 1]
    map_to_tensors["z"] = positions[:, 2]
    # normals placeholder
    map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
    map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
    map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)
    # ---- appearance ----
    if hasattr(model, "features_dc") and hasattr(model, "features_rest"):
        # DC term
        shs_0 = model.features_dc.detach().cpu().numpy()           # [N, 3]
        for i in range(shs_0.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
        # Higher-order SH terms
        shs_rest = model.features_rest.detach().cpu().numpy()
        # shapes vary; handle 2D or 3D
        if shs_rest.ndim == 3:  # [N, C, 3] like nerfstudio
            shs_rest = np.transpose(shs_rest, (0, 2, 1))  # [N, 3, C]
            shs_rest = shs_rest.reshape(n, -1)
        else:  # [N, C]
            shs_rest = shs_rest.reshape(n, -1)
        for i in range(shs_rest.shape[1]):
            map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
    else:
        # Fall back to RGB colors if no SH features present
        colors = torch.clamp(model.colors.clone(), 0.0, 1.0).detach().cpu().numpy()
        map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
    # ---- opacity ----
    map_to_tensors["opacity"] = model.opacities.detach().cpu().numpy()
    # ---- scales ----
    scales = model.scales.detach().cpu().numpy()
    for i in range(scales.shape[1]):
        map_to_tensors[f"scale_{i}"] = scales[:, i, None]
    # ---- rotation (quat) ----
    quats = model.quats.detach().cpu().numpy()
    for i in range(quats.shape[1]):
        map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    # ---- sanity: finite filter ----
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = select.sum()
        select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        n_after = select.sum()
        if n_after < n_before:
            print(f"{n_before - n_after} NaN/Inf elements in {k}")

    if select.sum() < n:
        print(f"values have NaN/Inf, only export {select.sum()}/{n}")
        for k in list(map_to_tensors.keys()):
            map_to_tensors[k] = map_to_tensors[k][select]
        n = select.sum()

    return n, map_to_tensors

def write_ply(filename, count, map_to_tensors):
        
    # Ensure count matches the length of all tensors
    if not all(len(tensor) == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")
    
    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")
    
    with open(filename, "wb") as ply_file:
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())
        ply_file.write(b"end_header\n")

        # Write binary data
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())

def make_outpath(model_path, out_path = None, out_dir = "outputs", levels_up = 3, suffix = ".ply"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        for parent in model_path.parents:
            if parent.name == "outputs":
                scene = model_path.relative_to(parent).parts[0]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"{scene}_{ts}_culled{suffix}"
    else:
        out_path = Path(out_path)

    return out_path

def run_cull(model_path, out_path, thr_xyz):
    """Culls 3DGS gaussians based on z-scores and exports to PLY."""
    tensors = load_config(model_path)
    mini_model = GaussParamStore(tensors)
    pipeline = types.SimpleNamespace(model=mini_model) 
    model = pipeline.model
    tensors = load_config(model_path)
    model   = GaussParamStore(tensors)
    means3D     = model.means.to("cpu") # cpu is faster for these operations
    center      = means3D.median(dim=0)
    std_dev     = means3D.std(dim=0)
    z_scores    = torch.abs((means3D - center.values) / std_dev)
    thr         = torch.tensor(thr_xyz)
    cull_mask   = (z_scores > thr).any(dim=1)
    keep = ~cull_mask
    print(f"Total culled: {(cull_mask).sum().item()}/{means3D.shape[0]}")
    for n in PARAMS:
        new = getattr(model, n)[keep].clone()
        delattr(model, n)
        model.register_buffer(n, new)

    count, map_to_tensors = setup_write_ply(model)
    out_path = make_outpath(model_path, out_path)
    write_ply(out_path, count, map_to_tensors)

    print(f"CULL COMPLETE: {out_path}")
