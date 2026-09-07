"""Inspect the RF-DETR-Nano torch model: structure, state_dict keys, output
shapes. Pins the exact architecture for the Vulkan port. Downloads the
checkpoint from HuggingFace on first run."""
import collections
import torch
from rfdetr import RFDETRNano


def find_module(m):
    # RFDETR wrapper -> .model (Model) -> .model (LWDETR nn.Module)
    for path in ("model.model", "model", ""):
        cur = m
        ok = True
        for p in [x for x in path.split(".") if x]:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and isinstance(cur, torch.nn.Module):
            return cur, path
    return None, None


def main():
    det = RFDETRNano()
    net, path = find_module(det)
    print(f"torch module found at: det.{path}  ({type(net).__name__})")
    net.eval()

    sd = net.state_dict()
    print(f"\nstate_dict: {len(sd)} tensors")
    # Group keys by top-2 prefix segments.
    groups = collections.Counter()
    for k in sd:
        parts = k.split(".")
        groups[".".join(parts[:2])] += 1
    for pfx, n in sorted(groups.items()):
        print(f"  {pfx:<40s} {n}")

    print("\n--- top-level children ---")
    for name, child in net.named_children():
        print(f"  {name}: {type(child).__name__}")

    # Backbone detail.
    print("\n--- backbone repr (truncated) ---")
    bb = getattr(net, "backbone", None)
    if bb is not None:
        s = repr(bb)
        print(s[:2500])

    # A few representative state_dict shapes per group.
    print("\n--- sample shapes ---")
    seen = set()
    for k, v in sd.items():
        pfx = ".".join(k.split(".")[:3])
        if pfx not in seen:
            seen.add(pfx)
            print(f"  {k:<60s} {tuple(v.shape)}")
        if len(seen) > 60:
            break


if __name__ == "__main__":
    main()
