"""Run a native baseline or an archived static output-projection patch."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local Qwen-Image-2.1 model directory')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--patch', type=Path)
    parser.add_argument('--variant', choices=['distill025', 'distill05', 'distill10', 'random_matched05'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if bool(args.patch) != bool(args.variant):
        parser.error('--patch and --variant must be supplied together')
    if args.output.exists():
        parser.error('Output directory already exists; use a new directory')
    if not args.image.is_file():
        parser.error('Input image does not exist')

    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this inference entry point')
    source = Image.open(args.image).convert('RGB')
    pipeline = QwenImage21Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to('cuda')
    pipeline.transformer.eval()
    weight = pipeline.transformer.proj_out.weight
    original = weight.detach().clone()
    replacement = None
    patch_sha = None
    if args.patch:
        patch_sha = hashlib.sha256(args.patch.read_bytes()).hexdigest()
        patch = torch.load(args.patch, map_location='cpu', weights_only=True)
        if not torch.equal(original.cpu(), patch['original_weight']):
            raise ValueError('Patch original weight does not exactly match the loaded model')
        replacement = patch['variants'][args.variant]
        if replacement.shape != weight.shape or not torch.isfinite(replacement).all():
            raise ValueError('Patch shape or values are invalid')
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        try:
            if replacement is not None:
                weight.copy_(replacement.to(device=weight.device, dtype=weight.dtype))
            result = pipeline(
                image=source, prompt=args.prompt, num_inference_steps=40,
                true_cfg_scale=1.0, output_resolution=1024, use_kv_cache=False,
                generator=torch.Generator('cuda').manual_seed(args.seed),
            ).images[0]
            result.save(args.output / 'output.png')
        finally:
            weight.copy_(original)
        restored = torch.equal(weight, original)
        if not restored:
            raise RuntimeError('Original weight restoration check failed')
    audit = dict(
        model=args.model, image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        prompt=args.prompt, seed=args.seed, variant=args.variant or 'native',
        patch_sha256=patch_sha, steps=40, cfg=1.0, resolution=1024,
        original_weight_restored=restored,
    )
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
