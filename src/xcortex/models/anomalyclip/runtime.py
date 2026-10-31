"""Inference bridge to the pinned official image AnomalyCLIP architecture.

This reconstructs learned prompt tensors from the checkpoint's complete buffers;
it does not replace the learned prompt with hand-written medical disease labels.
The image and patch-response equations match official test_one_example.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from .artifacts import BACKBONE, PROMPT, verify_artifact

DESIGN = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def prompt_tensors(state: dict, device):
    """Reconstruct the exact forward output of the upstream prompt learner."""
    import torch

    prompts, tokens = [], []
    for polarity in ("pos", "neg"):
        pieces = [state[f"token_prefix_{polarity}"], state[f"ctx_{polarity}"], state[f"token_suffix_{polarity}"]]
        prompt = torch.cat(pieces, dim=2)
        if tuple(prompt.shape) != (1, 1, 77, 768):
            raise ValueError("Unexpected AnomalyCLIP learned prompt shape")
        prompts.append(prompt.reshape(1, 77, 768))
        token = state[f"tokenized_prompts_{polarity}"]
        if tuple(token.shape) != (1, 1, 77):
            raise ValueError("Unexpected AnomalyCLIP prompt token shape")
        tokens.append(token.reshape(1, 77))
    deep = [state[f"compound_prompts_text.{i}"] for i in range(8)]
    if any(tuple(t.shape) != (4, 768) for t in deep):
        raise ValueError("Unexpected AnomalyCLIP deep prompt shape")
    return (torch.cat(prompts).float().to(device), torch.cat(tokens).to(device),
            [tensor.float().to(device) for tensor in deep])


def build_model(state: dict):
    """Construct the official ViT model from its backbone weights, CPU-first."""
    from ._upstream import AnomalyCLIP

    if "visual.proj" not in state:
        raise ValueError("Only the official ViT-L/14@336px backbone is supported")
    vision_width = state["visual.conv1.weight"].shape[0]
    patch_size = state["visual.conv1.weight"].shape[-1]
    layers = len([k for k in state if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    grid = round((state["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    text_width = state["ln_final.weight"].shape[0]
    text_layers = len({k.split(".")[2] for k in state if k.startswith("transformer.resblocks.")})
    if (vision_width, patch_size, layers, grid, text_width, text_layers) != (1024, 14, 24, 24, 768, 12):
        raise ValueError("Backbone architecture differs from pinned CLIP ViT-L/14@336px")
    model = AnomalyCLIP(
        state["text_projection"].shape[1], patch_size * grid, layers, vision_width, patch_size,
        state["positional_embedding"].shape[0], state["token_embedding.weight"].shape[0],
        text_width, text_width // 64, text_layers, design_details=DESIGN,
    )
    weights = {k: v for k, v in state.items() if k not in {"input_resolution", "context_length", "vocab_size"}}
    model.load_state_dict(weights, strict=True)
    model.visual.DAPM_replace(DPAM_layer=20)
    return model.eval().float()


class AnomalyCLIPRuntime:
    def __init__(self, backbone: Path, prompt: Path, *, device="cpu", image_size=518, features=(24,), sigma=4.0):
        import torch

        if not verify_artifact(backbone, BACKBONE) or not verify_artifact(prompt, PROMPT):
            raise FileNotFoundError("AnomalyCLIP artifacts missing or checksum mismatch; run xcortex download anomalyclip")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(v) for v in features)
        self.sigma = float(sigma)
        if self.image_size < 14 or self.image_size % 14 or not self.features or any(v not in (6, 12, 18, 24) for v in self.features):
            raise ValueError("AnomalyCLIP input size must be a positive multiple of 14; features must select 6/12/18/24")
        if len(set(self.features)) != len(self.features) or not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("AnomalyCLIP requires unique feature levels and a finite nonnegative smoothing sigma")
        # Official CLIP is a JIT archive. Extract tensors; execute our reviewed
        # architecture rather than the checkpoint's scripted forward graph.
        archive = torch.jit.load(str(backbone), map_location="cpu").eval()
        backbone_state = archive.state_dict()
        del archive
        with torch.random.fork_rng(devices=[]):
            self.model = build_model(backbone_state).to(self.device)
        del backbone_state
        checkpoint = torch.load(prompt, map_location="cpu", weights_only=True)
        prompts, tokens, deep = prompt_tensors(checkpoint["prompt_learner"], self.device)
        with torch.inference_mode():
            features = self.model.encode_text_learn(prompts, tokens, deep).float()
            features = torch.stack(torch.chunk(features, chunks=2, dim=0), dim=1)
            self.text_features = features / features.norm(dim=-1, keepdim=True)
        del checkpoint

    def infer(self, image: Image.Image) -> tuple[float, np.ndarray]:
        import torch
        import torch.nn.functional as functional
        from scipy.ndimage import gaussian_filter
        from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

        # Same explicit whole-image square resize as upstream utils.get_transform;
        # no center crop, flip or per-image heatmap normalization is introduced.
        transform = Compose([
            # Upstream uses antialias=None (legacy PIL behavior); preserve it.
            Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC, antialias=cast(Any, None)),
            ToTensor(), Normalize(MEAN, STD),
        ])
        tensor = cast(torch.Tensor, transform(image.convert("RGB"))).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(tensor, list(self.features), DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = ((image_features @ self.text_features.permute(0, 2, 1)) / 0.07).softmax(dim=-1)[:, 0, 1]
            maps = []
            for patch in patch_features:
                patch = patch / patch.norm(dim=-1, keepdim=True)
                similarity = ((patch.unsqueeze(2) * self.text_features[0][None, None]).sum(-1) / 0.07).softmax(-1)
                patch_side = int((similarity.shape[1] - 1) ** 0.5)
                if patch_side * patch_side != similarity.shape[1] - 1:
                    raise RuntimeError("AnomalyCLIP produced a non-square patch grid")
                similarity = similarity[:, 1:].reshape(1, patch_side, patch_side, 2).permute(0, 3, 1, 2)
                similarity = functional.interpolate(similarity, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
                maps.append((similarity[:, 1] + 1 - similarity[:, 0]) / 2)
            if len(maps) != len(self.features):
                raise RuntimeError("AnomalyCLIP did not return all requested feature levels")
            native_map = torch.stack(maps).sum(0)[0].float().cpu().numpy()
            native_map = gaussian_filter(native_map, sigma=self.sigma).astype(np.float32)
            image_score = float(score[0].item())
        if not np.isfinite(image_score) or not np.isfinite(native_map).all():
            raise RuntimeError("AnomalyCLIP returned nonfinite scores")
        return image_score, native_map
