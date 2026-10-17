from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from app.config import SETTINGS

@lru_cache(maxsize=1)
def _encoder():
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(SETTINGS.clip_model, pretrained=SETTINGS.clip_pretrained)
    model.eval().to(device)
    return model, preprocess, device

def embed_frames(frames: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """Return L2-normalized CLIP image embeddings; CLIP is never trained."""
    if not frames: raise ValueError("Window has no readable frames")
    model, preprocess, device = _encoder()
    batches = []
    with torch.no_grad():
        for offset in range(0, len(frames), batch_size):
            images = torch.stack([preprocess(Image.fromarray(f)) for f in frames[offset:offset + batch_size]]).to(device)
            vectors = model.encode_image(images)
            vectors /= vectors.norm(dim=-1, keepdim=True)
            batches.append(vectors.cpu().numpy())
    return np.concatenate(batches).astype(np.float32)
