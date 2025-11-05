# --- MsPacman preprocessing (exactly as specified) ---
import numpy as np
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    # obs: HxWx3 uint8
    img = obs[1:176:2, ::2]               # crop + downsample -> (88, 80, 3)
    img = img.sum(axis=2)                 # to grayscale via channel sum
    img[img == mspacman_color] = 0        # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize to [-128, 127]
    return img.reshape(88, 80, 1)         # H, W, C=1 (int8)