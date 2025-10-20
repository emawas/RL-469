import numpy as np

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocess 210x160x3 uint8 frame into 80x80 float32 (0/1) image.
    Matches the homework-provided function (with dtype modernized).
    """
    # crop
    image = image[35:195]
    # downsample by factor of 2, take R channel
    image = image[::2, ::2, 0]
    # erase background colors
    image[image == 144] = 0
    image[image == 109] = 0
    # everything else set to 1
    image[image != 0] = 1
    # reshape to (80, 80) float32
    return np.reshape(image.astype(np.float32).ravel(), (80, 80))

def frame_input(curr_rgb: np.ndarray, prev_proc: np.ndarray | None):
    """
    Returns the model input array:
      - If prev_proc is given: use frame difference (curr - prev) to emphasize motion
      - Else: use current preprocessed frame
    Output shape: (1, 80, 80)  (CHW with C=1 later in torch)
    """
    curr_proc = preprocess(curr_rgb)
    if prev_proc is None:
        inp = curr_proc
    else:
        inp = curr_proc - prev_proc
    return inp[np.newaxis, ...], curr_proc  # (1,80,80), curr_proc to store as prev