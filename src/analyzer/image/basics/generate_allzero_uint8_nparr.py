import numpy as np


def generate_allzero_uint8_nparr(width: int, height: int):
    result = []
    for i_col in range(height):
        px_col = []
        for i_row in range(width):
            px_col.append([0, 0, 0])

        result.append(px_col)

    return np.array(result, dtype="uint8")
