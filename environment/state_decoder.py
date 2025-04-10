import numpy as np

def decode_vector_observation(observations: np.ndarray, grid_size: int = 5) -> dict:
    state = {
        "tomato": {
            "x": int(observations[0] * grid_size),
            "y": int(observations[1] * grid_size),
            "progress": observations[2],
        },
        "lettuce": {
            "x": int(observations[3] * grid_size),
            "y": int(observations[4] * grid_size),
            "progress": observations[5],
        },
        "onion": {
            "x": int(observations[6] * grid_size),
            "y": int(observations[7] * grid_size),
            "progress": observations[8],
        },
        "plate1": {
            "x": int(observations[9] * grid_size),
            "y": int(observations[10] * grid_size),
        },
        "plate2": {
            "x": int(observations[11] * grid_size),
            "y": int(observations[12] * grid_size),
        },
        "knife1": {
            "x": int(observations[13] * grid_size),
            "y": int(observations[14] * grid_size),
        },
        "knife2": {
            "x": int(observations[15] * grid_size),
            "y": int(observations[16] * grid_size),
        },
        "delivery": {
            "x": int(observations[17] * grid_size),
            "y": int(observations[18] * grid_size),
        },
        "human": {
            "x": int(observations[19] * grid_size),
            "y": int(observations[20] * grid_size),
        },
        "ai": {
            "x": int(observations[21] * grid_size),
            "y": int(observations[22] * grid_size),
        }
    }
    return state
