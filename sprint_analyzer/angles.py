"""Joint Angele calculations using 2d coordinates."""

import numpy as np

def calculate_angle(hip, knee, ankle):
    """Return the angel at the knee in degrees."""

    hip = np.asarray(hip, dtype=float)
    knee = np.asarray(knee, dtype=float)
    ankle = np.asarray(ankle, dtype=float)

    #Vectors Originating at the knee
    upper_leg = hip - knee
    lower_leg = ankle - knee

    # Length of each vector 
    magnitude = (
        np.linalg.norm(upper_leg) * np.linalg.norm(lower_leg)
    )

    if magnitude == 0:
        return None
    
    #Dot Product formula
    cosine = np.dot(upper_leg, lower_leg) / magnitude
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return float(angle)