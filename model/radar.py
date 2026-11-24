""" 
    Radar Model -> Radar has fixed position in 2D coordinate space,
    along with other properties
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Radar():
    """ Radar Model """
    x_pos: float
    y_pos: float
    max_transmission_power: float # P_t
    transmit_antenna_gain: float # G_t
    receiver_antenna_gain: float # G_r
    radar_wavelength: float # lambda
    temp: float # T_s
    revisit_interval: float
    cross_section: float