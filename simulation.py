""" Simulation.py -> This file contains simulation setup for our research work """

from model.target import Corridor, Target
from model.radar import Radar
from model.function import param
from bo.bo import Dataset


setup = {
    "corridors": [
        Corridor(
            id=1,
            height=120,
            dimensions=[]
        ),
        Corridor(
            id=2,
            height=120,
            dimensions=[]
        ),
        Corridor(
            id=3,
            height=120,
            dimensions=[]
        ),
        Corridor(
            id=4,
            height=120,
            dimensions=[]
        ),
    ],
    "radar": [
        Radar(
            x_pos=0,
            y_pos=0,
            max_transmission_power=param["peak_transmission_power"],
            temp=param["system_temp"],
            revisit_interval=param["revisit_interval"],
            cross_section=param["cross_section"],
            radar_wavelength=param["radar_wavelength"],
            transmit_antenna_gain=param['transmit_antenna_gain'],
            receiver_antenna_gain=param["receiver_antenna_gain"]
        )
    ],
    "time": 10, # Each simualtion has 10 times slots
}

def run_simulation():
    """ Simulate the work """
    

if __name__ == "__main__":
    run_simulation()