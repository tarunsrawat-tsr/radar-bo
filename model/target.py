""" 
    Target Model - A target moves in airspace, has (x,y) positon
    horizontal and vertical velocity and acceleration

"""
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import List
from radar import Radar
import numpy as np

@dataclass
class Target():
    """ Target Model class """
    id: int
    x_coordinate: float
    y_coordinate: float
    horizontal_velocity: float
    vertical_velocity: float
    horizontal_acceleration: float = 0
    vertical_acceleration: float = 0
    dwell_time: float
    power: float
    time_slot: int
    model_history: list[np.array] = field(init=False)
    state_estimation: list[np.array] = field(init=False)
    corridor: int

    def __post_init__(self):
        self.model_history = self.init_model()
        self.state_estimation = self.init_state_estimation()

    @property
    def x_pos(self):
        return self.x_coordinate

    @property
    def y_pos(self):
        return self.y_coordinate

    @property
    def xdot(self):
        return self.horizontal_velocity

    @property
    def ydot(self):
        return self.vertical_velocity

    @property
    def dwell_time(self):
        return self.dwell_time

    @property
    def power(self):
        return self.power

    @x_pos.setter
    def x_pos(self, new_pos: float):
        self.x_coordinate = new_pos

    @y_pos.setter
    def y_pos(self, new_pos: float):
        self.y_coordinate = new_pos

    @xdot.setter
    def xdot(self, new_velocity: float):
        self.horizontal_velocity = new_velocity

    @ydot.setter
    def ydot(self, new_velocity: float):
        self.vertical_velocity = new_velocity

    @dwell_time.setter
    def dwell_time(self, new_time: float):
        self.dwell_time = new_time

    @power.setter
    def power(self, new_pow: float):
        self.power = new_pow

    def init_model(self):
        """ Creates state space model for our target at time slot (t = 0)"""
        self.model_history.append(np.zeros((4,1)))

    def init_state_estimation(self):
        """ Stores state estimation covariance matrix for target """
        self.state_estimation.append(np.eye(4,4))

@dataclass
class Corridor():
    id: int
    targets: list[Target] = Field(ge=0)
    height: float = Field(ge=0)
    dimensions: list[float, float, float, float, float, float, float, float]


@dataclass
class SystemModel():
    radars: list[Radar]