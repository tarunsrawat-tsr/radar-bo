""" 
    Target Model - A target moves in airspace, has (x,y) positon
    horizontal and vertical velocity and acceleration

"""
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import List
from function import SNR
from radar import Radar
import numpy as np

@dataclass
class Target():
    """ Target Model class """
    id: int = field(init=False)
    x_coordinate: float = field(init=False)
    y_coordinate: float = field(init=False)
    horizontal_velocity: float = field(init=False)
    vertical_velocity: float = field(init=False)
    dwell_time: float | None
    revisit_interval: float | None
    power: float | None
    time_slot: int | None
    model_history: list[np.array] = field(init=False)
    state_estimation: list[np.array] = field(init=False)
    priority: int | None
    corridor: int
    horizontal_acceleration: float = 0
    vertical_acceleration: float = 0

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
    targets: list[Target] | None
    height: float = Field(ge=0)
    dimensions: list[float, float, float, float, float, float, float, float]
    power: list[float, float] = field(init=False)

    def __post_init__(self):
        if self.targets is None:
            self.add_targets()

    # # To determine range of power at corners of each corridor, it is important for us to know min SNR that
    # # we must provide the UAV to survive
    # def power_range(self):
    #     """ Define power range for corridor """
    #     min_power, max_power = 
    #     def SNR(
    #             radar: Radar,
    #             coord: list[float, float]
    #         ):
    #         """ Compute SNR for each edge of the corridor """
    #         distance = np.sqrt((coord[0] - radar.x_pos) ** 2 + (coord[1] - radar.y_pos) ** 2)
    #         numerator = param["peak_transmittion_power"] * param["time_duration_per_beam"] * param["transmit_antenna_gain"] * param["receiver_antenna_gain"] * (param["radar_wavelength"] ** 2) * param["cross_section"]
    #         denominator = ((4 * np.pi) ** 3) * (distance ** 2) * (distance ** 2) * param["loss factor"] * param["boltzmann_constant"] * param["system_temp"]
    #         snr = numerator/denominator
    #         return snr
        
    def add_targets(
            self,
            coord: list[float, float] | None,
            velocity: list[float, float],
            count: int = Field(gt=1),
    ):
        """ Add multiple new targets into the corridor """
        def find_range(self):
            """ Determine (min,max) x and y coordinates for target coordinates"""
            X = [self.dimensions[0], self.dimensions[2], self.dimensions[4], self.dimensions[6]]
            Y = [self.dimensions[1], self.dimensions[3], self.dimensions[5], self.dimensions[7]]
            ans = []
            # find min x
            min, max = X[0], X[0]
            for i in range(1, len(X)):
                if min > X[i]:
                    min = X[i]
                if max < X[i]:
                    max = X[i]
        
            ans.append(min)
            ans.append(max)
        
            # find min y
            min, max = Y[0], Y[0]
            for i in range(1, len(Y)):
                if min > Y[i]:
                    min = Y[i]
                if max < Y[i]:
                    max = Y[i]
        
            ans.append(min)
            ans.append(max)
            return ans
        while count != 0:
            if coord is None:
                coordinates = find_range()
                x_min, x_max, y_min, y_max = coordinates
                coord[0] = np.random.uniform(low=x_min, max=x_max)
                coord[1] = np.random.uniform(low=y_min, max=y_max)

            new_target = Target(
                    id=f"{self.id}t{count}", 
                    x_coordinate= coord[0], 
                    y_coordinate=coord[1], 
                    horizontal_velocity=velocity[0], 
                    vertical_velocity=velocity[1],
                    corridor= self
                )

            self.targets.insert(index=count, object=new_target)
            count -= 1
    
    def new_target(
            self,
            coord: list[float, float] | None,
            velocity: list[float, float]
        ):
        """ Add single new target into the corridor """        
        if coord is None:
            coordinates = self.find_range()
            x_min, x_max, y_min, y_max = coordinates
            coord[0] = np.random.uniform(low=x_min, max=x_max)
            coord[1] = np.random.uniform(low=y_min, max=y_max)

        count = len(self.targets) + 1

        new_target = Target(
                id=f"{self.id}t{count}", 
                x_coordinate= coord[0], 
                y_coordinate=coord[1], 
                horizontal_velocity=velocity[0], 
                vertical_velocity=velocity[1],
                corridor= self
            )

        self.targets.insert(index=count, object=new_target)

    def remove_target(
            self,
            id: int
        ):
        """ Remove target from corridor """
        for i in range(len(self.targets)):
            if self.targets[i].id == id:
                self.targets.pop(index=i)