import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from model.function import param
from pydantic import Field
from pydantic.dataclasses import dataclass

@dataclass
class Dataset():
    features: int = Field(gt = 0)
    size: int = Field(gt = 0)
    outputs: int = Field(gt = 0)
    min: float = Field(gt=0)
    max: float = Field(le=param["revisit_interval"])

    def validator(
            sample: np.array,
            size: int,
            condition: float
        ):
        """ Check if each sample satisfies its constraint """
        sum = 0
        for i in range(size):
            sum += sample[i]
        return sample if sum > condition else None


    def create_sample(self):
        """ Create a single sample instance for dataset and validate """
        dt_range, min, max = self.features - 1, self.min, self.max
        X = np.array((1,self.features+1))
        for i in range(dt_range):
            X[i] = (np.random.uniform(min,max))

        # Check if sample is valid
        X = self.validator(X, len(X), param["revisit_interval"])

        # Find its objective value

    def create_initial_dataset(
            self, 
            sample_size: int
        ):
        """ Create entire dataset of sample_size """
        dt, counter = [], sample_size
        while (counter):
            X = self.create_sample()
            if X is not None:
                dt.append(X)
                counter -= 1

        return dt

    def candidate_dataset(
            index1: int,
            index2: int,
            X: np.array
        ):
        """ Create dataset with candidate samples for each pair of (t,p) """
        

@dataclass
class BayesianOpt():
    dataset: np.array
