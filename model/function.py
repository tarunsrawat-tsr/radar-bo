""" Helper functions used in simulation """

import numpy as np
from target import Target
from radar import Radar

param = {
    "peak_transmittion_power": 0,
    "transmit_antenna_gain": 0,
    "receiver_antenna_gain": 0,
    "radar_wavelength": 0,
    "cross_section": 0,
    "loss_factor": 0,
    "revisit_interval": 2.5, # We have considered same revisit interval for all targets (2.5 sec)
    "maneuverability_constant": 25,
    "ref_SNR": 0,
    "ref_dwell_time": 0,
    "ref_target_distance": 0,
    "ref_distance_cov": 0,
    "ref_azimuth_cov": 0,
    "phase_delay":0,
    "time_duration_per_beam": 0,
    "boltzmann_constant":0,
    "system_temp": 0
}

def transition_matrix(
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Constructing transition matrix for state space model of target """
    f = np.array([
        [1, 0, revisit_interval, 0],
        [0, 1, 0, revisit_interval],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return f

def updated_target_model(
        target: Target,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Generate state space model for target at time slot (t) """
    xt = target.model_history[-1]
    wt = maneuverability_noise()
    f = transition_matrix(revisit_interval)
    xt_new = f @ xt + wt
    target.model_history.append(xt_new)
    return xt_new


def maneuverability_covariance(
        revisit_interval: float = param["revisit_interval"],
        maneuverability_constant: float = param["maneuverability_constant"]
    ):
    """ Generate covariance matrix for target state space model at time slot (t)"""
    qt = maneuverability_constant * np.array([
        [revisit_interval ** 4 / 4, 0, revisit_interval ** 3 / 2, 0],
        [0, revisit_interval ** 4 / 4, 0, revisit_interval ** 3 / 2],
        [revisit_interval ** 3 / 2, 0, revisit_interval ** 2, 0],
        [0, revisit_interval ** 3 / 2, 0, revisit_interval ** 2]
    ])
    return qt

def maneuverability_noise(
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Generate zero mean gaussian noise for target state space model at time slot (t)"""
    mean = np.zeros(4)
    cov = maneuverability_covariance(revisit_interval)
    wt = np.random.multivariate_normal(mean, cov).reshape(4,1)
    return wt


def SNR(
        target: Target,
        radar: Radar,
    ):
    """ Calculates SNR of target from Radar at time slot (t)"""
    # Add power parameters as well
    distance = np.sqrt((target.x_pos - radar.x_pos) ** 2 + (target.y_pos - radar.y_pos) ** 2)
    snr = param["ref_SNR"] * (target.dwell_time/param["ref_dwell_time"]) * ((distance / param["ref_target_distance"]) ** -4)
    return snr

def r_variance(
        target: Target,
        radar: Radar
    ):
    """ Compute distance covariance for measurement noise """
    var = param["ref_distance_cov"] / SNR(target, radar)
    return var

def theta_variance(
        target: Target,
        radar: Radar
    ):
    """ Compute azimuth covariance for measurement noise """
    var = param["ref_azimuth_cov"] / SNR(target, radar)
    return var

def measurement_covariance(
        target: Target,
        radar: Radar
    ):
    """ Compute covariance matrix for radar measurement """
    cov = np.array([
        [r_variance(target, radar), 0],
        [0, theta_variance(target, radar)]
    ])
    return cov

def measurement_noise(
        target: Target,
        radar: Radar
    ):
    """ Compute measurement noise for target from radar """
    noise = np.array([
        [r_variance(target, radar)],
        [theta_variance(target, radar)]
    ])
    return noise

def non_linear_fun(
        x1: float,
        x2: float,
        y1: float,
        y2: float
    ):
    """ Maps linear function into non linear fun """
    ht = np.array([
        [np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)],
        [np.arctan2(y1, x1)]
    ])
    return ht

def radar_model(
        target: Target,
        radar: Radar
    ):
    """ Compute distance and angle of target from radar eq(4) """
    t = target.model_history[-1]
    ht = non_linear_fun(t[0], radar.x_pos, t[1], radar.y_pos)
    zt = ht + measurement_noise(target, radar)
    return zt

def EKF_Jacobian(
        target: Target,
    ):
    """ Compute jacobian matrix for EKF eq(7) """
    t = target.model_history[-1]
    ht = np.array([
        [(t[0]/np.sqrt(t[0] ** 2 + t[1] ** 2)), (t[1]/np.sqrt(t[0] ** 2 + t[1] ** 2)), 0, 0],
        [(-t[1]/t[0] ** 2 + t[1] ** 2), (t[0]/t[0] ** 2 + t[1] ** 2), 0, 0]
    ])
    return ht

def prior_state_estimation_covariance(
        target: Target,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Compute state estimation covariance matrix at time slot (t) given at time slot (t-1) eq(9) """
    ft = transition_matrix(revisit_interval)
    pt_prev = target.state_estimation[-1]
    qt = maneuverability_covariance(revisit_interval)
    pt_new = (ft @ pt_prev @ ft.T) + qt
    return pt_new

def prior_state_estimate(
        target: Target,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Estimates state of target at time slot (t) given information about state at time slot (t-1) eq(10) """
    ft = transition_matrix(revisit_interval)
    xt_prev = target.model_history[-1]
    xt_estimated = ft @ xt_prev
    return xt_estimated

def kalman_gain(
        target: Target,
        radar: Radar,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Computes Kalma Gain eq(11) """
    pt = prior_state_estimation_covariance(target, revisit_interval)
    ht = EKF_Jacobian(target)
    rt = measurement_covariance(target, radar)

    kt = pt @ ht.T @ ((ht @ pt @ ht.T + rt) ** -1)
    return kt

def post_state_estimate(
        target: Target,
        radar: Radar,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Corrects the estimated state eq(12) """
    t = target.model_history[-1]
    xt_prev = prior_state_estimate(target,revisit_interval)
    kt = kalman_gain(target,radar,revisit_interval)
    zt = radar_model(target,radar)
    ht_prev = non_linear_fun(t[0], radar.x_pos, t[1], radar.y_pos)
    xt_post = xt_prev + kt @ (zt - ht_prev)
    target.model_history.append(xt_post)
    return xt_post

def post_state_estimation_covariance(
        target: Target,
        radar: Radar,
        revisit_interval: float = param["revisit_interval"]
    ):
    """ Corrects radar's estimation about covariance eq(13) """
    i = np.eye(4,4)
    kt = kalman_gain(target,radar,revisit_interval)
    ht = EKF_Jacobian(target)
    pt_prev = prior_state_estimation_covariance(target, revisit_interval)
    pt_post = (i - (kt @ ht)) @ pt_prev
    target.state_estimation.append(pt_post)
    return pt_post

def tracking_cost(
        target: Target,
    ):
    """ Computes tracking cost of target at time slot (t) eq(14) """
    pt = target.state_estimation[-1]
    E = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    ct = np.trace(E @ pt @ E.T)
    return ct

def scanning_time(
        time_duration_per_beam: float = param["time_duration_per_beam"]
    ):
    """ Compute total scanning time in time slot (t) eq(16) """
    ts = (360/param["phase_delay"]) * time_duration_per_beam
    return ts

def received_power(
        target: Target,
        radar: Radar,
    ):
    """ The power received by radar eq(17) """
    t = target.model_history[-1]
    rt = np.sqrt((t[0] - radar.x_pos) ** 2 + (t[1] - radar.y_pos) ** 2)
    numerator = param["peak_transmittion_power"] * param["transmit_antenna_gain"] * param["receiver_antenna_gain"] * (param["radar_wavelength"] ** 2) * param["cross_section"]
    denominator = ((4 * np.pi) ** 3) * (rt ** 2) * (rt ** 2) * param["loss_factor"]
    pr = numerator/denominator
    return pr

def receiver_noise():
    """ Total noise power at radar receiver eq(18) """
    noise = (param["boltzmann_constant"] * param["system_temp"]) / param["time_duration_per_beam"]
    return noise

def SNR_scan(
        target: Target,
        radar: Radar,
    ):
    """ Scanning SNR """
    t = target.model_history[-1]
    distance = np.sqrt((t[0] - radar.x_pos) ** 2 + (t[1] - radar.y_pos) ** 2)
    numerator = param["peak_transmittion_power"] * param["time_duration_per_beam"] * param["transmit_antenna_gain"] * param["receiver_antenna_gain"] * (param["radar_wavelength"] ** 2) * param["cross_section"]
    denominator = ((4 * np.pi) ** 3) * (distance ** 2) * (distance ** 2) * param["loss factor"] * param["boltzmann_constant"] * param["system_temp"]
    snr_scan = numerator/denominator
    return snr_scan

def maximum_detectable_range():
    """ Maximum radius the radar can cover eq(20) """
    numerator = param["peak_transmittion_power"] * param["time_duration_per_beam"] * param["transmit_antenna_gain"] * param["receiver_antenna_gain"] * (param["radar_wavelength"] ** 2) * param["cross_section"]
    denominator = ((4 * np.pi) ** 3) * param["loss_factor"] * param["boltzmann_constant"] * param["system_temp"] * SNR_scan()
    r4 = numerator/denominator
    r = (r4) ** (1/4)
    return r