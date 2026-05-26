import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functools
import logging
import multiprocessing
import operator
import random
import math
import datetime
import numpy as np
import pandas as pd
import pymannkendall as mk
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Reshape, Input, Conv2DTranspose, UpSampling2D, GlobalAveragePooling2D
from deap import base, creator, tools
from statsmodels.nonparametric.smoothers_lowess import lowess


class History:
    def __init__(self):
        self.pop = None
        self.best = None
        self.maxs = None
        self.avgs = None


class CAE:
    def __init__(self, time_w, n_features, n_filters=16, activation='elu'):
        self.time_w = time_w
        self.n_features = n_features
        self.n_filters = n_filters
        self.activation = activation

        self.model = self._build_model()

        particle_size = 0
        for w in self.model.weights:
            particle_size += np.size(w)
        self.particle_size = particle_size

        self.pop = None
        self.best = None
        self.maxs = []
        self.avgs = []

    def _build_model(self):
        inp = Input(shape=(self.time_w, self.n_features, 1))
        enc = Conv2D(self.n_filters, (self.time_w, 1), activation=self.activation, trainable=False)(inp)
        enc = Conv2D(1, (1, self.n_features), activation=self.activation, trainable=False)(enc)

        dec = UpSampling2D((self.n_features, 1))(enc)
        dec = Conv2DTranspose(self.n_filters, (self.n_features, 1), padding='same', activation=self.activation)(dec)
        dec = UpSampling2D((1, self.time_w))(dec)
        dec = Conv2DTranspose(1, (1, self.time_w), padding='same', activation=self.activation)(dec)
        dec = Reshape((self.time_w, self.n_features, 1))(dec)

        model = Model(inp, [enc, dec])
        return model

    def _eval_fitness(self, particle, data):
        fits = []
        for rtf in data:
            fits.append(_fitness(model=self.model, ind=particle, rtf=rtf, window=self.time_w))
        return np.mean(fits),

    def _set_weights(self, ind):
        weights = []
        s = 0
        for w in self.model.weights:
            weights.append(np.reshape(ind[s:s + np.size(w)], tuple(w.shape)))
            s += np.size(w)
        self.model.set_weights(weights)

    def train(self, data, n_gen=200, pop_size=20, log_filepath='data/output/logs/'):
        logger = _get_logger(log_filepath)

        if "FitnessMax" in creator.__dict__:
            del creator.FitnessMax
        if "Particle" in creator.__dict__:
            del creator.Particle

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

        def generate(size, pmin, pmax, smin, smax):
            part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
            part.speed = [random.uniform(smin, smax) for _ in range(size)]
            part.smin = smin
            part.smax = smax
            return part

        pool = multiprocessing.Pool(processes=8)
        toolbox = base.Toolbox()
        toolbox.register("particle", generate, size=self.particle_size, pmin=-0.1, pmax=0.1, smin=-0.005, smax=0.005)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", _update_particle, phi1=2.0, phi2=2.0)
        toolbox.register("evaluate", functools.partial(self._eval_fitness, data=data))
        toolbox.register("map", pool.map)

        if not self.pop:
            self.pop = toolbox.population(n=pop_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        avgs_ma = []
        logger.debug("PSO started running...")
        for g in range(n_gen):
            # Evaluate all particles in parallel
            fitnesses = toolbox.map(toolbox.evaluate, self.pop)
            for part, fit in zip(self.pop, fitnesses):
                part.fitness.values = fit
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not self.best or self.best.fitness < part.fitness:
                    self.best = creator.Particle(part)
                    self.best.fitness.values = part.fitness.values

            # Update particles
            for part in self.pop:
                toolbox.update(part, self.best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(self.pop), **stats.compile(self.pop))
            logger.debug(logbook.stream)

            self.maxs.append(self.best.fitness.values[0])
            self.avgs.append(logbook[-1]['avg'])

            if g > 10:
                avgs_ma.append(np.mean(self.avgs[-10:]))

        pool.close()
        pool.join()

        history = History()
        
        history.pop = self.pop
        history.best = self.best
        history.maxs = self.maxs
        history.avgs = self.avgs

        self._set_weights(self.best)

        _close_logger(logger)

        return history

    def get_hi(self, data):
        x = _apply_time_window(data, self.time_w)
        return _monotonize_lowess(np.squeeze(self.model.predict(x, verbose=0)[0]))


def _apply_time_window(arr, w):
    n = arr.shape[1] - w + 1  # target shape: (1, samples, features, 1)
    l = []
    for i in range(n):
        l.append(arr[:, i:i + w, :, :])
    return np.concatenate(l, axis=0)


def _update_particle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


def _fitness(model, ind, rtf, window):
    weights = []
    s = 0
    for w in model.weights:
        weights.append(np.reshape(ind[s:s + np.size(w)], tuple(w.shape)))
        s += np.size(w)
    model.set_weights(weights)
    x = _apply_time_window(rtf, window)
    hi = np.squeeze(model.predict(x, verbose=0)[0])
    mk_tau = mk.original_test(hi).Tau
    mse = np.mean(np.square(x - model.predict(x, verbose=0)[1]))
    std = np.mean([hi[i:i + 30].std() for i in range(len(hi) - 30)])
    hi = _smoothen(_monotonize_lowess(hi, frac=0.05), frac=0.2)
    dif_0 = np.abs(hi[0])
    dif_1 = np.abs(1 - hi[-1])
    return mk_tau - 3 * mse - dif_0 - dif_1 - 3 * std


def _get_logger(log_filepath):
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    path = os.path.join(
        log_filepath, log_filename)
    fh = logging.FileHandler(path)
    ch = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def _smoothen(hi, frac):
    return lowess(hi, np.arange(len(hi)), frac=frac, return_sorted=False)


def _monotonize_lowess(hi, frac=0.05):
    trend = lowess(hi, np.arange(len(hi)), frac=frac, return_sorted=False)
    res = hi - trend
    trend_mon = np.squeeze(pd.DataFrame(trend).cummax().to_numpy())
    return trend_mon + res
