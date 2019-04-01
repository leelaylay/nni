# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import json
import math
import os
import random
from copy import deepcopy
from functools import total_ordering
from queue import PriorityQueue

import lightgbm as lgb
import numpy as np

from nni.networkmorphism_tuner.graph import graph_to_json, json_to_graph
from nni.networkmorphism_tuner.graph_transformer import transform
from nni.networkmorphism_tuner.layers import is_layer
from nni.networkmorphism_tuner.utils import Constant, OptimizeMode


class IncrementalRegressionProcess:
    """IncrementalRegressionProcess.
    """

    def __init__(self):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 15,
            'learning_rate': 1e-2,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.n_estimators = 100
        self.gbm = None

        self.num_limit = 20
        self._x = None
        self._y = None

    def fit(self, train_x, train_y):
        """ Fit the regressor with more data.
        Args:
            train_x: A list of network features.
            train_y: A list of metric values.
        """
        if self._x is None and self._y is None:
            self._x = np.array(train_x)
            self._y = np.array(train_y)
        else:
            self.incremental_fit(train_x, train_y)

    def incremental_fit(self, train_x, train_y):
        """ Incrementally fit the regressor. """
        train_x, train_y = np.array(train_x), np.array(train_y)
        self._x = np.concatenate((self._x, train_x), axis=0)
        self._y = np.concatenate((self._y, train_y), axis=0)
        return self

    def predict(self, valid_x):
        """Predict the result.
        Args:
            train_x: A list of network features.
        Returns:
            y_pred: The predicted value.
        """
        if len(self._x) <= self.num_limit:
            return np.array([0.0] * len(valid_x))

        lgb_train = lgb.Dataset(self._x, self._y)
        self.gbm = lgb.train(
            self.params, lgb_train, self.n_estimators, verbose_eval=False)
        valid_x = np.array(valid_x)
        y_pred = self.gbm.predict(
            valid_x, num_iteration=self.gbm.best_iteration)

        return y_pred


class NNOptimizer:
    """ Optimizer for neural architectures.
    Attributes:
        searcher: The Searcher which is calling the optimizer.
        t_min: The minimum temperature for simulated annealing.
        metric: An instance of the Metric subclasses.
        value_predictor: A Simple Value Predictor for optimization.
        search_tree: The network morphism search tree.
    """

    def __init__(self, searcher, t_min, optimizemode):
        self.searcher = searcher
        self.t_min = t_min
        self.optimizemode = optimizemode
        self.value_predictor = IncrementalRegressionProcess()
        self.search_tree = SearchTree()

    def fit(self, x_queue, y_queue):
        """ Fit the optimizer with new architectures and performances.
        Args:
            x_queue: A list of nn features.
            y_queue: A list of metric values.
        """
        self.value_predictor.fit(x_queue, y_queue)

    def generate(self, features):
        """Generate new architecture.
        Args:
            features: All the searched neural architectures.
        Returns:
            graph: An instance of Graph. A morphed neural network with weights.
            father_id: The father node ID in the search tree.
        """
        model_ids = self.search_tree.adj_list.keys()

        target_graph = None
        father_id = None
        features = deepcopy(features)
        elem_class = Elem
        if self.optimizemode is OptimizeMode.Maximize:
            elem_class = ReverseElem

        # Initialize the priority queue.
        pq = PriorityQueue()
        temp_list = []
        for model_id in model_ids:
            metric_value = self.searcher.get_metric_value_by_id(model_id)
            temp_list.append((metric_value, model_id))
        temp_list = sorted(temp_list)
        for metric_value, model_id in temp_list:
            graph = self.searcher.load_model_by_id(model_id)
            graph.clear_operation_history()
            graph.clear_weights()
            pq.put(elem_class(metric_value, model_id, graph))

        t = 1.0
        t_min = self.t_min
        alpha = 0.9
        opt_acq = self._get_init_opt_acq_value()
        temp_step_graph_list = []
        print("pq.qsize():{}".format(pq.qsize()))
        while not pq.empty() and t > t_min:
            elem = pq.get()
            if self.optimizemode is OptimizeMode.Maximize:
                temp_exp = min((elem.metric_value - opt_acq) / t, 1.0)
            else:
                temp_exp = min((opt_acq - elem.metric_value) / t, 1.0)
            ap = math.exp(temp_exp)
            if ap >= random.uniform(0, 1):
                for temp_graph in transform(elem.graph):
                    temp_features = temp_graph.extract_features()
                    if contain(features, temp_features):
                        continue
                    temp_acq_value = self.acq(temp_features)
                    features.append(temp_features)
                    pq.put(elem_class(temp_acq_value, elem.father_id, temp_graph))
                    temp_step_graph_list.append([temp_acq_value, temp_graph])
                    if self._accept_new_acq_value(opt_acq, temp_acq_value):
                        opt_acq = temp_acq_value
                        father_id = elem.father_id
                        target_graph = deepcopy(temp_graph)
            t *= alpha

        print("Father_id():{}".format(father_id))
        # Did not found a not duplicated architecture
        if father_id is None:
            return None, None, []
        nm_graph = self.searcher.load_model_by_id(father_id)
        for args in target_graph.operation_history:
            getattr(nm_graph, args[0])(*list(args[1:]))
        return nm_graph, father_id, temp_step_graph_list, opt_acq

    def acq(self, graph_features):
        ''' estimate the value of generated graph
        '''
        predict_value = self.value_predictor.predict(
            np.array([graph_features]))
        return predict_value

    def _get_init_opt_acq_value(self):
        if self.optimizemode is OptimizeMode.Maximize:
            return -np.inf
        return np.inf

    def _accept_new_acq_value(self, opt_acq, temp_acq_value):
        if temp_acq_value >= opt_acq and self.optimizemode is OptimizeMode.Maximize:
            return True
        if temp_acq_value <= opt_acq and self.optimizemode is OptimizeMode.Minimize:
            return True
        return False

    def add_child(self, father_id, model_id):
        ''' add child to the search tree
        Arguments:
            father_id {int} -- father id
            model_id {int} -- model id
        '''

        self.search_tree.add_child(father_id, model_id)


@total_ordering
class Elem:
    """Elements to be sorted according to metric value."""

    def __init__(self, metric_value, father_id, graph):
        self.father_id = father_id
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    """Elements to be reversely sorted according to metric value."""

    def __lt__(self, other):
        return self.metric_value > other.metric_value


def contain(features, target_features):
    """Check if the target descriptor is in the features."""
    target_features_array = np.array(target_features)
    for item in features:
        if (item == target_features_array).all():
            return True
    return False


class SearchTree:
    """The network morphism search tree."""

    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        ''' add child to search tree itself.
        Arguments:
            u {int} -- father id
            v {int} --  child id
        '''

        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []

    def get_dict(self, u=None):
        """ A recursive function to return the content of the tree in a dict."""
        if u is None:
            return self.get_dict(self.root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {"name": u, "children": children}
        return ret
