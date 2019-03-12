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

from copy import deepcopy
from random import randrange, sample

from nni.networkmorphism_tuner.graph import NetworkDescriptor
from nni.networkmorphism_tuner.layers import (
    StubAvgPooling33, StubConv7117, StubDense, StubDilConv33, StubDilConv55,
    StubMaxPooling33, StubReLU, StubSepConv33, StubSepConv55, StubSepConv77,
    get_batch_norm_class, get_conv_class, get_dropout_class, get_pooling_class,
    is_layer)
from nni.networkmorphism_tuner.utils import Constant


def to_wider_graph(graph):
    ''' wider graph
    '''
    weighted_layer_ids = graph.wide_layer_ids()
    weighted_layer_ids = list(
        filter(lambda x: graph.layer_list[x].output.shape[-1],
               weighted_layer_ids))
    wider_layers = sample(weighted_layer_ids, 1)

    for layer_id in wider_layers:
        layer = graph.layer_list[layer_id]
        if is_layer(layer, "Conv"):
            n_add = layer.filters
        else:
            n_add = layer.units

        graph.to_wider_model(layer_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    ''' skip connection graph
    '''
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layer_ids = graph.skip_connection_layer_ids()
    valid_connection = []
    for skip_type in sorted(
        [NetworkDescriptor.ADD_CONNECT, NetworkDescriptor.CONCAT_CONNECT]):
        for index_a in range(len(weighted_layer_ids)):
            for index_b in range(len(weighted_layer_ids))[index_a + 1:]:
                valid_connection.append((index_a, index_b, skip_type))

    if not valid_connection:
        return graph
    for index_a, index_b, skip_type in sample(valid_connection, 1):
        a_id = weighted_layer_ids[index_a]
        b_id = weighted_layer_ids[index_b]
        if skip_type == NetworkDescriptor.ADD_CONNECT:
            graph.to_add_skip_model(a_id, b_id)
        else:
            graph.to_concat_skip_model(a_id, b_id)
    return graph


def create_new_layer(layer, n_dim):
    ''' create  new layer for the graph
    '''

    input_shape = layer.output.shape
    conv_deeper_classes = [StubConv7117, StubDilConv33, StubDilConv55, StubMaxPooling33, StubAvgPooling33, StubSepConv33, StubSepConv55, StubSepConv77]

    # It is in the conv layer part.
    layer_class = sample(conv_deeper_classes, 1)[0]

    if layer_class is StubAvgPooling33 or layer_class is StubMaxPooling33:
        new_layer = layer_class(kernel_size=3, stride=1, padding=1)
    elif layer_class is StubConv7117:
        new_layer = layer_class(input_shape[-1])
    elif layer_class is StubDilConv33:
        new_layer = layer_class(input_shape[-1], input_shape[-1], 3, 1, 2, 2)
    elif layer_class is StubDilConv55:
        new_layer = layer_class(input_shape[-1], input_shape[-1], 5, 1, 4, 2)
    elif layer_class is StubSepConv33:
        new_layer = layer_class(input_shape[-1], input_shape[-1], 3, 1, 1)
    elif layer_class is StubSepConv55:
        new_layer = layer_class(input_shape[-1], input_shape[-1], 5, 1, 2)
    elif layer_class is StubSepConv77:
        new_layer = layer_class(input_shape[-1], input_shape[-1], 7, 1, 3)

    return new_layer


def to_deeper_graph(graph):
    ''' deeper graph
    '''
    # we only deep the conv block here
    weighted_layer_ids = graph.deep_conv_layer_ids()
    if len(weighted_layer_ids) >= Constant.MAX_LAYERS:
        return None

    deeper_layer_ids = sample(weighted_layer_ids, 1)

    for layer_id in deeper_layer_ids:
        layer = graph.layer_list[layer_id]
        new_layer = create_new_layer(layer, graph.n_dim)
        graph.to_deeper_model(layer_id, new_layer)
    return graph


def legal_graph(graph):
    '''judge if a graph is legal or not.
    '''

    descriptor = graph.extract_descriptor()
    skips = descriptor.skip_connections
    if len(skips) != len(set(skips)):
        return False
    return True


def transform(graph):
    '''core transform function for graph.
    '''

    graphs = []
    for _ in range(Constant.N_NEIGHBOURS * 2):
        random_num = randrange(2)
        temp_graph = None
        if random_num == 0:
            temp_graph = to_deeper_graph(deepcopy(graph))
        elif random_num == 1:
            temp_graph = to_skip_connection_graph(deepcopy(graph))

        if temp_graph is not None and temp_graph.size(
        ) <= Constant.MAX_MODEL_SIZE:
            graphs.append(temp_graph)

        if len(graphs) >= Constant.N_NEIGHBOURS:
            break

    return graphs
