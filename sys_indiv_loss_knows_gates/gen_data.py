import numpy as np
import matplotlib.pyplot as plt

def gen_block(width, height):
    block = np.zeros((height, width))
    step_width = int(width/height)
    for i in range(height):
        block[i,i*step_width:i*step_width+step_width] = 1
    return block

def gen_hier_data(num_objects):
    data = gen_block(num_objects, 1)
    height = 2
    while height <= num_objects:
        new_block = gen_block(num_objects, height)
        data = np.vstack([data, new_block])
        height = height*2
    return data

def gen_ring():
    max_val = 0.5
    ring_data = np.array([
                 np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])*max_val,
                 np.array([0.0,0.33,0.66,1.0,1.0,0.66,0.33,0.0])*max_val,
                 np.array([0.33,0.66,1.0,1.0,0.66,0.33,0.0,0.0])*max_val,
                 np.array([0.66,1.0,1.0,0.66,0.33,0.0,0.0,0.33])*max_val,
                 np.array([1.0,1.0,0.66,0.33,0.0,0.0,0.33,0.66])*max_val,
                 np.array([1.0,0.66,0.33,0.0,0.0,0.33,0.66,1.0])*max_val,
                 np.array([0.66,0.33,0.0,0.0,0.33,0.66,1.0,1.0])*max_val,
                 np.array([0.33,0.0,0.0,0.33,0.66,1.0,1.0,0.66])*max_val,
                 np.array([0.0,0.0,0.33,0.66,1.0,1.0,0.66,0.33])*max_val,
                 np.array([0.0,0.33,0.66,1.0,1.0,0.66,0.33,0.0])*max_val,
                 np.array([0.33,0.66,1.0,1.0,0.66,0.33,0.0,0.0])*max_val,
                 np.array([0.66,1.0,1.0,0.66,0.33,0.0,0.0,0.33])*max_val,
                 np.array([1.0,1.0,0.66,0.33,0.0,0.0,0.33,0.66])*max_val,
                 np.array([1.0,0.66,0.33,0.0,0.0,0.33,0.66,1.0])*max_val,
                 np.array([0.66,0.33,0.0,0.0,0.33,0.66,1.0,1.0])*max_val,])
    return ring_data

def gen_data3(num_obj, diff_struct):
    hier_data = gen_hier_data(num_obj)
    zeros = np.zeros((int(2*num_obj - 1), num_obj))
    if diff_struct:
        ring_data = gen_ring()*2.0
        cross_data = np.copy(hier_data)
        cross_data[3] = np.array([1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        cross_data[4] = np.array([0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0])
        cross_data[5] = np.array([0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0])
        cross_data[6] = np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0])
        cross_data = cross_data*1.0

        common = np.hstack([hier_data,hier_data,hier_data])
        context1 = np.hstack([hier_data,zeros,zeros])
        context2 = np.hstack([zeros,ring_data,zeros])
        context3 = np.hstack([zeros,zeros,cross_data])
        Y = np.vstack([common, context1, context2, context3])
        X = np.hstack([np.eye(num_obj) for _ in range(3)])
        contr = np.hstack([np.ones(num_obj)*i for i in range(3)]).astype(int)
        context = np.zeros((3,X.shape[1]))
        context[contr, np.arange(X.shape[1])] = 1
        X = np.vstack([X, context])
    else:
        common = np.hstack([hier_data,hier_data,hier_data])
        context1 = np.hstack([hier_data,zeros,zeros])
        context2 = np.hstack([zeros,hier_data,zeros])
        context3 = np.hstack([zeros,zeros,hier_data])
        Y = np.vstack([common, context1, context2, context3])
        X = np.hstack([np.eye(num_obj) for _ in range(3)])
        contr = np.hstack([np.ones(num_obj)*i for i in range(3)]).astype(int)
        context = np.zeros((3,X.shape[1]))
        context[contr, np.arange(X.shape[1])] = 1
        X = np.vstack([X, context])
    return X,Y

def gen_data4(num_obj):
    hier_data = gen_hier_data(num_obj)
    zeros = np.zeros((int(2*num_obj - 1), num_obj))
    common = np.hstack([hier_data,hier_data,hier_data,hier_data])
    context1 = np.hstack([hier_data,zeros,zeros,zeros])
    context2 = np.hstack([zeros,hier_data,zeros,zeros])
    context3 = np.hstack([zeros,zeros,hier_data,zeros])
    context4 = np.hstack([zeros,zeros,zeros,hier_data])
    Y = np.vstack([common, context1, context2, context3, context4])
    X = np.hstack([np.eye(num_obj) for _ in range(4)])
    contr = np.hstack([np.ones(num_obj)*i for i in range(4)]).astype(int)
    context = np.zeros((4,X.shape[1]))
    context[contr, np.arange(X.shape[1])] = 1
    X = np.vstack([X, context])
    return X,Y

def gen_data5(num_obj):
    hier_data = gen_hier_data(num_obj)
    zeros = np.zeros((int(2*num_obj - 1), num_obj))
    common = np.hstack([hier_data,hier_data,hier_data,hier_data,hier_data])
    context1 = np.hstack([hier_data,zeros,zeros,zeros,zeros])
    context2 = np.hstack([zeros,hier_data,zeros,zeros,zeros])
    context3 = np.hstack([zeros,zeros,hier_data,zeros,zeros])
    context4 = np.hstack([zeros,zeros,zeros,hier_data,zeros])
    context5 = np.hstack([zeros,zeros,zeros,zeros,hier_data])
    Y = np.vstack([common, context1, context2, context3, context4, context5])
    X = np.hstack([np.eye(num_obj) for _ in range(5)])
    contr = np.hstack([np.ones(num_obj)*i for i in range(5)]).astype(int)
    context = np.zeros((5,X.shape[1]))
    context[contr, np.arange(X.shape[1])] = 1
    X = np.vstack([X, context])
    return X,Y
