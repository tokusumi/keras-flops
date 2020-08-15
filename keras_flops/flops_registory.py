import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import graph_util
from tensorflow.python.profiler.internal.flops_registry import _reduction_op_flops


@ops.RegisterStatistics("FusedBatchNormV3", "flops")
def _FusedBatchNormV3(graph, node):
    """inference is supportted"""
    return ops.OpStats("flops", 10)


@ops.RegisterStatistics("Max", "flops")
def _flops_max(graph, node):
    """inference is supportted"""
    # reduction - comparison, no finalization
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

