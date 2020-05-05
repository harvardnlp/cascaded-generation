import sys
import os
import time
import torch
import numpy as np

sys.path.append('/tvm/python')
sys.path.append('/tvm/topi/python')
sys.path.append('/tvm/vta/python')
os.environ['TVM_HOME'] = '/tvm'

import tvm
from tvm import autotvm


@autotvm.template
def logsummul(dtype):
    nn = 512
    bb = 32
    n = tvm.var('n')
    n = tvm.convert(nn)
    b = tvm.var('b')
    b = tvm.convert(bb)
    m, l = n, n
    A = tvm.placeholder((b, n, l), name='A', dtype=dtype)
    B = tvm.placeholder((b, m, l), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, l), name='k')
    k2 = tvm.reduce_axis((0, l), name='k2')

    M = tvm.compute(
        (b, m, n),
        lambda bb, ii, jj: tvm.max(A[bb, jj, k] + B[bb, ii, k], axis=k),
        name='M'
    )
    M2 = tvm.compute(
        (b, m, n),
        lambda bb, ii, jj: tvm.sum(tvm.exp(A[bb, jj, k2] + B[bb, ii, k2]- M[bb, ii, jj]), axis=k2),
        #lambda bb, ii, jj: tvm.sum(tvm.exp(A[bb, jj, k2] + B[bb, ii, k2]- M[bb, ii, jj]), axis=k2),
        name='M2')

    C = tvm.compute(
        (b, m, n),
        lambda bb, ii, jj: tvm.log(M2[bb, ii, jj]) + M[bb, ii, jj],
        name='C')

    s = tvm.create_schedule(C.op)

    AA = s.cache_read(A, "shared", [M])
    AL = s.cache_read(AA, "local", [M])
    BB = s.cache_read(B, "shared", [M])
    BL = s.cache_read(BB, "local", [M])

    AA2 = s.cache_read(A, "shared", [M2])
    AL2 = s.cache_read(AA2, "local", [M2])
    BB2 = s.cache_read(B, "shared", [M2])
    BL2 = s.cache_read(BB2, "local", [M2])

    cfg = autotvm.get_config()
    cfg.define_knob("y_bn", [32, 64, 128])
    cfg.define_knob("x_bn", [32, 64, 128])
    cfg.define_knob("y_t", [8, 32, 64])
    cfg.define_knob("x_t", [2, 4, 8, 32])
    cfg.define_knob("k_split", [1, 2, 8, 16])
    unroll = True

    #cfg.define_knob("y_bn", [64])
    #cfg.define_knob("x_bn", [ 64])
    #cfg.define_knob("y_t", [8])
    #cfg.define_knob("x_t", [8])
    #cfg.define_knob("k_split", [8])

    b, y, x = s[C].op.axis
    y_bn = cfg["y_bn"].val
    x_bn = cfg["x_bn"].val
    by, y = s[C].split(y, y_bn)
    bx, x = s[C].split(x, x_bn)

    y_nthreads = cfg["y_t"].val
    x_nthreads = cfg["x_t"].val
    ty, yi = s[C].split(y, nparts=y_nthreads)
    tx, xi = s[C].split(x, nparts=x_nthreads)
    thread_x = tvm.thread_axis((0, x_nthreads), "threadIdx.x")
    thread_y = tvm.thread_axis((0, y_nthreads), "threadIdx.y")

    s[C].reorder(b, by, bx, ty, tx, yi, xi)
    s[C].bind(b, tvm.thread_axis("blockIdx.z"))
    s[C].bind(by, tvm.thread_axis("blockIdx.y"))
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    if unroll:
        s[C].pragma(yi, "auto_unroll_max_step", 16)

    def cache_split(shared):
        s[shared].compute_at(s[C], tx)
        _, yi, xi = s[shared].op.axis
        k, = s[shared].op.reduce_axis
        ko, ki = s[shared].split(k, cfg["k_split"].val)
        s[shared].reorder(ko, ki, yi, xi)
        if unroll:
            s[shared].pragma(ki, "auto_unroll_max_step", 16)
        return ko, ki
    ko, ki = cache_split(M)
    ko2, ki2 = cache_split(M2)

    def cache_read(shared, AA, AL, BB, BL, ko, ki):
        s[AA].compute_at(s[shared], ko)
        s[AL].compute_at(s[shared], ki)
        s[BB].compute_at(s[shared], ko)
        s[BL].compute_at(s[shared], ki)

        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=y_nthreads)
        tx, ki = s[AA].split(k, nparts=x_nthreads)
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        if unroll:
            s[AA].pragma(yi, "auto_unroll_max_step", 16)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=y_nthreads)
        tx, ki = s[BB].split(k, nparts=x_nthreads)
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        if unroll:
            s[BB].pragma(xi, "auto_unroll_max_step", 16)
    cache_read(M, AA, AL, BB, BL, ko, ki)
    cache_read(M2, AA2, AL2, BB2, BL2, ko2, ki2)

    return s, [A, B, C]


task = autotvm.task.create(logsummul, args=('float32',), target='cuda', target_host="llvm")
with autotvm.apply_history_best('best.log'):
    with tvm.target.create("cuda"):
        s_mult, arg_bufs = logsummul('float32')
        mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
from tvm.contrib.dlpack import to_pytorch_func
logsum_pytorch = to_pytorch_func(mod)

if __name__ == "__main__":
    from tvm import autotvm

    task = autotvm.task.create(logsummul, args=('float32',), target='cuda', target_host="llvm")
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=5),
        runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=100,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])

    def get_abc(shape, constructor=None):
        """Return random a, b and empty c with the same shape.
        """
        np.random.seed(0)
        a = np.random.normal(size=shape).astype(np.float32)
        b = np.random.normal(size=shape).astype(np.float32)
        c = np.empty_like(a)
        if constructor:
            a, b, c = [constructor(x) for x in (a, b, c)]
        return a, b, c

    autotvm.record.pick_best("matmul.log", "best.log")
    with autotvm.apply_history_best('best.log'):
        with tvm.target.create("cuda"):
            s_mult, arg_bufs = logsummul('float32')
            mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
            a, b, c, = get_abc((32, 512, 512), lambda x: tvm.nd.array(x, ctx=tvm.gpu()))

    k = torch.rand(32, 512, 512).cuda()
    import time
    num_trials = 10
    start_time = time.time()
    for _ in range(num_trials):
        #z = torch.matmul(k, k)
        mod(a, b, c)
    torch.cuda.synchronize()
    end_time = time.time()
    op = ""
    print(f"{op}: {(end_time - start_time) / num_trials}")
