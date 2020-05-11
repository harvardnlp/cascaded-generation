import sys
import os
import time
import torch
import numpy as np

#TVM_HOME = '/n/home11/yuntian/tvm/python/tvm'
#os.environ['LD_LIBRARY_PATH'] = (
#            os.environ['LD_LIBRARY_PATH']
#                + f":{TVM_HOME}/build"
##                    #+ os.environ["LLVM_LIB"]
#                    )

import tvm
from tvm import autotvm
import tvm.runtime
import tvm.te

@autotvm.template('hmm_runner_max')
def hmm_runner_max(dtype, nn):
    #nn = 256
    #bb = 32
    n = tvm.runtime.convert(nn)
    m = n
    b =  tvm.te.var("batch")
    t = tvm.te.var("num_step")
    l = n

    k = tvm.te.reduce_axis((0, l), name='k')
    X = tvm.te.placeholder((t, b, n, m), name="X", dtype=dtype)

    s_state = tvm.te.placeholder((t, b, n))
    s_init = tvm.te.compute((1, b, n), lambda a, b, c: 0.0)

    # X is log-potentials
    # bb is batch
    # t is time
    # ii is current table position
    # k is previous table position

    # Algorithm to compute
    M = tvm.te.compute(
        (t, b, n),
        lambda t, bb, ii:
        tvm.te.max(s_state[t-1, bb, k] + X[t-1, bb, k, ii], axis=k),
        name="M")

    s_scan = tvm.te.scan(s_init, M, s_state, inputs=[X])
    # End algorithm to compute

    s = tvm.te.create_schedule(s_scan.op)
    cfg = autotvm.get_config()
    cfg.define_knob("y_t", [8])
    cfg.define_knob("x_t", [16])
    cfg.define_knob("sm", [24])
    cfg.add_flop(1)

    num_thread_y = cfg["y_t"].val
    num_thread_x = cfg["x_t"].val * 3
    num_sm = cfg["sm"].val

    PERSIST_KERNEL = False
    DETECT_GLOBAL_BARRIER = False
    detect_global_barrier = DETECT_GLOBAL_BARRIER

    s = tvm.te.create_schedule(s_scan.op)
    CL = M
    SS = s.cache_read(s_state, "shared", [M])
    SL = s.cache_read(SS, "local", [M])
    # SS2 = s.cache_read(s_state, "shared", [M2])
    # SL2 = s.cache_read(SS2, "local", [M2])

    WhhL = s.cache_read(X, "local", [M])
    # WhhL2 = s.cache_read(X, "local", [M2])

    #First
    ko, ki = s[M].split(s[M].op.reduce_axis[0], nparts=num_thread_y)
    MLF = s.rfactor(M, ko)
    #Second
    # ko2, ki2 = s[M2].split(s[M2].op.reduce_axis[0], nparts=num_thread_y)
    # MLF2 = s.rfactor(M2, ko2)

    block_x = tvm.te.thread_axis((0, num_sm), "blockIdx.x")
    thread_x = tvm.te.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.te.thread_axis((0, num_thread_y), "threadIdx.y")
    block_z = tvm.te.thread_axis((0, b), "blockIdx.z")

    if PERSIST_KERNEL:
        s[s_scan.op].env_threads([block_x, thread_y, thread_x])

    bx, xi = s[s_init].split(s_init.op.axis[2], nparts=num_sm)
    tx, xi = s[s_init].split(xi, nparts=num_thread_x)
    s[s_init].bind(bx, block_x)
    s[s_init].bind(tx, thread_x)
    s[s_init].bind(s_init.op.axis[1], block_z)


    bx, xi = s[CL].split(s[CL].op.axis[2], nparts=num_sm)
    tx, xi = s[CL].split(xi, nparts=num_thread_x)
    s[CL].bind(bx, block_x)
    s[CL].bind(tx, thread_x)

    #s[M].compute_at(s[CL], tx)
    s[M].bind(s[M].op.reduce_axis[0], thread_y)
    s[MLF].compute_at(s[M], s[M].op.reduce_axis[0])
    s[M].bind(s[M].op.axis[1], block_z)


    # Repeat
    # s[M2].compute_at(s[CL], tx)
    # s[M2].bind(s[M2].op.reduce_axis[0], thread_y)
    # s[MLF2].compute_at(s[M2], s[M2].op.reduce_axis[0])
    s[WhhL].compute_at(s[MLF], MLF.op.axis[3])
    # s[WhhL2].compute_at(s[MLF2], MLF2.op.axis[3])

    kr, ki = s[MLF].split(MLF.op.reduce_axis[0], nparts=1)
    ko, ki = s[MLF].split(ki, factor=4)
    s[SS].compute_at(s[MLF], kr)
    s[SL].compute_at(s[MLF], ko)

    xo, xi = s[SS].split(SS.op.axis[2], factor=num_thread_x * num_thread_y * 3)
    ty, xi = s[SS].split(xi, nparts=num_thread_y)
    tx, xi = s[SS].split(xi, nparts=num_thread_x)

    s[SS].bind(tx, thread_x)

    return s, [X, s_scan]







def log_eye(K, dtype, device):
    x = torch.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

def log_eye_cat(x):
    K = x.shape[-1]
    batch = x.shape[1]
    return torch.cat([
    x,
    log_eye(K, x.dtype, x.device).view(1, 1, K, K).expand(1, batch, K, K),
    ], dim=0)


def fb_max(size):
    #with autotvm.apply_history_best(f'best_hmm_k{size}.log'):
    with tvm.target.create("cuda"):
        s_mult, arg_bufs = hmm_runner_max('float32', size)
        from tvm.contrib.dlpack import to_pytorch_func
        mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
        hmm_pytorch_max = to_pytorch_func(mod)
    def fb(x):
        time, batch, size, _ = x.shape
        forward = torch.zeros(time+1, batch, size).cuda()
        y = log_eye_cat(x).cuda()
        hmm_pytorch_max(y, forward)
        del y
        #torch.cuda.empty_cache()

        backward = torch.zeros(time+1, batch, size).cuda()
        y = log_eye_cat(x.flip(0).transpose(-2, -1)).contiguous().cuda()
        hmm_pytorch_max(y, backward)
        del y
        #torch.cuda.empty_cache()

        #check = (forward.view(time+1, batch, size)+
        #    backward.flip(0).view(time+1, batch, size))
        y = x.view(time, batch, size, size).transpose(-2, -1).contiguous().cuda()
        y += forward[:-1].view(time, batch, 1, size)
        y += backward[:-1].flip(0).view(time, batch, size, 1)
        marginals = y.transpose(-2, -1)

        #marginals = (forward[:-1].view(time, batch, 1, size) +
        #             backward[:-1].flip(0).view(time, batch, size, 1) +
        #             x.view(time, batch, size, size).transpose(-2, -1)).transpose(-2, -1)
        #return forward, backward, marginals, check
        return marginals
    return fb



sizes = [26, 50, 64, 128, 250, 256, 500, 512, 1024]

if __name__ == "__main__":
    from tvm import autotvm
    import logging
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    for size in sizes:
        task = autotvm.task.create('hmm_runner_max', args=('float32', size),
                                   target='cuda', target_host="llvm")

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=5),
            runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=100,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(f'hmm_k{size}.log')])

        autotvm.record.pick_best(f"hmm_k{size}.log", f"best_hmm_k{size}.log")

    print(fb_max(256)(torch.ones(10, 32, 256, 256).cuda()))
