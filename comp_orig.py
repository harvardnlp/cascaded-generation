import torch
import genbmm, math
import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home11/yuntian/genbmm/opt/hmm.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
fbs = {}
import time
import torch.cuda
torch.cuda.synchronize()

def max_marginals(scores):
    B, N, C, C = scores.shape
    def combine(a, b):
        return genbmm.maxbmm(a.view(-1, C, C).contiguous(),
                             b.view(-1, C, C).contiguous()).view(B, -1, C, C)
    N_sq = int(math.log(N, 2))
    chart = scores
    charts = []
    charts.append(chart)
    for i in range(1, N_sq+1):
        chart = combine(chart[:, ::2], chart[:, 1::2])
        charts.append(chart)
    P, S = 0, 1
    ps = torch.zeros(2, B, 1, C, C).cuda()
    for i in range(N_sq-1, -1, -1):
        ps2 = torch.zeros(2, B, int(2**(N_sq-i)), C, C).cuda()
        ps2[P, :, ::2] = ps[P, :, :]
        ps2[P, :, 1::2] = combine(ps[P, :, :], charts[i][:, ::2])
        
        ps2[S, :, ::2] = combine(charts[i][:, 1::2], ps[S, :, :])
        ps2[S, :, 1::2] = ps[S, :, :]
        ps = ps2
    suffix = ps[S, :, :]
    prefix = ps[P, :, :] 
    return prefix.max(-2, keepdim=True)[0] + scores.transpose(-2, -1) + suffix.max(-1, keepdim=True)[0]

if 16 not in fbs:
    fbs[16] = foo.fb_max(16)
if 32 not in fbs:
    fbs[32] = foo.fb_max(32)
if 64 not in fbs:
    fbs[64] = foo.fb_max(64)
if 128 not in fbs:
    fbs[128] = foo.fb_max(128)

with torch.no_grad():
    B, N, C, C = 1, 32, 64, 64
    time_start = time.time()
    #
    ttt=0
    for i in range(2000):
        scores = torch.rand(B, N, C, C).cuda()
        edge_max_marginals2 = max_marginals(scores).transpose(-1,-2) # bsz, length-1, K, K
        ttt += edge_max_marginals2.min()
    print (time.time() - time_start)
    
    time_start = time.time()
    ttt=0
    for i in range(2000):
        scores = torch.rand(B, N, C, C).cuda()
        fb = fbs[C]
        edge_max_marginals1 = fb(scores.transpose(0,1)) # bsz, length-1, K, K
        ttt += edge_max_marginals1.min()
    
    print (time.time() - time_start)
    #
    B, N, C, C = 1, 64, 64, 64
    time_start = time.time()
    #
    ttt=0
    for i in range(2000):
        scores = torch.rand(B, N, C, C).cuda()
        edge_max_marginals2 = max_marginals(scores).transpose(-1,-2) # bsz, length-1, K, K
        ttt += edge_max_marginals2.min()
    print (time.time() - time_start)
    
    time_start = time.time()
    ttt=0
    for i in range(2000):
        scores = torch.rand(B, N, C, C).cuda()
        fb = fbs[C]
        edge_max_marginals1 = fb(scores.transpose(0,1)) # bsz, length-1, K, K
        ttt += edge_max_marginals1.min()
    
    print (time.time() - time_start)
#
