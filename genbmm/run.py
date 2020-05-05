import math
from torch.autograd import Function
import torch
import _genbmm
torch.manual_seed(42)
a = torch.rand(5, 2, 7).cuda()
b = torch.rand(5, 7, 2).cuda()
q1 = _genbmm.forward(a,  b)

a2 = a.clone().requires_grad_(True)
b2 = b.clone().requires_grad_(True)

a3 = a2.unsqueeze(-1)
b3 = b2.unsqueeze(-3)
c = torch.logsumexp(a3 + b3, dim=-2)
#print(q1[0])
q2 = _genbmm.backward(a, b, torch.ones_like(q1[0]), q1[0])
#print(q2[0], q2[1])
ag, bg = torch.autograd.grad(c, (a2, b2), torch.ones_like(q1[0]))
#print(q2[0].shape, ag.shape)

print(torch.isclose(ag, q2[0]))
print(torch.isclose(bg, q2[1]))
