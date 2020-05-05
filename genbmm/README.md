# genbmm

This library is a collection of specialized matrix multiply operations implemented in PyTorch. It was developed to provide operators needed for [PyTorch-Struct](https://github.com/harvardnlp/pytorch-struct). 

The library has two main components. Currently it only supports CUDA operations.  

* Generalized matrix-multiplication with gradients (log-space, max, sample)
* Banded sparse matrices


## Quickstart 

```python
!pip install -qU git+https://github.com/harvardnlp/genbmm
```
### Generalized Matrix Multiplication

```python

import genbmm

a = torch.rand(10, 3, 4).cuda().requires_grad_(True)
b = torch.rand(10, 4, 5).cuda().requires_grad_(True)

# Log-Sum-Exp
c = genbmm.logbmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)
# Grad
prob_a, prob_b = torch.autograd.grad(c.sum(), (a, b))

# Max
c = genbmm.maxbmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2, = (a + b).max(-2)
# Grad
argmax_a, argmax_b = torch.autograd.grad(c.sum(), (a, b))

# Sample
c = genbmm.samplebmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)
# Grad
sample_a, sample_b = torch.autograd.grad(c.sum(), (a, b))
# c2 = (a + b).softmax(-2).sample(-2)
```

### Banded Sparse Matrices

See notebooks. 




