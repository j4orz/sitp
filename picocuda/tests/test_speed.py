# import unittest
# import numpy as np
# import torch
# import picograd

# torch_device = torch.device('mps' if getenv("MPS", 0) else ('cuda' if getenv("TORCHCUDA", 0) else 'cpu'))
# if str(torch_device) == "mps":
#   import torch.mps
#   def sync(): torch.mps.synchronize()
# elif str(torch_device) == "cuda":
#   import torch.cuda
#   def sync(): torch.cuda.synchronize()
# else:
#   def sync(): pass

# def helper_test_generic_square(name, N, f1, f2, onearg=False):
#   torch.manual_seed(0)
#   x_tch = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
#   y_tch = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device) if not onearg else None

#   x_pg = picograd.tensor(x_tch.cpu().numpy())
#   y_pg = picograd.tensor(y_tch.cpu().numpy()) if not onearg else None

#   helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", f1, (torch_a, torch_b), TinyJit(f2), (tiny_a, tiny_b))

# class TestSpeed(unittest.TestCase):
#   def test_gemm(self):
#     def f(a, b): return a @ b
#     helper_test_generic_square('gemm', 1024, f, f)

# if __name__ == '__main__':
#   unittest.main()