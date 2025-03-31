import unittest
import numpy as np
import torch
import picograd
import os

# see: https://pytorch.org/docs/stable/notes/numerical_accuracy.html

# ********************************************* HELPERS **********************************************
def assrt(input_shapes, f_tch, f_pg=None, l=-2, h=2, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, rg=False):
  if f_pg is None: f_pg = f_tch
  xtch, xpg = gen_inputs(input_shapes, l, h, rg)
  # print("x:", xtch)
  ytch, ypg = f_tch(*xtch), f_pg(*xpg)
  # print("y shape/stride", ytch.shape, ytch.stride())
  verify_outputs("forward pass", ytch.detach().numpy(), ypg.detach().numpy(), atol=atol, rtol=rtol)
  # compare(f"backward pass tensor {i}", tt_grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

def gen_inputs(input_shapes, l, h, rg=False):
  # 1. x_tch NOTE: for now shapes is always specified and randomly generated. future: input: sum(shapes, data)
  np.random.seed(0)
  np_data = [np.random.uniform(low=l, high=h, size=d).astype(np.float32) for d in input_shapes]
  xtch = [torch.tensor(data, requires_grad=rg) for data in np_data]
  for i in range(len(xtch)):
    if xtch[i].dtype == torch.int64: xtch[i] = xtch[i].type(torch.int32) # NOTE: torch default int64 for python ints input
      
  # 2. x_pg
  xpg = [picograd.tensor(x.detach().numpy()) for x in xtch] # TODO: support requires_grad
  return xtch, xpg

def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)

def verify_outputs(s, ytch, ypg, atol, rtol):
    if PRINT_TENSORS:
      print("expected (torch)", ytch)
      print("actual (pico)", ypg)
    try:
      assert ytch.shape == ypg.shape, f"shape mismatch: expected={ytch.shape} | actual={ypg.shape}"
      assert ytch.dtype == ypg.dtype, f"dtype mismatch: expected={ytch.dtype} | actual={ypg.dtype}"

      if np.issubdtype(ytch.dtype, np.floating):
        np.testing.assert_allclose(ytch, ypg, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(ytch, ypg)
    except AssertionError as e:
      raise AssertionError(f"{s} failed (shape={ytch.shape}) - {str(e)}") from None

# ********************************************* TESTS **********************************************
class TestViewOps(unittest.TestCase):
  def test_reshape(self):
    assrt([(4,3,6,6)], lambda x: x.reshape((12,6,6)))
    assrt([(4,3,6,6)], lambda x: x.reshape((-1,3,6,6)))
    assrt([(4,3,6,6)], lambda x: x.reshape((-1,1,6,6)))
    assrt([()], lambda x: x.reshape(()))
    assrt([(1,)], lambda x: x.reshape(()))
    assrt([()], lambda x: x.reshape((1,)))
    assrt([()], lambda x: x.reshape((1,1,1)))
    # self.assrt_exception([(3,4)], lambda x: x.reshape((-1,-1,2)), expected=RuntimeError)
    # self.assrt_exception([(3,4)], lambda x: x.reshape((-1,-1,-1,2)), expected=RuntimeError)
    # with self.assertRaises(ValueError):
    #   x = Tensor.ones((4,3,6,6))
    #   x.reshape([])

  def test_permute(self):
    assrt([(4,3)], lambda x: x.permute((1,0)))
    # TODO:
    # assrt([(4,3)], lambda x: x.permute((0,1)))
    # assrt([(1,2,3,4)], lambda x: x.permute((3,0,2,1)))
    # assrt([(3,4,5,6)], lambda x: x.permute((3,2,1,0)))
    # assrt([(3,4,5,6)], lambda x: x.permute((-2,-1,1,0)))
    # assrt([()], lambda x: x.permute(()))
    # self.assrt_exception([(3,4,5,6)], lambda x: x.permute((0,2)), lambda x: x.permute((0,2)), expected=RuntimeError)
    # self.assrt_exception([(3,4,5,6)], lambda x: x.permute((0,1,2,3,3,3)), lambda x: x.permute((0,1,2,3,3,3)), expected=RuntimeError)
    # self.assrt_exception([(3,4,5,6)], lambda x: x.permute((0,0,1,2,3)), lambda x: x.permute((0,0,1,2,3)), expected=RuntimeError)

  def test_transpose(self):
    pass

  def test_getitem_embedding(self):
    B, T, V, E = 32, 3, 27, 10
    C_VEtch, C_VEpg = gen_inputs([(V, E)], -2, 2, rg=False)
    C_VEtch, C_VEpg = C_VEtch[0], C_VEpg[0]
    
    X_BTnp = np.random.randint(0, V, (B, T))
    X_BTtch, X_BTpg = torch.tensor(X_BTnp), picograd.tensor(X_BTnp)
    X_BTEtch, X_BTEpg = C_VEtch[X_BTtch], C_VEpg[X_BTpg]

    np.testing.assert_allclose(X_BTEtch.numpy(), X_BTEpg.numpy(), atol=1e-6, rtol=1e-6)# f: ℝ^d -> [0,1]^kr

  def test_gather(self):
    pass

  def test_scatter(self):
    pass

  def test_cat(self):
    pass

  def test_stack(self):
    pass

  def test_squeeze(self):
    assrt([(1,3,6,6)], lambda x: x.squeeze(0))
    assrt([(4,3,1,6)], lambda x: x.squeeze(1))
    assrt([(4,3,6,6)], lambda x: x.squeeze(3))
    assrt([(4,3,6,1)], lambda x: x.squeeze(-1))
    assrt([(4,3,6,6)], lambda x: x.squeeze())
    assrt([(1,3,6,6)], lambda x: x.squeeze())
    assrt([(2,3,1)], lambda x: x.squeeze())
    # self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, 50), lambda x: x.squeeze(dim=50), expected=IndexError)
    # self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, -50), lambda x: x.squeeze(dim=-50), expected=IndexError)
    # self.helper_test_exception([()], lambda x: torch.squeeze(x, 10), lambda x: x.squeeze(dim=10), expected=IndexError)
    # self.helper_test_exception([()], lambda x: torch.squeeze(x, 1), lambda x: x.squeeze(dim=1), expected=IndexError)
    # self.helper_test_exception([()], lambda x: torch.squeeze(x, -2), lambda x: x.squeeze(dim=-2), expected=IndexError)

  def test_unsqueeze(self):
    assrt([(4,3,6,6)], lambda x: x.unsqueeze(0))
    assrt([(4,3,6,6)], lambda x: x.unsqueeze(1))
    assrt([(4,3,6,6)], lambda x: x.unsqueeze(4))
    assrt([(4,3,6,6)], lambda x: x.unsqueeze(-1))
    assrt([(4,3,6,6)], lambda x: x.unsqueeze(-3))
    assrt([()], lambda x: x.unsqueeze(0))

  def test_flatten(self):
    pass

  def test_unflatten(self):
    pass

class TestUOps(unittest.TestCase):
  def test_tanh(self):
    assrt([(45,65)], lambda x: torch.tanh(x), lambda x: picograd.tanh(x))

  def test_exp(self):
    assrt([(3,3)], lambda x: torch.exp(x), lambda x: picograd.exp(x))

  def test_log(self):
    assrt([(3,3)], lambda x: torch.log(x), lambda x: picograd.log(x))

  def test_sum(self):
    assrt([(45,3)], lambda x: x.sum())
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=3))
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=(1,3)))
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=(0,2)))
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=(1,2)))
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=1))
    # assrt([(3,4,5,6)], lambda x: x.sum(axis=1, keepdim=True))
    # assrt([()], lambda x: x.sum())
    # assrt([()], lambda x: x.sum(0))
    # assrt([()], lambda x: x.sum(-1))
    # assrt([()], lambda x: x.sum(()))
    # self.helper_test_exception([(3,4,5,6)], lambda x: x.sum(5), lambda x: x.sum(5), expected=IndexError)
    # self.helper_test_exception([()], lambda x: x.sum(1), lambda x: x.sum(1), expected=IndexError)
    # self.helper_test_exception([()], lambda x: x.sum((1,)), lambda x: x.sum((1,)), expected=IndexError)

class TestBinOps(unittest.TestCase):
  def test_add(self):
    assrt([(3,3), (3,3)], lambda x,y: x+y)

  def test_add_scalar(self):
    assrt([(3,3)], lambda x: x+8)
    assrt([(3,3)], lambda x: x+-1)

  def test_sub(self):
    assrt([(3,3), (3,3)], lambda x,y: x-y)

  def test_sub_scalar(self):
    assrt([(3,3)], lambda x: x-8)
    assrt([(3,3)], lambda x: x+-1)

  def test_mul(self):
    assrt([(3,3), (3,3)], lambda x,y: x*y)
    assrt([(3,1), (1,3)], lambda x,y: x*y)

  def test_mul_scalar(self):
    assrt([(3,3)], lambda x: x*8)
    # assrt([(3,3)], lambda x: 8*x)
    assrt([(3,3)], lambda x: x*-1)

  def test_div(self):
    assrt([(3,3), (3,3)], lambda x,y: x/y)

  def test_div_scalar(self):
    assrt([(3,3)], lambda x: x/8)
    assrt([(3,3)], lambda x: x/-1)

  def test_matmul(self):
    assrt([(3), (3,10)], lambda x,y: x.matmul(y), lambda x,y: x @ y)
    assrt([(3,10), (10)], lambda x,y: x.matmul(y), lambda x,y: x @ y)
    assrt([(3,3), (3,3)], lambda x,y: x.matmul(y), lambda x,y: x @ y)

class TestReduceOps(unittest.TestCase):
  def test_sum(self):
    pass

  def test_prod(self):
    pass

  def test_max(self):
    pass

  def test_min(self):
    pass

  def test_mean(self):
    pass

  def test_var(self):
    pass

  def test_std(self):
    pass

  def test_softmax(self):
    pass

  def test_argmax(self):
    pass

  def test_logsumexp(self):
    pass

  def test_logcumsumexp(self):
    pass

# class TestProcessingOps(unittest.TestCase):
#   def test_matmul(self):
#     pass

#   def test_dot(self):
#     pass

#   # def test_conv2d(self):
#   #   pass

#   # def test_avgpool2d(self):
#   #   pass

#   # def test_maxpool2d(self):
#   #   pass

# #   def test_topk(self):
# #     pass

class TestNetworkOps(unittest.TestCase):
  def test_linear(self):
    pass

  def test_sequential(self):
    pass

  def test_layernorm(self):
    pass

  def test_batchnorm(self):
    pass

  def test_crossentropyloss(self):
    pass

  def test_nllloss(self):
    pass  

# def test_softmax(self):
#     assrt([(3), (3)], lambda x: torch.nn.functional.softmax(x, dim=1), lambda x: picograd.nn.functional.softmax(x, 1))

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main()
