import unittest
import numpy as np
import torch
import picograd

# see: https://pytorch.org/docs/stable/notes/numerical_accuracy.html

# ********************************************* HELPERS **********************************************
def assrt(input_shapes, f_tch, f_pg=None, l=-2, h=2, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, rg=False):
  if f_pg is None: f_pg = f_tch
  xtch, xpg = gen_inputs(input_shapes, l, h, rg)
  ytch, ypg = f_tch(*xtch), f_pg(*xpg)
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
  xpg = [picograd.tensor(x.detach().numpy()) for x in xtch] # TODO: support rerquires_grad
  return xtch, xpg

def verify_outputs(s, ytch, ypg, atol, rtol):
    # if PRINT_TENSORS: print(s, xtch, x_pico)
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
class TestUOps(unittest.TestCase):
  def test_tanh(self):
    assrt([(3,3)], lambda x: torch.tanh(x), lambda x: picograd.tanh(x))

  def test_exp(self):
    assrt([(3,3)], lambda x: torch.exp(x), lambda x: picograd.exp(x))

  def test_log(self):
    assrt([(3,3)], lambda x: torch.log(x), lambda x: picograd.log(x))

  def test_sum(self):
    assrt([(3,3)], lambda x: x.sum(dim=1, keepdim=True), lambda x: picograd.sum(x, 1, True))

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
    # assrt([(64), (64,99)], lambda x,y: x.matmul(y), lambda x,y: x @ y)
    assrt([(3,3), (3,3)], lambda x,y: x.matmul(y), lambda x,y: x @ y)

# class TestReduceOps(unittest.TestCase):

# class TestViewOps(unittest.TestCase):
#   def test_getitem_tensorindex(self):
#   B, T, V, E = 32, 3, 27, 10 # embedding for character language model
#   C_VEtch, C_VEpg = torch.randn((V, E)), picograd.randn((V, E))

#   X_BT = np.random.randint(0, V, (B, T))
#   X_BTtch, X_BTpg = torch.tensor(X_BT), picograd.tensor(X_BT)
#   print(X_BTtch.shape, X_BTpg.shape)
#   X_BTEtch, X_BTEpg = C_VEtch[X_BTtch], C_VEpg[X_BTpg]
#   np.testing.assert_allclose(X_BTEtch.numpy(), X_BTEpg.numpy(), atol=1e-6, rtol=1e-6)

# class TestNNOps(unittest.TestCase):
    # def test_cross_entropy(self):
    #     assrt([(3), (3)], lambda p,q: torch.nn.functional.cross_entropy(p, q), lambda p,q: picograd.nn.functional.cross_entropy(p, q))

    # def test_softmax(self):
    #     assrt([(3), (3)], lambda x: torch.nn.functional.softmax(x, dim=1), lambda x: picograd.nn.functional.softmax(x, 1))

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main()
