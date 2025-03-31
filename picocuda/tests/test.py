import unittest
import os
import numpy as np
import torch
import picograd

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
    pass
    # assrt([(4,3)], lambda x: x.permute((1,0)))
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
    assrt([(45,65)], torch.exp, Tensor.exp)
    assrt(None, torch.exp, Tensor.exp, vals=[[math.inf, -math.inf, math.nan]])
    assrt([()], torch.exp, Tensor.exp)

  def test_log(self):
    pass

class TestBinOps(unittest.TestCase):
  def test_add(self):
    assrt([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
    assrt([(45,68), (45,68)], lambda x,y: x+y)
    assrt([(), ()], lambda x,y: x+y)
    assrt([(45,65), (45,1)], lambda x,y: x+y)
    assrt([(45,65), ()], lambda x,y: x+y)
    assrt([(45,65), (65,)], lambda x,y: x+y)
    assrt([(3,3)], lambda x: x+8)
    assrt([(3,3)], lambda x: x+-1)

  def test_sub(self):
    assrt([(3,3), (3,3)], lambda x,y: x-y)
    assrt([(3,3)], lambda x: x-8)
    assrt([(3,3)], lambda x: x+-1)

  def test_mul(self):
    assrt([(3,3), (3,3)], lambda x,y: x*y)
    assrt([(3,1), (1,3)], lambda x,y: x*y)
    assrt([(3,3)], lambda x: x*8)
    # TODO: assrt([(3,3)], lambda x: 8*x)
    assrt([(3,3)], lambda x: x*-1)
    assrt([(45,65)], lambda x: x*2)
    assrt([(45,65)], lambda x: x*-1)
    assrt([(45,65)], lambda x: 255*x)
    assrt([(45,65)], lambda x: 2*x)
    assrt([()], lambda x: x*2)
    assrt([()], lambda x: 2*x)

  def test_div(self):
    assrt([(3,3), (3,3)], lambda x,y: x/y)
    assrt([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
    assrt([(45,65), (45,65)], lambda x,y: x/y)
    assrt([(), ()], lambda x,y: x/y)
    assrt([(3,3)], lambda x: x/8)
    assrt([(3,3)], lambda x: x/-1)
    helper_test_op([(45,65)], lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x)
    helper_test_op([()], lambda x: x/2)
    helper_test_op([()], lambda x: 2/x)

  def test_matmul(self):
    assrt([(3), (3,10)], lambda x,y: x.matmul(y), lambda x,y: x @ y)
    assrt([(3,10), (10)], lambda x,y: x.matmul(y), lambda x,y: x @ y)
    assrt([(3,3), (3,3)], lambda x,y: x.matmul(y), lambda x,y: x @ y)

class TestReduceOps(unittest.TestCase):
  def test_sum(self):
    pass
    # TODO:  for fnn verify picograd.sum(softmax()) == 1
    # assrt([(45,3)], lambda x: x.sum())
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

  # TODO:
  def test_softmax(self):
    helper_test_op([(45,65)], torch.nn.Softmax(dim=1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=-1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
  def test_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.softmax(0), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(1), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(2), atol=1e-7, grad_atol=2e-7)

# SAMPLING TODO:
# - softmax
# - sum (for user debugging)
# - multinomial. kolmgorov-smirnov.

# TODO:
# - https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
# - https://gist.github.com/devries/11405101
def assrt_distribution(tiny_func, torch_func=None, numpy_func=None, shape=(40, 43), alpha=0.04):
  assert not (torch_func is None and numpy_func is None), "no function to compare with"
  Tensor.manual_seed(1337)
  torch.manual_seed(1337)
  np.random.seed(1337)

  x1, x2 = tiny_func(*shape).numpy().flatten(), tiny_func(shape).numpy().flatten()
  if numpy_func is not None: y = numpy_func(shape).flatten()
  if torch_func is not None: z = torch_func(shape).numpy().flatten()

  return (numpy_func is None or (kstest(x1, y) >= alpha and kstest(x2, y) >= alpha)) and \
    (torch_func is None or (kstest(x1, z) >= alpha and kstest(x2, z) >= alpha))

def ksprob(a):
  fac, total, termbf = 2.0, 0.0, 0.0
  a2 = -2.0 * a * a
  for j in range(1, 101):
    term = fac * math.exp(a2 * j * j)
    total += term
    if math.fabs(term) <= 0.001 * termbf or math.fabs(term) <= 1e-8 * total:
      return total
    fac = -fac
    termbf = math.fabs(term)
  return 1.0

def kstest(l1, l2):
  n1, n2 = len(l1), len(l2)
  l1.sort()
  l2.sort()
  j1, j2, d, fn1, fn2 = 0, 0, 0.0, 0.0, 0.0
  while j1 < n1 and j2 < n2:
    d1, d2 = l1[j1], l2[j2]
    if d1 <= d2:
      fn1 = (float(j1) + 1.0) / float(n1)
      j1 += 1
    if d2 <= d1:
      fn2 = (float(j2) + 1.0) / float(n2)
      j2 += 1
    dtemp = math.fabs(fn2 - fn1)
    if dtemp > d:
      d = dtemp
  ne = float(n1 * n2) / float(n1 + n2)
  nesq = math.sqrt(ne)
  prob = ksprob((nesq + 0.12 + 0.11 / nesq) * d)
  return prob

def equal_distribution(tiny_func, torch_func=None, numpy_func=None, shape=(40, 43), alpha=0.04):
  Tensor.manual_seed(1337)
  torch.manual_seed(1337)
  np.random.seed(1337)
  assert not (torch_func is None and numpy_func is None), "no function to compare with"
  x1 = tiny_func(*shape).numpy().flatten()
  x2 = tiny_func(shape).numpy().flatten()
  if numpy_func is not None: y = numpy_func(shape).flatten()
  if torch_func is not None: z = torch_func(shape).numpy().flatten()
  return (numpy_func is None or (kstest(x1, y) >= alpha and kstest(x2, y) >= alpha)) and \
    (torch_func is None or (kstest(x1, z) >= alpha and kstest(x2, z) >= alpha))

def normal_test(func, shape=(20, 23), alpha=0.05): return equal_distribution(func, numpy_func=lambda x: np.random.randn(*x), shape=shape, alpha=alpha)

class TestRandom(unittest.TestCase):
  def test_multinomial(self):
    def _check_with_torch(w, num_samples, replacement):
      tiny_res = Tensor(w).multinomial(num_samples, replacement=replacement)
      torch_res = torch.tensor(w).multinomial(num_samples, replacement=replacement)
      self.assertEqual(tiny_res.shape, torch_res.shape)
      if torch_res.ndim == 1:
        tiny_res = tiny_res.unsqueeze(0)
        torch_res = torch_res.unsqueeze(0)
      for i in range(torch_res.shape[0]):
        self.assertTrue(equal_distribution(lambda *_: tiny_res[i], lambda _: torch_res[i]))

    _check_with_torch(w=[0.231, 0., 1., 0.5], num_samples=2000, replacement=True)
    _check_with_torch(w=[[0.2, 0.8]], num_samples=2000, replacement=True)  # 2D but only 1 row
    _check_with_torch(w=[[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]], num_samples=2000, replacement=True)  
    # no-replacement isn't supported, unless taking only one sample
  
    # self.assertRaises(AssertionError, lambda: Tensor(w).multinomial(100, replacement=False))
    # self.assertRaises(AssertionError, lambda: Tensor(2).multinomial(1, replacement=False))
    # self.assertRaises(AssertionError, lambda: Tensor([1, 9]).multinomial(0, replacement=False))

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main()
