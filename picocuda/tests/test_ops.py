import unittest
import numpy as np
import torch
import picograd

# *****************************************************************************************************************
# ********************************************* HELPERS **********************************************
# *****************************************************************************************************************

def assrt(input_shapes, f_tch, f_pg=None, l=-2, h=2, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, rg=False):
    if f_pg is None: f_pg = f_tch
    xtch, x_pg = gen_inputs(input_shapes, l, h, rg)
    print("moose", xtch)
    ytch, y_pg = f_tch(*xtch), f_pg(*x_pg)
    verify_outputs("forward pass", ytch.detach().numpy(), y_pg, atol=atol, rtol=rtol)
    # compare(f"backward pass tensor {i}", tt_grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

def gen_inputs(input_shapes, l, h, rg=False):
    # 1. x_tch NOTE: for now shapes is always specified and randomly generated. future: input: sum(shapes, data)
    np.random.seed(0)
    np_data = [np.random.uniform(low=l, high=h, size=d).astype(np.float32) for d in input_shapes] #(_to_np_dtype(dtypes.default_float)) for size in shapes]
    x_tch = [torch.tensor(data, requires_grad=rg) for data in np_data]
    for i in range(len(x_tch)):
        if x_tch[i].dtype == torch.int64: x_tch[i] = x_tch[i].type(torch.int32) # NOTE: torch default int64 for python ints input
        
    # 2. x_pg
    x_pg = [picograd.tensor(x.detach().numpy()) for x in x_tch] # TODO: support rerquires_grad
    return x_tch, x_pg

def verify_outputs(s, xtch, x_pico, atol, rtol):
        # if PRINT_TENSORS: print(s, xtch, x_pico)
        try:
            assert xtch.shape == x_pico.shape, f"shape mismatch: expected={xtch.shape} | actual={x_pico.shape}"
            assert xtch.dtype == x_pico.dtype, f"dtype mismatch: expected={xtch.dtype} | actual={x_pico.dtype}"\

            if np.issubdtype(xtch.dtype, np.floating):
                np.testing.assert_allclose(xtch, x_pico, atol=atol, rtol=rtol)
            else:
                np.testing.assert_equal(xtch, x_pico)
        except Exception as e:
            raise Exception(f"{s} failed shape {xtch.shape}: {e}")

# *****************************************************************************************************************
# ********************************************* TESTS **********************************************
# *****************************************************************************************************************

class TestOps(unittest.TestCase):
    def test_add(self):
        assrt([(3, 3), (3, 3)], lambda x,y: x+y)

    # def test_sub(self):
    #     helper_test_op([3, 3], lambda x,y: x-y)

    # def test_mul(self):
    #     helper_test_op([3, 3], lambda x,y: x*y)


    # def test_div(self):
    #     helper_test_op([3, 3], lambda x,y: x/y)

    # def test_matmul(self):
    #     helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), lambda x,y: x @ y)

if __name__ == '__main__':
    np.random.seed(1337)
    unittest.main(verbosity=2)
