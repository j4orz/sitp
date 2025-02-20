"""
model: Neural Language Models (Bengio et al. 2003) URL: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Dimension key:
# windows
B: batch size
T: sequence length

# input/output
V: vocabulary size
E: embedding dimension
D: model dimension
"""
import picograd

# *********************MODEL*********************
# import matplotlib.pyplot as plt
# %matplotlib inline
# from jaxtyping import ...
# g = picograd.Generator().manual_seed(1337) # for .randn()

B, T = 32, 3
V, E, D = 27, 10, 200

# step: 0/200000, loss 27.63208770751953
# -> expected loss = nll = p(c) = -picograd.tensor(1/V=27).log() = 3.2958
# -> self.W = picograd.randn() is sampling from N(0, 1)
# -> self.W * [gain/sqrt(D_in)] (picograd.init_kaimingnormal())

# residuals + normalization + Adam/RMSprop has made initialization less fragile
# -> b/c initialization is fragile/intractable with *deep* neural networks

class Linear:
    def __init__(self, D_in, D_out, bias=True):
        # TODO: generator self.W_DiDo = picograd.randn((D_in, D_out), generator=g) * (5/3)/D_in**0.5 # kaiming init (He et al. 2015)
        self.W_DiDo = picograd.randn((D_in, D_out)) * (5/3)/D_in**0.5 # kaiming init (He et al. 2015)
        self.b_Do = picograd.zeros(D_out) if bias else None

    def __call__(self, X_Di):
        self.X_Do = X_Di @ self.W_DiDo
        if self.b_Do is not None:
            self.X_Do += self.b_Do
        self.out = self.X_Do
        return self.X_Do

    def parameters(self):
        return [self.W_DiDo] + ([] if self.b_Do is None else [self.b_Do])

class Tanh:
    def __call__(self, X_BD):
        self.X_BD = picograd.tanh(X_BD)
        # plt.hist(self.X_BD.view(-1).tolist(), 50); # distribution of weights
        # plt.imshow(self.X_BD.abs() > 0.99, cmap='gray', interpolation='nearest') # vanishing gradients
        self.out = self.X_BD
        return self.X_BD
    
    def parameters(self):
        return []

# model = [
#     Linear(T * E, D, bias=False), BatchNorm1D(D), Tanh(),
#     Linear(D, D, bias=False), BatchNorm1D(D), Tanh(),
#     Linear(D, V, bias=False), BatchNorm1D(V)
# ]

model = [
    Linear(T * E, D, bias=False), Tanh(),
    Linear(D, D, bias=False), Tanh(),
    Linear(D, V, bias=False)
]

C = picograd.randn((V,E)) #, generator=g)
params = [C] + [p for l in model for p in l.parameters()]
for p in params:
    p.requires_grad = True

print("model loaded to cpu")

# *********************TRAINING LOOP*********************
# 1. dataloader
import picograd.nn.functional as F
import random

words = open('./tests/names.txt', 'r').read().splitlines()
v = sorted(list(set(''.join(words))))
encode = { c:i+1 for i,c in enumerate(v) }
encode['.'] = 0
decode = { i:c for c,i in encode.items() }

def gen_dataset(words):
    X, Y = [], []
    for w in words[:3]:
        context = [0] * T;
        for c in w + '.':
            X.append(context)
            Y.append(encode[c])
            # print(''.join(decode[i] for i in context), '-->', decode[encode[c]])
            context = context[1:] + [encode[c]]

    X, Y = picograd.tensor(X), picograd.tensor(Y) # X:(N,C) Y:(N)
    return X, X

random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr, Ytr = gen_dataset(words[:n1])
Xdev, Ydev = gen_dataset(words[n1:n2])
Xte, Yte = gen_dataset(words[n2:])

# # 2. training loop
# N = Xtr.shape[0]
# losses, steps = [], []
# for step in range(200000):
#     # 1. forward
#     indices_B = picograd.randint(0, N, (B,)) # 6. picograd.randint
#     X_B, Y_B = Xtr[indices_B], Ytr[indices_B]

#     X_BD = C[X_B].view(-1, T * E)
#     for layer in model:
#         X_BD = layer(X_BD)
#     loss = F.cross_entropy(X_BD, Y_B) # 5. picograd.cross_entropy

#     # 2. backward
#     for layer in model:
#         layer.out.retain_grad() # 6 .retain_grad()
#     for p in params:
#         p.grad = None
#     loss.backward()

#     # 3. update
#     for p in params:
#         p.data += -0.01 * p.grad

#     steps.append(step)
#     losses.append(loss.log10().item())
#     if step % 10000 == 0:
#         print(f"step: {step}/{200000}, loss {loss.item()}")

# plt.plot(steps, losses)


# *********************INFERENCE LOOP*********************
# for layer in model:
#   if isinstance(layer, BatchNorm1D):
#       layer.training = False

token_terminal = 0
for _ in range(20):
  output, context = [], [0] * T
  while True:
        emb = C[picograd.tensor([context])]
        X_BD = emb.view(emb.shape[0], -1) # 2. .view
        for h in model:
            X_BD = h(X_BD)
            logits = X_BD
            probs = F.softmax(logits, dim=1) # 3. softmax

            token = picograd.multinomial(probs, num_samples=1, replacement=True).item()#, generator=g).item() # 4. multinomial
            context = context[1:] + [token]
            output.append(decode[token])
            if token == token_terminal:
                break
  print(''.join(output))