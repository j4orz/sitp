# picoc

```rust
                 __________________________________________________________________
                 |      sem                     opt                  gen          |
                 |   ____________             ________         ________________   |
                 |   |      ast |             | cfg--|-->cfg-->|\ isel/isched |   |             ____
    o            |   |type  / \ |             | /    |         | \ ra         |   |            ||""||
 _ /<. -->c0 u8--|-->|parse/   \|-->bril u8-->|/     |         |  \ enc--exp--|---|-->r5 elf-->||__||
(*)>(*)          |   -----------              --------         ----------------   |            [ -=.]`)
                 |   OLD front(1)            NEW mid(2)      UPDATED back(3)      |            ====== 0
                 -----------------------------------------------------------------|
                                           PICOC
```

# Contributing
picoc is the aot optimizing compiler for [Tensor Compilers: Zero to Hero](https://j4orz.ai/zero-to-hero/).

**C0**

**Bril**

**RISCV**

**ROCDL**