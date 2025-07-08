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
`picoc` vendors `bril` in two ways:

1. rust crates via cargo git dependencies in `Cargo.toml` (see [the cargo book section 3.3](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories))
```sh
cargo run
```

2. artifacts and binaries with git submodules in `vendor/bril` (see [the git book section 7.11](https://git-scm.com/book/en/v2/Git-Tools-Submodules)):
```sh
git clone --recurse-submodules https://github.com/j4orz/picoc # if already cloned, then git submodule update --init --recursive
cd picoc
git -C vendor/bril sparse-checkout init --cone
git -C vendor/bril sparse-checkout set benchmarks tests bril2json-rs
```

if you need to materalize additional source from upstream, run
```sh
git -C vendor/bril sparse-checkout add /path/to/new/source
```

**RISCV**

**ROCDL**