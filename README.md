```
           ,,                                     ,,                      ,,                   
 .M"""bgd  db                                   `7MM                      db   mm              
,MI    "Y                                         MM                           MM              
`MMb.    `7MM  `7MMpMMMb.  .P"Ybmmm `7MM  `7MM    MM   ,6"Yb.  `7Mb,od8 `7MM mmMMmm `7M'   `MF'
  `YMMNq.  MM    MM    MM :MI  I8     MM    MM    MM  8)   MM    MM' "'   MM   MM     VA   ,V  
.     `MM  MM    MM    MM  WmmmP"     MM    MM    MM   ,pm9MM    MM       MM   MM      VA ,V   
Mb     dM  MM    MM    MM 8M          MM    MM    MM  8M   MM    MM       MM   MM       VVV    
P"Ybmmd" .JMML..JMML  JMML.YMMMMMb    `Mbod"YML..JMML.`Moo9^Yo..JMML.   .JMML. `Mbmo    ,V     
                          6'     dP                                                    ,V      
                          Ybmmmd'                                                   OOb"
                                                                  
 .M"""bgd                   mm                                    
,MI    "Y                   MM                                    
`MMb.  `7M'   `MF',pP"Ybd mmMMmm .gP"Ya `7MMpMMMb.pMMMb.  ,pP"Ybd 
  `YMMNq.VA   ,V  8I   `"   MM  ,M'   Yb  MM    MM    MM  8I   `" 
.     `MM VA ,V   `YMMMa.   MM  8M""""""  MM    MM    MM  `YMMMa. 
Mb     dM  VVV    L.   I8   MM  YM.    ,  MM    MM    MM  L.   I8 
P"Ybmmd"   ,V     M9mmmP'   `Mbmo`Mbmmd'.JMML  JMML  JMML.M9mmmP' 
          ,V                                                      
       OOb"
```

Serial Compilation: C on CPUs [lec]() [txt]() [src]()
---
1. AST + Stack Spills: [lec]() [txt](https://j4orz.ai/zero-to-hero/ch1.html) [src]()
2. CFG + Chaitin-Briggs: [lec]() [txt](https://j4orz.ai/zero-to-hero/ch2.html) [src]()
3. CFG-SSA + Hack/Pereira: [lec]() [txt](https://j4orz.ai/zero-to-hero/ch3.html) [src]()
4. SoN + Chaitin-Briggs-Click: [lec]() [txt](https://j4orz.ai/zero-to-hero/ch4.html) [src]()
5. Parsing llm.c [lec]() [txt]() [src]()

Parallel Compilation: CUDA C on GPUs
---
6. Vectorization: [lec]() [txt]() [src]()
7. SIMT: [lec]() [txt]() [src]()

Differentiable Compilation: PyTorch
---





picoc vendors bril artifacts and binaries with git submodules in `vendor/bril`
```sh
git clone --recurse-submodules https://github.com/j4orz/picoc # fresh
git submodule update --remote # already cloned
```