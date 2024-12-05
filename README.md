- this repo is archived.
  - videos are still useful conceptually
  - but implementations contain bugs (not fuzzed)
- checkout picoprob for dsa implemented in rust
  - this will be actively maintained to support cp

# iruka
![](https://i.imgur.com/q6N56c9.png)

![](https://img.shields.io/npm/v/dsa-ts)
[![Tests](https://github.com/jeffzh4ng/dsa-ts/workflows/Tests/badge.svg)](https://github.com/jeffzh4ng/algorithms-and-data-structures/actions?query=branch%3Amaster++)
![](https://img.shields.io/codecov/c/github/jeffzh4ng/dsa-ts)
![](https://img.shields.io/github/license/jeffzh4ng/algorithms-and-data-structures)

A collection of classical [data structures](https://github.com/jeffzh4ng/dsa-ts#data-structures) and [algorithms](https://github.com/jeffzh4ng/dsa-ts#algorithms) implemented in Typescript. Click the 📹 emoji for tutorials.

The repository's primary goal is educational. Hence, all implementations include a prolific number of comments which guide the reader. The name of the project, iruka, is an ode to Iruka sensei from Naruto. He became a teacher to pass on the Will of Fire, and teach the future ninja of the leaf village. Likewise, this project is here to teach the future software engineers of earth.

You can use this package in your projects if you so wish. Test coverage will be kept at 100%. To install the package, use npm or yarn:

```
yarn add dsa-ts
```

## Data Structures
- [x] Sequences
  - [x] [📹](https://www.youtube.com/watch?v=oXXLFvtG6-Q&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=5) [Linked List](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/sequences/linked-list/linked-list.ts)
  - [x] [📹](https://www.youtube.com/watch?v=7l4YHzc3iXU&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=7) [Stack](https://github.com/jeffzh4ng/dsa-ts/tree/master/src/data-structures/sequences/stack/stack.ts)
  - [x] [📹](https://www.youtube.com/watch?v=E1I8IcKv_cQ&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=9) [Queue](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/sequences/queue/queue.ts)
  - [x] [Double-ended Queue](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/sequences/queue/deque.ts)
  - [x] [📹](https://www.youtube.com/watch?v=39HHWATPcwY&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=11) [Circular Buffer](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/sequences/circular-buffer/circular-buffer.ts)
- [x] Priority Queues
  - [x] [📹](https://www.youtube.com/watch?v=xpSa8YOqeWQ&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=13) [Binary Heap](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/priority-queues/min-binary-heap.ts)
  - [x] [D-Heap](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/priority-queues/min-d-heap.ts)
  - [x] [📹](https://www.youtube.com/watch?v=TlYPo28sw9E) [Indexed Binary Heap](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/data-structures/priority-queues/min-indexed-d-heap.ts)
  - [x] Mergeable Heaps
    - [x] [📹](https://www.youtube.com/watch?v=m8rsw3KfDKs&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=17) [Binomial Heap](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/priority-queues/mergeable-heaps/min-binomial-heap.ts)
    - [x] [📹](https://www.youtube.com/watch?v=-IOse-LEJtw&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=19) [Lazy Binomial Heap](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/priority-queues/mergeable-heaps/lazy-min-binomial-heap.ts)
    - [x] [📹](https://www.youtube.com/watch?v=E_AgyAI8lsc&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=23) [Fibonnaci Heap](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/priority-queues/mergeable-heaps/min-fibonacci-heap.ts)
- [x] Search Trees
  - [x] [📹](https://www.youtube.com/watch?v=zEOWjb7wDik&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=24) [Binary Search Tree](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/trees/binary-search-tree/index.ts)
  - [x] [📹](https://www.youtube.com/watch?v=ietVegtgG_s&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=25) [AVL Tree](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/trees/avl-tree/index.ts)
  - [x] [Red-black Tree](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/trees/red-black-tree/index.ts)
  - [x] [B-Tree](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/trees/b-tree/b-tree.ts)
- [x] Hash Tables
  - [x] [📹](https://www.youtube.com/watch?v=_lJFWYFroYs&list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd&index=26) [Hash Table (Separate Chaining)](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/hash-tables/hash-table-separate-chaining.ts)
  - [x] [Hash Table (Open Addressing)](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/hash-tables/hash-table-open-addressing-base.ts)
    - [x] [Hash Table (Linear Probing)](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/hash-tables/hash-table-linear-probing.ts)
    - [x] [Hash Table (Quadratic Probing)](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/hash-tables/hash-table-quadratic-probing.ts)
    - [x] [Hash Table (Double Hashing)](https://github.com/jeffzh4ng/algorithms-and-data-structures/blob/master/src/data-structures/hash-tables/hash-table-double-hashing.ts)
    
## Algorithms
  - [x] Search
    - [x] [Binary Search](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/search/binary-search.ts)
    - [x] [Breadth-first Search](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/search/breadth-first-search.ts)
    - [x] [Depth-first Search](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/search/depth-first-search.ts)
  - [x] Sorting
    - [x] [Merge Sort](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/merge-sort.ts)
    - [x] [Quick Sort](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/quick-sort.ts)
    - [x] [Heap Sort](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/heap-sort.ts)
    - [x] [Topological Sort (DFS)](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/topological-sort-dfs.ts)
    - [x] [Topological Sort (Kahns)](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/topological-sort-kahn.ts)
    - [x] [Counting Sort](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/sort/counting-sort.ts)
    - [ ] [Bucket Sort]()
  - [] Graph Theory
    - [] Graphs
      - [x] Shortest Paths
        - [x] [Dijkstra's SSSP](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/graphs/shortest-paths/dijkstras-shortest-path.ts)
        - [x] [Bellman-Ford SSSP](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/graphs/shortest-paths/bellman-ford-shortest-path.ts)
        - [x] [Floyd-Warshall APSP](https://github.com/jeffzh4ng/dsa-ts/blob/master/src/algorithms/graphs/shortest-paths/floyd-warshall-shortest-path.ts)
      - [ ] Minimum Spanning Tree
    - [ ] Network Flow
    
## References
- [Fundamental Data Structures](https://en.wikipedia.org/wiki/Book:Fundamental_Data_Structures)
- [Introduction to Algorithms by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein](https://en.wikipedia.org/wiki/Introduction_to_Algorithms)
- [Algorithm Design by Jon Kleinberg and Éva Tardos](https://www.amazon.ca/Algorithm-Design-Jon-Kleinberg/dp/0321295358)
- [Algorithms by William Fiset](https://github.com/williamfiset/Algorithms)
- [Algorithms by Jeff Erickson](http://jeffe.cs.illinois.edu/teaching/algorithms/)
- [Stanford CS 166](https://web.stanford.edu/class/cs166/)
- [Harvard CS 224](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uP4rJgf5ayhHWgw7akUWSf)

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

*Commiting Process*

The commit process does not directly use git. dsa-ts uses commitizen to ensure we are comitting semantic commits. To commit, use the command `yarn commit`, and follow the citizen cli which prompts you for various metainfo regarding the commit. After the pre-commit tests are run successfully, you are able to push to your development branch with `git push`.

## License
This repository is released under the MIT license. In short, this means you are free to use this software in any personal, open-source or commercial projects
