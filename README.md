# `perm-vec`

Provides a `Perm` type for permuting vectors of data.

```toml
[dependencies]
perm-vec = "0.1"
```

It is a very common practice in programming to work with arrays holding reordered indices of something.  (e.g. the indices into an array that sort it by some field, or that change it to match some different ordering convention).  However, this is usually done in an entirely ad-hoc manner, which is unfortunate because not only can these indices be difficult to reason about sometimes, but they also have a great deal of frequently-overlooked mathematical properties that can make reasoning about them so much easier.

`Perm` is a type which implements many of these properties.

```rust
use perm_vec::{Perm, Permute};

fn main() {
    // The vec that permutes "abcd" into "bcda".
    let perm_shl = Perm::from_vec(vec![1, 2, 3, 0]).unwrap();
    assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_shl), vec![10, 20, 30, 0]);
    
    // The permutation that reverses a vector
    let perm_rev = Perm::from_vec((0..4).rev().collect()).unwrap();
    assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_rev), vec![30, 20, 10, 0]);
    
    // Let's compose them!
    let perm_comp_1 = perm_shl.then(&perm_rev);  // this one shifts, then reverses
    let perm_comp_2 = perm_rev.then(&perm_shl);  // this one reverses, then shifts
    assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_comp_1), vec![0, 30, 20, 10]);
    assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_comp_2), vec![20, 10, 0, 30]);
}
```

Various methods are provided for calculating things like:

* The inverse of a permutation.
* Integer powers of a permutation.
* Direct sums.
* Direct products.
