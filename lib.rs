/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

use std::fmt;

/// Represents a reordering operation on an array.
///
/// See the [`Permute`] trait for more information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Perm {
    inv: PermVec,
}

// This is a Perm stored in the form that's easiest to reason about.
// (documented in `Perm::from_vec`)
//
// It exists solely for clarity of implementation, because implementing
// `Perm::shift_right` as `inv: self.inv.prefix_shift_left()` says it a lot
// better than any comment I could write above the inlined method body.
//
// Basically, method bodies on Perm describe the relationship between
// the perm and its inverse, while method bodies on PermVec do the real work.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PermVec( // PermVec<Src, Dest>
    Vec<usize>, // Indexed<Dest, Vec<Src>>
);

impl fmt::Debug for PermVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// An error which indicates that a permutation vector is invalid.
///
/// A permutation vector `vec` is valid if and only if it contains one copy of every
/// index in the range `0..vec.len()`.
#[derive(Debug)]
pub struct InvalidPermutationError { _private: () }

impl fmt::Display for InvalidPermutationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt("Tried to construct an invalid permutation.", f)
    }
}

impl Perm {
    /// Construct the identity perm of a given length.
    pub fn eye(n: usize) -> Perm
    { Perm { inv: PermVec::eye(n) } }

    /// Get the length of the permutation.
    pub fn len(&self) -> usize
    { self.inv.0.len() }

    /// Compute the `Perm` that, when applied to the input slice, would (stably) sort it.
    pub fn argsort<T: Ord>(xs: &[T]) -> Perm
    { Perm { inv: PermVec::argsort(xs).inverted() } }

    /// Compute a `Perm` that, when applied to the input slice, would sort it. (not necessarily stably)
    pub fn argsort_unstable<T: Ord>(xs: &[T]) -> Perm
    { Perm { inv: PermVec::argsort_unstable(xs).inverted() } }

    /// Construct a perm. Useful for literals in unit tests.
    ///
    /// The representation accepted by this is comparable to indexing with an
    /// integer array in numpy.  If the `k`th element of the permutation vector
    /// is `value`, then applying the permutation will *pull* the data at index
    /// `value` into index `k`.
    ///
    /// This performs O(n log n) validation on the data to verify that it
    /// satisfies the invariants of Perm.  It also inverts the perm (an O(n)
    /// operation).
    pub fn from_vec(vec: Vec<usize>) -> Result<Perm, InvalidPermutationError>
    { Ok(Perm { inv: PermVec::from_vec(vec)? }.inverted()) }

    /// Construct a perm from the vector internally used to represent it.
    ///
    /// The format taken by this method is actually **the inverse** of the format
    /// accepted by `from_vec`. If the `k`th element of the permutation vector is
    /// `value`, then applying the permutation will *push* the data at index `k`
    /// over to index `value`. This format is generally trickier to think about,
    /// but is superior to the `from_vec` representation in terms of efficiency.
    ///
    /// This performs O(n log n) validation on the data to verify that it
    /// satisfies the invariants of `Perm`.
    pub fn from_raw_inv(inv: Vec<usize>) -> Result<Perm, InvalidPermutationError>
    { Ok(Perm { inv: PermVec::from_vec(inv)? }) }

    /// No-op constructor.  Still performs checking in debug builds.
    ///
    /// # Safety
    ///
    /// `inv` must contain every element in `(0..inv.len())`,
    /// or else the behavior is undefined.
    pub unsafe fn from_raw_inv_unchecked(inv: Vec<usize>) -> Perm
    { Perm { inv: PermVec(inv).debug_validated() } }

    /// Construct a permutation of length `a.len() + b.len()`.
    ///
    /// Mathematically speaking, this computes the "direct sum" of two permutations.
    ///
    /// The inserted elements will be shifted by this permutation's length,
    /// so that they operate on an entirely independent set of data from
    /// the existing elements.
    pub fn append_mut(&mut self, other: &Perm)
    {
        // (a direct sum of inverses is the inverse of the direct sum)
        self.inv.append_mut(&other.inv);
    }

    /// Construct a random permutation of the given length.
    pub fn random(n: usize) -> Perm
    {
        use rand::Rng;

        let mut inv: Vec<_> = (0..n).collect();
        rand::thread_rng().shuffle(&mut inv);
        Perm { inv: PermVec(inv) }
    }

    /// Recover the vector representation of the permutation.
    ///
    /// See [`Perm::from_vec`] for more details about this representation.
    ///
    /// This has a runtime cost of O(n), because [`Perm`] does not actually store this
    /// vector.  See [`Perm::into_raw_inv`] for a constant-time alternative.
    pub fn into_vec(self) -> Vec<usize>
    { self.inverted().inv.0 }

    /// Obtain the vector that is internally used to represent the permutation.
    ///
    /// This representation is actually the inverse of the representation produced by
    /// [`Perm::into_vec`].  See [`Perm::from_raw_inv`] for more information.
    pub fn into_raw_inv(self) -> Vec<usize>
    { self.inv.0 }

    /// Get the inverse of this permutation.
    #[must_use = "not an in-place operation"]
    pub fn inverted(&self) -> Perm
    {
        // (the inverse of the inverse is... you know...)
        Perm { inv: self.inv.inverted() }
    }

    // (this might sound niche, but it's not like we can safely expose `&mut [u32]`,
    //  so what's the harm in having a niche method?)
    //
    /// Compose with the permutation that shifts elements forward (performing `self` first)
    ///
    /// To construct the shift permutation itself, use `Perm::eye(n).shift_right(amt)`.
    pub fn shift_right(self, amt: usize) -> Self
    { Perm { inv: self.inv.prefix_shift_left(amt) } }

    /// Compose with the permutation that shifts elements backward (performing `self` first)
    ///
    /// To construct the shift permutation itself, use `Perm::eye(n).shift_left(amt)`.
    pub fn shift_left(self, amt: usize) -> Self
    { Perm { inv: self.inv.prefix_shift_right(amt) } }

    /// Compose with the permutation that shifts elements to the right by a signed offset.
    pub fn shift_signed(self, n: isize) -> Self
    {
        if n < 0 {
            assert_ne!(n, std::isize::MIN, "(exasperated sigh)");
            self.shift_left((-n) as usize)
        } else {
            self.shift_right(n as usize)
        }
    }

    /// Apply the permutation to an index. O(1).
    ///
    /// Calling this on the indices contained in a sparse-format data structure will
    /// produce the same indices as if the corresponding dense-format data structure
    /// were permuted.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds for the permutation length.
    pub fn permute_index(&self, i: usize) -> usize {
        // F.Y.I. this method is **literally the entire reason** that we store the inverse.
        self.inv.0[i]
    }

    /// Construct the outer product of self and `slower`, with `self`
    /// being the fast (inner) index.
    ///
    /// The resulting `Perm` will permute blocks of size `self.len()`
    /// according to `slower`, and will permute elements within each
    /// block by `self`.
    pub fn with_outer(&self, slower: &Perm) -> Perm
    {
        // the inverse of the outer product is the outer product of inverses
        Perm { inv: self.inv.with_outer(&slower.inv) }
    }

    /// Construct the outer product of self and `faster`, with `self`
    /// being the slow (outer) index.
    pub fn with_inner(&self, faster: &Perm) -> Perm
    { faster.with_outer(self) }

    /// Compute the permutation that applies this permutation `exp` times in a row.
    ///
    /// This uses exponentiation by squaring to run in `O(log(exp))` time.
    pub fn pow_unsigned(&self, mut exp: u64) -> Perm {
        // Exponentiation by squaring (permutations form a monoid)

        // NOTE: there's plenty of room to optimize the number of heap
        //       allocations here
        let mut acc = Perm::eye(self.len());
        let mut base = self.clone();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc.permuted_by(&base);
            }
            base = base.clone().permuted_by(&base);
            exp /= 2;
        }
        acc
    }

    /// Compute the permutation that applies this permutation `exp` times in a row.
    ///
    /// This version of the function takes a signed value, so that negative values can produce
    /// powers of the inverse.
    ///
    /// This uses exponentiation by squaring to run in `O(log(exp))` time.
    pub fn pow_signed(&self, exp: i64) -> Perm {
        if exp < 0 {
            assert_ne!(exp, std::i64::MIN);  // technically possible to support but... come on, man
            self.inverted().pow_unsigned((-exp) as u64)
        } else {
            self.pow_unsigned(exp as u64)
        }
    }
}

impl PermVec {
    fn eye(n: usize) -> PermVec
    { PermVec((0..n).collect()) }

    fn argsort<T: Ord>(xs: &[T]) -> PermVec
    {
        let mut perm: Vec<_> = (0..xs.len()).collect();
        perm.sort_by(|&a, &b| xs[a].cmp(&xs[b]));
        PermVec(perm)
    }

    fn argsort_unstable<T: Ord>(xs: &[T]) -> PermVec
    {
        let mut perm: Vec<_> = (0..xs.len()).collect();
        perm.sort_unstable_by(|&a, &b| xs[a].cmp(&xs[b]));
        PermVec(perm)
    }

    fn from_vec(vec: Vec<usize>) -> Result<PermVec, InvalidPermutationError>
    {
        if !Self::validate_data(&vec) {
            return Err(InvalidPermutationError { _private: () });
        }
        Ok(PermVec(vec))
    }

    // Checks invariants required by Perm for unsafe code.
    #[must_use = "doesn't assert"]
    fn validate_data(xs: &[usize]) -> bool {
        let mut vec = xs.to_vec();
        vec.sort();
        vec.into_iter().eq(0..xs.len())
    }

    fn debug_validated(self) -> PermVec {
        debug_assert!(PermVec::validate_data(&self.0));
        self
    }

    fn append_mut(&mut self, other: &Self)
    {
        let offset = self.0.len();
        self.0.extend(other.0.iter().map(|&i| i + offset));
        debug_assert!(PermVec::validate_data(&self.0));
    }

    #[must_use = "not an in-place operation"]
    fn inverted(&self) -> Self
    {
        let mut inv = vec![::std::usize::MAX; self.0.len()]; // [Src] -> Dest
        for (i, &x) in self.0.iter().enumerate() { // i: Dest, x: Src
            inv[x] = i;
        }
        PermVec(inv).debug_validated()
    }

    // The perm that does `self`, then shifts right.
    #[allow(unused)]
    fn postfix_shift_right(mut self, amt: usize) -> PermVec
    {
        let n = self.0.len();
        self.0.rotate_right(amt % n);
        self.debug_validated()
    }

    // The perm that does `self`, then shifts left.
    #[allow(unused)]
    fn postfix_shift_left(mut self, amt: usize) -> PermVec
    {
        let n = self.0.len();
        self.0.rotate_left(amt % n);
        self.debug_validated()
    }

    // The perm that shifts left, then applies `self`
    fn prefix_shift_left(mut self, amt: usize) -> PermVec {
        // Add amt to each value.
        let n = self.0.len();
        let amt = amt % n;
        for x in &mut self.0 {
            *x = (*x + amt) % n;
        }
        self.debug_validated()
    }

    // The perm that shifts right, then applies `self`
    fn prefix_shift_right(self, amt: usize) -> PermVec {
        // Subtract amt from each value.
        // ...or rather, shift left by `(-amt) mod len`
        //
        // (technically, this puts it into the range `[1, len]` instead of `[0, len)`
        //  due to a silly edge case, but that doesn't matter)
        let len = self.0.len();
        self.prefix_shift_left(len - amt % len)
    }

    fn with_outer(&self, slower: &PermVec) -> PermVec
    {
        assert!(self.0.len().checked_mul(slower.0.len()).is_some());

        let mut perm = Vec::with_capacity(self.0.len() * slower.0.len());

        for &block_index in &slower.0 {
            let offset = self.0.len() * block_index;
            perm.extend(self.0.iter().map(|&x| x + offset));
        }
        PermVec(perm).debug_validated()
    }

    // Perm that applies self then other.
    fn then(&self, other: &PermVec) -> PermVec
    {
        assert_eq!(self.0.len(), other.0.len(), "Incorrect permutation length");

        let mut out = vec![::std::usize::MAX; self.0.len()];

        for (out_i, &self_i) in other.0.iter().enumerate() {
            out[out_i] = self.0[self_i];
        }

        PermVec(out).debug_validated()
    }
}

impl Perm {
    /// Flipped group operator, which composes left-to-right.
    ///
    /// Very simply. `a.then(b) == b.of(a)`.  The flipped order can feel more natural
    /// when using method syntax, or if you are dealing with matrix equations written
    /// in a row-centric formalism.
    ///
    /// Additionally, it has a straightforward relation to the group action:
    /// ```text
    /// x.permuted_by(a).permuted_by(b) == x.permuted_by(a.then(b))
    /// ```
    pub fn then(&self, other: &Perm) -> Perm
    {
        // The inverses compose in reverse.
        Perm { inv: other.inv.then(&self.inv) }
    }

    /// Conventional group operator, which composes right-to-left.
    pub fn of(&self, other: &Perm) -> Perm
    { other.then(self) }
}

/// Trait for applying a permutation operation.
///
/// Impls of `Permute` do not always necessarily apply the permutation directly to
/// vectors contained in the type.  For instance, data in a sparse format will likely
/// use the permutation to transform stored indices.
///
/// As a conceptual tool, it can help to consider indices as having distinct types
/// (e.g. vertex indices, component indices; or more specific things like "indices of
/// rows when sorted by column 2"); in this model, `Perm` can be thought of as being
/// parametrized over two index types, where `Perm<Src, Dest>` transforms data indexed
/// by type `Src` into data indexed by type `Dest`.  Again, this is just a conceptual
/// tool; `Perm` does not actually have these type parameters!
/// (More about this in [this blog post](https://exphp.github.io/blog/2018/07/30/that-weekend-i-wasted-on-newtyped-indices.html))
///
/// # Laws
///
/// All implementations of `Permute` must satisfy the following properties,
/// which give `Permute::permuted_by` the qualities of a group action.
/// (whose group operator is, incidentally, also `Permute::permuted_by`!)
///
/// * **Identity:**
///   ```text
///   data.permuted_by(Perm::eye(data.len())) == data
///   ```
/// * **Compatibility:**
///   ```text
///   data.permuted_by(a).permuted_by(b) == data.permuted_by(a.permuted_by(b))
///   ```
///
/// When envisioning `Perm` as generic over `Src` and `Dest` types, it could
/// perhaps be said that `Perm`s are the morphisms of a category. (brushing
/// aside issues of mismatched length)
pub trait Permute: Sized {
    // awkward name, but it makes it makes two things clear
    // beyond a shadow of a doubt:
    // - The receiver gets permuted, not the argument.
    //   (relevant when Self is Perm)
    // - The permutation is not in-place.
    fn permuted_by(self, perm: &Perm) -> Self;
}

// (module to protect from lollipop model; the unsafety here
//  is extremely localized)
mod unsafe_impls {
    use super::*;

    pub(super) fn inv_permute_to_new_vec<T>(vec: Vec<T>, inv: &PermVec) -> Vec<T> {
        let mut out = Vec::with_capacity(vec.len());
        inv_permute_to_mut_vec(vec, inv, &mut out);
        out
    }

    pub(super) fn inv_permute_to_mut_vec<T>(mut vec: Vec<T>, inv: &PermVec, out: &mut Vec<T>) {
        assert_eq!(
            vec.len(), inv.0.len(),
            "Incorrect permutation length",
        );

        out.clear();
        out.reserve_exact(inv.0.len());

        //------------------------------------------------
        // You are now entering a PANIC FREE ZONE

        { // scope ptrs so we can reason about them
            let vec_ptr = vec.as_ptr();
            let out_ptr = out.as_mut_ptr();

            // a perm holds indices into the data vec, so the inverse holds indices into `out`.
            for (vec_i, &out_i) in inv.0.iter().enumerate() {
                // SAFETY:
                //
                //  * vec_i < vec.len() because:
                //    * vec_i comes from the standard library implementation of `impl Iterator for std::iter::Enumerate`,
                //      and is thus guaranteed to be `< inv.0.len()`.
                //    * We asserted earlier that `inv.0.len() == vec.len()`.
                //
                //  * vec[vec_i] will not be double-dropped, because:
                //    * we perform `vec.set_len(0)` after this loop.
                //    * we cannot possibly panic before this occurs.
                let value = unsafe { vec_ptr.offset(vec_i as isize).read() };

                // SAFETY:
                //
                //  * out_i < out.capacity() because:
                //    * A privacy-protected invariant of PermVec guarantees that `out_i < inv.0.len()`.
                //    * We called `Vec::reserve_exact` to ensure that `inv.0.len() <= out.capacity()`.
                let dest_ptr = unsafe { out_ptr.offset(out_i as isize) };
                unsafe { dest_ptr.write(value) };
            }
        }

        // Don't drop the original items, but do allow the original
        // vec to fall out of scope so the memory can be freed.

        // SAFETY:
        //
        // * All elements in out[0..vec.len()] are initialized because:
        //   * A privacy-protected invariant of PermVec guarantees that, in the above `for` loop,
        //     every index from 0..vec.len() will have appeared exactly once as `out_i`.
        unsafe { out.set_len(vec.len()); }
        unsafe { vec.set_len(0); }

        // Thank you for flying with us. You may now PANIC!
        //------------------------------------------------
    }
}

impl<T> Permute for Vec<T> {
    fn permuted_by(self, perm: &Perm) -> Vec<T>
    { self::unsafe_impls::inv_permute_to_new_vec(self, &perm.inv) }
}

// `Permute` doubles as the group operator.
// (think of it as matrix multiplication in the matrix representation)
impl Permute for PermVec {
    fn permuted_by(self, perm: &Perm) -> PermVec
    { PermVec(self.0.permuted_by(perm)) }
}

impl Permute for Perm {
    fn permuted_by(self, other: &Perm) -> Perm
    { self.then(other) }
}

// rsp2-tasks needs this
impl<T: Clone> Permute for std::rc::Rc<[T]> {
    fn permuted_by(self, perm: &Perm) -> Self
    {
        // this could be done with less copying...
        // (though it absolutely has to make at least one full copy)
        let vec = self.to_vec();  // an O(n) copy
        let vec = vec.permuted_by(perm);  // O(n) work
        let slice = vec.into_boxed_slice();
        slice.into()  // another O(n) copy to embed refcount
    }
}

impl<T: Permute + Clone> Permute for std::rc::Rc<T> {
    fn permuted_by(self, perm: &Perm) -> Self {
        Box::new((*self).clone().permuted_by(perm)).into()
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    use self::drop_pusher::DropPusher;
    mod drop_pusher {
        use std::rc::Rc;
        use std::cell::RefCell;

        /// Helper for testing panic/drop safety.
        pub struct DropPusher<T: Copy>(Rc<RefCell<Vec<T>>>, T);

        impl<T: Copy + 'static> DropPusher<T> {
            /// Create a shared vector, and a `new` function which constructs
            /// `DropPushers` tied to that vector.
            pub fn new_trial() -> (Rc<RefCell<Vec<T>>>, Box<dyn Fn(T) -> DropPusher<T>>)
            {
                let history = Rc::new(RefCell::new(vec![]));
                let new = {
                    let history = history.clone();
                    Box::new(move |x| DropPusher(history.clone(), x))
                };
                (history, new)
            }
        }

        impl<T: Copy> Drop for DropPusher<T> {
            fn drop(&mut self) {
                self.0.borrow_mut().push(self.1);
            }
        }
    }

    #[test]
    fn inverse()
    {
        let perm = Perm::random(20);
        let inv = perm.inverted();

        assert_eq!(perm.clone().permuted_by(&inv), Perm::eye(20));
        assert_eq!(inv.permuted_by(&perm), Perm::eye(20));
    }

    #[test]
    fn inverse_is_argsort()
    {
        let perm = Perm::random(20);
        assert_eq!(
            Perm::argsort(&perm.clone().into_vec()).into_vec(),
            perm.inverted().into_vec(),
        );
    }

    #[test]
    fn invalid() {
        assert!(matches!(
            Perm::from_vec(vec![0, 1, 3, 3]),
            Err(InvalidPermutationError {..}),
        ));

        assert!(matches!(
            Perm::from_vec(vec![1, 2, 3]),
            Err(InvalidPermutationError {..}),
        ));
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn incompatible() {
        // another requirement for the Vec impl's safety
        let _ = vec![4, 2, 1].permuted_by(&Perm::eye(2));
    }

    #[test]
    fn drop_safety() {
        let (drop_history, dp) = DropPusher::new_trial();
        {
            let vec = vec![dp(0), dp(1), dp(2), dp(3), dp(4)];

            let vec2 = vec.permuted_by(&Perm::from_vec(vec![3, 1, 0, 4, 2]).unwrap());
            assert_eq!(drop_history.borrow().len(), 0);

            drop(vec2);
            assert_eq!(drop_history.borrow().len(), 5);
        }
        assert_eq!(drop_history.borrow().len(), 5);
    }

    #[test]
    fn argsort_is_stable()
    {
        use rand::Rng;

        // a long vector of only two unique values; a prime target for stability checks
        let n = 300;
        let values: Vec<_> = (0..n).map(|_| rand::thread_rng().gen_range(0, 2)).collect();

        let perm = Perm::argsort(&values);
        let permuted_indices = (0..n).collect::<Vec<_>>().permuted_by(&perm);
        let permuted_values = values.permuted_by(&perm);

        let error = format!("not your lucky day, Mister one-in-{:e}", 2f64.powi(n));
        let first_one = permuted_values.iter().position(|&x| x == 1).expect(&error);

        let is_strictly_sorted = |xs: &[_]| xs.windows(2).all(|w| w[0] < w[1]);
        assert!(is_strictly_sorted(&permuted_indices[..first_one]));
        assert!(is_strictly_sorted(&permuted_indices[first_one..]));

        let error = format!("DEFINITELY not your lucky day, Mister one-in-{}-factorial!!", n);
        assert!(!is_strictly_sorted(&permuted_indices[..]), "{}", error);
    }

    #[test]
    fn associativity()
    {
        let xy = Perm::from_vec(vec![1, 0, 2]).unwrap();
        let zx = Perm::from_vec(vec![2, 1, 0]).unwrap();
        let xyzx = Perm::from_vec(vec![2, 0, 1]).unwrap();
        assert_eq!(xy.clone().permuted_by(&zx), xyzx);
        assert_eq!(xy.then(&zx), xyzx);
        assert_eq!(zx.of(&xy), xyzx);
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![0,1,2].permuted_by(&xyzx),
        );
        assert_eq!(
            vec![0,1,2].permuted_by(&xy).permuted_by(&zx),
            vec![2,0,1],
        );

        for _ in 0..10 {
            use rand::Rng;

            let mut rng = rand::thread_rng();
            let n = rng.gen_range(10, 20);
            let s = b"abcdefghijklmnopqrstuvwxyz"[..n].to_vec();
            let a = Perm::random(n);
            let b = Perm::random(n);
            let c = Perm::random(n);
            let bc = b.clone().permuted_by(&c);
            assert_eq!(
                a.clone().permuted_by(&b).permuted_by(&c),
                a.clone().permuted_by(&bc),
                "compatibility, for Self = Perm (a.k.a. associativity)",
            );
            assert_eq!(
                a.inv.clone().permuted_by(&b).permuted_by(&c),
                a.inv.clone().permuted_by(&bc),
                "compatibility, for Self = PermVec",
            );
            assert_eq!(
                s.clone().permuted_by(&b).permuted_by(&c),
                s.clone().permuted_by(&bc),
                "compatibility, for Self = Vec",
            );
        }
    }

    #[test]
    fn append()
    {
        let mut a = Perm::from_vec(vec![1, 0]).unwrap();
        let b = Perm::from_vec(vec![1, 2, 0]).unwrap();
        a.append_mut(&b);
        assert_eq!(
            vec![00, 01, /* -- */ 10, 11, 12].permuted_by(&a),
            vec![01, 00, /* -- */ 11, 12, 10],
        );
    }

    #[test]
    fn outer()
    {
        let use_outer = |a, b| {
            let a = Perm::from_vec(a).unwrap();
            let b = Perm::from_vec(b).unwrap();
            let xs: Vec<_> =
                (0..b.len()).flat_map(|slow| {
                    (0..a.len()).map(move |fast| 10 * slow + fast)
                }).collect();
            xs.permuted_by(&a.with_outer(&b))
        };

        assert_eq!(
            use_outer(
                vec![1, 0, 2, 3],
                vec![1, 2, 0],
            ),
            vec![
                11, 10, 12, 13,
                21, 20, 22, 23,
                01, 00, 02, 03,
            ],
        );

        // empty perms
        assert_eq!(use_outer(vec![1, 0], vec![]), vec![]);

        assert_eq!(use_outer(vec![], vec![1, 0]), vec![]);
    }

    #[test]
    fn shift() {
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_right(8)),
            vec![4, 5, 0, 1, 2, 3],
        );
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_left(8)),
            vec![2, 3, 4, 5, 0, 1],
        );
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_signed(8)),
            vec![4, 5, 0, 1, 2, 3],
        );
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_signed(-8)),
            vec![2, 3, 4, 5, 0, 1],
        );
        // potentially dumb edge case
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_signed(6)),
            vec![0, 1, 2, 3, 4, 5],
        );
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5].permuted_by(&Perm::eye(6).shift_signed(-6)),
            vec![0, 1, 2, 3, 4, 5],
        );
    }

    #[test]
    fn pow_unsigned() {
        for &len in &[0, 1, 4, 20] {
            for _ in 0..5 {
                let perm = Perm::random(len);
                for &exp in &[0, 1, 4, 20, 21] {
                    let original = b"abcdefghijklmnopqrstuvwxyz"[..len as usize].to_owned();

                    let mut brute_force = original.clone();
                    for _ in 0..exp {
                        brute_force = brute_force.permuted_by(&perm);
                    }

                    let fast = original.permuted_by(&perm.pow_unsigned(exp));
                    assert_eq!(fast, brute_force);
                }
            }
        }
    }

    #[test]
    /// ```rust
    /// // IMPORTANT:  If you modify this test, UPDATE THE README!!!
    ///
    /// use perm_vec::{Perm, Permute};
    ///
    /// fn main() {
    ///     // The vec that permutes "abcd" into "bcda".
    ///     let perm_shl = Perm::from_vec(vec![1, 2, 3, 0]).unwrap();
    ///     assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_shl), vec![10, 20, 30, 0]);
    ///
    ///     // The permutation that reverses a vector
    ///     let perm_rev = Perm::from_vec((0..4).rev().collect()).unwrap();
    ///     assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_rev), vec![30, 20, 10, 0]);
    ///
    ///     // Let's compose them!
    ///     let perm_comp_1 = perm_shl.then(&perm_rev);  // this one shifts, then reverses
    ///     let perm_comp_2 = perm_rev.then(&perm_shl);  // this one reverses, then shifts
    ///     assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_comp_1), vec![0, 30, 20, 10]);
    ///     assert_eq!(vec![0, 10, 20, 30].permuted_by(&perm_comp_2), vec![20, 10, 0, 30]);
    /// }
    /// ```
    fn _readme_doctest() {}
}
