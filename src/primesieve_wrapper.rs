#![allow(non_camel_case_types, internal_features)]
#![feature(core_intrinsics, ptr_as_ref_unchecked, impl_trait_in_assoc_type)]
use std::{convert::{AsRef, AsMut}, borrow::{Borrow, BorrowMut}, ops::{Deref, DerefMut, Index, IndexMut}, ffi::CStr, marker::PhantomData, mem::{self, MaybeUninit}, iter::{FusedIterator}};

/**
 * This crate is a wrapper of Kim Walisch's primesieve, see [the C documentation](https://github.com/kimwalisch/primesieve/blob/master/doc/C_API.md) for more details.
*/

/**
 * Used with raw functions [`raw::primesieve_generate_primes`] and [`raw::primesieve_generate_n_primes`] to select the data type.
 * When using the high level wrappers [`generate_primes`] and [`generate_n_primes`], the output type is automatically inferred based on the
 * inputs.  When the inputs are integer literals with no suffix, the compiler might successfully infer the type from the output usage, but you
 * may need to add a suffix.  The iterator interface [`PrimeIterator`] always returns `u64` primes.
*/
#[repr(C)]
pub enum Type {
	SHORT_PRIMES,
	USHORT_PRIMES,
	INT_PRIMES,
	UINT_PRIMES,
	LONG_PRIMES,
	ULONG_PRIMES,
	LONGLONG_PRIMES,
	ULONGLONG_PRIMES,
	INT16_PRIMES,
	UINT16_PRIMES,
	INT32_PRIMES,
	UINT32_PRIMES,
	INT64_PRIMES,
	UINT64_PRIMES
}

pub trait PrimeType {
	const TYPE: Type;
	fn into_u64(self) -> u64;
	fn from_u64(x: u64) -> Self;
}

impl PrimeType for i16 {
	const TYPE: Type = Type::INT16_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

impl PrimeType for u16 {
	const TYPE: Type = Type::UINT16_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

impl PrimeType for i32 {
	const TYPE: Type = Type::INT32_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

impl PrimeType for u32 {
	const TYPE: Type = Type::UINT32_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

impl PrimeType for i64 {
	const TYPE: Type = Type::INT64_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

impl PrimeType for u64 {
	const TYPE: Type = Type::UINT64_PRIMES;
	fn into_u64(self) -> u64 { self as _ }
	fn from_u64(x: u64) -> Self { x as _ }
}

pub mod raw {
	use crate::Type;

	#[repr(C)]
	pub struct primesieve_iterator {
		pub i: usize,
		pub size: usize,
		pub start: u64,
		pub stop_hint: u64,
		pub primes: *mut u64,
		pub memory: *mut libc::c_void,
		pub is_error: libc::c_int
	}

	#[link(name = "primesieve")]
	unsafe extern "C" {
		pub fn primesieve_generate_primes(start: u64, stop: u64, size: *mut usize, r#type: Type) -> *mut libc::c_void;
		pub fn primesieve_generate_n_primes(n: u64, start: u64, r#type: Type) -> *mut libc::c_void;
		pub fn primesieve_nth_prime(n: i64, start: u64) -> u64;
		pub fn primesieve_count_primes(start: u64, stop: u64) -> u64;
		pub fn primesieve_count_twins(start: u64, stop: u64) -> u64;
		pub fn primesieve_count_triplets(start: u64, stop: u64) -> u64;
		pub fn primesieve_count_quadruplets(start: u64, stop: u64) -> u64;
		pub fn primesieve_count_quintuplets(start: u64, stop: u64) -> u64;
		pub fn primesieve_count_sextuplets(start: u64, stop: u64) -> u64;
		pub fn primesieve_print_primes(start: u64, stop: u64);
		pub fn primesieve_print_twins(start: u64, stop: u64);
		pub fn primesieve_print_triplets(start: u64, stop: u64);
		pub fn primesieve_print_quadruplets(start: u64, stop: u64);
		pub fn primesieve_print_quintuplets(start: u64, stop: u64);
		pub fn primesieve_print_sextuplets(start: u64, stop: u64);
		pub fn primesieve_get_max_stop() -> u64;
		pub fn primesieve_get_sieve_size() -> libc::c_int;
		pub fn primesieve_get_num_threads() -> libc::c_int;
		pub fn primesieve_set_sieve_size(sieve_size: libc::c_int);
		pub fn primesieve_set_num_threads(num_threads: libc::c_int);
		pub fn primesieve_free(primes: *mut libc::c_void);
		pub fn primesieve_version() -> *const libc::c_char;
		pub fn primesieve_init(it: *mut primesieve_iterator);
		pub fn primesieve_free_iterator(it: *mut primesieve_iterator);
		pub fn primesieve_clear(it: *mut primesieve_iterator);
		pub fn primesieve_jump_to(it: *mut primesieve_iterator, start: u64, stop_hint: u64);
		pub fn primesieve_generate_next_primes(it: *mut primesieve_iterator);
		pub fn primesieve_generate_prev_primes(it: *mut primesieve_iterator);
	}
}

/**
 * Wrapper that owns an allocation from primesieve and will correctly free it when it is dropped.
 * This wrapper acts like a slice of `T`, similar to Rust's `Vec`.
*/
pub struct PrimeArray<T> {
	_buf: *mut libc::c_void,
	_len: usize,
	_elem: PhantomData<T>
}

impl<T> AsRef<[T]> for PrimeArray<T> {
	fn as_ref<'a>(&'a self) -> &'a [T] {
		unsafe { std::slice::from_raw_parts(self._buf.cast(), self._len) }
	}
}

impl<T> AsMut<[T]> for PrimeArray<T> {
	fn as_mut<'a>(&'a mut self) -> &'a mut [T] {
		unsafe { std::slice::from_raw_parts_mut(self._buf.cast(), self._len) }
	}
}

impl<T> Borrow<[T]> for PrimeArray<T> {
	fn borrow<'a>(&'a self) -> &'a [T] {
		self.as_ref()
	}
}

impl<T> BorrowMut<[T]> for PrimeArray<T> {
	fn borrow_mut<'a>(&'a mut self) -> &'a mut [T] {
		self.as_mut()
	}
}

impl<T> Deref for PrimeArray<T> {
	type Target = [T];
	fn deref(&self) -> &Self::Target {
		self.as_ref()
	}
}

impl<T> DerefMut for PrimeArray<T> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.as_mut()
	}
}

impl<T> Drop for PrimeArray<T> {
	fn drop(&mut self) {
		unsafe { raw::primesieve_free(self._buf) }
	}
}

impl<T> Index<usize> for PrimeArray<T> {
	type Output = T;
	fn index(&self, idx: usize) -> &Self::Output {
		assert!(idx < self._len, "Index out of bound!");
		unsafe { self._buf.cast::<T>().add(idx).as_ref_unchecked() }
	}
}

impl<T> IndexMut<usize> for PrimeArray<T> {
	fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
		assert!(idx < self._len, "Index out of bound!");
		unsafe { self._buf.cast::<T>().add(idx).as_mut_unchecked() }
	}
}

/**
 * By-value iterator for [`PrimeArray`] that "correctly" drops any un-consumed values if the iterator is dropped without consuming all its elements.
 * This is a bit over the top, since [`PrimeArray`] can only be obtained with `T` being a primitive integer type with trivial drop, so we don't have
 * to bother dropping elements.  And also, [`PrimeArray`] itself only frees its allocation, it does not drop elements.  But given that this is
 * EXACTLY the same as `MallocArrayIterator` from another of my wrapper crates, I made this future-proof implementatation in case I ever add functionality
 * to re-use buffers in place.  The reasonable way to do that would involve custom allocators, but Rust's allocator API does not actually jive with
 * real-world C libraries like gmp, flint, primesieve, etc that have some malloc-wrapper like [`raw::primesieve_free`], so we can only handle this in
 * unreasonable ways.
*/
pub struct PrimeArrayIterator<T> {
	arr: PrimeArray<MaybeUninit<T>>,
	a: usize,
	b: usize,
}

impl<T> Iterator for PrimeArrayIterator<T> {
	type Item = T;
	fn next(&mut self) -> Option<T> {
		if self.a == self.b {
			None
		} else {
			self.a += 1;
			Some(unsafe { self.arr[self.a - 1].assume_init_read() })
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let res = self.b - self.a;
		(res, Some(res))
	}
}

impl<T> DoubleEndedIterator for PrimeArrayIterator<T> {
	fn next_back(&mut self) -> Option<T> {
		if self.a == self.b {
			None
		} else {
			self.b -= 1;
			Some(unsafe { self.arr[self.b].assume_init_read() })
		}
	}
}

impl<T> ExactSizeIterator for PrimeArrayIterator<T> {}
impl<T> FusedIterator for PrimeArrayIterator<T> {}

impl<T> Drop for PrimeArrayIterator<T> {
	fn drop(&mut self) {
		self.count(); // consume any remaining elements to ensure Drop is called, since PrimeArrayIterator transmutes the inner type to MaybeUninit to avoid double dropping
	}
}

impl<T> IntoIterator for PrimeArray<T> {
	type Item = T;
	type IntoIter = PrimeArrayIterator<T>;
	fn into_iter(self) -> Self::IntoIter {
		Self::IntoIter { b: self._len, arr: unsafe { mem::transmute(self) }, a: 0 }
	}
}

impl<'a, T> IntoIterator for &'a PrimeArray<T> {
	type Item = &'a T;
	type IntoIter = impl Iterator<Item=Self::Item> + 'a;
	fn into_iter(self) -> Self::IntoIter {
		let tmp: &[_] = self;
		tmp.into_iter()
	}
}

impl<'a, T> IntoIterator for &'a mut PrimeArray<T> {
	type Item = &'a mut T;
	type IntoIter = impl Iterator<Item=Self::Item> + 'a;
	fn into_iter(self) -> Self::IntoIter {
		let tmp: &mut [_] = self;
		tmp.into_iter()
	}
}

/**
 * This is a more flexible interface into primesive than [`generate_primes`] and [`generate_n_primes`]: using the iterator interface,
 * we can decrease memory usage, amortize prime generation, split work across threads, and more.
 * 
 * We need to understand prime sieves at a basic level to know how this interface works.
 * 
 * Basically, a primesieve with an upper bount of `MAX` consists of all the "sieving primes" up to the square root of `MAX`,
 * plus a number of "segments"/"buckets".  Strictly speaking, the range where we want to find primes is split into multiple "segments",
 * and we load "one" segment at a time into a "bucket", so that we re-use the buckets throughout the sieve.
 * 
 * This distinction between "buckets" and "segments" is just a distinction I use when writing about prime sieves, other authors might use these interchangably.
 * 
 * If we were writing a sieve, we would choose the bucket size to be the L2 cache size divided by the number of threads, and each thread would
 * use its bucket to process one segment at a time.  This process goes like:
 * - the bucket has a bit corresponding to every potentially prime number in the segment.  Initialize the bucket so all these bits are 0.
 * - for every sieving prime p, set the bits for all its multiples starting at p^2 to 1.
 * - any potentially prime in the segment whose bit is still 0 is prime.
 * 
 * "Potentially prime numbers in the segment" refers to an optimization called "wheel factorization" where we completely skip all numbers that are multiples
 * of some small primes (2, 3, sometimes 5, sometimes 7).  In theory we could use wheel sizes of 2, 6, 30, 210, or even larger or non-product-of-consecutive-prime
 * values, but in practice the wheel sizes of 6, 30, and 210 are usually optimal for realistic sieve sizes.  For a wheel size, we can skip all remainders mod
 * the wheel size that are not coprime to it, so there are very diminishing returns on what percent of numbers we skip, but the costs go up fast in terms of code size.
 * 
 * Additionally, when we can choose a bucket size that's reasonably larger than the square root of `MAX`, we can just follow this definition pretty directly,
 * but for large sieve sizes, the bucket size will actually be small relative to the square root of `MAX`.
 * 
 * This splits the sieving primes into three qualitative categories (not counting "wheel primes" like 2, 3, etc that we don't even need to sieve out multiples of):
 * - small primes that have many multiples per bucket
 * - medium primes that have at least one and probably "several" multiples per bucket
 * - large primes that might not have a multiple in every bucket, or have at most "several" multiples per bucket
 * 
 * A good prime sieve, like primesieve, will treat these three sizes of primes differently.
 * 
 * Ok, so now we can understand what `PrimeIterator` actually does: it will generate sieving primes, incrementally adding more if needed, and then lazily sieve
 * segments one bucket at a time to generate primes requested from `next()`.
 * 
 * But this interface gives us a lot of control over this to ensure good performance:
 * - [`PrimeIterator::jump_to`]: specify where to start and a hint of where to stop generating primes.
 *   This allows the iterator to pre-generate ALL the needed sieving primes so that it won't have to incrementally add more, and if `stop_hint` is specified
 *   correctly, it also will choose segment/bucket sizes smartly.  If you don't know how large of a range you will need to sieve, you can just guess here,
 *   and it will work fine if wrong, just with slightly worse performance.
 * 
 * This is also the preferred way to use primesieve in a multi-threaded way: split the sieving range into one chunk for each thread, then construct
 * and consume a `PrimeIterator` for each of these chunks in its own thread.  For a basic example:
 * ```
 * use std::thread::spawn;
 * use primesieve_wrapper::{get_num_threads, PrimeIterator};
 * 
 * let b = 1_000_000_000;
 * let t = get_num_threads() as u64;
 * let mut workers: Vec<_> = (0..t).map(|i|spawn(move||{
 *     let mut it = PrimeIterator::new();
 *     let (my_a, my_b) = (b*i/t, b*(i+1)/t);
 *     it.jump_to(my_a, my_b);
 *     it.take_while(|&p|p<my_b).count()
 * })).collect();
 * assert_eq!(workers.into_iter().map(|h|h.join().unwrap()).sum::<usize>(), 50_847_534);
 * ```
 * 
 * In a real example, we would probably do something more interesting than `.count()` in each thread.  We can do whatever we want, but note that for large ranges,
 * actually materializing all primes in memory at once could be impossible.  For example, storing all primes up to 1 trillion at once would take 280 gigabytes.
 * Yet primesieve can generate all these primes in under 10 seconds on my computer.
 * 
 * Also, this is a `DoubleEndedIterator`.  When reverse iterating, we generate primes starting at the first one <= `start`.  If `start < 2` or we call `next_back`
 * after 2 is returned, the iterator will return 0.  And when we `jump_to` before reverse iterating, the `stop_hint` should just be `start`.
*/
pub struct PrimeIterator {
	_raw: raw::primesieve_iterator
}

impl PrimeIterator {
	pub fn new() -> Self {
		unsafe {
			let mut inner = MaybeUninit::uninit();
			raw::primesieve_init(inner.as_mut_ptr());
			Self { _raw: inner.assume_init() }
		}
	}

	pub fn clear(&mut self) {
		unsafe { raw::primesieve_clear(&mut self._raw) }
	}

	pub fn jump_to(&mut self, start: u64, stop_hint: u64) {
		unsafe { raw::primesieve_jump_to(&mut self._raw, start, stop_hint) }
	}

	pub fn get_raw(&mut self) -> &mut raw::primesieve_iterator {
		&mut self._raw
	}
}

impl Default for PrimeIterator {
	fn default() -> Self {
		Self::new()
	}
}

impl Iterator for PrimeIterator {
	type Item = u64;
	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self._raw.i += 1;
		if std::intrinsics::unlikely(self._raw.i >= self._raw.size) {
			unsafe { raw::primesieve_generate_next_primes(&mut self._raw) }
		}
		Some(unsafe { std::slice::from_raw_parts(self._raw.primes, self._raw.size)[self._raw.i] })
	}
}

impl DoubleEndedIterator for PrimeIterator {
	#[inline]
	fn next_back(&mut self) -> Option<Self::Item> {
		if std::intrinsics::unlikely(self._raw.i == 0) {
			unsafe { raw::primesieve_generate_prev_primes(&mut self._raw) }
		}
		self._raw.i -= 1;
		Some(unsafe { std::slice::from_raw_parts(self._raw.primes, self._raw.size)[self._raw.i] })
	}
}

impl Drop for PrimeIterator {
	fn drop(&mut self) {
		unsafe { raw::primesieve_free_iterator(&mut self._raw) }
	}
}

/*
impl ParallelIterator for PrimeIterator {
	
}
*/

/**
 * Generate all primes in the range `[start, stop]` (both ends are inclusive).
 * The type is inferred from the inputs.  If the inputs are untyped integer literals, then the compiler might fail to choose the right type
 * depending on how the output is constrained, so you may need to add a suffix like `1_000_000u64` to the input.
 * 
 * This function uses a single thread, and stores all primes found in a big array.
 * So for large ranges, the iterator interface [`PrimeIterator`] is preferrable, since it's easy to make multithreaded and uses far less memory.
 */
pub fn generate_primes<T: PrimeType>(start: T, stop: T) -> PrimeArray<T> {
	unsafe {
		let mut size = MaybeUninit::uninit();
		let buf = raw::primesieve_generate_primes(start.into_u64(), stop.into_u64(), size.as_mut_ptr(), T::TYPE);
		PrimeArray::<T> {
			_buf: buf,
			_len: size.assume_init(),
			_elem: PhantomData
		}
	}
}

/**
 * Generate `n` primes starting at `start` (including `start` if it is prime).
 * The type is inferred from the inputs.  If the inputs are untyped integer literals, then the compiler might fail to choose the right type
 * depending on how the output is constrained, so you may need to add a suffix like `1_000_000u64` to the input.
 * 
 * This function uses a single thread, and stores all primes found in a big array.
 * So for large ranges, the iterator interface [`PrimeIterator`] is preferrable, since it's easy to make multithreaded and uses far less memory.
 */
pub fn generate_n_primes<T: PrimeType>(n: usize, start: T) -> PrimeArray<T> {
	unsafe {
		let buf = raw::primesieve_generate_n_primes(n as _ , start.into_u64(), T::TYPE);
		PrimeArray::<T> {
			_buf: buf,
			_len: n,
			_elem: PhantomData
		}
	}
}

/**
 * NOTE: we can compute the `n`th prime MUCH faster than actually generating all primes up to (`n`th prime after `start`), so this function should generally not be used.
 * Find the `n`th prime >= `start`.  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn nth_prime<T: PrimeType>(n: isize, start: T) -> T {
	T::from_u64(unsafe { raw::primesieve_nth_prime(n as _, start.into_u64()) })
}

/**
 * NOTE: there are MUCH faster algorithms to count primes than actually generating all primes, so this function should generally not be used.
 * Count primes in `[start, stop]` (both ends inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_primes(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_primes(start, stop) as _ }
}


/**
 * Count twin primes, ie pairs where `p` and `p+2` are both prime and BOTH in the interval `[start, stop]` (both inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_twins(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_twins(start, stop) as _ }
}

/**
 * Count triplet primes, ie triples where `p`, `p+2`, and `p+6`; or `p`, `p+4`, and `p+6` are all prime and ALL in the interval `[start, stop]` (both inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_triplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_triplets(start, stop) as _ }
}

/**
 * Count quadruplet primes, ie quadruplets where `p`, `p + 2`, `p + 6`, and `p + 8` are all prime and ALL in the interval `[start, stop]` (both inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_quadruplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_quadruplets(start, stop) as _ }
}

/**
 * Count quintuplet primes, ie quintuplets where `p`, `p + 2`, `p + 6`, `p + 8`, and either `p - 4` or `p + 12` are all prime and ALL in the interval `[start, stop]` (both inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_quintuplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_quintuplets(start, stop) as _ }
}

/**
 * Count sextuplet primes, ie sextuplets where `p - 4`, `p`, `p + 2`, `p + 6`, `p + 8`, and `p + 12` are all prime and ALL in the interval `[start, stop]` (both inclusive).  This function is multithreaded, see [`set_num_threads`] (defaults to all threads).
 */
pub fn count_sextuplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_sextuplets(start, stop) as _ }
}

pub fn print_primes(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_primes(start, stop) }
}

pub fn print_twins(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_twins(start, stop) }
}

pub fn print_triplets(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_triplets(start, stop) }
}

pub fn print_quadruplets(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_quadruplets(start, stop) }
}

pub fn print_quintuplets(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_quintuplets(start, stop) }
}

pub fn print_sextuplets(start: u64, stop: u64) {
	unsafe { raw::primesieve_print_sextuplets(start, stop) }
}

pub fn get_max_stop() -> u64 {
	unsafe { raw::primesieve_get_max_stop() }
}

pub fn get_sieve_size() -> usize {
	unsafe { raw::primesieve_get_sieve_size() as _ }
}

/**
 * get the number of threads primesieve's multithreaded functions will use, which defaults to the number of hardware threads, but can
 * be overridden with [`set_num_threads`].  Most CPUs have a feature called "hyperthreading" where each CPU core can run 2 hardware threads at once,
 * but some CPUs (mostly Intel and mobile) have "efficiency cores" which are slower and don't support hyperthreading.  Some CPUs like Apple's M chips
 * even have more core tiers than just efficiency and "performance", and I'm not sure generally how primesieve chooses the number of threads in this case.
 * In theory we could use one thread per hardware thread but allocate less work to efficiency cores, but in practice the threads we spawn would not actually
 * get pinned to certain cores without OS level muddling, so just spawning one thread per hardware thread on the highest performance tier of cores would probably
 * be ideal.
 * 
 * Primesieve will also try to choose a reasonable bucket size based on the number of threads.  Using more threads than hardware threads will probably make this
 * challenging and could result in very bad performance.
 */
pub fn get_num_threads() -> usize {
	unsafe { raw::primesieve_get_num_threads() as _ }
}

/**
 * set the number of threads primesieve's multithreaded functions will use, which defaults to the number of hardware threads.
 * Most CPUs have a feature called "hyperthreading" where each CPU core can run 2 hardware threads at once,
 * but some CPUs (mostly Intel and mobile) have "efficiency cores" which are slower and don't support hyperthreading.  Some CPUs like Apple's M chips
 * even have more core tiers than just efficiency and "performance", and I'm not sure generally how primesieve chooses the number of threads in this case.
 * In theory we could use one thread per hardware thread but allocate less work to efficiency cores, but in practice the threads we spawn would not actually
 * get pinned to certain cores without OS level muddling, so just spawning one thread per hardware thread on the highest performance tier of cores would probably
 * be ideal.
 * 
 * Primesieve will also try to choose a reasonable bucket size based on the number of threads.  Using more threads than hardware threads will probably make this
 * challenging and could result in very bad performance.
*/
pub fn set_sieve_size(n: usize) {
	unsafe { raw::primesieve_set_sieve_size(n as _) }
}

pub fn set_num_threads(n: usize) {
	unsafe { raw::primesieve_set_num_threads(n as _) }
}

pub fn get_version() -> &'static CStr {
	unsafe { CStr::from_ptr(raw::primesieve_version()) }
}
