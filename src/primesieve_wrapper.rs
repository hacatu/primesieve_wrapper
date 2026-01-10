#![allow(non_camel_case_types, internal_features)]
#![feature(core_intrinsics, ptr_as_ref_unchecked)]
use std::{convert::{AsRef, AsMut}, borrow::{Borrow, BorrowMut}, ops::{Deref, DerefMut, Index, IndexMut}, ffi::CStr, marker::PhantomData, mem::MaybeUninit};

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

pub fn nth_prime<T: PrimeType>(n: isize, start: T) -> T {
	T::from_u64(unsafe { raw::primesieve_nth_prime(n as _, start.into_u64()) })
}

pub fn count_primes(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_primes(start, stop) as _ }
}

pub fn count_twins(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_twins(start, stop) as _ }
}

pub fn count_triplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_triplets(start, stop) as _ }
}

pub fn count_quadruplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_quadruplets(start, stop) as _ }
}

pub fn count_quintuplets(start: u64, stop: u64) -> usize {
	unsafe { raw::primesieve_count_quintuplets(start, stop) as _ }
}

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

pub fn get_num_threads() -> usize {
	unsafe { raw::primesieve_get_num_threads() as _ }
}

pub fn set_sieve_size(n: usize) {
	unsafe { raw::primesieve_set_sieve_size(n as _) }
}

pub fn set_num_threads(n: usize) {
	unsafe { raw::primesieve_set_num_threads(n as _) }
}

pub fn get_version() -> &'static CStr {
	unsafe { CStr::from_ptr(raw::primesieve_version()) }
}
