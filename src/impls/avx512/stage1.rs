#![allow(dead_code)]
use crate::{
    Stage1Parse,
    macros::{static_cast_i32, static_cast_i64, static_cast_u32},
};
#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

use arch::{
    __m512i, _mm512_loadu_si512, _mm512_storeu_si512,
    _mm512_set1_epi8, _mm512_cmpeq_epi8_mask, _mm512_cmpgt_epu8_mask,
    _mm512_shuffle_epi8, _mm512_and_si512, _mm512_set_epi8,
    _mm512_srli_epi32, _mm512_set_epi32, _mm512_add_epi32,
    __mmask64, _mm_clmulepi64_si128, _mm_set_epi64x, _mm_set1_epi8,
};

macro_rules! low_nibble_mask {
    () => {
        _mm512_set_epi8(
            16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0,
            16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0,
            16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0,
            16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0,
        )
    };
}

macro_rules! high_nibble_mask {
    () => {
        _mm512_set_epi8(
            8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0,
            8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0,
            8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0,
            8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0,
        )
    };
}

#[derive(Debug)]
pub(crate) struct SimdInput {
    v0: __m512i,
}

impl Stage1Parse for SimdInput {
    type Utf8Validator = simdutf8::basic::imp::x86::avx2::ChunkedUtf8ValidatorImp;
    type SimdRepresentation = __m512i;
    
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_ptr_alignment)]
    #[target_feature(enable = "avx512f")]
    unsafe fn new(ptr: &[u8]) -> Self {
        unsafe {
            Self {
                v0: _mm512_loadu_si512(ptr.as_ptr().cast::<__m512i>()),
            }
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    #[target_feature(enable = "avx512f")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_quote_mask(quote_bits: u64) -> u64 {
        unsafe {
            std::arch::x86_64::_mm_cvtsi128_si64(_mm_clmulepi64_si128(
                _mm_set_epi64x(0, static_cast_i64!(quote_bits)),
                _mm_set1_epi8(-1_i8),
                0,
            )) as u64
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    #[target_feature(enable = "avx512f")]
    #[cfg(target_arch = "x86")]
    unsafe fn compute_quote_mask(quote_bits: u64) -> u64 {
        let mut quote_mask: u64 = quote_bits ^ (quote_bits << 1);
        quote_mask = quote_mask ^ (quote_mask << 2);
        quote_mask = quote_mask ^ (quote_mask << 4);
        quote_mask = quote_mask ^ (quote_mask << 8);
        quote_mask = quote_mask ^ (quote_mask << 16);
        quote_mask = quote_mask ^ (quote_mask << 32);
        quote_mask
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn cmp_mask_against_input(&self, m: u8) -> u64 {
        unsafe {
            let mask: __m512i = _mm512_set1_epi8(m as i8);
            let cmp_res: __mmask64 = _mm512_cmpeq_epi8_mask(self.v0, mask);
            cmp_res
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn unsigned_lteq_against_input(&self, maxval: __m512i) -> u64 {
        unsafe {
            let cmp_res: __mmask64 = _mm512_cmpgt_epu8_mask(self.v0, maxval);
            !cmp_res
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn find_whitespace_and_structurals(&self, whitespace: &mut u64, structurals: &mut u64) {
        unsafe {
            let low_nibble_mask: __m512i = low_nibble_mask!();
            let high_nibble_mask: __m512i = high_nibble_mask!();

            let structural_shufti_mask: __m512i = _mm512_set1_epi8(0x7);
            let whitespace_shufti_mask: __m512i = _mm512_set1_epi8(0x18);

            let v_lo: __m512i = _mm512_and_si512(
                _mm512_shuffle_epi8(low_nibble_mask, self.v0),
                _mm512_shuffle_epi8(
                    high_nibble_mask,
                    _mm512_and_si512(_mm512_srli_epi32(self.v0, 4), _mm512_set1_epi8(0x7f)),
                ),
            );

            let structural_mask: __mmask64 = _mm512_cmpeq_epi8_mask(
                _mm512_and_si512(v_lo, structural_shufti_mask),
                _mm512_set1_epi8(0),
            );
            *structurals = !structural_mask;

            let whitespace_mask: __mmask64 = _mm512_cmpeq_epi8_mask(
                _mm512_and_si512(v_lo, whitespace_shufti_mask),
                _mm512_set1_epi8(0),
            );
            *whitespace = !whitespace_mask;
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap, clippy::cast_ptr_alignment)]
    #[target_feature(enable = "avx512f")]
    unsafe fn flatten_bits(base: &mut Vec<u32>, idx: u32, mut bits: u64) {
        unsafe {
            let cnt: usize = bits.count_ones() as usize;
            let mut l = base.len();
            let idx_minus_64 = idx.wrapping_sub(64);
            let idx_64_v = _mm512_set_epi32(
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
            );

            base.reserve(64);
            let final_len = l + cnt;

            while bits != 0 {
                let v0 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v1 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v2 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v3 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v4 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v5 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v6 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v7 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v8 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v9 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v10 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v11 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v12 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v13 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v14 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v15 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);

                let v: __m512i = _mm512_set_epi32(
                    v15, v14, v13, v12, v11, v10, v9, v8,
                    v7, v6, v5, v4, v3, v2, v1, v0
                );
                let v: __m512i = _mm512_add_epi32(idx_64_v, v);
                _mm512_storeu_si512(base.as_mut_ptr().add(l).cast::<__m512i>(), v);
                l += 16;
            }
            base.set_len(final_len);
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[target_feature(enable = "avx512f")]
    unsafe fn fill_s8(n: i8) -> __m512i {
        unsafe { _mm512_set1_epi8(n) }
    }
}
