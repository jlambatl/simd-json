#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

use arch::{
    __m512i, _mm512_loadu_si512, _mm512_storeu_si512,
    _mm512_cmpeq_epi8_mask, _mm512_set1_epi8,
};

use crate::{
    Deserializer, Result, SillyWrapper,
    error::ErrorType,
    macros::static_cast_u32,
    safer_unchecked::GetSaferUnchecked,
    stringparse::{ESCAPE_MAP, handle_unicode_codepoint},
};

#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[allow(
    clippy::if_not_else,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
#[cfg_attr(not(feature = "no-inline"), inline)]
pub(crate) unsafe fn parse_str<'invoke, 'de>(
    input: SillyWrapper<'de>,
    data: &'invoke [u8],
    buffer: &'invoke mut [u8],
    mut idx: usize,
) -> Result<&'de str> {
    unsafe {
        use ErrorType::{InvalidEscape, InvalidUnicodeCodepoint};

        let input = input.input;
        idx += 1;

        let src: &[u8] = data.get_kinda_unchecked(idx..);
        let mut src_i: usize = 0;
        let mut len = src_i;
        loop {
            #[allow(clippy::cast_ptr_alignment)]
            let v: __m512i = _mm512_loadu_si512(src.as_ptr().add(src_i).cast::<__m512i>());

            let bs_mask = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\\' as i8));
            let quote_mask = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\"' as i8));
            
            if (bs_mask.wrapping_sub(1) & quote_mask) != 0 {
                let quote_dist: u32 = quote_mask.trailing_zeros();
                len += quote_dist as usize;
                let v = std::str::from_utf8_unchecked(std::slice::from_raw_parts(input.add(idx), len));
                return Ok(v);
            }
            if (quote_mask.wrapping_sub(1) & bs_mask) == 0 {
                src_i += 64;
                len += 64;
            } else {
                let bs_dist: u32 = bs_mask.trailing_zeros();
                len += bs_dist as usize;
                src_i += bs_dist as usize;
                break;
            }
        }

        let mut dst_i: usize = 0;

        loop {
            #[allow(clippy::cast_ptr_alignment)]
            let v: __m512i = _mm512_loadu_si512(src.as_ptr().add(src_i).cast::<__m512i>());

            #[allow(clippy::cast_ptr_alignment)]
            _mm512_storeu_si512(buffer.as_mut_ptr().add(dst_i).cast::<__m512i>(), v);

            let bs_mask = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\\' as i8));
            let quote_mask = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\"' as i8));
            
            if (bs_mask.wrapping_sub(1) & quote_mask) != 0 {
                let quote_dist: u32 = quote_mask.trailing_zeros();
                dst_i += quote_dist as usize;
                input
                    .add(idx + len)
                    .copy_from_nonoverlapping(buffer.as_ptr(), dst_i);
                let v = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    input.add(idx),
                    len + dst_i,
                ));
                return Ok(v);
            }
            if (quote_mask.wrapping_sub(1) & bs_mask) != 0 {
                let bs_dist: u32 = bs_mask.trailing_zeros();
                let escape_char: u8 = *src.get_kinda_unchecked(src_i + bs_dist as usize + 1);
                if escape_char == b'u' {
                    src_i += bs_dist as usize;
                    dst_i += bs_dist as usize;
                    let (o, s) = handle_unicode_codepoint(
                        src.get_kinda_unchecked(src_i..),
                        buffer.get_kinda_unchecked_mut(dst_i..),
                    )
                    .map_err(|_| Deserializer::error_c(src_i, 'u', InvalidUnicodeCodepoint))?;

                    if o == 0 {
                        return Err(Deserializer::error_c(src_i, 'u', InvalidUnicodeCodepoint));
                    }
                    src_i += s;
                    dst_i += o;
                } else {
                    let escape_result: u8 = *ESCAPE_MAP.get_kinda_unchecked(escape_char as usize);
                    if escape_result == 0 {
                        return Err(Deserializer::error_c(
                            src_i,
                            escape_char as char,
                            InvalidEscape,
                        ));
                    }
                    *buffer.get_kinda_unchecked_mut(dst_i + bs_dist as usize) = escape_result;
                    src_i += bs_dist as usize + 2;
                    dst_i += bs_dist as usize + 1;
                }
            } else {
                src_i += 64;
                dst_i += 64;
            }
        }
    }
}
