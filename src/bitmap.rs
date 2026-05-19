use crate::MODULE_ID;
use frozen_core::{error::FrozenRes, fmmap::FrozenMMap, hints};
use std::sync::atomic;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

type Word = u64;
type Slot = [Word; 4];
type MMap = FrozenMMap<Slot, MODULE_ID>;

const _: () = assert!(std::mem::size_of::<Slot>() == 0x100 >> 3);

const PAGES_AT_INIT: usize = 2;
const FULL_WORD: Word = Word::MAX;
const SLOTS_PER_PAGE: usize = 1 + 0x0F;

const _: () = assert!(Word::MAX == FULL_WORD);
const _: () = assert!(SLOTS_PER_PAGE & (SLOTS_PER_PAGE - 1) == 0);

#[repr(C)]
struct PageHeader {
    available_bits: u64,
    current_bit_ptr: u64,
    current_word_ptr: u64,
    _reserved: u64,
}

impl PageHeader {
    #[inline(always)]
    fn new(slot: &Slot) -> &Self {
        unsafe { &*(slot as *const Slot as *const Self) }
    }

    #[inline(always)]
    fn new_mut(slot: &mut Slot) -> &mut Self {
        unsafe { &mut *(slot as *mut Slot as *mut Self) }
    }
}

enum ISA {
    #[cfg(target_arch = "x86_64")]
    SSE2,

    #[cfg(target_arch = "x86_64")]
    AVX2,

    #[cfg(target_arch = "aarch64")]
    NEON,
}

struct SIMD {
    isa: ISA,
}

impl SIMD {
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Self { isa: ISA::AVX2 };
            }

            return Self { isa: ISA::SSE2 };
        }

        #[cfg(target_arch = "aarch64")]
        return Self { isa: ISA::NEON };
    }

    #[inline(always)]
    unsafe fn is_slot_full(&self, slot: &Slot) -> bool {
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            ISA::SSE2 => unsafe { Self::is_slot_full_sse2(slot) },

            #[cfg(target_arch = "x86_64")]
            ISA::AVX2 => unsafe { Self::is_slot_full_avx2(slot) },

            #[cfg(target_arch = "aarch64")]
            ISA::NEON => unsafe { Self::is_slot_full_neon(slot) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn is_slot_full_sse2(slot: &Slot) -> bool {
        let full = _mm_set1_epi64x(FULL_WORD as i64);
        let ptr = slot.as_ptr() as *const __m128i;

        let lo = _mm_loadu_si128(ptr);
        let hi = _mm_loadu_si128(ptr.add(1));

        let cmp_lo = _mm_cmpeq_epi32(lo, full);
        let cmp_hi = _mm_cmpeq_epi32(hi, full);

        let lo_full = _mm_movemask_epi8(cmp_lo) == 0xFFFF;
        let hi_full = _mm_movemask_epi8(cmp_hi) == 0xFFFF;

        lo_full && hi_full
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_slot_full_avx2(slot: &Slot) -> bool {
        let full = _mm256_set1_epi64x(FULL_WORD as i64);
        let ymm = _mm256_loadu_si256(slot.as_ptr() as *const __m256i);
        let cmp = _mm256_cmpeq_epi64(ymm, full);

        _mm256_movemask_epi8(cmp) == -1
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn is_slot_full_neon(slot: &Slot) -> bool {
        let ptr = slot.as_ptr();
        let lo = vld1q_u64(ptr);
        let hi = vld1q_u64(ptr.add(2));

        let anded = vandq_u64(lo, hi);
        vgetq_lane_u64(anded, 0) == FULL_WORD && vgetq_lane_u64(anded, 1) == FULL_WORD
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[inline(always)]
    fn is_slot_full(slot: &Slot) -> bool {
        slot.iter().all(|&x| x == FULL_WORD)
    }

    fn validate_impl<F>(func: F)
    where
        F: Fn(&Slot) -> bool,
    {
        let cases = [
            (([FULL_WORD, FULL_WORD, FULL_WORD, FULL_WORD]), true),
            (([0, FULL_WORD, FULL_WORD, FULL_WORD]), false),
            (([FULL_WORD, 0, FULL_WORD, FULL_WORD]), false),
            (([FULL_WORD, FULL_WORD, 0, FULL_WORD]), false),
            (([FULL_WORD, FULL_WORD, FULL_WORD, 0]), false),
            (([0, 0, 0, 0]), false),
            (([1, 2, 3, 4]), false),
            (([FULL_WORD, FULL_WORD - 1, FULL_WORD, FULL_WORD]), false),
        ];

        for (slot, expected) in cases {
            assert_eq!(func(&slot), expected);
            assert_eq!(func(&slot), is_slot_full(&slot));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ok_sse2_isa() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }

        unsafe {
            validate_impl(|slot| SIMD::is_slot_full_sse2(slot));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ok_avx2_isa() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            validate_impl(|slot| SIMD::is_slot_full_avx2(slot));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ok_neon_isa() {
        unsafe {
            validate_impl(|slot| SIMD::is_slot_full_neon(slot));
        }
    }

    #[test]
    fn ok_runtime_dispatch_correctness() {
        let simd = SIMD::new();
        let test_cases = [
            [FULL_WORD, FULL_WORD, FULL_WORD, FULL_WORD],
            [0, FULL_WORD, FULL_WORD, FULL_WORD],
            [FULL_WORD, 0, FULL_WORD, FULL_WORD],
            [FULL_WORD, FULL_WORD, 0, FULL_WORD],
            [FULL_WORD, FULL_WORD, FULL_WORD, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
        ];

        unsafe {
            for slot in test_cases {
                assert_eq!(simd.is_slot_full(&slot), is_slot_full(&slot));
            }
        }
    }

    #[test]
    fn ok_is_full_works_with_many_patterns() {
        let simd = SIMD::new();
        let patterns = [
            0,
            1,
            0xFF,
            0xFFFF,
            0xFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            FULL_WORD - 1,
            FULL_WORD,
        ];

        unsafe {
            for &a in &patterns {
                for &b in &patterns {
                    for &c in &patterns {
                        for &d in &patterns {
                            let slot = [a, b, c, d];
                            assert_eq!(
                                is_slot_full(&slot),
                                simd.is_slot_full(&slot),
                                "failed for slot: {:?}",
                                slot
                            );
                        }
                    }
                }
            }
        }
    }
}
