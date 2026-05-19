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

    #[cfg(target_arch = "aarch64")]
    SVE,
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
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return Self { isa: ISA::SVE };
            }

            return Self { isa: ISA::NEON };
        }
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

            #[cfg(target_arch = "aarch64")]
            ISA::SVE => unsafe { Self::is_slot_full_sve(slot) },
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
        let full = vdupq_n_u64(FULL_WORD);
        let ptr = slot.as_ptr();

        let lo = vld1q_u64(ptr);
        let hi = vld1q_u64(ptr.add(2));

        let cmp_lo = vceqq_u64(lo, full);
        let cmp_hi = vceqq_u64(hi, full);

        let lo_full = vminvq_u64(cmp_lo) == FULL_WORD;
        let hi_full = vminvq_u64(cmp_hi) == FULL_WORD;

        lo_full && hi_full
    }

    #[cfg(all(target_arch = "aarch64"))]
    #[target_feature(enable = "sve")]
    unsafe fn is_slot_full_sve(slot: &Slot) -> bool {
        let pg = svptrue_b64();
        let vec = svld1_u64(pg, slot.as_ptr());
        let cmp = svcmpeq_n_u64(pg, vec, FULL_WORD);

        svptest_all(pg, cmp)
    }
}
