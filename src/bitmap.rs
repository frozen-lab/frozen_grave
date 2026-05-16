#![allow(unused)]

use crate::MODULE_ID;
use frozen_core::{error::FrozenRes, fmmap::FrozenMMap};

type WORD = u32;

/// A custom type used for [`FrozenMMap`]
///
/// Each type contains 8 * u32 slots, i.e. 256 bits. This is done to ensure that the [`SLOT`]
/// could easily fit into a single `avx2` register for best possible performance for lookup
type SLOT = [WORD; WORD_PER_SLOT];

/// Number of slots available per page.
///
/// Each page contains 16 * [`SLOT`], i.e. 512 bytes
const SLOT_PER_PAGE: u32 = 0x10;

const WORD_PER_SLOT: usize = 8;
const BITS_PER_PAGE: usize = BITS_PER_SLOT * SLOT_PER_PAGE as usize;
const BITS_PER_SLOT: usize = 8 * WORD_PER_SLOT * std::mem::size_of::<u32>();

/// Number of pages available when a new index is initialized from scratch
///
/// NOTE: First slot is reserved for [`Header`]
const INITIAL_SLOT_CAPACITY: usize = (SLOT_PER_PAGE as usize * 4) + 1;

struct BitMap {
    mmap: FrozenMMap<SLOT, MODULE_ID>,
    simd: SIMD,
}

impl BitMap {
    pub(crate) fn new<P: AsRef<std::path::Path>>(path: P, flush_duration: std::time::Duration) -> FrozenRes<Self> {
        let mmap = FrozenMMap::<SLOT, MODULE_ID>::new(
            path,
            frozen_core::fmmap::FMCfg {
                flush_duration,
                initial_count: INITIAL_SLOT_CAPACITY,
            },
        )?;

        Ok(Self {
            mmap,
            simd: SIMD::new(),
        })
    }

    #[inline(always)]
    unsafe fn lookup_2(&self) -> FrozenRes<Option<(usize, u32, u32)>> {
        let header = Header(self.mmap.read(0, |hdr| *hdr)?);

        let total_pages = header.total_pages();
        let start_page = header.current_page();
        let start_slot = header.current_slot();

        for rel_page in 0..total_pages {
            let slot_begin = if rel_page == 0 { start_slot } else { 0 };
            let page_idx = (start_page + rel_page) % total_pages;
            let page_off = 1 + (page_idx * SLOT_PER_PAGE);

            for slot_off in slot_begin..SLOT_PER_PAGE {
                let mmap_slot = (page_off + slot_off) as usize;
                let slot = unsafe { self.mmap.read(mmap_slot, |s| *s) }?;

                if self.simd.is_full(&slot) {
                    continue;
                }

                for word_idx in 0..WORD_PER_SLOT {
                    let word = slot[word_idx];
                    if word == WORD::MAX {
                        continue;
                    }

                    let inv = !word;
                    let candidate = inv & (inv >> 1);

                    if candidate == 0 {
                        continue;
                    }

                    let bit_idx = candidate.trailing_zeros() as usize;
                    let abs = (page_idx as usize * BITS_PER_PAGE)
                        + (slot_off as usize * BITS_PER_SLOT)
                        + (word_idx * 0x20)
                        + bit_idx;

                    return Ok(Some((abs, page_idx, slot_off)));
                }
            }
        }

        Ok(None)
    }
}

#[repr(C)]
struct Header(SLOT);

impl Header {
    const CURRENT_PAGE_INDEX: usize = 1;
    const CURRENT_SLOT_INDEX: usize = 2;
    const TOTAL_PAGES_INDEX: usize = 0;
    const AVAILABLE_FREE_SLOTS_INDEX: usize = 3;

    #[inline]
    fn total_pages(&self) -> u32 {
        self.0[Self::TOTAL_PAGES_INDEX]
    }

    #[inline]
    fn available_free_slots(&self) -> u32 {
        self.0[Self::AVAILABLE_FREE_SLOTS_INDEX]
    }

    #[inline]
    fn current_slot(&self) -> u32 {
        self.0[Self::CURRENT_SLOT_INDEX]
    }

    #[inline]
    fn current_page(&self) -> u32 {
        self.0[Self::CURRENT_PAGE_INDEX]
    }
}

enum ISA {
    SSE2,
    AVX2,
    NEON,
}

struct SIMD(ISA);

impl SIMD {
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Self(ISA::AVX2);
            }

            return Self(ISA::SSE2);
        }

        #[cfg(target_arch = "aarch64")]
        return Self(ISA::NEON);
    }

    unsafe fn is_full(&self, slot: &SLOT) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            return match self.0 {
                ISA::SSE2 => self._is_full_sse2(slot),
                ISA::AVX2 => self._is_full_avx2(slot),
                _ => unreachable!(),
            };
        }

        #[cfg(target_arch = "aarch64")]
        return self._is_full_neon(slot);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn _is_full_sse2(&self, slot: &SLOT) -> bool {
        use std::arch::x86_64::*;

        let full = _mm_set1_epi32(-1);
        let ptr = slot.as_ptr() as *const __m128i;

        let lo = unsafe { _mm_loadu_si128(ptr) };
        let hi = unsafe { _mm_loadu_si128(ptr.add(1)) };

        let cmp_lo = _mm_cmpeq_epi32(lo, full);
        let cmp_hi = _mm_cmpeq_epi32(hi, full);

        let lo_full = _mm_movemask_epi8(cmp_lo) == 0xFFFF;
        let hi_full = _mm_movemask_epi8(cmp_hi) == 0xFFFF;

        lo_full && hi_full
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn _is_full_avx2(&self, slot: &SLOT) -> bool {
        use std::arch::x86_64::*;

        let full = _mm256_set1_epi32(-1);
        let ymm = unsafe { _mm256_loadu_si256(slot.as_ptr() as *const __m256i) };

        let cmp = _mm256_cmpeq_epi32(ymm, full);
        _mm256_movemask_epi8(cmp) == -1
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn _is_full_neon(&self, slot: &SLOT) -> bool {
        use std::arch::aarch64::*;

        let ptr = slot.as_ptr();
        let full = vdupq_n_u32(WORD::MAX);

        let lo = unsafe { vld1q_u32(ptr) };
        let hi = unsafe { vld1q_u32(ptr.add(4)) };

        let cmp_lo = vceqq_u32(lo, full);
        let cmp_hi = vceqq_u32(hi, full);

        let lo_full = unsafe { vminvq_u32(cmp_lo) } == WORD::MAX;
        let hi_full = unsafe { vminvq_u32(cmp_hi) } == WORD::MAX;

        lo_full && hi_full
    }
}
