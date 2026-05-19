use crate::MODULE_ID;
use frozen_core::{error::FrozenRes, fmmap::FrozenMMap, hints};
use std::sync::atomic;

type WORD = u64;
type SLOT = [WORD; 4];
type MMap = FrozenMMap<SLOT, MODULE_ID>;

const _: () = assert!(std::mem::size_of::<SLOT>() == 0x100 >> 3);

const PAGES_AT_INIT: usize = 2;
const SLOTS_PER_PAGE: usize = 1 + 0x0F;

#[repr(C)]
struct PageHeader {
    available_bits: u64,
    current_bit_ptr: u64,
    current_word_ptr: u64,
    _reserved: u64,
}

impl PageHeader {
    #[inline(always)]
    fn new(slot: &SLOT) -> &Self {
        unsafe { &*(slot as *const SLOT as *const Self) }
    }

    #[inline(always)]
    fn new_mut(slot: &mut SLOT) -> &mut Self {
        unsafe { &mut *(slot as *mut SLOT as *mut Self) }
    }
}
