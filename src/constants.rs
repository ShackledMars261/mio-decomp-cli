pub const MAX_UINT64: u64 = u64::MAX;

pub const GIN_MAGIC_NUMBER: u32 = 0x004E4947; // Little endian ASCII: "GIN\0"
pub const GIN_VERSION: u32 = 2;

pub const GIN_SECTION_NAME_SIZE: usize = 64;
pub const GIN_SECTION_PARAM_COUNT: usize = 4;

pub const GIN_MAX_PATH: usize = 256;

pub const GIN_SECTION_DUMMY_ID: u64 = MAX_UINT64; // for non-queryable sections (ex: referenced by other sections, .reloc & co)

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Flags: u32 {
        const SERIALIZED = 1 << 0;
        const RELOC      = 1 << 1;
        const ALLOC      = 1 << 2;
        const SCHEMA     = 1 << 3;
        const ZSTD       = 1 << 4;
        const LZ4        = 1 << 5;
    }
}
