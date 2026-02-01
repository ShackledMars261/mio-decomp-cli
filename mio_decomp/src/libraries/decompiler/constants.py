import sys
from enum import IntEnum

GIN_MAGIC_NUMBER: int = 0x004E4947  # Little endian ascii translates this into "GIN\0"
GIN_VERSION: int = 2

GIN_SECTION_NAME_SIZE = 64
GIN_SECTION_PARAM_COUNT = 4

GIN_MAX_PATH = 256

GIN_SECTION_DUMMY_ID: int = (
    sys.maxsize
)  # for non-queryable sections (ex: referenced by other sections, .reloc & co)


class FLAGS(IntEnum):
    SERIALIZED: int = 1 << 0
    RELOC: int = 1 << 1
    ALLOC: int = 1 << 2
    SCHEMA: int = 1 << 3
    ZSTD: int = 1 << 4
    LZ4: int = 1 << 5
