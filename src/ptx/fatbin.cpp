#include "fatbin.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pxvk::fatbin {
namespace {

constexpr std::uint32_t kMagic = 0xBA55ED50u;
// Entry header versions: 0x101 (CUDA 5+), 0x100 (CUDA 4.x; longer headers
// that append the source file name, and no compression).
constexpr std::uint16_t kFileVersion = 0x101;
constexpr std::uint16_t kFileVersionCuda4 = 0x100;
constexpr std::uint16_t kKindPtx = 1;
constexpr std::uint64_t kFlagLz4 = 0x2000;
constexpr std::uint64_t kFlagZstd = 0x8000;
constexpr std::size_t kMaxBytes = 64u * 1024u * 1024u;

#pragma pack(push, 1)
struct Header {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t header_size;
    std::uint64_t files_size;
};
struct FileHeader {
    std::uint16_t kind;
    std::uint16_t version;
    std::uint32_t header_size;
    std::uint32_t payload_size;
    std::uint32_t unknown0;
    std::uint32_t compressed_size;
    std::uint32_t unknown1;
    std::uint32_t unknown2;
    std::uint32_t sm_version;
    std::uint32_t bit_width;
    std::uint32_t unknown3;
    std::uint64_t flags;
    std::uint64_t unknown5;
    std::uint64_t uncompressed_size;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 16, "fatbin header layout");
static_assert(sizeof(FileHeader) == 64, "fatbin file header layout");

std::string lz4_block(const unsigned char *src, std::size_t size, std::size_t expected)
{
    std::string out;
    out.reserve(expected);
    std::size_t pos = 0;
    auto length = [&](std::size_t base) {
        if (base != 15) return base;
        unsigned char extension = 255;
        while (extension == 255) {
            if (pos >= size) throw std::runtime_error("truncated LZ4 length");
            extension = src[pos++];
            base += extension;
            if (base > expected) throw std::runtime_error("invalid LZ4 length");
        }
        return base;
    };
    while (pos < size) {
        const unsigned char token = src[pos++];
        const std::size_t literals = length(token >> 4);
        if (pos + literals > size || out.size() + literals > expected)
            throw std::runtime_error("invalid LZ4 literal run");
        out.append(reinterpret_cast<const char *>(src + pos), literals);
        pos += literals;
        if (pos == size) break;
        if (pos + 2 > size) throw std::runtime_error("truncated LZ4 match offset");
        const std::size_t offset = src[pos] | (src[pos + 1] << 8);
        pos += 2;
        if (offset == 0 || offset > out.size()) throw std::runtime_error("invalid LZ4 match offset");
        const std::size_t match = length(token & 15u) + 4;
        if (out.size() + match > expected) throw std::runtime_error("invalid LZ4 match length");
        for (std::size_t i = 0; i < match; ++i) out.push_back(out[out.size() - offset]);
    }
    if (out.size() != expected) throw std::runtime_error("LZ4 size mismatch");
    return out;
}

} // namespace

std::size_t container_size(const void *data)
{
    Header header;
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != kMagic || header.version != 1 || header.header_size < sizeof(Header) ||
        header.files_size > kMaxBytes)
        return 0;
    return header.header_size + static_cast<std::size_t>(header.files_size);
}

std::vector<PtxPayload> extract_ptx(const void *data, std::size_t size)
{
    const auto *bytes = static_cast<const unsigned char *>(data);
    if (size < sizeof(Header)) throw std::runtime_error("truncated fatbin header");
    const std::size_t total = container_size(data);
    if (total == 0 || total > size) throw std::runtime_error("invalid fatbin header");
    Header header;
    std::memcpy(&header, bytes, sizeof(header));
    std::vector<PtxPayload> out;
    // Header fields are untrusted 32-bit values and size_t is 32 bits in the
    // x86 runtime: do the bounds arithmetic in 64 bits so it cannot wrap.
    std::uint64_t offset = header.header_size;
    while (offset + sizeof(FileHeader) <= total) {
        FileHeader file;
        std::memcpy(&file, bytes + offset, sizeof(file));
        const bool cuda4 = file.version == kFileVersionCuda4;
        if (cuda4) file.compressed_size = 0, file.flags = 0;
        const std::uint64_t stored = std::max<std::uint64_t>(file.payload_size, file.compressed_size);
        const std::uint64_t payload = offset + file.header_size;
        if ((file.version != kFileVersion && !cuda4) || file.header_size < sizeof(FileHeader) ||
            payload + stored > total)
            throw std::runtime_error("invalid fatbin file header");
        if (file.kind == kKindPtx) {
            PtxPayload ptx;
            ptx.sm_version = file.sm_version;
            if (file.flags & kFlagZstd) throw std::runtime_error("Zstd-compressed PTX is unsupported");
            if (file.flags & kFlagLz4) {
                if (file.uncompressed_size > kMaxBytes || file.compressed_size > stored)
                    throw std::runtime_error("invalid compressed PTX size");
                ptx.text = lz4_block(bytes + payload, static_cast<std::size_t>(file.compressed_size),
                                     static_cast<std::size_t>(file.uncompressed_size));
            } else {
                ptx.text.assign(reinterpret_cast<const char *>(bytes + payload),
                                static_cast<std::size_t>(file.payload_size));
            }
            while (!ptx.text.empty() && ptx.text.back() == '\0') ptx.text.pop_back();
            if (ptx.text.empty()) throw std::runtime_error("empty PTX payload");
            out.push_back(std::move(ptx));
        }
        offset = payload + stored;
    }
    return out;
}

} // namespace pxvk::fatbin
