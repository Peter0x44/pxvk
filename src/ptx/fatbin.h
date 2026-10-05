// Classic CUDA fatbin container (magic 0xBA55ED50) decoding.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pxvk::fatbin {

struct PtxPayload {
    std::string text;
    std::uint32_t sm_version = 0;
};

// Total container size from its 16-byte header, or 0 if the header is invalid.
std::size_t container_size(const void *data);

// Extracts every PTX payload. Throws std::runtime_error on malformed input;
// cubin-only containers yield an empty vector.
std::vector<PtxPayload> extract_ptx(const void *data, std::size_t size);

} // namespace pxvk::fatbin
