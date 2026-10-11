// PTX entry -> Vulkan compute SPIR-V.
//
// Shader interface (descriptor set 0), shared by every compiled kernel so the
// runtime can use one pipeline layout and one descriptor set:
//   binding 0  storage buffer  uint  heap[]   CUDA global memory (byte addr/4)
//   binding 1  storage buffer  uvec2 heap2[]  same buffer, 8-byte view
//   binding 2  storage buffer  uvec4 heap4[]  same buffer, 16-byte view
//   binding 4  storage buffer  uint  scratch[] storage-backed shared memory
//              (only with CompileOptions::shared_in_storage)
//   binding 3  uniform buffer  launch block (dynamic offset):
//                bytes [0, param_bytes)            packed kernel parameters
//                bytes [texture_table_offset, ...) per texref: {address,
//                                                   element count,
//                                                   components, 0}
//                bytes [symbol_table_offset, ...)  u32 CUDA address of each
//                                                   KernelInfo::symbols entry
// Specialization constants:
//   0,1,2  workgroup size x/y/z (%ntid)
//   3      shared memory size in 32-bit words (static + dynamic)
//   4      heap base: CUDA address of heap byte 0
#pragma once

#include "ptx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pxvk::ptx {

constexpr std::uint32_t kSpecWorkgroupX = 0;
constexpr std::uint32_t kSpecSharedWords = 3;
constexpr std::uint32_t kSpecHeapBase = 4;
constexpr std::uint32_t kBindingHeap = 0;
constexpr std::uint32_t kBindingHeap2 = 1;
constexpr std::uint32_t kBindingHeap4 = 2;
constexpr std::uint32_t kBindingLaunch = 3;
constexpr std::uint32_t kBindingScratch = 4;
constexpr std::uint32_t kTextureEntryBytes = 16;

struct CompileOptions {
    // Device supports VK_KHR_shader_float_controls denormFlushToZero for
    // fp32; kernels whose float arithmetic is entirely .ftz then use the
    // native execution mode instead of per-operation emulation.
    bool native_ftz = false;
    // Device supports the SignedZeroInfNanPreserve execution mode for fp32.
    // Without it Vulkan lets drivers assume NaN/Inf never occur, which breaks
    // PTX unordered comparisons (setp.ltu etc.) on some drivers.
    bool preserve_nan = false;
    // Forbid fusing f32 mul+add into fma (PTX permits fusion unless a
    // rounding modifier is present). Mainly for numerical experiments.
    bool no_contraction = false;
    // Place CUDA shared memory in a storage buffer (binding 4) instead of
    // Workgroup memory: each workgroup uses words [linear_id * S, (linear_id
    // + 1) * S) where S is spec constant 3. For launches whose shared memory
    // exceeds the device's maxComputeSharedMemorySize.
    bool shared_in_storage = false;
    // The device cannot guarantee subgroups of at least 32 lanes (e.g.
    // lavapipe's 8). Warp-synchronous code (volatile shared stores) is then
    // emulated with workgroup barriers around each store; `if` arms that
    // contain such code are predicated so that every invocation reaches
    // the barriers. The kernel no longer needs a subgroup size.
    bool emulate_warps = false;
    // Device supports shaderInt64; kernels with 64-bit integers fail to
    // compile without it.
    bool int64 = true;
    // Device supports VK_EXT_shader_atomic_float's shaderBufferFloat32AtomicAdd:
    // global atom/red.add.f32 use OpAtomicFAddEXT instead of a compare-and-swap
    // loop (not with a split heap; shared memory always uses the loop).
    bool float_atomic_add = false;
    // Split heap, for devices whose maxStorageBufferRange is smaller than
    // the CUDA heap: binding 0 is an array of heap_windows descriptors, each
    // covering 2^heap_window_shift bytes of the heap buffer, selected per
    // access by address (non-uniform descriptor indexing), plus one dummy
    // descriptor that out-of-range indices are clamped to. Vector accesses
    // become scalar ones and bindings 1-2 are unused. 0 = one descriptor.
    std::uint32_t heap_window_shift = 0;
    std::uint32_t heap_windows = 0;
};

struct KernelInfo {
    std::string name;
    std::uint32_t param_bytes = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> params; // offset, size per parameter
    std::uint32_t texture_table_offset = 0;
    std::uint32_t symbol_table_offset = 0;
    std::uint32_t launch_block_bytes = 0;
    std::uint32_t static_shared_bytes = 0;
    bool uses_dynamic_shared = false;
    std::uint32_t local_bytes = 0;
    std::uint32_t maxntid[3] = {0, 0, 0};
    std::uint32_t reqntid[3] = {0, 0, 0};
    std::vector<std::string> symbols;  // module .global/.const variables used
    std::vector<std::string> textures; // texrefs used, in texture-table order
    bool uses_int64 = false;
    bool uses_barrier = false;
    // Kernel relies on warp-synchronous shared memory (volatile accesses);
    // the runtime should request a subgroup size of 32.
    bool warp_synchronous = false;
    // Compiled with CompileOptions::emulate_warps and contains warp-
    // synchronous code, which is emulated with workgroup barriers.
    bool warp_emulated = false;
};

struct CompiledKernel {
    std::vector<std::uint32_t> spirv;
    KernelInfo info;
};

CompiledKernel compile_kernel(const Module &module, const std::string &entry,
                              const CompileOptions &options = {});

} // namespace pxvk::ptx
