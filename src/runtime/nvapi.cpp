// pxvk-nvapi.dll: the minimal NVAPI that PhysX needs to select hardware PhysX.
//
// The PhysX runtime (PhysXDevice.dll, PhysXCore.dll) asks NVAPI for a
// CUDA-capable GPU before it makes any CUDA call; without an NVIDIA driver
// there is no nvapi.dll and PhysX silently falls back to CPU PhysX. This DLL
// reports one PhysX-capable GPU. loader.cpp redirects only the PhysX DLLs'
// nvapi.dll loads to it (PhysXDevice uses LOAD_LIBRARY_SEARCH_SYSTEM32, so
// an application-local nvapi.dll is never found), only in processes
// started through pxvk-launch, and only when the system has no nvapi.dll
// (with an NVIDIA driver installed PhysX keeps using the real NVAPI). The
// application's own NVAPI use is untouched.
//
// NVAPI exports a single function, nvapi_QueryInterface(id), returning a
// cdecl function pointer or null for an unknown id ("no implementation",
// which callers treat as an absent feature). The implemented ids follow
// ZLUDA's NVAPI (nvapi/src/lib.rs at commit 8a7931be). The native Mirror's
// Edge trace also shows NvAPI_EnumLogicalGPUs, NvAPI_EnumPhysicalGPUs,
// NvAPI_Stereo_IsEnabled and private ids 0x84244524/0xcb7b09d2; like ZLUDA,
// these stay unimplemented until a workload shows they matter. The SDK
// samples select hardware PhysX without them.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstring>

namespace {

constexpr int kNvapiOk = 0;
constexpr int kNvapiInvalidArgument = -5;
constexpr int kNvapiIncompatibleStructVersion = -9;
// NV_COMPUTE_GPU flags reported for the PhysX-selected GPU (CUDA capable,
// PhysX capable and enabled, ...); the values are ZLUDA's.
constexpr std::uint32_t kPhysicsFlags =
    (1u << 0) | (1u << 1) | (1u << 3) | (1u << 4) | (1u << 16) | (1u << 21);

struct ComputeGpu {
    void *physical_gpu;
    std::uint32_t flags;
};

struct ComputeTopologyV1 {
    std::uint32_t version;
    std::uint32_t gpu_count;
    ComputeGpu compute_gpus[8];
};

struct ComputeTopologyV2 {
    std::uint32_t version;
    std::uint32_t gpu_count;
    ComputeGpu *compute_gpus;
};

struct DisplayDriverVersion {
    std::uint32_t version;
    std::uint32_t driver_version;
    std::uint32_t build_change_list;
    char build_branch[64];
    char adapter[64];
};

static_assert(sizeof(ComputeTopologyV1) == 72, "NVAPI V1 topology ABI mismatch");
static_assert(sizeof(ComputeTopologyV2) == 12, "NVAPI V2 topology ABI mismatch");
static_assert(sizeof(DisplayDriverVersion) == 140, "NVAPI driver version ABI mismatch");

constexpr std::uint32_t make_version(std::uint32_t size, std::uint32_t version)
{
    return size | (version << 16);
}

template <typename T>
constexpr std::uint32_t make_version(std::uint32_t version)
{
    return make_version(sizeof(T), version);
}

// The single GPU: opaque physical-GPU handle 1, GPU id 1.
ComputeGpu g_compute_gpu{reinterpret_cast<void *>(1), kPhysicsFlags};

int __cdecl initialize()
{
    return kNvapiOk;
}

int __cdecl unload()
{
    return kNvapiOk;
}

int __cdecl get_display_driver_version(void *, DisplayDriverVersion *version)
{
    if (version == nullptr) return kNvapiInvalidArgument;
    if (version->version != make_version<DisplayDriverVersion>(1))
        return kNvapiIncompatibleStructVersion;

    version->driver_version = 99999;
    version->build_change_list = 0;
    std::memset(version->build_branch, 0, sizeof(version->build_branch));
    std::memset(version->adapter, 0, sizeof(version->adapter));
    constexpr char adapter[] = "pxvk Vulkan device";
    std::memcpy(version->adapter, adapter, sizeof(adapter) - 1);
    return kNvapiOk;
}

int __cdecl get_driver_and_branch_version(std::uint32_t *driver_version,
                                           char branch[64])
{
    if (driver_version == nullptr || branch == nullptr)
        return kNvapiInvalidArgument;

    *driver_version = 99999;
    std::memset(branch, 0, 64);
    constexpr char fake_branch[] = "r999_99";
    std::memcpy(branch, fake_branch, sizeof(fake_branch) - 1);
    return kNvapiOk;
}

int __cdecl get_physical_gpu_from_id(std::uint32_t gpu_id, void **physical_gpu)
{
    if (gpu_id != 1 || physical_gpu == nullptr) return kNvapiInvalidArgument;
    *physical_gpu = g_compute_gpu.physical_gpu;
    return kNvapiOk;
}

int __cdecl get_id_from_physical_gpu(void *physical_gpu, std::uint32_t *gpu_id)
{
    if (physical_gpu != g_compute_gpu.physical_gpu || gpu_id == nullptr)
        return kNvapiInvalidArgument;
    *gpu_id = 1;
    return kNvapiOk;
}

int __cdecl cuda_enum_compute_capable_gpus(std::uint32_t *version)
{
    if (version == nullptr) return kNvapiInvalidArgument;

    if (*version == make_version<ComputeTopologyV1>(1)) {
        auto *topology = reinterpret_cast<ComputeTopologyV1 *>(version);
        topology->gpu_count = 1;
        topology->compute_gpus[0] = g_compute_gpu;
        return kNvapiOk;
    }
    if (*version == make_version<ComputeTopologyV2>(2)) {
        auto *topology = reinterpret_cast<ComputeTopologyV2 *>(version);
        topology->gpu_count = 1;
        topology->compute_gpus = &g_compute_gpu;
        return kNvapiOk;
    }
    return kNvapiIncompatibleStructVersion;
}

int __cdecl private_cuda_enum_compute_capable_gpus(ComputeTopologyV2 *topology)
{
    return cuda_enum_compute_capable_gpus(
        reinterpret_cast<std::uint32_t *>(topology));
}

} // namespace

extern "C" __declspec(dllexport) void *__cdecl nvapi_QueryInterface(
    std::uint32_t interface_id)
{
    switch (interface_id) {
    case 0x0150e828: // NvAPI_Initialize
        return reinterpret_cast<void *>(initialize);
    case 0xd22bdd7e: // NvAPI_Unload
        return reinterpret_cast<void *>(unload);
    case 0xf951a4d1: // NvAPI_GetDisplayDriverVersion
        return reinterpret_cast<void *>(get_display_driver_version);
    case 0x2926aaad: // NvAPI_SYS_GetDriverAndBranchVersion
        return reinterpret_cast<void *>(get_driver_and_branch_version);
    case 0x5380ad1a: // NvAPI_GetPhysicalGPUFromGPUID
        return reinterpret_cast<void *>(get_physical_gpu_from_id);
    case 0x6533ea3e: // NvAPI_GetGPUIDfromPhysicalGPU
        return reinterpret_cast<void *>(get_id_from_physical_gpu);
    case 0x5786cc6e: // NvAPI_GPU_CudaEnumComputeCapableGpus
        return reinterpret_cast<void *>(cuda_enum_compute_capable_gpus);
    case 0x36e39e6b: // private: the V2 compute-GPU enumeration
        return reinterpret_cast<void *>(private_cuda_enum_compute_capable_gpus);
    default: return nullptr;
    }
}
