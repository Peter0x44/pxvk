#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "cuda_abi.h"
#include "runtime_probe_spv.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using Init = CUresult(PXVK_CUDAAPI *)(unsigned int);
using DeviceGet = CUresult(PXVK_CUDAAPI *)(CUdevice *, int);
using ContextCreate = CUresult(PXVK_CUDAAPI *)(CUcontext *, unsigned int, CUdevice);
using ContextDetach = CUresult(PXVK_CUDAAPI *)(CUcontext);
using GetExportTable = CUresult(PXVK_CUDAAPI *)(const void **, const CUuuid *);
using ModuleLoadData = CUresult(PXVK_CUDAAPI *)(CUmodule *, const void *);
using ModuleGetFunction = CUresult(PXVK_CUDAAPI *)(CUfunction *, CUmodule, const char *);
using ModuleGetGlobal = CUresult(PXVK_CUDAAPI *)(CUdeviceptr *, size_t *, CUmodule, const char *);
using ModuleGetTexRef = CUresult(PXVK_CUDAAPI *)(CUtexref *, CUmodule, const char *);
using MemAlloc = CUresult(PXVK_CUDAAPI *)(CUdeviceptr *, size_t);
using MemFree = CUresult(PXVK_CUDAAPI *)(CUdeviceptr);
using MemHostAlloc = CUresult(PXVK_CUDAAPI *)(void **, size_t, unsigned int);
using MemFreeHost = CUresult(PXVK_CUDAAPI *)(void *);
using Memset = CUresult(PXVK_CUDAAPI *)(CUdeviceptr, unsigned char, size_t);
using CopyHtoD = CUresult(PXVK_CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream);
using CopyDtoH = CUresult(PXVK_CUDAAPI *)(void *, CUdeviceptr, size_t, CUstream);
using CopyDtoD = CUresult(PXVK_CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
using AddressRange = CUresult(PXVK_CUDAAPI *)(CUdeviceptr *, size_t *, CUdeviceptr);
using TexAddress = CUresult(PXVK_CUDAAPI *)(size_t *, CUtexref, CUdeviceptr, size_t);
using Launch = CUresult(PXVK_CUDAAPI *)(CUfunction, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
    CUstream, void **, void **);

bool expect(CUresult result, const char *operation)
{
    if (result == CUDA_SUCCESS) return true;
    std::fprintf(stderr, "%s returned CUDA error %d\n", operation, result);
    return false;
}

std::string directory_of(const char *path)
{
    std::string value(path);
    const std::size_t separator = value.find_last_of("\\/");
    return separator == std::string::npos ? "." : value.substr(0, separator);
}

template <typename T>
bool load_symbol(HMODULE module, const char *name, T &output)
{
    output = reinterpret_cast<T>(GetProcAddress(module, name));
    if (output != nullptr) return true;
    std::fprintf(stderr, "missing export %s\n", name);
    return false;
}

bool verify_export_presence(HMODULE module)
{
    const char *names[] = {
        "cuArray3DCreate_v2", "cuCtxCreate_v2", "cuCtxDetach", "cuCtxGetApiVersion",
        "cuCtxGetCurrent", "cuCtxGetDevice", "cuCtxSynchronize", "cuDeviceGet",
        "cuDeviceGetAttribute", "cuDeviceGetCount", "cuDeviceGetName",
        "cuDeviceTotalMem_v2", "cuDriverGetVersion", "cuFuncGetAttribute",
        "cuGetExportTable", "cuInit", "cuLaunchKernel", "cuMemAlloc_v2",
        "cuMemFreeHost", "cuMemFree_v2", "cuMemGetAddressRange_v2",
        "cuMemHostAlloc", "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cuMemcpyHtoDAsync_v2", "cuMemsetD8_v2", "cuModuleGetFunction",
        "cuModuleGetGlobal_v2", "cuModuleGetTexRef", "cuModuleLoadData",
        "cuModuleUnload", "cuTexRefSetAddress_v2", "cuTexRefSetAddressMode",
        "cuTexRefSetFilterMode", "cuTexRefSetFlags", "cuTexRefSetFormat",
        "cuTexRefSetMaxAnisotropy", "cuTexRefSetMipmapFilterMode",
        "cuTexRefSetMipmapLevelBias", "cuTexRefSetMipmapLevelClamp",
        "cuWGLGetDevice"};
    for (const char *name : names) {
        if (GetProcAddress(module, name) == nullptr) {
            std::fprintf(stderr, "missing representative export %s\n", name);
            return false;
        }
    }
    return true;
}

bool write_probe_shader(const std::string &directory)
{
    const std::string spirv = directory + "\\spirv";
    if (!CreateDirectoryA(spirv.c_str(), nullptr) &&
        GetLastError() != ERROR_ALREADY_EXISTS) {
        std::fprintf(stderr, "cannot create %s\n", spirv.c_str());
        return false;
    }
    const std::string path = spirv + "\\pxvk-runtime-probe.spv";
    FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    const bool ok = std::fwrite(kPxvkRuntimeProbeSpv, 1,
                                kPxvkRuntimeProbeSpvBytes, file) ==
                    kPxvkRuntimeProbeSpvBytes;
    std::fclose(file);
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: pxvk-runtime-smoke <nvcuda.dll>\n");
        return 2;
    }
    if (!write_probe_shader(directory_of(argv[1]))) {
        std::fprintf(stderr, "failed to write runtime probe shader\n");
        return 2;
    }
    HMODULE module = LoadLibraryA(argv[1]);
    if (module == nullptr) {
        std::fprintf(stderr, "LoadLibrary failed with %lu\n", GetLastError());
        return 2;
    }
    if (!verify_export_presence(module)) return 1;

    Init init{};
    DeviceGet device_get{};
    ContextCreate context_create{};
    ContextDetach context_detach{};
    GetExportTable get_export_table{};
    ModuleLoadData module_load{};
    ModuleGetFunction module_function{};
    ModuleGetGlobal module_global{};
    ModuleGetTexRef module_texture{};
    MemAlloc mem_alloc{};
    MemFree mem_free{};
    MemHostAlloc host_alloc{};
    MemFreeHost host_free{};
    Memset memset_d8{};
    CopyHtoD copy_h2d{};
    CopyDtoH copy_d2h{};
    CopyDtoD copy_d2d{};
    AddressRange address_range{};
    TexAddress texture_address{};
    Launch launch{};
#define PXVK_LOAD(variable, name) if (!load_symbol(module, name, variable)) return 1
    PXVK_LOAD(init, "cuInit");
    PXVK_LOAD(device_get, "cuDeviceGet");
    PXVK_LOAD(context_create, "cuCtxCreate_v2");
    PXVK_LOAD(context_detach, "cuCtxDetach");
    PXVK_LOAD(get_export_table, "cuGetExportTable");
    PXVK_LOAD(module_load, "cuModuleLoadData");
    PXVK_LOAD(module_function, "cuModuleGetFunction");
    PXVK_LOAD(module_global, "cuModuleGetGlobal_v2");
    PXVK_LOAD(module_texture, "cuModuleGetTexRef");
    PXVK_LOAD(mem_alloc, "cuMemAlloc_v2");
    PXVK_LOAD(mem_free, "cuMemFree_v2");
    PXVK_LOAD(host_alloc, "cuMemHostAlloc");
    PXVK_LOAD(host_free, "cuMemFreeHost");
    PXVK_LOAD(memset_d8, "cuMemsetD8_v2");
    PXVK_LOAD(copy_h2d, "cuMemcpyHtoDAsync_v2");
    PXVK_LOAD(copy_d2h, "cuMemcpyDtoHAsync_v2");
    PXVK_LOAD(copy_d2d, "cuMemcpyDtoDAsync_v2");
    PXVK_LOAD(address_range, "cuMemGetAddressRange_v2");
    PXVK_LOAD(texture_address, "cuTexRefSetAddress_v2");
    PXVK_LOAD(launch, "cuLaunchKernel");
#undef PXVK_LOAD

    if (!expect(init(0), "cuInit")) return 1;
    CUdevice device = -1;
    if (!expect(device_get(&device, 0), "cuDeviceGet")) return 1;
    CUcontext context = nullptr;
    if (!expect(context_create(&context, 0, device), "cuCtxCreate_v2")) return 1;

    const CUuuid guids[] = {
        {{0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a, 0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9}},
        {{0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74, 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66}},
        {{0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc}},
        {{0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93}},
        {{0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a, 0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86}},
        {{0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a, 0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae}},
    };
    for (const CUuuid &guid : guids) {
        const void *table = nullptr;
        if (!expect(get_export_table(&table, &guid), "cuGetExportTable") || table == nullptr)
            return 1;
    }

    CUmodule cuda_module = nullptr;
    const std::uint32_t dummy_image = 0;
    if (!expect(module_load(&cuda_module, &dummy_image), "cuModuleLoadData")) return 1;
    const char *global_names[] = {
        "cCollisionParameters", "updateInstanceData", "parameters_cmem",
        "gDeformableCollisionInstanceData", "gDeformableCollisionBatchedData",
        "solverGpuData",
    };
    const auto allocate_module_globals = [&](CUmodule target) {
        for (const char *name : global_names) {
            CUdeviceptr global = 0;
            if (!expect(module_global(&global, nullptr, target, name),
                        "cuModuleGetGlobal_v2(context lifecycle)"))
                return false;
        }
        return true;
    };
    if (!allocate_module_globals(cuda_module)) return 1;

    CUdeviceptr first = 0;
    CUdeviceptr second = 0;
    if (!expect(mem_alloc(&first, 32), "cuMemAlloc_v2(first)") ||
        !expect(mem_alloc(&second, 32), "cuMemAlloc_v2(second)")) return 1;
    const std::array<std::uint32_t, 4> input = {11, 22, 33, 44};
    std::array<std::uint32_t, 4> output{};
    if (!expect(copy_h2d(first, input.data(), sizeof(input), nullptr), "cuMemcpyHtoDAsync_v2") ||
        !expect(copy_d2d(second, first, sizeof(input), nullptr), "cuMemcpyDtoDAsync_v2") ||
        !expect(copy_d2h(output.data(), second, sizeof(output), nullptr), "cuMemcpyDtoHAsync_v2") ||
        output != input) {
        std::fprintf(stderr, "ordered copy contract failed\n");
        return 1;
    }
    if (!expect(memset_d8(second, 0, sizeof(output)), "cuMemsetD8_v2") ||
        !expect(copy_d2h(output.data(), second, sizeof(output), nullptr), "cuMemcpyDtoHAsync_v2 after memset"))
        return 1;
    for (std::uint32_t word : output) if (word != 0) return 1;
    CUdeviceptr allocation_base = 0;
    size_t allocation_size = 0;
    if (!expect(address_range(&allocation_base, &allocation_size, first + 4),
                "cuMemGetAddressRange_v2") || allocation_base != first || allocation_size != 32)
        return 1;

    constexpr size_t physx_heap_bytes = 128u * 1024u * 1024u;
    void *first_host = nullptr;
    if (!expect(host_alloc(&first_host, physx_heap_bytes, 0),
                "cuMemHostAlloc(PhysX heap first)") ||
        first_host == nullptr ||
        !expect(host_free(first_host), "cuMemFreeHost(PhysX heap first)"))
        return 1;
    void *second_host = nullptr;
    if (!expect(host_alloc(&second_host, physx_heap_bytes, 0),
                "cuMemHostAlloc(PhysX heap reused)") ||
        second_host != first_host ||
        !expect(host_free(second_host), "cuMemFreeHost(PhysX heap reused)")) {
        std::fprintf(stderr, "PhysX host heap was not reused across scene lifecycle\n");
        return 1;
    }

    CUtexref texture = nullptr;
    if (!expect(module_texture(&texture, cuda_module, "inputSortedPositionsTex"),
                "cuModuleGetTexRef") ||
        !expect(texture_address(nullptr, texture, first, 32),
                "cuTexRefSetAddress_v2(offset=null)")) return 1;

    CUfunction function = nullptr;
    if (!expect(module_function(&function, cuda_module, "pxvk_runtime_probe"),
                "cuModuleGetFunction")) return 1;
    const std::uint32_t destination_parameter = second;
    size_t parameter_size = sizeof(destination_parameter);
    void *extra[] = {
        reinterpret_cast<void *>(PXVK_LAUNCH_PARAM_BUFFER_POINTER),
        const_cast<std::uint32_t *>(&destination_parameter),
        reinterpret_cast<void *>(PXVK_LAUNCH_PARAM_BUFFER_SIZE),
        &parameter_size,
        reinterpret_cast<void *>(PXVK_LAUNCH_PARAM_END),
    };
    if (!expect(launch(function, 4, 1, 1, 8, 1, 1, 16, nullptr, nullptr, extra),
                "cuLaunchKernel(runtime probe)")) return 1;
    if (!expect(copy_d2h(output.data(), second, sizeof(output), nullptr),
                "cuMemcpyDtoHAsync_v2(probe result)")) return 1;
    const std::array<std::uint32_t, 4> expected = {7, 8, 9, 10};
    if (output != expected) {
        std::fprintf(stderr, "Vulkan dispatch produced %u,%u,%u,%u\n",
                     output[0], output[1], output[2], output[3]);
        return 1;
    }

    if (!expect(mem_free(second), "cuMemFree_v2(second)") ||
        !expect(mem_free(first), "cuMemFree_v2(first)")) return 1;
    CUdeviceptr reused = 0;
    if (!expect(mem_alloc(&reused, 32), "cuMemAlloc_v2(reused)") ||
        reused != first ||
        !expect(mem_free(reused), "cuMemFree_v2(reused)")) {
        std::fprintf(stderr, "device heap was not reused across scene lifecycle\n");
        return 1;
    }
    if (!expect(context_detach(context), "cuCtxDetach")) return 1;
    CUcontext recreated_context = nullptr;
    if (!expect(context_create(&recreated_context, 0, device),
                "cuCtxCreate_v2(recreated)"))
        return 1;
    CUmodule recreated_module = nullptr;
    if (!expect(module_load(&recreated_module, &dummy_image),
                "cuModuleLoadData(recreated)") ||
        !allocate_module_globals(recreated_module) ||
        !expect(context_detach(recreated_context), "cuCtxDetach(recreated)"))
        return 1;
    std::puts("pxvk runtime smoke passed: exports, Dark API, memory, Vulkan dispatch");
    FreeLibrary(module);
    return 0;
}
