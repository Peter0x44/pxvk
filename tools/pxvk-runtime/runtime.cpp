#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "cuda_abi.h"
#include "vulkan_backend.h"
#include "../nvcuda-trace/src/launch_params.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(sizeof(void *) == 4, "pxvk runtime requires a 32-bit build");
static_assert(sizeof(size_t) == 4, "CUDA x86 size_t must be 32-bit");

namespace {

constexpr std::uint32_t kModuleHandleBase = 0x20000000u;
constexpr std::uint32_t kFunctionHandleBase = 0x30000000u;
constexpr std::uint32_t kTextureHandleBase = 0x40000000u;
constexpr std::uint32_t kContextHandleBase = 0x10000000u;
constexpr std::uint32_t kHandleStride = 4u;
constexpr std::uint32_t kGlobalAddressStart = 0x100u;
constexpr std::uint32_t kDeviceAddressStart = 0x10000u;
constexpr std::size_t kMaxPackedParameters = 64u * 1024u;

HMODULE g_module = nullptr;
thread_local CUcontext g_current_context = nullptr;

using LoadLibraryWFn = HMODULE(WINAPI *)(LPCWSTR);
using LoadLibraryExWFn = HMODULE(WINAPI *)(LPCWSTR, HANDLE, DWORD);
using LdrDllNotificationFn = VOID(CALLBACK *)(ULONG, const void *, void *);
using LdrRegisterDllNotificationFn =
    LONG(NTAPI *)(ULONG, LdrDllNotificationFn, void *, void **);
using LdrUnregisterDllNotificationFn = LONG(NTAPI *)(void *);

struct PxvkUnicodeString {
    USHORT length;
    USHORT maximum_length;
    PWSTR buffer;
};

struct PxvkLoadedDllNotification {
    ULONG flags;
    const PxvkUnicodeString *full_name;
    const PxvkUnicodeString *base_name;
    void *base;
    ULONG image_size;
};

struct PxvkDllNotification {
    PxvkLoadedDllNotification loaded;
};

wchar_t g_module_path[32768]{};
LoadLibraryWFn g_original_load_library_w = nullptr;
LoadLibraryExWFn g_original_load_library_ex_w = nullptr;
LdrUnregisterDllNotificationFn g_ldr_unregister = nullptr;
void *g_dll_notification_cookie = nullptr;

bool driver_library_name(LPCWSTR path)
{
    if (path == nullptr) return false;
    const wchar_t *name = path;
    for (const wchar_t *cursor = path; *cursor != L'\0'; ++cursor) {
        if (*cursor == L'\\' || *cursor == L'/') name = cursor + 1;
    }
    return _wcsicmp(name, L"nvcuda.dll") == 0;
}

HMODULE WINAPI redirect_load_library_w(LPCWSTR path)
{
    return driver_library_name(path)
               ? g_original_load_library_w(g_module_path)
               : g_original_load_library_w(path);
}

HMODULE WINAPI redirect_load_library_ex_w(LPCWSTR path, HANDLE file, DWORD flags)
{
    return driver_library_name(path)
               ? g_original_load_library_ex_w(g_module_path, nullptr, 0)
               : g_original_load_library_ex_w(path, file, flags);
}

void patch_import(void **slot, void *replacement)
{
    DWORD old_protection = 0;
    if (VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old_protection) == FALSE)
        return;
    *slot = replacement;
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(*slot), old_protection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(*slot));
}

void patch_cudart_driver_load(void *module_base)
{
    auto *base = static_cast<unsigned char *>(module_base);
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS32 *>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    const IMAGE_DATA_DIRECTORY &directory =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (directory.VirtualAddress == 0u) return;
    auto *descriptor =
        reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + directory.VirtualAddress);
    for (; descriptor->Name != 0u; ++descriptor) {
        if (descriptor->OriginalFirstThunk == 0u || descriptor->FirstThunk == 0u)
            continue;
        auto *names = reinterpret_cast<IMAGE_THUNK_DATA32 *>(
            base + descriptor->OriginalFirstThunk);
        auto *slots = reinterpret_cast<IMAGE_THUNK_DATA32 *>(
            base + descriptor->FirstThunk);
        for (; names->u1.AddressOfData != 0u; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL32(names->u1.Ordinal)) continue;
            const auto *import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME *>(
                base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char *>(import->Name),
                            "LoadLibraryW") == 0) {
                patch_import(reinterpret_cast<void **>(&slots->u1.Function),
                             reinterpret_cast<void *>(&redirect_load_library_w));
            } else if (std::strcmp(reinterpret_cast<const char *>(import->Name),
                                   "LoadLibraryExW") == 0) {
                patch_import(reinterpret_cast<void **>(&slots->u1.Function),
                             reinterpret_cast<void *>(&redirect_load_library_ex_w));
            }
        }
    }
}

VOID CALLBACK dll_notification(ULONG reason, const void *raw_data, void *)
{
    constexpr ULONG loaded_reason = 1u;
    if (reason != loaded_reason || raw_data == nullptr) return;
    const auto *data = static_cast<const PxvkDllNotification *>(raw_data);
    if (data->loaded.base_name == nullptr ||
        data->loaded.base_name->buffer == nullptr) return;
    wchar_t name[64]{};
    const std::size_t characters = std::min<std::size_t>(
        data->loaded.base_name->length / sizeof(wchar_t), std::size(name) - 1u);
    std::wmemcpy(name, data->loaded.base_name->buffer, characters);
    if (_wcsicmp(name, L"cudart32_65.dll") == 0)
        patch_cudart_driver_load(data->loaded.base);
}

void install_driver_redirect()
{
    GetModuleFileNameW(g_module, g_module_path,
                       static_cast<DWORD>(std::size(g_module_path)));
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    g_original_load_library_w = reinterpret_cast<LoadLibraryWFn>(
        GetProcAddress(kernel, "LoadLibraryW"));
    g_original_load_library_ex_w = reinterpret_cast<LoadLibraryExWFn>(
        GetProcAddress(kernel, "LoadLibraryExW"));
    auto register_notification = reinterpret_cast<LdrRegisterDllNotificationFn>(
        GetProcAddress(ntdll, "LdrRegisterDllNotification"));
    g_ldr_unregister = reinterpret_cast<LdrUnregisterDllNotificationFn>(
        GetProcAddress(ntdll, "LdrUnregisterDllNotification"));
    if (g_original_load_library_w != nullptr &&
        g_original_load_library_ex_w != nullptr &&
        register_notification != nullptr) {
        register_notification(0, dll_notification, nullptr,
                              &g_dll_notification_cookie);
    }
}

bool checked_u32_range(std::uint32_t base, std::size_t bytes, std::uint32_t &end)
{
    if (bytes > std::numeric_limits<std::uint32_t>::max() - base) return false;
    end = base + static_cast<std::uint32_t>(bytes);
    return true;
}

std::uint32_t align_up(std::uint32_t value, std::uint32_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

template <typename T>
T handle_from_id(std::uint32_t base, std::uint32_t id)
{
    return reinterpret_cast<T>(static_cast<std::uintptr_t>(base + id * kHandleStride));
}

template <typename T>
bool id_from_handle(T handle, std::uint32_t base, std::size_t count, std::uint32_t &id)
{
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(handle);
    if (raw <= base || raw < base + kHandleStride ||
        ((raw - base) % kHandleStride) != 0u) return false;
    const std::uint32_t candidate = static_cast<std::uint32_t>((raw - base) / kHandleStride);
    if (candidate == 0u || candidate > count) return false;
    id = candidate;
    return true;
}

enum class Profile {
    none,
    deformable_update,
    deformable_collision,
    deformable_solver,
    fluid_density,
    fluid_force,
    runtime_probe
};

struct KernelSpec {
    const char *name;
    const char *shader;
    std::uint32_t packed_bytes;
    std::uint32_t block_x;
    std::uint32_t shared_bytes;
    bool storage_shared;
    Profile profile;
};

constexpr KernelSpec kKernelSpecs[] = {
    {"_Z39PxgCudaFluidCollisionConvexShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-convexes.spv", 20, 64, 0, false, Profile::none},
    {"_Z36PxgCudaFluidCollisionTrianglesBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK29PxgFluidCollisionWorkUnitMeshEPi", "fluid-triangles.spv", 20, 64, 0, false, Profile::none},
    {"_Z40PxgCudaFluidCollisionCapsuleShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-capsules.spv", 20, 64, 0, false, Profile::none},
    {"_Z15radixSortBlocksILj32ELj0ELb0EEvP5uint4S1_S1_S1_j", "radix-sort.spv", 20, 256, 4096, false, Profile::none},
    {"_Z38PxgCudaFluidCollisionConstraintsBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-constraints.spv", 20, 128, 0, false, Profile::none},
    {"_Z42PxgCudaFluidCollisionDynamicResponseBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-dynamic-response.spv", 20, 128, 0, false, Profile::none},
    {"_Z24PxgCudaDeformableUpdateDPKhjPi", "deformable-update.spv", 12, 64, 0, false, Profile::deformable_update},
    {"_Z41PxgCudaFluidCollisionStaticResponseBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-static-response.spv", 20, 128, 0, false, Profile::none},
    {"_Z19KernelFluidDensityD17PxgFixedSizeArrayIfES0_15DensityDebugGPUPiS_IK6float4ES_IKjES7_S7_S7_", "fluid-density.spv", 84, 192, 0, false, Profile::fluid_density},
    {"_Z29KernelDeformableUberCollisionPi", "deformable-collision.spv", 4, 64, 0, false, Profile::deformable_collision},
    {"_Z26KernelDeformableUberSolverILj512EEvjbPhPi", "deformable-solver.spv", 16, 64, 7072, false, Profile::deformable_solver},
    {"_Z26KernelDeformableUberSolverILj512EEvjbPhPi", "sample-cloth-solver.spv", 16, 512, 42240, true, Profile::deformable_solver},
    {"_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj", "fluid-calc-hash.spv", 52, 64, 0, false, Profile::none},
    {"_Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej", "fluid-reorder.spv", 76, 64, 260, false, Profile::none},
    {"_Z26KernelFluidSimpleParticleD17PxgFixedSizeArrayI12PxcVectorPadES_IKS0_E11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj", "fluid-simple-particle.spv", 44, 256, 0, false, Profile::none},
    {"_Z36PxgCudaFluidCollisionBoxShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi", "fluid-boxes.spv", 20, 64, 0, false, Profile::none},
    {"_Z17KernelFluidForceD17PxgFixedSizeArrayI12PxcVectorPadE13ForceDebugGPUPiS_IK6float4ES6_S_IKfES_IKjESA_SA_SA_", "fluid-force.spv", 100, 128, 0, false, Profile::fluid_force},
    {"pxvk_runtime_probe", "pxvk-runtime-probe.spv", 4, 8, 16, true, Profile::runtime_probe},
};

const KernelSpec *lookup_kernel_name(const char *name)
{
    if (name == nullptr) return nullptr;
    for (const KernelSpec &spec : kKernelSpecs) {
        if (std::strcmp(spec.name, name) == 0) return &spec;
    }
    return nullptr;
}

const KernelSpec *lookup_kernel_launch(const char *name, std::uint32_t block_x,
                                       std::uint32_t shared_bytes)
{
    if (name == nullptr) return nullptr;
    for (const KernelSpec &spec : kKernelSpecs) {
        if (spec.block_x == block_x && spec.shared_bytes == shared_bytes &&
            std::strcmp(spec.name, name) == 0) return &spec;
    }
    return nullptr;
}

struct ConstantSpec {
    const char *name;
    std::uint32_t offset;
    std::uint32_t size;
};

constexpr ConstantSpec kUpdateConstants[] = {{"updateInstanceData", 0, 7168}};
constexpr ConstantSpec kCollisionConstants[] = {
    {"gDeformableCollisionInstanceData", 0, 17920},
    {"gDeformableCollisionBatchedData", 17920, 72},
};
constexpr ConstantSpec kSolverConstants[] = {{"solverGpuData", 0, 5888}};
constexpr ConstantSpec kFluidConstants[] = {{"parameters_cmem", 0, 10240}};

constexpr const char *kCollisionTextures[] = {
    "texShapeAabbMin", "texShapeAabbMax", "texShapePoseQ", "texShapeNewPoseQ",
    "texShapePoseP", "texShapeNewPoseP", "texShapeDims", "texShapeFlags",
    "texShapeBodyID", "texShapeOffsetPrimitives", "texShapeNumPrimitives",
    "texTriangleOrigin", "texTriangleEdge0", "texTriangleEdge1",
    "texTriangleAabbMin", "texTriangleAabbMax", "texConvexPlane",
};
constexpr const char *kDensityTextures[] = {
    "inputSortedPositionsTex", "inputParticleHashKeyTex", "inputCellStartTex",
    "inputCellEndTex",
};
constexpr const char *kForceTextures[] = {
    "inputSortedPositionsTex", "inputSortedVelocitiesTex", "inputSortedDensitiesTex",
    "inputParticleHashKeyTex", "inputCellStartTex", "inputCellEndTex",
};

struct ResourceProfile {
    const ConstantSpec *constants;
    std::size_t constant_count;
    std::uint32_t constant_bytes;
    const char *const *textures;
    std::size_t texture_count;
};

ResourceProfile resources_for(Profile profile)
{
    switch (profile) {
    case Profile::deformable_update:
        return {kUpdateConstants, std::size(kUpdateConstants), 7168, nullptr, 0};
    case Profile::deformable_collision:
        return {kCollisionConstants, std::size(kCollisionConstants), 17992,
                kCollisionTextures, std::size(kCollisionTextures)};
    case Profile::deformable_solver:
        return {kSolverConstants, std::size(kSolverConstants), 5888, nullptr, 0};
    case Profile::fluid_density:
        return {kFluidConstants, std::size(kFluidConstants), 10240,
                kDensityTextures, std::size(kDensityTextures)};
    case Profile::fluid_force:
        return {kFluidConstants, std::size(kFluidConstants), 10240,
                kForceTextures, std::size(kForceTextures)};
    default:
        return {nullptr, 0, 0, nullptr, 0};
    }
}

std::uint32_t global_size(const char *name)
{
    if (name == nullptr) return 0;
    struct Pair { const char *name; std::uint32_t size; };
    constexpr Pair globals[] = {
        {"cCollisionParameters", 160},
        {"updateInstanceData", 7168},
        {"parameters_cmem", 10240},
        {"gDeformableCollisionInstanceData", 17920},
        {"gDeformableCollisionBatchedData", 72},
        {"solverGpuData", 5888},
    };
    for (const Pair &item : globals)
        if (std::strcmp(item.name, name) == 0) return item.size;
    return 0;
}

struct ModuleRecord {
    bool live = true;
    Profile profile = Profile::none;
};

struct FunctionRecord {
    bool live = true;
    std::uint32_t module_id = 0;
    std::string name;
    bool launch_logged = false;
};

struct TextureRecord {
    bool live = true;
    std::uint32_t module_id = 0;
    std::string name;
    std::uint32_t address = 0;
    std::uint32_t bytes = 0;
    int format = 0;
    int components = 1;
    unsigned int flags = 0;
};

struct AllocationRecord {
    bool live = true;
    bool global = false;
    std::uint32_t base = 0;
    std::uint32_t size = 0;
    std::uint32_t module_id = 0;
    std::string name;
};

struct HostAllocation {
    void *pointer = nullptr;
    std::size_t size = 0;
};

using ContextDestructor = void(PXVK_CUDAAPI *)(CUcontext, void *, void *);
struct ClsValue {
    void *value = nullptr;
    ContextDestructor destructor = nullptr;
};
struct ContextRecord {
    bool live = true;
    CUdevice device = 0;
    std::unordered_map<std::uintptr_t, ClsValue> cls;
};

struct State {
    std::mutex mutex;
    pxvk::VulkanBackend backend;
    bool initialized = false;
    bool deinitialized = false;
    std::uint32_t next_global_address = kGlobalAddressStart;
    std::uint32_t next_device_address = kDeviceAddressStart;
    std::vector<ModuleRecord> modules;
    std::vector<FunctionRecord> functions;
    std::vector<TextureRecord> textures;
    std::vector<AllocationRecord> allocations;
    std::vector<HostAllocation> host_allocations;
    std::vector<ContextRecord> contexts;
    FILE *log = nullptr;

    void open_log()
    {
        if (log != nullptr) return;
        char path[1024]{};
        const DWORD length = GetEnvironmentVariableA("PXVK_LOG", path,
                                                      static_cast<DWORD>(sizeof(path)));
        if (length == 0 || length >= sizeof(path)) return;
        log = std::fopen(path, "a");
    }

    void write_log(const char *format, ...)
    {
        open_log();
        if (log == nullptr) return;
        va_list args;
        va_start(args, format);
        std::vfprintf(log, format, args);
        va_end(args);
        std::fputc('\n', log);
        std::fflush(log);
    }
};

State &state()
{
    static State *instance = new State;
    return *instance;
}

CUresult initialize_locked(State &value)
{
    if (value.deinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (value.initialized) return CUDA_SUCCESS;
    std::string error;
    if (!value.backend.initialize(error)) {
        value.write_log("runtime_init_failed error=%s", error.c_str());
        return CUDA_ERROR_NO_DEVICE;
    }
    value.initialized = true;
    value.write_log("runtime_init device=%s", value.backend.device_name().c_str());
    return CUDA_SUCCESS;
}

CUresult require_initialized_locked(State &value)
{
    if (value.deinitialized) return CUDA_ERROR_DEINITIALIZED;
    return value.initialized ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

ModuleRecord *module_record(State &value, CUmodule handle, std::uint32_t *id_out = nullptr)
{
    std::uint32_t id = 0;
    if (!id_from_handle(handle, kModuleHandleBase, value.modules.size(), id)) return nullptr;
    ModuleRecord &record = value.modules[id - 1u];
    if (!record.live) return nullptr;
    if (id_out) *id_out = id;
    return &record;
}

FunctionRecord *function_record(State &value, CUfunction handle)
{
    std::uint32_t id = 0;
    if (!id_from_handle(handle, kFunctionHandleBase, value.functions.size(), id)) return nullptr;
    FunctionRecord &record = value.functions[id - 1u];
    return record.live ? &record : nullptr;
}

TextureRecord *texture_record(State &value, CUtexref handle)
{
    std::uint32_t id = 0;
    if (!id_from_handle(handle, kTextureHandleBase, value.textures.size(), id)) return nullptr;
    TextureRecord &record = value.textures[id - 1u];
    return record.live ? &record : nullptr;
}

ContextRecord *context_record(State &value, CUcontext handle, std::uint32_t *id_out = nullptr)
{
    std::uint32_t id = 0;
    if (!id_from_handle(handle, kContextHandleBase, value.contexts.size(), id)) return nullptr;
    ContextRecord &record = value.contexts[id - 1u];
    if (!record.live) return nullptr;
    if (id_out) *id_out = id;
    return &record;
}

AllocationRecord *owning_allocation(State &value, std::uint32_t address,
                                    std::size_t bytes)
{
    std::uint32_t end = 0;
    if (!checked_u32_range(address, bytes, end)) return nullptr;
    for (AllocationRecord &record : value.allocations) {
        if (!record.live) continue;
        const std::uint32_t record_end = record.base + record.size;
        if (address >= record.base && end <= record_end) return &record;
    }
    return nullptr;
}

AllocationRecord *global_allocation(State &value, std::uint32_t module_id,
                                    const char *name)
{
    for (AllocationRecord &record : value.allocations) {
        if (record.live && record.global && record.module_id == module_id &&
            record.name == name) return &record;
    }
    return nullptr;
}

std::string shader_path(const char *shader)
{
    char override_path[1024]{};
    const DWORD length = GetEnvironmentVariableA("PXVK_SHADER_DIR", override_path,
                                                  static_cast<DWORD>(sizeof(override_path)));
    std::string directory;
    if (length != 0 && length < sizeof(override_path)) {
        directory.assign(override_path, length);
    } else {
        char module_path[1024]{};
        const DWORD module_length = GetModuleFileNameA(g_module, module_path,
                                                        static_cast<DWORD>(sizeof(module_path)));
        directory.assign(module_path, module_length);
        const std::size_t separator = directory.find_last_of("\\/");
        if (separator != std::string::npos) directory.resize(separator);
        directory += "\\spirv";
    }
    if (!directory.empty() && directory.back() != '\\' && directory.back() != '/')
        directory.push_back('\\');
    return directory + shader;
}

bool read_current_process(void *, const void *address, void *output, size_t size)
{
    SIZE_T read = 0;
    return ReadProcessMemory(GetCurrentProcess(), address, output, size, &read) != FALSE &&
           read == size;
}

CUresult create_module_locked(State &value, CUmodule *output)
{
    if (output == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        value.modules.push_back({});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const std::uint32_t id = static_cast<std::uint32_t>(value.modules.size());
    *output = handle_from_id<CUmodule>(kModuleHandleBase, id);
    value.write_log("module_register id=%u", id);
    return CUDA_SUCCESS;
}

CUresult create_context_locked(State &value, CUcontext *output, unsigned int flags,
                               CUdevice device)
{
    if (output == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    try {
        ContextRecord record;
        record.device = device;
        value.contexts.push_back(std::move(record));
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const std::uint32_t id = static_cast<std::uint32_t>(value.contexts.size());
    *output = handle_from_id<CUcontext>(kContextHandleBase, id);
    g_current_context = *output;
    value.write_log("context_create id=%u flags=%u", id, flags);
    return CUDA_SUCCESS;
}

CUresult backend_failure(State &value, const char *operation, const std::string &error)
{
    value.write_log("%s_failed error=%s", operation, error.c_str());
    return error.find("memory") != std::string::npos
               ? CUDA_ERROR_OUT_OF_MEMORY
               : CUDA_ERROR_UNKNOWN;
}

CUresult collect_constants(State &value, std::uint32_t module_id, Profile profile,
                           std::vector<std::uint8_t> &output)
{
    const ResourceProfile resources = resources_for(profile);
    output.assign(resources.constant_bytes, 0u);
    std::string error;
    for (std::size_t i = 0; i < resources.constant_count; ++i) {
        const ConstantSpec &spec = resources.constants[i];
        AllocationRecord *allocation = global_allocation(value, module_id, spec.name);
        if (allocation == nullptr) continue;
        if (!value.backend.read_heap(output.data() + spec.offset, allocation->base,
                                     spec.size, error))
            return backend_failure(value, "constant_read", error);
    }
    return CUDA_SUCCESS;
}

CUresult collect_textures(State &value, std::uint32_t module_id, Profile profile,
                          std::vector<std::uint32_t> &output)
{
    const ResourceProfile resources = resources_for(profile);
    output.assign(resources.texture_count * 3u, 0u);
    for (std::size_t i = 0; i < resources.texture_count; ++i) {
        for (const TextureRecord &texture : value.textures) {
            if (!texture.live || texture.module_id != module_id ||
                texture.name != resources.textures[i]) continue;
            const std::uint32_t components =
                static_cast<std::uint32_t>(std::max(texture.components, 1));
            output[i * 3u + 0u] = texture.address / 4u;
            output[i * 3u + 1u] =
                components == 0u ? 0u : texture.bytes / (components * 4u);
            output[i * 3u + 2u] = components;
            break;
        }
    }
    return CUDA_SUCCESS;
}

bool guid_equal(const CUuuid *id, const unsigned char (&bytes)[16])
{
    return id != nullptr && std::memcmp(id->bytes, bytes, 16) == 0;
}

constexpr unsigned char kCudartGuid[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
constexpr unsigned char kToolsRuntimeGuid[16] = {
    0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};
constexpr unsigned char kToolsTlsGuid[16] = {
    0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
    0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc};
constexpr unsigned char kClsGuid[16] = {
    0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
    0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};
constexpr unsigned char kContextBypassGuid[16] = {
    0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a,
    0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86};
constexpr unsigned char kHeapGuid[16] = {
    0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a,
    0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae};

CUresult PXVK_CUDAAPI dark_get_module(CUmodule *, const PxvkFatbinWrapper *);
CUresult PXVK_CUDAAPI dark_cudart_context(CUcontext *, CUdevice);
void PXVK_CUDAAPI dark_unknown_buffer1(void **, size_t *);
void PXVK_CUDAAPI dark_unknown_buffer2(void **, size_t *);
CUresult PXVK_CUDAAPI dark_cls_put(CUcontext, void *, void *, ContextDestructor);
CUresult PXVK_CUDAAPI dark_cls_delete(CUcontext, void *);
CUresult PXVK_CUDAAPI dark_cls_get(void **, CUcontext, void *);
CUresult PXVK_CUDAAPI dark_context_create(CUcontext *, unsigned int, CUdevice);
CUresult PXVK_CUDAAPI dark_heap_alloc(const void **, size_t, size_t);
CUresult PXVK_CUDAAPI dark_heap_free(const void *, size_t *);

const void *const *cudart_table()
{
    static std::array<void *, 13> table = [] {
        std::array<void *, 13> result{};
        result[0] = reinterpret_cast<void *>(13u * sizeof(void *));
        result[1] = reinterpret_cast<void *>(&dark_get_module);
        result[2] = reinterpret_cast<void *>(&dark_cudart_context);
        return result;
    }();
    return table.data();
}

const void *const *tools_runtime_table()
{
    static std::array<void *, 7> table = [] {
        std::array<void *, 7> result{};
        result[0] = reinterpret_cast<void *>(7u * sizeof(void *));
        result[2] = reinterpret_cast<void *>(&dark_unknown_buffer1);
        result[6] = reinterpret_cast<void *>(&dark_unknown_buffer2);
        return result;
    }();
    return table.data();
}

const void *const *tools_tls_table()
{
    static std::array<void *, 4> table = [] {
        std::array<void *, 4> result{};
        result[0] = reinterpret_cast<void *>(4u * sizeof(void *));
        return result;
    }();
    return table.data();
}

const void *const *cls_table()
{
    static std::array<void *, 4> table = {
        reinterpret_cast<void *>(&dark_cls_put),
        reinterpret_cast<void *>(&dark_cls_delete),
        reinterpret_cast<void *>(&dark_cls_get), nullptr};
    return table.data();
}

const void *const *context_bypass_table()
{
    static std::array<void *, 2> table = {
        reinterpret_cast<void *>(2u * sizeof(void *)),
        reinterpret_cast<void *>(&dark_context_create)};
    return table.data();
}

const void *const *heap_table()
{
    static std::array<void *, 3> table = {
        reinterpret_cast<void *>(3u * sizeof(void *)),
        reinterpret_cast<void *>(&dark_heap_alloc),
        reinterpret_cast<void *>(&dark_heap_free)};
    return table.data();
}

} // namespace

PXVK_EXPORT CUresult PXVK_CUDAAPI cuInit(unsigned int flags)
{
    if (flags != 0u) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    return initialize_locked(value);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDriverGetVersion(int *version)
{
    if (version == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result == CUDA_SUCCESS) *version = 12000;
    return result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetCount(int *count)
{
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result == CUDA_SUCCESS) *count = 1;
    return result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
    if (device == nullptr || ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result == CUDA_SUCCESS) *device = 0;
    return result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetName(char *name, int length, CUdevice device)
{
    if (name == nullptr || length <= 0 || device != 0) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result != CUDA_SUCCESS) return result;
    const std::string reported = "pxvk Vulkan (" + value.backend.device_name() + ")";
    std::strncpy(name, reported.c_str(), static_cast<std::size_t>(length));
    name[length - 1] = '\0';
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice device)
{
    if (bytes == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result == CUDA_SUCCESS) {
        *bytes = static_cast<size_t>(std::min<std::uint64_t>(
            value.backend.total_device_memory(), std::numeric_limits<std::uint32_t>::max()));
    }
    return result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice device)
{
    return cuDeviceTotalMem_v2(bytes, device);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                                             CUdevice device)
{
    if (major == nullptr || minor == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    *major = 5;
    *minor = 2;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetAttribute(int *output,
                                                        CUdevice_attribute attribute,
                                                        CUdevice device)
{
    if (output == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result != CUDA_SUCCESS) return result;
    switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *output = static_cast<int>(value.backend.max_workgroup_invocations()); break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
        *output = static_cast<int>(value.backend.max_workgroup_size_x()); break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *output = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *output = 64; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        *output = static_cast<int>(std::min<std::uint32_t>(
            value.backend.max_grid_size_x(), std::numeric_limits<int>::max())); break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *output = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        *output = static_cast<int>(std::max<std::uint32_t>(
            value.backend.max_shared_memory(), 49152u)); break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: *output = 65536; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *output = 32; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: *output = std::numeric_limits<int>::max(); break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: *output = 65536; break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: *output = 512; break;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: *output = 1; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *output = static_cast<int>(value.backend.compute_unit_count()); break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *output = 5; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *output = 2; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: *output = 0; break;
    default:
        *output = attribute == static_cast<int>(0x20080402u) ? 256 : 0;
        break;
    }
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetProperties(CUdevprop *properties,
                                                         CUdevice device)
{
    if (properties == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    std::memset(properties, 0, sizeof(*properties));
    properties->maxThreadsPerBlock = 1024;
    properties->maxThreadsDim[0] = 1024;
    properties->maxThreadsDim[1] = 1024;
    properties->maxThreadsDim[2] = 64;
    properties->maxGridSize[0] = std::numeric_limits<int>::max();
    properties->maxGridSize[1] = 65535;
    properties->maxGridSize[2] = 65535;
    properties->sharedMemPerBlock = 49152;
    properties->totalConstantMemory = 65536;
    properties->SIMDWidth = 32;
    properties->memPitch = std::numeric_limits<int>::max();
    properties->regsPerBlock = 65536;
    properties->textureAlign = 512;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxCreate_v2(CUcontext *context,
                                                  unsigned int flags,
                                                  CUdevice device)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    return result == CUDA_SUCCESS ? create_context_locked(value, context, flags, device)
                                  : result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxCreate(CUcontext *context,
                                               unsigned int flags,
                                               CUdevice device)
{
    return cuCtxCreate_v2(context, flags, device);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetCurrent(CUcontext context)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (context != nullptr && context_record(value, context) == nullptr)
        return CUDA_ERROR_INVALID_CONTEXT;
    g_current_context = context;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetCurrent(CUcontext *context)
{
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *context = g_current_context;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPushCurrent_v2(CUcontext context)
{
    return cuCtxSetCurrent(context);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPushCurrent(CUcontext context)
{
    return cuCtxSetCurrent(context);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPopCurrent_v2(CUcontext *context)
{
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *context = g_current_context;
    g_current_context = nullptr;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPopCurrent(CUcontext *context)
{
    return cuCtxPopCurrent_v2(context);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetApiVersion(CUcontext context,
                                                      unsigned int *version)
{
    if (version == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (context == nullptr) context = g_current_context;
    if (context_record(value, context) == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
    *version = 3020;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetDevice(CUdevice *device)
{
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    ContextRecord *context = context_record(value, g_current_context);
    if (context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
    *device = context->device;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSynchronize(void)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result != CUDA_SUCCESS) return result;
    std::string error;
    return value.backend.synchronize(error) ? CUDA_SUCCESS
                                            : backend_failure(value, "synchronize", error);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDetach(CUcontext context)
{
    State &value = state();
    std::vector<std::pair<std::uintptr_t, ClsValue>> callbacks;
    {
        std::lock_guard<std::mutex> lock(value.mutex);
        if (context == nullptr) context = g_current_context;
        std::uint32_t id = 0;
        ContextRecord *record = context_record(value, context, &id);
        if (record == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
        for (const auto &item : record->cls) callbacks.push_back(item);
        record->cls.clear();
        record->live = false;
        if (g_current_context == context) g_current_context = nullptr;
        value.write_log("context_detach id=%u", id);
    }
    for (const auto &item : callbacks) {
        if (item.second.destructor != nullptr)
            item.second.destructor(context, reinterpret_cast<void *>(item.first),
                                   item.second.value);
    }
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDestroy_v2(CUcontext context)
{
    return cuCtxDetach(context);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDestroy(CUcontext context)
{
    return cuCtxDetach(context);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetCacheConfig(CUfunc_cache) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig) { return CUDA_SUCCESS; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetExportTable(const void **output,
                                                    const CUuuid *id)
{
    if (output == nullptr || id == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    if (result != CUDA_SUCCESS) return result;
    if (guid_equal(id, kCudartGuid)) *output = cudart_table();
    else if (guid_equal(id, kToolsRuntimeGuid)) *output = tools_runtime_table();
    else if (guid_equal(id, kToolsTlsGuid)) *output = tools_tls_table();
    else if (guid_equal(id, kClsGuid)) *output = cls_table();
    else if (guid_equal(id, kContextBypassGuid)) *output = context_bypass_table();
    else if (guid_equal(id, kHeapGuid)) *output = heap_table();
    else return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image)
{
    if (image == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    return result == CUDA_SUCCESS ? create_module_locked(value, module) : result;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleLoadFatBinary(CUmodule *module,
                                                         const void *fatbin)
{
    return cuModuleLoadData(module, fatbin);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleUnload(CUmodule module)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    std::uint32_t id = 0;
    ModuleRecord *record = module_record(value, module, &id);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    record->live = false;
    for (FunctionRecord &function : value.functions)
        if (function.module_id == id) function.live = false;
    for (TextureRecord &texture : value.textures)
        if (texture.module_id == id) texture.live = false;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetFunction(CUfunction *output,
                                                       CUmodule module,
                                                       const char *name)
{
    if (output == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    std::uint32_t module_id = 0;
    ModuleRecord *module_value = module_record(value, module, &module_id);
    if (module_value == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    const KernelSpec *spec = lookup_kernel_name(name);
    if (spec != nullptr && spec->profile != Profile::none &&
        spec->profile != Profile::runtime_probe) {
        if (module_value->profile != Profile::none && module_value->profile != spec->profile)
            return CUDA_ERROR_INVALID_IMAGE;
        module_value->profile = spec->profile;
    }
    try {
        value.functions.push_back({true, module_id, name});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const std::uint32_t id = static_cast<std::uint32_t>(value.functions.size());
    *output = handle_from_id<CUfunction>(kFunctionHandleBase, id);
    value.write_log("function_lookup module=%u name=%s translated=%u", module_id, name,
                    spec != nullptr ? 1u : 0u);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *pointer,
                                                        size_t *bytes,
                                                        CUmodule module,
                                                        const char *name)
{
    if (pointer == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    std::uint32_t module_id = 0;
    if (module_record(value, module, &module_id) == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    AllocationRecord *existing = global_allocation(value, module_id, name);
    if (existing != nullptr) {
        *pointer = existing->base;
        if (bytes != nullptr) *bytes = existing->size;
        return CUDA_SUCCESS;
    }
    const std::uint32_t size = global_size(name);
    if (size == 0u) return CUDA_ERROR_NOT_FOUND;
    const std::uint32_t base = align_up(value.next_global_address, 16u);
    std::uint32_t end = 0;
    if (!checked_u32_range(base, size, end) || end > kDeviceAddressStart)
        return CUDA_ERROR_OUT_OF_MEMORY;
    std::string error;
    if (!value.backend.ensure_heap(end, error)) return backend_failure(value, "global_alloc", error);
    try {
        value.allocations.push_back({true, true, base, size, module_id, name});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    value.next_global_address = end;
    *pointer = base;
    if (bytes != nullptr) *bytes = size;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetGlobal(CUdeviceptr *pointer,
                                                     size_t *bytes,
                                                     CUmodule module,
                                                     const char *name)
{
    return cuModuleGetGlobal_v2(pointer, bytes, module, name);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetTexRef(CUtexref *output,
                                                     CUmodule module,
                                                     const char *name)
{
    if (output == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    std::uint32_t module_id = 0;
    if (module_record(value, module, &module_id) == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    for (std::size_t i = 0; i < value.textures.size(); ++i) {
        TextureRecord &texture = value.textures[i];
        if (texture.live && texture.module_id == module_id && texture.name == name) {
            *output = handle_from_id<CUtexref>(kTextureHandleBase,
                                               static_cast<std::uint32_t>(i + 1u));
            return CUDA_SUCCESS;
        }
    }
    try {
        value.textures.push_back({true, module_id, name});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *output = handle_from_id<CUtexref>(kTextureHandleBase,
        static_cast<std::uint32_t>(value.textures.size()));
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemHostAlloc(void **pointer, size_t bytes,
                                                  unsigned int)
{
    if (pointer == nullptr || bytes == 0u) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    void *allocation = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (allocation == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    try {
        value.host_allocations.push_back({allocation, bytes});
    } catch (...) {
        VirtualFree(allocation, 0, MEM_RELEASE);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pointer = allocation;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFreeHost(void *pointer)
{
    if (pointer == nullptr) return CUDA_SUCCESS;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    const auto found = std::find_if(value.host_allocations.begin(),
        value.host_allocations.end(), [pointer](const HostAllocation &item) {
            return item.pointer == pointer;
        });
    if (found == value.host_allocations.end()) return CUDA_ERROR_INVALID_VALUE;
    if (VirtualFree(pointer, 0, MEM_RELEASE) == FALSE) return CUDA_ERROR_OPERATING_SYSTEM;
    value.host_allocations.erase(found);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes)
{
    if (pointer == nullptr || bytes == 0u) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult initialized = require_initialized_locked(value);
    if (initialized != CUDA_SUCCESS) return initialized;
    const std::uint32_t base = align_up(value.next_device_address, 256u);
    std::uint32_t end = 0;
    if (!checked_u32_range(base, bytes, end)) return CUDA_ERROR_OUT_OF_MEMORY;
    std::string error;
    if (!value.backend.ensure_heap(end, error)) return backend_failure(value, "device_alloc", error);
    try {
        value.allocations.push_back({true, false, base,
                                     static_cast<std::uint32_t>(bytes), 0, {}});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    value.next_device_address = end;
    *pointer = base;
    value.write_log("device_alloc base=0x%08x bytes=%u", base,
                    static_cast<unsigned int>(bytes));
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAlloc(CUdeviceptr *pointer, size_t bytes)
{
    return cuMemAlloc_v2(pointer, bytes);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFree_v2(CUdeviceptr pointer)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    for (AllocationRecord &record : value.allocations) {
        if (record.live && !record.global && record.base == pointer) {
            record.live = false;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFree(CUdeviceptr pointer)
{
    return cuMemFree_v2(pointer);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *base,
                                                           size_t *bytes,
                                                           CUdeviceptr pointer)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    AllocationRecord *record = owning_allocation(value, pointer, 0);
    if (record == nullptr) return CUDA_ERROR_NOT_FOUND;
    if (base != nullptr) *base = record->base;
    if (bytes != nullptr) *bytes = record->size;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetAddressRange(CUdeviceptr *base,
                                                        size_t *bytes,
                                                        CUdeviceptr pointer)
{
    return cuMemGetAddressRange_v2(base, bytes, pointer);
}

CUresult copy_h_to_d(CUdeviceptr destination, const void *source, size_t bytes)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (owning_allocation(value, destination, bytes) == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    return value.backend.write_heap(destination, source, bytes, error)
               ? CUDA_SUCCESS : backend_failure(value, "copy_h2d", error);
}
CUresult copy_d_to_h(void *destination, CUdeviceptr source, size_t bytes)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (owning_allocation(value, source, bytes) == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    return value.backend.read_heap(destination, source, bytes, error)
               ? CUDA_SUCCESS : backend_failure(value, "copy_d2h", error);
}
CUresult copy_d_to_d(CUdeviceptr destination, CUdeviceptr source, size_t bytes)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (owning_allocation(value, destination, bytes) == nullptr ||
        owning_allocation(value, source, bytes) == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    return value.backend.move_heap(destination, source, bytes, error)
               ? CUDA_SUCCESS : backend_failure(value, "copy_d2d", error);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { return copy_h_to_d(d, s, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *s, size_t n, CUstream stream) { return stream == nullptr ? copy_h_to_d(d, s, n) : CUDA_ERROR_NOT_SUPPORTED; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr d, const void *s, size_t n, CUstream stream) { return cuMemcpyHtoDAsync_v2(d, s, n, stream); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { return copy_d_to_h(d, s, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoHAsync_v2(void *d, CUdeviceptr s, size_t n, CUstream stream) { return stream == nullptr ? copy_d_to_h(d, s, n) : CUDA_ERROR_NOT_SUPPORTED; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoHAsync(void *d, CUdeviceptr s, size_t n, CUstream stream) { return cuMemcpyDtoHAsync_v2(d, s, n, stream); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) { return copy_d_to_d(d, s, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream stream) { return stream == nullptr ? copy_d_to_d(d, s, n) : CUDA_ERROR_NOT_SUPPORTED; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream stream) { return cuMemcpyDtoDAsync_v2(d, s, n, stream); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8_v2(CUdeviceptr destination,
                                                 unsigned char value_byte,
                                                 size_t bytes)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (owning_allocation(value, destination, bytes) == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    return value.backend.fill_heap(destination, value_byte, bytes, error)
               ? CUDA_SUCCESS : backend_failure(value, "memset", error);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return cuMemsetD8_v2(d, v, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream stream) { return stream == nullptr ? cuMemsetD8_v2(d, v, n) : CUDA_ERROR_NOT_SUPPORTED; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddress_v2(size_t *offset,
                                                        CUtexref texture,
                                                        CUdeviceptr pointer,
                                                        size_t bytes)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    TextureRecord *record = texture_record(value, texture);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    if ((pointer != 0u || bytes != 0u) &&
        owning_allocation(value, pointer, bytes) == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    record->address = pointer;
    record->bytes = static_cast<std::uint32_t>(bytes);
    if (offset != nullptr) *offset = 0;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddress(size_t *o, CUtexref t, CUdeviceptr p, size_t n) { return cuTexRefSetAddress_v2(o, t, p, n); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFormat(CUtexref texture,
                                                     CUarray_format format,
                                                     int components)
{
    if (components < 1 || components > 4) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    TextureRecord *record = texture_record(value, texture);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    record->format = format;
    record->components = components;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFlags(CUtexref texture,
                                                    unsigned int flags)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    TextureRecord *record = texture_record(value, texture);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    record->flags = flags;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFilterMode(CUtexref, CUfilter_mode) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref, CUfilter_mode) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref, float) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref, float, float) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref, unsigned int) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddressMode(CUtexref, int dimension, CUaddress_mode) { return dimension == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncGetAttribute(int *output,
                                                      CUfunction_attribute attribute,
                                                      CUfunction function)
{
    if (output == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    FunctionRecord *record = function_record(value, function);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    const KernelSpec *spec = lookup_kernel_name(record->name.c_str());
    switch (attribute) {
    case 0: *output = spec != nullptr ? static_cast<int>(spec->block_x) : 1024; break;
    case 5: *output = 41; break;
    case 6: *output = 52; break;
    default: *output = 0; break;
    }
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncSetCacheConfig(CUfunction, CUfunc_cache) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncSetSharedMemConfig(CUfunction, CUsharedconfig) { return CUDA_SUCCESS; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuLaunchKernel(
    CUfunction function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_bytes, CUstream stream, void **kernel_params, void **extra)
{
    if (stream != nullptr || kernel_params != nullptr || extra == nullptr ||
        grid_x == 0u || grid_y == 0u || grid_z == 0u ||
        block_y != 1u || block_z != 1u)
        return CUDA_ERROR_NOT_SUPPORTED;
    pxvk_launch_params launch{};
    const pxvk_launch_params_status status =
        pxvk_parse_launch_extra(extra, read_current_process, nullptr, &launch);
    if (status != PXVK_LAUNCH_PARAMS_OK || launch.buffer == nullptr ||
        launch.size > kMaxPackedParameters)
        return CUDA_ERROR_INVALID_VALUE;

    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    FunctionRecord *record = function_record(value, function);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    const KernelSpec *spec = lookup_kernel_launch(record->name.c_str(), block_x, shared_bytes);
    if (spec == nullptr) {
        value.write_log("unsupported_launch name=%s block=%u shared=%u",
                        record->name.c_str(), block_x, shared_bytes);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (launch.size != spec->packed_bytes) return CUDA_ERROR_INVALID_VALUE;

    ModuleRecord &module = value.modules[record->module_id - 1u];
    if (!module.live) return CUDA_ERROR_INVALID_HANDLE;
    std::vector<std::uint8_t> constants;
    std::vector<std::uint32_t> textures;
    CUresult result = collect_constants(value, record->module_id, spec->profile, constants);
    if (result != CUDA_SUCCESS) return result;
    result = collect_textures(value, record->module_id, spec->profile, textures);
    if (result != CUDA_SUCCESS) return result;

    pxvk::DispatchInfo dispatch;
    dispatch.shader_path = shader_path(spec->shader);
    dispatch.parameters = launch.buffer;
    dispatch.parameter_bytes = launch.size;
    dispatch.constants = constants.empty() ? nullptr : constants.data();
    dispatch.constant_bytes = constants.size();
    dispatch.texture_metadata = textures.empty() ? nullptr : textures.data();
    dispatch.texture_metadata_bytes = textures.size() * sizeof(std::uint32_t);
    dispatch.grid_x = grid_x;
    dispatch.grid_y = grid_y;
    dispatch.grid_z = grid_z;
    dispatch.shared_bytes_per_group = shared_bytes;
    dispatch.storage_backed_shared = spec->storage_shared;
    std::string error;
    if (!value.backend.dispatch(dispatch, error))
        return backend_failure(value, "launch", error);
    if (!record->launch_logged) {
        value.write_log("launch name=%s grid=%ux%ux%u block=%u shared=%u",
                        record->name.c_str(), grid_x, grid_y, grid_z,
                        block_x, shared_bytes);
        record->launch_logged = true;
    }
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetErrorName(CUresult result, const char **text)
{
    if (text == nullptr) return CUDA_ERROR_INVALID_VALUE;
    switch (result) {
    case CUDA_SUCCESS: *text = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *text = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *text = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *text = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case CUDA_ERROR_INVALID_HANDLE: *text = "CUDA_ERROR_INVALID_HANDLE"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *text = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *text = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetErrorString(CUresult result, const char **text)
{
    if (text == nullptr) return CUDA_ERROR_INVALID_VALUE;
    switch (result) {
    case CUDA_SUCCESS: *text = "no error"; break;
    case CUDA_ERROR_INVALID_VALUE: *text = "invalid value"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *text = "out of memory"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *text = "driver not initialized"; break;
    case CUDA_ERROR_INVALID_HANDLE: *text = "invalid handle"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *text = "operation not supported by pxvk"; break;
    default: *text = "unknown pxvk CUDA error"; break;
    }
    return CUDA_SUCCESS;
}

namespace {

CUresult PXVK_CUDAAPI dark_get_module(CUmodule *module,
                                       const PxvkFatbinWrapper *wrapper)
{
    if (module == nullptr || wrapper == nullptr || wrapper->magic != 0x466243b1u ||
        wrapper->version != 1u || wrapper->data == nullptr ||
        wrapper->data->magic != 0xba55ed50u || wrapper->data->header_size < 16u)
        return CUDA_ERROR_INVALID_IMAGE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    CUresult result = require_initialized_locked(value);
    return result == CUDA_SUCCESS ? create_module_locked(value, module) : result;
}

CUresult PXVK_CUDAAPI dark_cudart_context(CUcontext *context, CUdevice device)
{
    if (context == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    *context = reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0xdeadbeefu));
    return CUDA_SUCCESS;
}

std::array<unsigned char, 1024> g_unknown_buffer1{};
std::array<unsigned char, 14> g_unknown_buffer2{};
void PXVK_CUDAAPI dark_unknown_buffer1(void **pointer, size_t *size)
{
    if (pointer != nullptr) *pointer = g_unknown_buffer1.data();
    if (size != nullptr) *size = g_unknown_buffer1.size();
}
void PXVK_CUDAAPI dark_unknown_buffer2(void **pointer, size_t *size)
{
    if (pointer != nullptr) *pointer = g_unknown_buffer2.data();
    if (size != nullptr) *size = g_unknown_buffer2.size();
}

CUresult PXVK_CUDAAPI dark_cls_put(CUcontext context, void *key, void *stored,
                                    ContextDestructor destructor)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (context == nullptr) context = g_current_context;
    ContextRecord *record = context_record(value, context);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    try {
        record->cls[reinterpret_cast<std::uintptr_t>(key)] = {stored, destructor};
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_cls_delete(CUcontext context, void *key)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (context == nullptr) context = g_current_context;
    ContextRecord *record = context_record(value, context);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    record->cls.erase(reinterpret_cast<std::uintptr_t>(key));
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_cls_get(void **output, CUcontext context, void *key)
{
    if (output == nullptr) return CUDA_ERROR_INVALID_VALUE;
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (context == nullptr) context = g_current_context;
    ContextRecord *record = context_record(value, context);
    if (record == nullptr) return CUDA_ERROR_INVALID_HANDLE;
    const auto found = record->cls.find(reinterpret_cast<std::uintptr_t>(key));
    if (found == record->cls.end()) return CUDA_ERROR_INVALID_HANDLE;
    *output = found->second.value;
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_context_create(CUcontext *context, unsigned int flags,
                                           CUdevice device)
{
    return cuCtxCreate_v2(context, flags, device);
}

CUresult PXVK_CUDAAPI dark_heap_alloc(const void **output, size_t first, size_t second)
{
    if (output == nullptr) return CUDA_ERROR_INVALID_VALUE;
    size_t *record = new (std::nothrow) size_t[2];
    if (record == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    record[0] = first;
    record[1] = second;
    *output = record;
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_heap_free(const void *opaque, size_t *second)
{
    if (opaque == nullptr) return CUDA_ERROR_INVALID_VALUE;
    const size_t *record = static_cast<const size_t *>(opaque);
    if (second != nullptr) *second = record[1];
    delete[] record;
    return CUDA_SUCCESS;
}

} // namespace

PXVK_EXPORT CUresult __cdecl pxvkNotSupportedStub(void)
{
    State &value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    return value.deinitialized ? CUDA_ERROR_DEINITIALIZED : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        g_module = module;
        DisableThreadLibraryCalls(module);
        install_driver_redirect();
    } else if (reason == DLL_PROCESS_DETACH &&
               g_dll_notification_cookie != nullptr &&
               g_ldr_unregister != nullptr) {
        g_ldr_unregister(g_dll_notification_cookie);
        g_dll_notification_cookie = nullptr;
    }
    return TRUE;
}
