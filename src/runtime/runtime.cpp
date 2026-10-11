// pxvk nvcuda.dll: CUDA driver API for classic 32-bit GPU PhysX on Vulkan.
//
// Architecture (see docs/architecture.md):
//   * Modules arrive as classic fatbins (through CUDART's private export
//     table) or PTX text. Their PTX is parsed once; module globals are
//     allocated immediately; each kernel is compiled to SPIR-V on first use.
//   * CUDA device memory is a single device-local Vulkan buffer. Device
//     pointers are byte offsets into it (the first 64 KiB are never used, so
//     null and small values stay invalid).
//   * Pinned host memory (cuMemHostAlloc) is host-visible Vulkan memory, so
//     async copies to/from it are plain recorded GPU copies.
//   * All work is recorded into one ordered batch and submitted only when the
//     host must observe results (synchronous copies, synchronize calls). All
//     streams share that order, which CUDA permits.
#include "cuda_abi.h"
#include "runtime.h"

#include "compiler.h"
#include "device.h"
#include "fatbin.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

static_assert(sizeof(void *) == 4, "pxvk runtime requires a 32-bit build");

namespace {

using namespace pxvk;

constexpr std::uint32_t kHeapReserve = 0x10000;
constexpr std::uint32_t kAllocationAlignment = 256;
constexpr std::uint32_t kWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinMagic = 0xBA55ED50u;

HMODULE g_module = nullptr;

// ---- environment ----------------------------------------------------------
// All variables are described in docs/environment.md.
std::string env(const char *name)
{
    char value[1024];
    const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    return length && length < sizeof(value) ? std::string(value, length) : std::string();
}

// A switch is on when set to anything but empty, 0, off, false or no.
bool env_flag(const char *name)
{
    const std::string value = env(name);
    for (const char *off : {"", "0", "off", "false", "no"})
        if (_stricmp(value.c_str(), off) == 0) return false;
    return true;
}

// Creates a directory and any missing parents; existing ones are fine.
void make_directories(const std::string &path)
{
    for (std::size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '\\' || path[i] == '/') {
            const std::string prefix = path.substr(0, i);
            if (!prefix.empty() && prefix.back() != ':') CreateDirectoryA(prefix.c_str(), nullptr);
        }
    }
}

// ---- logging ------------------------------------------------------------
class Log {
public:
    void write(const char *format, ...)
    {
        if (!opened_) open();
        if (!file_) return;
        va_list args;
        va_start(args, format);
        std::vfprintf(file_, format, args);
        va_end(args);
        std::fputc('\n', file_);
        std::fflush(file_);
    }

private:
    // PXVK_LOG: a new log per run; missing folders are created.
    void open()
    {
        opened_ = true;
        const std::string path = env("PXVK_LOG");
        if (path.empty()) return;
        const std::size_t slash = path.find_last_of("\\/");
        if (slash != std::string::npos) make_directories(path.substr(0, slash));
        file_ = std::fopen(path.c_str(), "w");
    }
    bool opened_ = false;
    FILE *file_ = nullptr;
};

// ---- runtime objects --------------------------------------------------------
struct Module;

struct TexRef {
    Module *module = nullptr;
    std::string name;
    std::uint32_t address = 0, bytes = 0;
    int format = 0x20; // CU_AD_FORMAT_FLOAT
    int components = 1;
};

// A compiled kernel variant. Entries live in State::kernels, keyed by PTX
// content, so modules that are unloaded and registered again (PhysX does
// this on every scene change) reuse kernels, shaders and pipelines.
struct CompiledVariant {
    ptx::CompiledKernel kernel;
    vk::Shader *shader = nullptr;
    std::string error;
};

struct Function {
    Module *module = nullptr;
    std::string name;
    CompiledVariant *variants[2] = {nullptr, nullptr}; // [0] Workgroup shared, [1] storage-backed
    std::uint64_t launches = 0;
};

struct Module {
    ptx::Module ast; // freed on unload; the shell stays so stale handles stay invalid
    CUcontext context = nullptr; // destroying it unloads the module, as in CUDA
    std::uint64_t ptx_hash = 0;
    std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> globals; // address, size
    std::map<std::string, std::unique_ptr<TexRef>> texrefs;
    std::map<std::string, std::unique_ptr<Function>> functions;
};

struct HostRegion {
    vk::HostBuffer *buffer = nullptr;
    char *pointer = nullptr;
    std::size_t size = 0;
    unsigned flags = 0;
};

using ContextDestructor = void(PXVK_CUDAAPI *)(CUcontext, void *, void *);
struct ClsValue {
    void *value = nullptr;
    ContextDestructor destructor = nullptr;
};
struct Context {
    unsigned flags = 0;
    std::map<std::uintptr_t, ClsValue> cls;
};
struct Event {
    bool recorded = false;
    LARGE_INTEGER time{};
};
struct Stream {
    unsigned flags = 0;
};

struct State {
    std::recursive_mutex mutex;
    Log log;
    vk::Device device;
    bool initialized = false;
    CUresult init_result = CUDA_SUCCESS;
    ptx::CompileOptions options;
    std::string dump_directory;
    bool sync_everything = false; // PXVK_SYNC=1: wait after every operation (debugging)
    bool profile = false;         // PXVK_PROFILE=1: per-kernel GPU timing in the stats log
    std::uint64_t syncs = 0;
    std::uint64_t last_stats_launches = 0;
    std::string pipeline_cache_path;      // empty: no persistent cache
    std::uint64_t saved_pipelines = 0;    // pipeline count at the last save
    DWORD pipelines_changed_at = 0;
    bool pending_host_copies = false; // async DtoH into pinned memory recorded since the last sync

    // Device-memory allocator: first fit over freed blocks, else bump.
    std::map<std::uint32_t, std::uint32_t> free_blocks; // address -> size
    std::map<std::uint32_t, std::uint32_t> used;        // address -> size
    std::uint32_t heap_end = kHeapReserve;
    std::uint64_t heap_limit = 0;

    std::vector<std::unique_ptr<Module>> modules;
    std::map<std::string, std::unique_ptr<CompiledVariant>> kernels; // ptx hash + entry + variant
    std::set<Module *> live_modules;
    std::set<Function *> functions;
    std::set<TexRef *> texrefs;
    std::map<char *, HostRegion> host;
    std::set<Context *> contexts;
    std::set<Event *> events;
    std::set<Stream *> streams;

    // Periodic statistics.
    std::uint64_t launches = 0;
    DWORD last_stats = 0;
};

State &state()
{
    static State *s = new State; // never destroyed: CUDA calls may outlive DLL detach
    return *s;
}

thread_local std::vector<CUcontext> t_context_stack;

CUcontext current_context()
{
    return t_context_stack.empty() ? nullptr : t_context_stack.back();
}

void set_current_context(CUcontext context)
{
    if (t_context_stack.empty()) t_context_stack.push_back(context);
    else t_context_stack.back() = context;
}

// ---- helpers ------------------------------------------------------------------
std::uint64_t fnv1a(const std::string &text)
{
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) h = (h ^ c) * 1099511628211ull;
    return h;
}

// PXVK_API_LOG: one line per memory/sync/launch call, tagged with the
// harness frame number, in the same format as tools/nvcuda-trace.
bool api_log_enabled()
{
    static const bool enabled = env_flag("PXVK_API_LOG");
    return enabled;
}
#define API_LOG(...)                                                                     do {                                                                                     if (api_log_enabled()) {                                                                 char api_buffer_[256];                                                               std::snprintf(api_buffer_, sizeof(api_buffer_), __VA_ARGS__);                        state().log.write("api f=%s t=%lu %s", env("PXVK_FRAME").c_str(), GetCurrentThreadId(), api_buffer_);         }                                                                                } while (0)

CUresult failure(State &s, const char *what, const std::string &error, CUresult code = CUDA_ERROR_UNKNOWN)
{
    s.log.write("error %s: %s", what, error.c_str());
    return code;
}

CUresult initialize(State &s)
{
    if (s.initialized) return s.init_result;
    s.initialized = true;
    std::string error;
    if (!s.device.initialize(env("PXVK_DEVICE"), error)) {
        s.log.write("init failed: %s", error.c_str());
        s.init_result = CUDA_ERROR_NO_DEVICE;
        return s.init_result;
    }
    const auto &caps = s.device.caps();
    s.options.native_ftz = caps.native_ftz;
    s.options.preserve_nan = caps.preserve_nan;
    s.options.emulate_warps = caps.narrow_subgroups();
    s.options.int64 = caps.int64;
    s.options.float_atomic_add = caps.float_atomic_add;
    s.options.shader_clock = caps.shader_clock;
    s.options.heap_window_shift = caps.heap_window_shift;
    s.options.heap_windows = caps.heap_windows;
    s.heap_limit = caps.max_heap_bytes;
    s.dump_directory = env("PXVK_DUMP_SPIRV");
    if (!s.dump_directory.empty()) make_directories(s.dump_directory);
    s.sync_everything = env_flag("PXVK_SYNC");
    s.profile = env_flag("PXVK_PROFILE");
    // Persistent pipeline cache: %LOCALAPPDATA%\pxvk (PXVK_CACHE_DIR
    // overrides, "off" disables), one file per driver cache identity.
    std::string cache_dir = env("PXVK_CACHE_DIR");
    if (cache_dir.empty() && !env("LOCALAPPDATA").empty()) cache_dir = env("LOCALAPPDATA") + "\\pxvk";
    if (!cache_dir.empty() && _stricmp(cache_dir.c_str(), "off") != 0) {
        make_directories(cache_dir);
        s.pipeline_cache_path = cache_dir + "\\pipelines-" + caps.pipeline_cache_uuid + ".bin";
        s.device.load_pipeline_cache(s.pipeline_cache_path);
    }
    s.device.set_timing(s.profile);
    s.log.write("init device=\"%s\" vendor=%04x subgroup=%u native_ftz=%d subgroup_control=%d preserve_nan=%d shared=%u heap_limit=%lluMiB heap_windows=%u warp_emulation=%d float_atomics=%d nvapi=%s",
                caps.name.c_str(), caps.vendor_id, caps.subgroup_size, caps.native_ftz, caps.subgroup_control,
                caps.preserve_nan, caps.max_shared_bytes, static_cast<unsigned long long>(s.heap_limit >> 20),
                caps.heap_windows, s.options.emulate_warps, s.options.float_atomic_add, pxvk::runtime::nvapi_mode());
    return CUDA_SUCCESS;
}

CUresult require(State &s)
{
    if (!s.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return s.init_result;
}

// ---- device memory ---------------------------------------------------------------
// Returns [start, start + size) to the free list, merging it with adjacent
// free blocks; space that reaches the top of the heap lowers heap_end
// instead, so later bump allocations can use it.
void add_free(State &s, std::uint32_t start, std::uint32_t size)
{
    auto next = s.free_blocks.find(start + size);
    if (next != s.free_blocks.end()) {
        size += next->second;
        s.free_blocks.erase(next);
    }
    auto prev = s.free_blocks.lower_bound(start);
    if (prev != s.free_blocks.begin()) {
        --prev;
        if (prev->first + prev->second == start) {
            start = prev->first;
            size += prev->second;
            s.free_blocks.erase(prev);
        }
    }
    if (std::uint64_t(start) + size == s.heap_end) s.heap_end = start;
    else s.free_blocks[start] = size;
}

CUresult allocate(State &s, std::size_t requested, std::uint32_t alignment, std::uint32_t &address)
{
    const std::uint64_t size = (std::uint64_t(requested) + kAllocationAlignment - 1) & ~std::uint64_t(kAllocationAlignment - 1);
    if (size == 0 || size > 0xFFFFFFFFu) return CUDA_ERROR_INVALID_VALUE;
    alignment = std::max(alignment, kAllocationAlignment);
    for (auto it = s.free_blocks.begin(); it != s.free_blocks.end(); ++it) {
        const std::uint64_t start = (std::uint64_t(it->first) + alignment - 1) & ~std::uint64_t(alignment - 1);
        const std::uint64_t end = std::uint64_t(it->first) + it->second;
        if (start + size > end) continue;
        const std::uint32_t block = it->first, block_size = it->second;
        s.free_blocks.erase(it);
        if (start > block) s.free_blocks[block] = static_cast<std::uint32_t>(start - block);
        if (start + size < end) s.free_blocks[static_cast<std::uint32_t>(start + size)] = static_cast<std::uint32_t>(end - start - size);
        (void)block_size;
        address = static_cast<std::uint32_t>(start);
        s.used[address] = static_cast<std::uint32_t>(size);
        return CUDA_SUCCESS;
    }
    const std::uint64_t start = (std::uint64_t(s.heap_end) + alignment - 1) & ~std::uint64_t(alignment - 1);
    if (start + size > s.heap_limit) return CUDA_ERROR_OUT_OF_MEMORY;
    std::string error;
    const std::uint64_t heap_before = s.device.heap_size();
    if (!s.device.ensure_heap(start + size, error)) return failure(s, "heap", error, CUDA_ERROR_OUT_OF_MEMORY);
    if (s.device.heap_size() != heap_before)
        s.log.write("heap grow %lluKiB -> %lluKiB", static_cast<unsigned long long>(heap_before >> 10),
                    static_cast<unsigned long long>(s.device.heap_size() >> 10));
    const std::uint32_t gap_start = s.heap_end;
    s.heap_end = static_cast<std::uint32_t>(start + size);
    if (start > gap_start) add_free(s, gap_start, static_cast<std::uint32_t>(start - gap_start));
    address = static_cast<std::uint32_t>(start);
    s.used[address] = static_cast<std::uint32_t>(size);
    return CUDA_SUCCESS;
}

bool release(State &s, std::uint32_t address)
{
    const auto found = s.used.find(address);
    if (found == s.used.end()) return false;
    const std::uint32_t start = found->first, size = found->second;
    s.used.erase(found);
    add_free(s, start, size);
    return true;
}

// The allocation containing [address, address + bytes), if any.
bool valid_range(const State &s, std::uint32_t address, std::size_t bytes)
{
    auto it = s.used.upper_bound(address);
    if (it == s.used.begin()) return false;
    --it;
    return std::uint64_t(address) + bytes <= std::uint64_t(it->first) + it->second;
}

const HostRegion *pinned(const State &s, const void *pointer, std::size_t bytes)
{
    char *p = static_cast<char *>(const_cast<void *>(pointer));
    auto it = s.host.upper_bound(p);
    if (it == s.host.begin()) return nullptr;
    --it;
    const HostRegion &region = it->second;
    if (p >= region.pointer && p + bytes <= region.pointer + region.size) return &region;
    return nullptr;
}

// ---- modules --------------------------------------------------------------
CUresult load_ptx(State &s, const std::string &text, CUmodule *output)
{
    auto module = std::make_unique<Module>();
    try {
        module->ast = ptx::parse(text);
    } catch (const std::exception &e) {
        s.log.write("module parse failed: %s", e.what());
        return CUDA_ERROR_INVALID_IMAGE;
    }
    if (module->ast.address_size != 32) {
        s.log.write("module rejected: .address_size %d", module->ast.address_size);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    module->ptx_hash = fnv1a(text);
    std::string error;
    // A load that fails partway gives back what it already took.
    auto abandon = [&](CUresult r) {
        for (auto &g : module->globals) release(s, g.second.first);
        for (auto &t : module->texrefs) s.texrefs.erase(t.second.get());
        return r;
    };
    for (const auto &v : module->ast.vars) {
        if (v.space == ptx::Space::tex) {
            auto ref = std::make_unique<TexRef>();
            ref->module = module.get();
            ref->name = v.name;
            s.texrefs.insert(ref.get());
            module->texrefs[v.name] = std::move(ref);
            continue;
        }
        if (v.space != ptx::Space::global && v.space != ptx::Space::constant) continue;
        if (v.external) continue;
        const std::uint64_t size = v.size_bytes();
        if (size == 0) continue;
        std::uint32_t address = 0;
        const CUresult r = allocate(s, static_cast<std::size_t>(size), std::max<std::uint32_t>(v.align, 16), address);
        if (r != CUDA_SUCCESS) return abandon(r);
        module->globals[v.name] = {address, static_cast<std::uint32_t>(size)};
        std::vector<std::uint8_t> init(static_cast<std::size_t>(size), 0);
        std::memcpy(init.data(), v.init.data(), std::min(init.size(), v.init.size()));
        if (!s.device.upload(address, init.data(), init.size(), error))
            return abandon(failure(s, "module global", error));
    }
    std::size_t entries = 0;
    for (const auto &f : module->ast.functions) entries += f.is_entry;
    s.log.write("module load hash=%016llx bytes=%u entries=%u globals=%u texrefs=%u",
                static_cast<unsigned long long>(module->ptx_hash), static_cast<unsigned>(text.size()),
                static_cast<unsigned>(entries), static_cast<unsigned>(module->globals.size()),
                static_cast<unsigned>(module->texrefs.size()));
    module->context = current_context();
    *output = reinterpret_cast<CUmodule>(module.get());
    s.live_modules.insert(module.get());
    s.modules.push_back(std::move(module));
    return CUDA_SUCCESS;
}

// Releases a module's globals and its parsed PTX. The Module, Function and
// TexRef objects stay allocated (small) because the application may still
// hold their handles, which must remain safely invalid. Compiled kernels
// live on in State::kernels, keyed by PTX content, for re-registration.
void unload_module(State &s, Module *m)
{
    for (auto &g : m->globals) release(s, g.second.first);
    m->globals.clear();
    for (auto &f : m->functions) s.functions.erase(f.second.get());
    for (auto &t : m->texrefs) s.texrefs.erase(t.second.get());
    s.live_modules.erase(m);
    m->ast = ptx::Module{};
    s.log.write("module unload hash=%016llx", static_cast<unsigned long long>(m->ptx_hash));
}

CUresult load_fatbin(State &s, const void *container, CUmodule *output)
{
    try {
        const std::size_t size = fatbin::container_size(container);
        if (size == 0) return CUDA_ERROR_INVALID_IMAGE;
        const auto payloads = fatbin::extract_ptx(container, size);
        if (payloads.empty()) {
            s.log.write("fatbin has no PTX (cubin-only modules are unsupported)");
            return CUDA_ERROR_NO_BINARY_FOR_GPU;
        }
        // Prefer the PTX built for the highest virtual architecture.
        const auto best = std::max_element(payloads.begin(), payloads.end(),
            [](const fatbin::PtxPayload &a, const fatbin::PtxPayload &b) { return a.sm_version < b.sm_version; });
        return load_ptx(s, best->text, output);
    } catch (const std::exception &e) {
        // Name the binary that registered it and dump its header, so new
        // container formats can be identified from a user's log.
        char owner[MAX_PATH] = "?";
        HMODULE module = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<const char *>(container), &module))
            GetModuleFileNameA(module, owner, MAX_PATH);
        char header[3 * 32 + 1] = "";
        const auto *bytes = static_cast<const unsigned char *>(container);
        for (int i = 0; i < 32; ++i) std::snprintf(header + 3 * i, 4, "%02x ", bytes[i]);
        s.log.write("fatbin decode failed: %s (from %s, header %s)", e.what(), owner, header);
        return CUDA_ERROR_INVALID_IMAGE;
    }
}

CUresult load_image(State &s, const void *image, CUmodule *output)
{
    std::uint32_t magic = 0;
    std::memcpy(&magic, image, 4);
    if (magic == kWrapperMagic) {
        const auto *wrapper = static_cast<const PxvkFatbinWrapper *>(image);
        return load_fatbin(s, wrapper->data, output);
    }
    if (magic == kFatbinMagic) return load_fatbin(s, image, output);
    if (magic == 0x464c457fu) return CUDA_ERROR_NO_BINARY_FOR_GPU; // ELF cubin
    return load_ptx(s, static_cast<const char *>(image), output);
}

Module *module_of(State &s, CUmodule handle)
{
    Module *m = reinterpret_cast<Module *>(handle);
    return s.live_modules.count(m) ? m : nullptr;
}

Function *function_of(State &s, CUfunction handle)
{
    Function *f = reinterpret_cast<Function *>(handle);
    return s.functions.count(f) && s.live_modules.count(f->module) ? f : nullptr;
}

TexRef *texref_of(State &s, CUtexref handle)
{
    TexRef *t = reinterpret_cast<TexRef *>(handle);
    return s.texrefs.count(t) ? t : nullptr;
}

void dump_spirv(State &s, const Function &f, int variant, const std::vector<std::uint32_t> &words)
{
    if (s.dump_directory.empty()) return;
    char name[64];
    std::snprintf(name, sizeof(name), "%016llx_%08llx_%d.spv", static_cast<unsigned long long>(f.module->ptx_hash),
                  static_cast<unsigned long long>(fnv1a(f.name) & 0xffffffffu), variant);
    if (FILE *out = std::fopen((s.dump_directory + "\\" + name).c_str(), "wb")) {
        std::fwrite(words.data(), 4, words.size(), out);
        std::fclose(out);
    }
}

CompiledVariant *variant(State &s, Function &f, int which)
{
    CompiledVariant *&slot = f.variants[which];
    if (!slot) {
        char key_prefix[32];
        std::snprintf(key_prefix, sizeof(key_prefix), "%016llx:%d:", static_cast<unsigned long long>(f.module->ptx_hash),
                      which);
        auto &entry = s.kernels[key_prefix + f.name];
        if (!entry) {
            entry = std::make_unique<CompiledVariant>();
            const DWORD start = GetTickCount();
            try {
                ptx::CompileOptions options = s.options;
                options.shared_in_storage = which == 1;
                entry->kernel = ptx::compile_kernel(f.module->ast, f.name, options);
                dump_spirv(s, f, which, entry->kernel.spirv);
                entry->shader = s.device.create_shader(entry->kernel.spirv, entry->kernel.info.launch_block_bytes,
                                                       entry->kernel.info.warp_synchronous, which == 1, entry->error);
            } catch (const std::exception &e) {
                entry->error = e.what();
            }
            s.log.write("compile %s variant=%d words=%u time=%lums%s%s%s", f.name.c_str(), which,
                        static_cast<unsigned>(entry->kernel.spirv.size()), GetTickCount() - start,
                        entry->kernel.info.warp_emulated ? " warp_emulated" : "",
                        entry->error.empty() ? "" : " error=", entry->error.c_str());
        }
        slot = entry.get();
    }
    return slot->shader ? slot : nullptr;
}

void log_stats(State &s)
{
    const DWORD now = GetTickCount();
    if (s.last_stats == 0) s.last_stats = now;
    if (now - s.last_stats < 5000 && !(s.profile && s.launches - s.last_stats_launches >= 500)) return;
    s.last_stats_launches = s.launches;
    const auto &st = s.device.stats();
    s.log.write("stats launches=%llu syncs=%llu submits=%llu pipelines=%llu up=%lluKiB down=%lluKiB "
                "wait=%.1fms pipeline_create=%.1fms",
                static_cast<unsigned long long>(s.launches), static_cast<unsigned long long>(s.syncs),
                static_cast<unsigned long long>(st.submits), static_cast<unsigned long long>(st.pipelines),
                static_cast<unsigned long long>(st.upload_bytes >> 10),
                static_cast<unsigned long long>(st.download_bytes >> 10), st.wait_ms, st.pipeline_ms);
    if (s.profile) {
        for (const auto &k : s.device.take_profile())
            s.log.write("profile %-60.60s count=%llu total=%.2fms avg=%.4fms", k.first.c_str(),
                        static_cast<unsigned long long>(k.second.count), k.second.total_ms,
                        k.second.total_ms / double(k.second.count));
    }
    s.last_stats = now;
}

// ---- launch capture ---------------------------------------------------------------
// Same format as tools/nvcuda-trace/src/capture.c, so pxvk and native runs of
// the same frames can be compared (PXVK_CAPTURE_DIR + PXVK_CAPTURE_FRAMES; the
// harness publishes the current frame in PXVK_FRAME).
bool capture_selected(State &s)
{
    static const std::string frames = env("PXVK_CAPTURE_FRAMES");
    if (frames.empty() || env("PXVK_CAPTURE_DIR").empty()) return false;
    static const bool directory_ready = (make_directories(env("PXVK_CAPTURE_DIR")), true);
    (void)directory_ready;
    const std::string frame = env("PXVK_FRAME");
    std::size_t start = 0;
    while (start <= frames.size()) {
        std::size_t end = frames.find(',', start);
        if (end == std::string::npos) end = frames.size();
        if (frames.compare(start, end - start, frame) == 0 && !frame.empty()) return true;
        start = end + 1;
    }
    (void)s;
    return false;
}

void capture_snapshot(State &s, unsigned ordinal, const char *suffix)
{
    char path[1200];
    std::snprintf(path, sizeof(path), "%s\\capture_%06u.%s", env("PXVK_CAPTURE_DIR").c_str(), ordinal, suffix);
    FILE *file = std::fopen(path, "wb");
    if (!file) return;
    std::string error;
    std::vector<std::uint8_t> buffer;
    for (const auto &u : s.used) {
        buffer.resize(u.second);
        if (!s.device.download(buffer.data(), u.first, u.second, error)) break;
        for (std::uint32_t page = 0; page < u.second; page += 4096) {
            const std::uint32_t length = std::min<std::uint32_t>(4096, u.second - page);
            bool nonzero = false;
            for (std::uint32_t i = 0; i < length && !nonzero; ++i) nonzero = buffer[page + i] != 0;
            if (!nonzero) continue;
            const std::uint32_t header[2] = {u.first + page, length};
            std::fwrite(header, 4, 2, file);
            std::fwrite(buffer.data() + page, 1, length, file);
        }
    }
    std::fclose(file);
}

unsigned capture_begin(State &s, const Function &f, const std::uint32_t grid[3], const std::uint32_t block[3],
                       unsigned shared, const std::uint8_t *params, std::size_t param_bytes)
{
    static unsigned ordinal = 0;
    ++ordinal;
    if (!capture_selected(s)) return 0;
    char path[1200];
    std::snprintf(path, sizeof(path), "%s\\capture_%06u.txt", env("PXVK_CAPTURE_DIR").c_str(), ordinal);
    if (FILE *meta = std::fopen(path, "w")) {
        std::fprintf(meta, "kernel %s\nframe %s\nlaunch %u\nfatbin 0\n", f.name.c_str(), env("PXVK_FRAME").c_str(),
                     ordinal);
        std::fprintf(meta, "grid %u %u %u\nblock %u %u %u\nshared %u\nparams ", grid[0], grid[1], grid[2], block[0],
                     block[1], block[2], shared);
        for (std::size_t i = 0; i < param_bytes; ++i) std::fprintf(meta, "%02x", params[i]);
        std::fprintf(meta, "\n");
        for (const auto &g : f.module->globals)
            std::fprintf(meta, "global %s %08x %u\n", g.first.c_str(), g.second.first, g.second.second);
        for (const auto &t : f.module->texrefs)
            std::fprintf(meta, "texref %s %08x %u %d %d\n", t.first.c_str(), t.second->address, t.second->bytes,
                         t.second->format, t.second->components);
        for (const auto &u : s.used) std::fprintf(meta, "allocation %08x %u\n", u.first, u.second);
        std::fclose(meta);
    }
    capture_snapshot(s, ordinal, "before");
    return ordinal;
}

// ---- copies ---------------------------------------------------------------------
CUresult copy_htod(State &s, CUdeviceptr dst, const void *src, std::size_t bytes, bool async)
{
    if (bytes == 0) return CUDA_SUCCESS;
    if (src == nullptr || !valid_range(s, dst, bytes)) return CUDA_ERROR_INVALID_VALUE;
    API_LOG("%s dst=%08x bytes=%u", async ? "htod_async" : "htod", dst, static_cast<unsigned>(bytes));
    // The source is captured now even for async copies from pinned memory.
    // Real hardware starts such DMA almost immediately, and PhysX reuses its
    // pinned staging ring relying on that; deferring the read to the next
    // batch submission would copy data written after the call.
    (void)async;
    std::string error;
    // Stream order: an earlier async DtoH into this pinned memory must land
    // before its bytes are read here.
    if (s.pending_host_copies && pinned(s, src, bytes)) {
        if (!s.device.synchronize(error)) return failure(s, "sync", error);
        s.pending_host_copies = false;
    }
    if (!s.device.upload(dst, src, bytes, error)) return failure(s, "memcpy HtoD", error);
    return s.sync_everything && !s.device.synchronize(error) ? failure(s, "sync", error) : CUDA_SUCCESS;
}

CUresult copy_dtoh(State &s, void *dst, CUdeviceptr src, std::size_t bytes, bool async)
{
    if (bytes == 0) return CUDA_SUCCESS;
    if (dst == nullptr || !valid_range(s, src, bytes)) return CUDA_ERROR_INVALID_VALUE;
    API_LOG("%s src=%08x bytes=%u", async ? "dtoh_async" : "dtoh", src, static_cast<unsigned>(bytes));
    std::string error;
    const HostRegion *region = async ? pinned(s, dst, bytes) : nullptr;
    const bool ok = region
        ? s.device.copy_to_host(region->buffer, static_cast<char *>(dst) - region->pointer, src, bytes, error)
        : s.device.download(dst, src, bytes, error);
    if (!ok) return failure(s, "memcpy DtoH", error);
    if (region) s.pending_host_copies = true;
    return s.sync_everything && !s.device.synchronize(error) ? failure(s, "sync", error) : CUDA_SUCCESS;
}

CUresult copy_dtod(State &s, CUdeviceptr dst, CUdeviceptr src, std::size_t bytes)
{
    if (bytes == 0) return CUDA_SUCCESS;
    if (!valid_range(s, dst, bytes) || !valid_range(s, src, bytes)) return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    if (!s.device.copy(dst, src, bytes, error)) return failure(s, "memcpy DtoD", error);
    return s.sync_everything && !s.device.synchronize(error) ? failure(s, "sync", error) : CUDA_SUCCESS;
}

void save_pipeline_cache(State &s)
{
    // Save shortly after new pipelines appear, not on every one: PhysX
    // creates several in a burst at scene start.
    const std::uint64_t pipelines = s.device.stats().pipelines;
    if (s.pipeline_cache_path.empty() || pipelines == s.saved_pipelines) return;
    const DWORD now = GetTickCount();
    if (s.pipelines_changed_at == 0) s.pipelines_changed_at = now;
    if (now - s.pipelines_changed_at < 1000) return;
    // A failed save (read-only directory, file held open) is retried only
    // once more pipelines appear, not every second.
    if (!s.device.save_pipeline_cache(s.pipeline_cache_path))
        s.log.write("pipeline cache save failed: %s", s.pipeline_cache_path.c_str());
    s.saved_pipelines = pipelines;
    s.pipelines_changed_at = 0;
}

CUresult synchronize(State &s)
{
    API_LOG("sync");
    ++s.syncs;
    save_pipeline_cache(s);
    std::string error;
    log_stats(s);
    if (!s.device.synchronize(error)) return failure(s, "synchronize", error);
    s.pending_host_copies = false;
    return CUDA_SUCCESS;
}

// ---- private export tables ----------------------------------------------------------
bool guid_equal(const CUuuid *id, const unsigned char (&bytes)[16])
{
    return id != nullptr && std::memcmp(id->bytes, bytes, 16) == 0;
}

constexpr unsigned char kCudartGuid[16] = {0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
                                           0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
constexpr unsigned char kToolsRuntimeGuid[16] = {0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
                                                 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};
constexpr unsigned char kToolsTlsGuid[16] = {0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
                                             0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc};
constexpr unsigned char kClsGuid[16] = {0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
                                        0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};
constexpr unsigned char kContextBypassGuid[16] = {0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a,
                                                  0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86};
constexpr unsigned char kHeapGuid[16] = {0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a,
                                         0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae};

CUresult PXVK_CUDAAPI dark_get_module(CUmodule *module, const PxvkFatbinWrapper *wrapper);
CUresult PXVK_CUDAAPI dark_cudart_context(CUcontext *context, CUdevice device);
void PXVK_CUDAAPI dark_unknown_buffer1(void **pointer, size_t *size);
void PXVK_CUDAAPI dark_unknown_buffer2(void **pointer, size_t *size);
CUresult PXVK_CUDAAPI dark_cls_put(CUcontext context, void *key, void *value, ContextDestructor destructor);
CUresult PXVK_CUDAAPI dark_cls_delete(CUcontext context, void *key);
CUresult PXVK_CUDAAPI dark_cls_get(void **output, CUcontext context, void *key);
CUresult PXVK_CUDAAPI dark_context_create(CUcontext *context, unsigned int flags, CUdevice device);
CUresult PXVK_CUDAAPI dark_heap_alloc(const void **output, size_t first, size_t second);
CUresult PXVK_CUDAAPI dark_heap_free(const void *opaque, size_t *second);

const void *const *cudart_table()
{
    static void *table[13] = {reinterpret_cast<void *>(13 * sizeof(void *)),
                              reinterpret_cast<void *>(&dark_get_module),
                              reinterpret_cast<void *>(&dark_cudart_context)};
    return table;
}
const void *const *tools_runtime_table()
{
    static void *table[7] = {reinterpret_cast<void *>(7 * sizeof(void *)), nullptr,
                             reinterpret_cast<void *>(&dark_unknown_buffer1), nullptr, nullptr, nullptr,
                             reinterpret_cast<void *>(&dark_unknown_buffer2)};
    return table;
}
const void *const *tools_tls_table()
{
    static void *table[4] = {reinterpret_cast<void *>(4 * sizeof(void *))};
    return table;
}
const void *const *cls_table()
{
    static void *table[4] = {reinterpret_cast<void *>(&dark_cls_put), reinterpret_cast<void *>(&dark_cls_delete),
                             reinterpret_cast<void *>(&dark_cls_get), nullptr};
    return table;
}
const void *const *context_bypass_table()
{
    static void *table[2] = {reinterpret_cast<void *>(2 * sizeof(void *)),
                             reinterpret_cast<void *>(&dark_context_create)};
    return table;
}
const void *const *heap_table()
{
    static void *table[3] = {reinterpret_cast<void *>(3 * sizeof(void *)),
                             reinterpret_cast<void *>(&dark_heap_alloc), reinterpret_cast<void *>(&dark_heap_free)};
    return table;
}

CUresult create_context(State &s, CUcontext *output, unsigned flags, CUdevice device)
{
    if (output == nullptr || device != 0) return CUDA_ERROR_INVALID_VALUE;
    Context *context = new Context;
    context->flags = flags;
    s.contexts.insert(context);
    *output = reinterpret_cast<CUcontext>(context);
    set_current_context(*output);
    s.log.write("context create %p flags=%u", static_cast<void *>(context), flags);
    return CUDA_SUCCESS;
}

Context *context_of(State &s, CUcontext handle)
{
    Context *c = reinterpret_cast<Context *>(handle);
    return s.contexts.count(c) ? c : nullptr;
}

#define LOCK State &s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex)
#define REQUIRE do { const CUresult r_ = require(s); if (r_ != CUDA_SUCCESS) return r_; } while (0)

} // namespace

// ============================================================================
// Initialization and device queries
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuInit(unsigned int flags)
{
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    return initialize(s);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDriverGetVersion(int *version)
{
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    *version = 6050; // the driver API level of the classic CUDART that PhysX ships
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetCount(int *count)
{
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    *count = 1;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
    if (!device || ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    LOCK;
    REQUIRE;
    *device = 0;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetName(char *name, int length, CUdevice device)
{
    if (!name || length <= 0 || device != 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    const std::string text = "pxvk (" + s.device.caps().name + ")";
    std::strncpy(name, text.c_str(), static_cast<std::size_t>(length));
    name[length - 1] = '\0';
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice device)
{
    if (!bytes || device != 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    *bytes = static_cast<size_t>(std::min<std::uint64_t>(s.heap_limit, 0xFFFFFFFFu));
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice device) { return cuDeviceTotalMem_v2(bytes, device); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes)
{
    LOCK;
    REQUIRE;
    std::uint64_t in_use = 0;
    for (const auto &u : s.used) in_use += u.second;
    if (total_bytes) *total_bytes = static_cast<size_t>(std::min<std::uint64_t>(s.heap_limit, 0xFFFFFFFFu));
    if (free_bytes) *free_bytes = static_cast<size_t>(std::min<std::uint64_t>(s.heap_limit - in_use, 0xFFFFFFFFu));
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetInfo(size_t *f, size_t *t) { return cuMemGetInfo_v2(f, t); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice device)
{
    if (!major || !minor || device != 0) return CUDA_ERROR_INVALID_VALUE;
    *major = 5;
    *minor = 2;
    return CUDA_SUCCESS;
}

// The virtual device is an sm_52 GPU. Attribute values mirror what a real
// NVIDIA driver reports (CUDART validates e.g. texture bindings against the
// texture limits, so unmodelled attributes must not read as zero), adjusted
// to an sm_52 persona and clamped to what the Vulkan device can run.
const int kPersona[91] = {0, 1024, 1024, 1024, 64, 2147483647, 65535, 65535, 49152, 65536, 32, 2147483647, 65536, 1500000, 512, 1, 16, 1, 0, 1, 0, 131072, 131072, 65536, 16384, 16384, 16384, 32768, 32768, 2048, 512, 1, 0, 1, 0, 0, 3500000, 256, 2097152, 2048, 1, 0, 32768, 2048, 1, 32768, 32768, 8192, 8192, 32768, 0, 32, 32768, 32768, 2046, 32768, 131072, 65536, 16384, 16384, 16384, 32768, 2048, 32768, 32768, 2048, 32768, 32768, 2046, 268435456, 131072, 65000, 2097120, 32768, 32768, 5, 2, 32768, 1, 1, 1, 98304, 65536, 0, 0, 0, 0, 64, 0, 0, 1};

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetAttribute(int *output, CUdevice_attribute attribute, CUdevice device)
{
    if (!output || device != 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    const auto &caps = s.device.caps();
    if (attribute <= 0 || attribute > 90) {
        *output = 0;
        return CUDA_ERROR_INVALID_VALUE;
    }
    int value = kPersona[attribute];
    switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        value = static_cast<int>(std::min<std::uint32_t>(1024, caps.max_workgroup_invocations));
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
        value = static_cast<int>(std::min<std::uint32_t>(
            static_cast<std::uint32_t>(value), caps.max_workgroup_size[attribute - CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X]));
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
        value = static_cast<int>(std::min<std::uint32_t>(
            static_cast<std::uint32_t>(value), caps.max_workgroup_count[attribute - CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X]));
        break;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: value = caps.discrete ? 0 : 1; break;
    default: break;
    }
    *output = value;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceGetProperties(CUdevprop *p, CUdevice device)
{
    if (!p || device != 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    std::memset(p, 0, sizeof(*p));
    cuDeviceGetAttribute(&p->maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0);
    cuDeviceGetAttribute(&p->maxThreadsDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
    cuDeviceGetAttribute(&p->maxThreadsDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, 0);
    cuDeviceGetAttribute(&p->maxThreadsDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, 0);
    cuDeviceGetAttribute(&p->maxGridSize[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0);
    cuDeviceGetAttribute(&p->maxGridSize[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, 0);
    cuDeviceGetAttribute(&p->maxGridSize[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, 0);
    p->sharedMemPerBlock = 49152;
    p->totalConstantMemory = 65536;
    p->SIMDWidth = 32;
    p->memPitch = 0x7fffffff;
    p->regsPerBlock = 65536;
    p->clockRate = 1500000;
    p->textureAlign = 512;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuDeviceCanAccessPeer(int *can, CUdevice, CUdevice)
{
    if (!can) return CUDA_ERROR_INVALID_VALUE;
    *can = 0;
    return CUDA_SUCCESS;
}

// ============================================================================
// Contexts
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device)
{
    LOCK;
    REQUIRE;
    return create_context(s, context, flags, device);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxCreate(CUcontext *c, unsigned int f, CUdevice d) { return cuCtxCreate_v2(c, f, d); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetCurrent(CUcontext context)
{
    LOCK;
    if (context && !context_of(s, context)) return CUDA_ERROR_INVALID_CONTEXT;
    set_current_context(context);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetCurrent(CUcontext *context)
{
    if (!context) return CUDA_ERROR_INVALID_VALUE;
    *context = current_context();
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPushCurrent_v2(CUcontext context)
{
    LOCK;
    if (!context_of(s, context)) return CUDA_ERROR_INVALID_CONTEXT;
    t_context_stack.push_back(context);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPushCurrent(CUcontext c) { return cuCtxPushCurrent_v2(c); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPopCurrent_v2(CUcontext *context)
{
    if (t_context_stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (context) *context = t_context_stack.back();
    t_context_stack.pop_back();
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxPopCurrent(CUcontext *c) { return cuCtxPopCurrent_v2(c); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetApiVersion(CUcontext context, unsigned int *version)
{
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    if (!context) context = current_context();
    if (!context_of(s, context)) return CUDA_ERROR_INVALID_CONTEXT;
    *version = 3020;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetDevice(CUdevice *device)
{
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    if (!context_of(s, current_context())) return CUDA_ERROR_INVALID_CONTEXT;
    *device = 0;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSynchronize(void)
{
    LOCK;
    REQUIRE;
    return synchronize(s);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDetach(CUcontext context)
{
    State &s = state();
    std::map<std::uintptr_t, ClsValue> callbacks;
    {
        std::lock_guard<std::recursive_mutex> lock(s.mutex);
        if (!context) context = current_context();
        Context *c = context_of(s, context);
        if (!c) return CUDA_ERROR_INVALID_CONTEXT;
        synchronize(s);
        // Like CUDA, destroying a context unloads its modules. PhysX relies
        // on this: it never calls cuModuleUnload, it destroys the context
        // on every scene change and registers its modules again.
        std::vector<Module *> owned;
        for (Module *m : s.live_modules)
            if (m->context == context) owned.push_back(m);
        for (Module *m : owned) unload_module(s, m);
        callbacks.swap(c->cls);
        s.contexts.erase(c);
        delete c;
        for (auto &entry : t_context_stack)
            if (entry == context) entry = nullptr;
        s.log.write("context destroy %p", static_cast<void *>(context));
    }
    for (const auto &item : callbacks)
        if (item.second.destructor)
            item.second.destructor(context, reinterpret_cast<void *>(item.first), item.second.value);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDestroy_v2(CUcontext c) { return cuCtxDetach(c); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxDestroy(CUcontext c) { return cuCtxDetach(c); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetCacheConfig(CUfunc_cache) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *config)
{
    if (config) *config = 0;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *config)
{
    if (config) *config = 1; // 4-byte banks
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetLimit(size_t *value, int limit)
{
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    switch (limit) {
    case 0: *value = 1024; return CUDA_SUCCESS;             // stack size
    case 1: *value = 1048576; return CUDA_SUCCESS;          // printf FIFO
    case 2: *value = 8 * 1024 * 1024; return CUDA_SUCCESS;  // malloc heap
    default: return CUDA_ERROR_INVALID_VALUE;
    }
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxSetLimit(int, size_t) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuCtxGetStreamPriorityRange(int *least, int *greatest)
{
    if (least) *least = 0;
    if (greatest) *greatest = 0;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetExportTable(const void **output, const CUuuid *id)
{
    if (!output || !id) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    if (guid_equal(id, kCudartGuid)) *output = cudart_table();
    else if (guid_equal(id, kToolsRuntimeGuid)) *output = tools_runtime_table();
    else if (guid_equal(id, kToolsTlsGuid)) *output = tools_tls_table();
    else if (guid_equal(id, kClsGuid)) *output = cls_table();
    else if (guid_equal(id, kContextBypassGuid)) *output = context_bypass_table();
    else if (guid_equal(id, kHeapGuid)) *output = heap_table();
    else {
        s.log.write("unknown export table %02x%02x%02x%02x...", id->bytes[0], id->bytes[1], id->bytes[2], id->bytes[3]);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

// ============================================================================
// Modules
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image)
{
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    return load_image(s, image, module);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int, void *, void **)
{
    return cuModuleLoadData(module, image);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *image)
{
    return cuModuleLoadData(module, image);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleUnload(CUmodule handle)
{
    LOCK;
    Module *m = module_of(s, handle);
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    synchronize(s);
    unload_module(s, m);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetFunction(CUfunction *output, CUmodule handle, const char *name)
{
    if (!output || !name) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Module *m = module_of(s, handle);
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    auto found = m->functions.find(name);
    if (found == m->functions.end()) {
        const ptx::Function *decl = m->ast.find(name);
        if (!decl || !decl->is_entry || !decl->has_body) return CUDA_ERROR_NOT_FOUND;
        auto f = std::make_unique<Function>();
        f->module = m;
        f->name = name;
        s.functions.insert(f.get());
        found = m->functions.emplace(name, std::move(f)).first;
    }
    *output = reinterpret_cast<CUfunction>(found->second.get());
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *pointer, size_t *bytes, CUmodule handle, const char *name)
{
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Module *m = module_of(s, handle);
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    const auto found = m->globals.find(name);
    if (found == m->globals.end()) return CUDA_ERROR_NOT_FOUND;
    if (pointer) *pointer = found->second.first;
    if (bytes) *bytes = found->second.second;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetGlobal(CUdeviceptr *p, size_t *b, CUmodule m, const char *n)
{
    return cuModuleGetGlobal_v2(p, b, m, n);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuModuleGetTexRef(CUtexref *output, CUmodule handle, const char *name)
{
    if (!output || !name) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Module *m = module_of(s, handle);
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    const auto found = m->texrefs.find(name);
    if (found == m->texrefs.end()) return CUDA_ERROR_NOT_FOUND;
    *output = reinterpret_cast<CUtexref>(found->second.get());
    return CUDA_SUCCESS;
}

// ============================================================================
// Memory
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes)
{
    if (!pointer || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    std::uint32_t address = 0;
    // Match the NVIDIA driver's alignment: applications sub-allocate inside
    // large blocks and rely on the result meeting the 512-byte texture
    // alignment (CUDART's cudaBindTexture rejects misaligned pointers before
    // calling the driver). Large blocks get 2 MiB alignment like the native
    // large-page allocator.
    const std::uint32_t alignment = bytes >= (2u << 20) ? (2u << 20) : 4096u;
    const CUresult r = allocate(s, bytes, alignment, address);
    if (r != CUDA_SUCCESS) {
        s.log.write("cuMemAlloc failed bytes=%u result=%d", static_cast<unsigned>(bytes), r);
        return r;
    }
    // The NVIDIA driver hands out zeroed memory in practice and PhysX relies
    // on it: SampleParticleFluid frees and reallocates its pool on a scene
    // change and diverged from native while the reused block still held the
    // previous scene's data.
    std::string error;
    if (!s.device.fill(address, 0, bytes, error)) {
        release(s, address);
        return failure(s, "cuMemAlloc clear", error);
    }
    *pointer = address;
    s.log.write("cuMemAlloc address=%08x bytes=%u", address, static_cast<unsigned>(bytes));
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAlloc(CUdeviceptr *p, size_t b) { return cuMemAlloc_v2(p, b); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFree_v2(CUdeviceptr pointer)
{
    if (pointer == 0) return CUDA_SUCCESS;
    LOCK;
    // Pending work may still use the memory; the batch keeps executing in
    // order, and reuse is ordered after it, so no synchronization is needed.
    return release(s, pointer) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFree(CUdeviceptr p) { return cuMemFree_v2(p); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *bytes, CUdeviceptr pointer)
{
    LOCK;
    auto it = s.used.upper_bound(pointer);
    if (it == s.used.begin()) return CUDA_ERROR_NOT_FOUND;
    --it;
    if (pointer >= it->first + it->second) return CUDA_ERROR_NOT_FOUND;
    if (base) *base = it->first;
    if (bytes) *bytes = it->second;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemGetAddressRange(CUdeviceptr *b, size_t *n, CUdeviceptr p)
{
    return cuMemGetAddressRange_v2(b, n, p);
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemHostAlloc(void **pointer, size_t bytes, unsigned int flags)
{
    if (!pointer || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    REQUIRE;
    void *mapped = nullptr;
    std::string error;
    vk::HostBuffer *buffer = s.device.allocate_host(bytes, &mapped, error);
    if (!buffer) return failure(s, "cuMemHostAlloc", error, CUDA_ERROR_OUT_OF_MEMORY);
    HostRegion region;
    region.buffer = buffer;
    region.pointer = static_cast<char *>(mapped);
    region.size = bytes;
    region.flags = flags;
    s.host[region.pointer] = region;
    *pointer = mapped;
    s.log.write("cuMemHostAlloc pointer=%p bytes=%u flags=%u", mapped, static_cast<unsigned>(bytes), flags);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAllocHost_v2(void **p, size_t b) { return cuMemHostAlloc(p, b, 0); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemAllocHost(void **p, size_t b) { return cuMemHostAlloc(p, b, 0); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemFreeHost(void *pointer)
{
    if (!pointer) return CUDA_SUCCESS;
    LOCK;
    const auto found = s.host.find(static_cast<char *>(pointer));
    if (found == s.host.end()) return CUDA_ERROR_INVALID_VALUE;
    s.device.free_host(found->second.buffer);
    s.host.erase(found);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemHostGetFlags(unsigned int *flags, void *pointer)
{
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    const HostRegion *region = pinned(s, pointer, 1);
    if (!region) return CUDA_ERROR_INVALID_VALUE;
    *flags = region->flags;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr d, const void *src, size_t n) { LOCK; return copy_htod(s, d, src, n, false); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoD(CUdeviceptr d, const void *src, size_t n) { return cuMemcpyHtoD_v2(d, src, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *src, size_t n, CUstream) { LOCK; return copy_htod(s, d, src, n, true); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr d, const void *src, size_t n, CUstream st) { return cuMemcpyHtoDAsync_v2(d, src, n, st); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n) { LOCK; return copy_dtoh(s, dst, src, n, false); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t n) { return cuMemcpyDtoH_v2(dst, src, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n, CUstream) { LOCK; return copy_dtoh(s, dst, src, n, true); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t n, CUstream st) { return cuMemcpyDtoHAsync_v2(dst, src, n, st); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr src, size_t n) { LOCK; return copy_dtod(s, d, src, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr src, size_t n) { return cuMemcpyDtoD_v2(d, src, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr src, size_t n, CUstream) { LOCK; return copy_dtod(s, d, src, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr d, CUdeviceptr src, size_t n, CUstream st) { return cuMemcpyDtoDAsync_v2(d, src, n, st); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8_v2(CUdeviceptr destination, unsigned char value, size_t bytes)
{
    LOCK;
    API_LOG("memset dst=%08x value=%u bytes=%u", destination, value, static_cast<unsigned>(bytes));
    if (bytes == 0) return CUDA_SUCCESS;
    if (!valid_range(s, destination, bytes)) return CUDA_ERROR_INVALID_VALUE;
    std::string error;
    if (!s.device.fill(destination, value, bytes, error)) return failure(s, "memset", error);
    return s.sync_everything && !s.device.synchronize(error) ? failure(s, "sync", error) : CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return cuMemsetD8_v2(d, v, n); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream) { return cuMemsetD8_v2(d, v, n); }

// ============================================================================
// Texture references
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddress_v2(size_t *offset, CUtexref handle, CUdeviceptr pointer, size_t bytes)
{
    LOCK;
    API_LOG("texbind ref=%p ptr=%08x bytes=%u", static_cast<void *>(handle), pointer, static_cast<unsigned>(bytes));
    TexRef *t = texref_of(s, handle);
    if (!t) return CUDA_ERROR_INVALID_HANDLE;
    if (bytes != 0 && !valid_range(s, pointer, bytes)) return CUDA_ERROR_INVALID_VALUE;
    t->address = pointer;
    t->bytes = static_cast<std::uint32_t>(bytes);
    // No hardware alignment requirement: fetches use the exact address.
    if (offset) *offset = 0;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddress(size_t *o, CUtexref t, CUdeviceptr p, size_t n) { return cuTexRefSetAddress_v2(o, t, p, n); }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFormat(CUtexref handle, CUarray_format format, int components)
{
    if (components < 1 || components > 4) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    TexRef *t = texref_of(s, handle);
    if (!t) return CUDA_ERROR_INVALID_HANDLE;
    // Only 32-bit element formats are lowered (u32, s32, f32).
    if (format != 0x03 && format != 0x0a && format != 0x20) {
        s.log.write("texture %s: unsupported format 0x%x", t->name.c_str(), format);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    t->format = format;
    t->components = components;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFlags(CUtexref handle, unsigned int flags)
{
    LOCK;
    TexRef *t = texref_of(s, handle);
    if (!t) return CUDA_ERROR_INVALID_HANDLE;
    // tex1Dfetch on linear memory uses integer coordinates and returns raw
    // elements; normalized coordinates/reads are not lowered.
    if (flags & ~1u) s.log.write("texture %s: ignoring flags 0x%x", t->name.c_str(), flags);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetFilterMode(CUtexref, CUfilter_mode) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref, CUfilter_mode) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref, float) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref, float, float) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref, unsigned int) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode) { return CUDA_SUCCESS; }

// ============================================================================
// Functions and launches
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncGetAttribute(int *output, CUfunction_attribute attribute, CUfunction handle)
{
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Function *f = function_of(s, handle);
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    CompiledVariant *v = variant(s, *f, 0);
    if (!v) return CUDA_ERROR_INVALID_PTX;
    const ptx::KernelInfo &info = v->kernel.info;
    switch (attribute) {
    case 0: { // MAX_THREADS_PER_BLOCK
        std::uint32_t limit = std::min<std::uint32_t>(1024, s.device.caps().max_workgroup_invocations);
        const std::uint32_t declared = info.maxntid[0] ? info.maxntid[0] * std::max(1u, info.maxntid[1]) * std::max(1u, info.maxntid[2]) : 0;
        if (declared) limit = std::min(limit, declared);
        *output = static_cast<int>(limit);
        break;
    }
    case 1: *output = static_cast<int>(info.static_shared_bytes); break;
    case 2: {
        std::uint32_t bytes = 0;
        for (const auto &name : info.symbols)
            for (const auto &v2 : f->module->ast.vars)
                if (v2.name == name && v2.space == ptx::Space::constant) bytes += static_cast<std::uint32_t>(v2.size_bytes());
        *output = static_cast<int>(bytes);
        break;
    }
    case 3: *output = static_cast<int>(info.local_bytes); break;
    case 4: *output = 32; break;  // registers (not modelled)
    case 5: *output = 52; break;  // PTX virtual architecture
    case 6: *output = 52; break;  // binary architecture
    default: *output = 0; break;
    }
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncSetCacheConfig(CUfunction, CUfunc_cache) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuFuncSetSharedMemConfig(CUfunction, CUsharedconfig) { return CUDA_SUCCESS; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, CUfunction, int block_size, size_t)
{
    if (!blocks || block_size <= 0) return CUDA_ERROR_INVALID_VALUE;
    *blocks = std::max(1, 2048 / block_size);
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuLaunchKernel(CUfunction handle, unsigned int grid_x, unsigned int grid_y,
                                                 unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                                                 unsigned int block_z, unsigned int shared_bytes, CUstream,
                                                 void **kernel_params, void **extra)
{
    LOCK;
    REQUIRE;
    Function *f = function_of(s, handle);
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    CompiledVariant *normal = variant(s, *f, 0);
    if (!normal) return CUDA_ERROR_INVALID_PTX;
    const ptx::KernelInfo &info = normal->kernel.info;

    // Launch block: packed parameters followed by the texture table.
    std::vector<std::uint8_t> block(info.launch_block_bytes, 0);
    if (extra) {
        const void *buffer = nullptr;
        std::size_t size = 0;
        for (std::size_t i = 0; extra[i] != nullptr; i += 2) {
            const auto key = reinterpret_cast<std::uintptr_t>(extra[i]);
            if (key == PXVK_LAUNCH_PARAM_BUFFER_POINTER) buffer = extra[i + 1];
            else if (key == PXVK_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<std::size_t *>(extra[i + 1]);
            else return CUDA_ERROR_INVALID_VALUE;
        }
        if (size < info.param_bytes || (size && !buffer)) return CUDA_ERROR_INVALID_VALUE;
        if (info.param_bytes) std::memcpy(block.data(), buffer, info.param_bytes);
    } else if (kernel_params) {
        for (std::size_t i = 0; i < info.params.size(); ++i)
            std::memcpy(block.data() + info.params[i].first, kernel_params[i], info.params[i].second);
    } else if (info.param_bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (std::size_t t = 0; t < info.textures.size(); ++t) {
        std::uint32_t entry[4] = {0, 0, 0, 0};
        const auto found = f->module->texrefs.find(info.textures[t]);
        if (found != f->module->texrefs.end()) {
            const TexRef &ref = *found->second;
            entry[0] = ref.address;
            entry[1] = ref.bytes / (4u * static_cast<std::uint32_t>(ref.components));
            entry[2] = static_cast<std::uint32_t>(ref.components);
        }
        std::memcpy(block.data() + info.texture_table_offset + t * ptx::kTextureEntryBytes, entry, sizeof(entry));
    }
    for (std::size_t i = 0; i < info.symbols.size(); ++i) {
        const auto found = f->module->globals.find(info.symbols[i]);
        const std::uint32_t address = found != f->module->globals.end() ? found->second.first : 0;
        std::memcpy(block.data() + info.symbol_table_offset + 4 * i, &address, 4);
    }

    vk::DispatchShape shape;
    shape.grid[0] = grid_x;
    shape.grid[1] = grid_y;
    shape.grid[2] = grid_z;
    shape.block[0] = block_x;
    shape.block[1] = block_y;
    shape.block[2] = block_z;
    shape.shared_bytes = info.static_shared_bytes + shared_bytes;
    if (shape.shared_bytes > 49152) return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    CompiledVariant *v = normal;
    if (shape.shared_bytes > s.device.caps().max_shared_bytes) {
        v = variant(s, *f, 1);
        if (!v) return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    const std::uint32_t grid_dims[3] = {grid_x, grid_y, grid_z}, block_dims[3] = {block_x, block_y, block_z};
    const unsigned capture = capture_begin(s, *f, grid_dims, block_dims, shared_bytes, block.data(), info.param_bytes);
    std::string error;
    if (!s.device.dispatch(v->shader, shape, block.data(), block.size(), error, f->name.c_str())) {
        s.log.write("launch %s grid=%ux%ux%u block=%ux%ux%u shared=%u failed: %s", f->name.c_str(), grid_x, grid_y,
                    grid_z, block_x, block_y, block_z, shared_bytes, error.c_str());
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    if (capture) capture_snapshot(s, capture, "after");
    if (f->launches++ == 0)
        s.log.write("first launch %s grid=%ux%ux%u block=%ux%ux%u shared=%u%s", f->name.c_str(), grid_x, grid_y, grid_z,
                    block_x, block_y, block_z, shared_bytes, v == normal ? "" : " (storage shared)");
    ++s.launches;
    API_LOG("launch %.40s grid=%ux%ux%u block=%u shared=%u", f->name.c_str(), grid_x, grid_y, grid_z, block_x, shared_bytes);
    if (s.sync_everything && !s.device.synchronize(error)) return failure(s, "sync", error);
    return CUDA_SUCCESS;
}

// ============================================================================
// Streams and events: every stream shares the single ordered queue.
// ============================================================================

PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamCreate(CUstream *stream, unsigned int flags)
{
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Stream *st = new Stream;
    st->flags = flags;
    s.streams.insert(st);
    *stream = reinterpret_cast<CUstream>(st);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamCreateWithPriority(CUstream *stream, unsigned int flags, int)
{
    return cuStreamCreate(stream, flags);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamDestroy_v2(CUstream stream)
{
    LOCK;
    Stream *st = reinterpret_cast<Stream *>(stream);
    if (!s.streams.erase(st)) return CUDA_ERROR_INVALID_HANDLE;
    delete st;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamDestroy(CUstream stream) { return cuStreamDestroy_v2(stream); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamSynchronize(CUstream st) { LOCK; REQUIRE; API_LOG("stream_sync %p", static_cast<void *>(st)); return synchronize(s); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamQuery(CUstream st) { LOCK; REQUIRE; API_LOG("stream_query %p", static_cast<void *>(st)); return synchronize(s); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamWaitEvent(CUstream, CUevent, unsigned int) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int *flags)
{
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Stream *st = reinterpret_cast<Stream *>(stream);
    *flags = s.streams.count(st) ? st->flags : 0;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuStreamGetPriority(CUstream, int *priority)
{
    if (priority) *priority = 0;
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventCreate(CUevent *event, unsigned int)
{
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Event *e = new Event;
    s.events.insert(e);
    *event = reinterpret_cast<CUevent>(e);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventDestroy_v2(CUevent event)
{
    LOCK;
    Event *e = reinterpret_cast<Event *>(event);
    if (!s.events.erase(e)) return CUDA_ERROR_INVALID_HANDLE;
    delete e;
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventDestroy(CUevent event) { return cuEventDestroy_v2(event); }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventRecord(CUevent event, CUstream)
{
    LOCK;
    Event *e = reinterpret_cast<Event *>(event);
    if (!s.events.count(e)) return CUDA_ERROR_INVALID_HANDLE;
    e->recorded = true;
    API_LOG("event_record %p", static_cast<void *>(event));
    QueryPerformanceCounter(&e->time);
    return CUDA_SUCCESS;
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventQuery(CUevent event)
{
    LOCK;
    API_LOG("event_query %p", static_cast<void *>(event));
    if (!s.events.count(reinterpret_cast<Event *>(event))) return CUDA_ERROR_INVALID_HANDLE;
    return synchronize(s);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventSynchronize(CUevent event)
{
    LOCK;
    API_LOG("event_sync %p", static_cast<void *>(event));
    if (!s.events.count(reinterpret_cast<Event *>(event))) return CUDA_ERROR_INVALID_HANDLE;
    return synchronize(s);
}
PXVK_EXPORT CUresult PXVK_CUDAAPI cuEventElapsedTime(float *ms, CUevent start, CUevent end)
{
    if (!ms) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    Event *a = reinterpret_cast<Event *>(start), *b = reinterpret_cast<Event *>(end);
    if (!s.events.count(a) || !s.events.count(b)) return CUDA_ERROR_INVALID_HANDLE;
    if (!a->recorded || !b->recorded) return CUDA_ERROR_NOT_READY;
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    *ms = static_cast<float>(double(b->time.QuadPart - a->time.QuadPart) * 1000.0 / double(frequency.QuadPart));
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuProfilerInitialize(const char *, const char *, int) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuProfilerStart(void) { return CUDA_SUCCESS; }
PXVK_EXPORT CUresult PXVK_CUDAAPI cuProfilerStop(void) { return CUDA_SUCCESS; }

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetErrorName(CUresult result, const char **text)
{
    if (!text) return CUDA_ERROR_INVALID_VALUE;
    switch (result) {
    case CUDA_SUCCESS: *text = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *text = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *text = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *text = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case CUDA_ERROR_NO_DEVICE: *text = "CUDA_ERROR_NO_DEVICE"; break;
    case CUDA_ERROR_INVALID_IMAGE: *text = "CUDA_ERROR_INVALID_IMAGE"; break;
    case CUDA_ERROR_INVALID_HANDLE: *text = "CUDA_ERROR_INVALID_HANDLE"; break;
    case CUDA_ERROR_NOT_FOUND: *text = "CUDA_ERROR_NOT_FOUND"; break;
    case CUDA_ERROR_INVALID_PTX: *text = "CUDA_ERROR_INVALID_PTX"; break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: *text = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *text = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *text = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

PXVK_EXPORT CUresult PXVK_CUDAAPI cuGetErrorString(CUresult result, const char **text)
{
    return cuGetErrorName(result, text);
}

// ============================================================================
// Private export-table entry points
// ============================================================================

namespace {

CUresult PXVK_CUDAAPI dark_get_module(CUmodule *module, const PxvkFatbinWrapper *wrapper)
{
    if (!module || !wrapper || wrapper->magic != kWrapperMagic || !wrapper->data) return CUDA_ERROR_INVALID_IMAGE;
    LOCK;
    REQUIRE;
    return load_fatbin(s, wrapper->data, module);
}

CUresult PXVK_CUDAAPI dark_cudart_context(CUcontext *context, CUdevice device)
{
    if (!context || device != 0) return CUDA_ERROR_INVALID_VALUE;
    // CUDART only checks for success here and then creates its context
    // through the context-bypass table.
    *context = reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0xdeadbeefu));
    return CUDA_SUCCESS;
}

unsigned char g_unknown_buffer1[1024];
unsigned char g_unknown_buffer2[14];
void PXVK_CUDAAPI dark_unknown_buffer1(void **pointer, size_t *size)
{
    if (pointer) *pointer = g_unknown_buffer1;
    if (size) *size = sizeof(g_unknown_buffer1);
}
void PXVK_CUDAAPI dark_unknown_buffer2(void **pointer, size_t *size)
{
    if (pointer) *pointer = g_unknown_buffer2;
    if (size) *size = sizeof(g_unknown_buffer2);
}

CUresult PXVK_CUDAAPI dark_cls_put(CUcontext context, void *key, void *value, ContextDestructor destructor)
{
    LOCK;
    if (!context) context = current_context();
    Context *c = context_of(s, context);
    if (!c) return CUDA_ERROR_INVALID_HANDLE;
    c->cls[reinterpret_cast<std::uintptr_t>(key)] = {value, destructor};
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_cls_delete(CUcontext context, void *key)
{
    LOCK;
    if (!context) context = current_context();
    Context *c = context_of(s, context);
    if (!c) return CUDA_ERROR_INVALID_HANDLE;
    c->cls.erase(reinterpret_cast<std::uintptr_t>(key));
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_cls_get(void **output, CUcontext context, void *key)
{
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    LOCK;
    if (!context) context = current_context();
    Context *c = context_of(s, context);
    if (!c) return CUDA_ERROR_INVALID_HANDLE;
    const auto found = c->cls.find(reinterpret_cast<std::uintptr_t>(key));
    if (found == c->cls.end()) return CUDA_ERROR_INVALID_HANDLE;
    *output = found->second.value;
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_context_create(CUcontext *context, unsigned int flags, CUdevice device)
{
    return cuCtxCreate_v2(context, flags, device);
}

CUresult PXVK_CUDAAPI dark_heap_alloc(const void **output, size_t first, size_t second)
{
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    auto *record = new (std::nothrow) size_t[2];
    if (!record) return CUDA_ERROR_OUT_OF_MEMORY;
    record[0] = first;
    record[1] = second;
    *output = record;
    return CUDA_SUCCESS;
}

CUresult PXVK_CUDAAPI dark_heap_free(const void *opaque, size_t *second)
{
    if (!opaque) return CUDA_ERROR_INVALID_VALUE;
    const auto *record = static_cast<const size_t *>(opaque);
    if (second) *second = record[1];
    delete[] record;
    return CUDA_SUCCESS;
}

} // namespace

extern "C" BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        g_module = module;
        // No DisableThreadLibraryCalls: the statically linked CRT frees
        // thread_local state (the context stack) on thread detach.
        pxvk::runtime::install_driver_redirect(module);
    } else if (reason == DLL_PROCESS_DETACH) {
        pxvk::runtime::remove_driver_redirect();
    }
    return TRUE;
}

void pxvk_log_unsupported(const char *name)
{
    State &s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    static std::set<std::string> logged;
    if (logged.insert(name).second) s.log.write("unsupported call %s", name);
}
