// pxvk-replay: differential replay of native CUDA kernel captures on Vulkan.
//
//   pxvk-replay [--device N] [--no-native-ftz] [--time N] [--verbose]
//               [--mismatches] <capture-dir> [capture-ordinal...]
//
// Each capture (see tools/nvcuda-trace/src/capture.c) is replayed with the
// exact native memory image, parameters, texture bindings and module global
// addresses. The resulting memory is compared with the native result:
//   exact     words identical to native
//   close     differing words within 4 ULP when read as fp32
//   wrong     everything else (--verbose prints the first few); wrong_int
//             counts those that look like integers (zero exponent field),
//             which approximation differences cannot explain
// --mismatches prints every differing word with its grade.
#include "compiler.h"
#include "device.h"
#include "fatbin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Region {
    std::uint32_t address;
    std::vector<std::uint8_t> bytes;
};

struct Capture {
    std::string kernel;
    unsigned fatbin = 0, frame = 0;
    std::uint32_t grid[3] = {1, 1, 1}, block[3] = {1, 1, 1}, shared = 0;
    std::vector<std::uint8_t> params;
    std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> globals; // name -> base,size
    struct Tex {
        std::uint32_t base, bytes;
        int format, components;
    };
    std::map<std::string, Tex> textures;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> allocations;
};

std::vector<Region> read_sparse(const fs::path &path)
{
    std::vector<Region> out;
    std::ifstream in(path, std::ios::binary);
    std::uint32_t header[2];
    while (in.read(reinterpret_cast<char *>(header), 8)) {
        Region r;
        r.address = header[0];
        r.bytes.resize(header[1]);
        in.read(reinterpret_cast<char *>(r.bytes.data()), header[1]);
        out.push_back(std::move(r));
    }
    return out;
}

Capture read_meta(const fs::path &path)
{
    Capture c;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string key;
        s >> key;
        if (key == "kernel") s >> c.kernel;
        else if (key == "fatbin") s >> c.fatbin;
        else if (key == "frame") s >> c.frame;
        else if (key == "grid") s >> c.grid[0] >> c.grid[1] >> c.grid[2];
        else if (key == "block") s >> c.block[0] >> c.block[1] >> c.block[2];
        else if (key == "shared") s >> c.shared;
        else if (key == "params") {
            std::string hex;
            s >> hex;
            for (std::size_t i = 0; i + 1 < hex.size(); i += 2)
                c.params.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
        } else if (key == "global") {
            std::string name, base;
            std::uint32_t size;
            s >> name >> base >> size;
            c.globals[name] = {static_cast<std::uint32_t>(std::stoul(base, nullptr, 16)), size};
        } else if (key == "texref") {
            std::string name, base;
            Capture::Tex t{};
            s >> name >> base >> t.bytes >> t.format >> t.components;
            t.base = static_cast<std::uint32_t>(std::stoul(base, nullptr, 16));
            c.textures[name] = t;
        } else if (key == "allocation") {
            std::string base;
            std::uint32_t size;
            s >> base >> size;
            c.allocations.push_back({static_cast<std::uint32_t>(std::stoul(base, nullptr, 16)), size});
        }
    }
    return c;
}

int ulp_distance(std::uint32_t a, std::uint32_t b)
{
    float fa, fb;
    std::memcpy(&fa, &a, 4);
    std::memcpy(&fb, &b, 4);
    if (std::isnan(fa) || std::isnan(fb) || std::isinf(fa) || std::isinf(fb)) return 1 << 30;
    auto ordered = [](std::uint32_t v) -> std::int64_t {
        return (v & 0x80000000u) ? -static_cast<std::int64_t>(v & 0x7fffffffu) : static_cast<std::int64_t>(v);
    };
    const std::int64_t d = std::llabs(ordered(a) - ordered(b));
    return d > (1 << 30) ? (1 << 30) : static_cast<int>(d);
}

struct ModuleCache {
    pxvk::ptx::Module module;
    std::map<std::string, pxvk::ptx::CompiledKernel> kernels;
    std::map<std::string, pxvk::vk::Shader *> shaders;
    std::map<std::string, pxvk::ptx::CompiledKernel> storage_kernels; // shared memory in a buffer
    std::map<std::string, pxvk::vk::Shader *> storage_shaders;
};

} // namespace

int main(int argc, char **argv)
{
    std::string device_selector;
    bool native_ftz = true;
    bool force_emulate_warps = false; // emulate warp-synchronous code even with 32-lane subgroups
    bool force_native_ftz = false; // timing experiments: skip FTZ emulation even if unsupported
    bool verbose = false;
    bool all_mismatches = false;
    bool no_contraction = false;
    bool force_storage_shared = false;
    int timing_runs = 0;
    fs::path dir, ptx_override;
    std::string spirv_filter; // command run as "<filter> in.spv out.spv" on every kernel (experiments)
    std::vector<unsigned> wanted;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device_selector = argv[++i];
        else if (arg == "--no-native-ftz") native_ftz = false;
        else if (arg == "--force-native-ftz") force_native_ftz = true;
        else if (arg == "--emulate-warps") force_emulate_warps = true;
        else if (arg == "--no-contract") no_contraction = true;
        else if (arg == "--storage-shared") force_storage_shared = true;
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--mismatches") all_mismatches = true;
        else if (arg == "--time" && i + 1 < argc) timing_runs = std::atoi(argv[++i]);
        else if (arg == "--ptx" && i + 1 < argc) ptx_override = argv[++i];
        else if (arg == "--spirv-filter" && i + 1 < argc) spirv_filter = argv[++i];
        else if (dir.empty()) dir = arg;
        else wanted.push_back(static_cast<unsigned>(std::stoul(arg)));
    }
    if (dir.empty()) {
        std::fprintf(stderr, "usage: pxvk-replay [--device N] [--no-native-ftz] [--time N] [--verbose] [--mismatches] dir [ordinal...]\n");
        return 2;
    }
    // Missing native captures are an error, not "0/0 failed".
    const char *capture_hint = "Captures are recorded from native CUDA with tools/nvcuda-trace on an NVIDIA "
                               "machine and copied to other machines; see docs/testing.md.";
    if (!fs::is_directory(dir)) {
        std::fprintf(stderr, "capture folder not found: %s\n%s\n", dir.string().c_str(), capture_hint);
        return 2;
    }

    pxvk::vk::Device device;
    std::string error;
    if (!device.initialize(device_selector, error)) {
        std::fprintf(stderr, "vulkan: %s\n", error.c_str());
        return 2;
    }
    const auto &caps = device.caps();
    std::printf("device: %s (subgroup %u, native ftz %d, subgroup control %d, float atomics %d, max heap %llu MiB)\n",
                caps.name.c_str(), caps.subgroup_size, caps.native_ftz, caps.subgroup_control, caps.float_atomic_add,
                static_cast<unsigned long long>(caps.max_heap_bytes >> 20));
    pxvk::ptx::CompileOptions options;
    options.native_ftz = force_native_ftz || (native_ftz && caps.native_ftz);
    options.preserve_nan = caps.preserve_nan;
    options.emulate_warps = force_emulate_warps || caps.narrow_subgroups();
    options.int64 = caps.int64;
    options.float_atomic_add = caps.float_atomic_add;
    options.heap_window_shift = caps.heap_window_shift;
    options.heap_windows = caps.heap_windows;
    options.no_contraction = no_contraction;
    device.set_timing(timing_runs > 0);

    std::vector<fs::path> metas;
    for (const auto &entry : fs::directory_iterator(dir))
        if (entry.path().extension() == ".txt" && entry.path().filename().string().rfind("capture_", 0) == 0)
            metas.push_back(entry.path());
    std::sort(metas.begin(), metas.end());
    if (metas.empty()) {
        std::fprintf(stderr, "no captures (capture_*.txt) in %s\n%s\n", dir.string().c_str(), capture_hint);
        return 2;
    }

    std::map<unsigned, std::unique_ptr<ModuleCache>> modules;
    int failures = 0, total = 0;
    for (const fs::path &meta : metas) {
        // Any exception (corrupt capture, compile error, allocation failure)
        // fails this capture only, so the remaining ones are still graded.
        unsigned ordinal = 0;
        std::string short_name = meta.filename().string();
        bool counted = false;
        try {
        ordinal = static_cast<unsigned>(std::stoul(meta.stem().string().substr(8)));
        if (!wanted.empty() && std::find(wanted.begin(), wanted.end(), ordinal) == wanted.end()) continue;
        const Capture c = read_meta(meta);
        fs::path base = meta;
        base.replace_extension("");
        ++total;
        counted = true;
        short_name = c.kernel.substr(0, 48);
        auto fail = [&](const std::string &why) {
            std::printf("%06u %-48s FAIL %s\n", ordinal, short_name.c_str(), why.c_str());
            ++failures;
        };

        // Module and kernel compilation (cached per fatbin).
        auto &cache = modules[c.fatbin];
        try {
            if (!cache) {
                cache = std::make_unique<ModuleCache>();
                char name[32];
                std::snprintf(name, sizeof(name), "fatbin_%04u.bin", c.fatbin);
                std::ifstream in(dir / name, std::ios::binary);
                std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                const auto ptx = pxvk::fatbin::extract_ptx(bytes.data(), bytes.size());
                if (ptx.empty()) throw std::runtime_error("fatbin has no PTX");
                // The PTX the runtime would load: highest virtual architecture.
                const auto best = std::max_element(ptx.begin(), ptx.end(), [](const auto &a, const auto &b) {
                    return a.sm_version < b.sm_version;
                });
                cache->module = pxvk::ptx::parse(best->text);
                if (!ptx_override.empty()) {
                    // The override replaces only the module it was edited from:
                    // one that defines every entry of this fatbin.
                    std::ifstream override_in(ptx_override, std::ios::binary);
                    const std::string text((std::istreambuf_iterator<char>(override_in)), std::istreambuf_iterator<char>());
                    pxvk::ptx::Module edited = pxvk::ptx::parse(text);
                    bool covers = true;
                    for (const auto &f : cache->module.functions)
                        if (f.is_entry && !edited.find(f.name)) covers = false;
                    if (covers) cache->module = std::move(edited);
                }
            }
            if (!cache->kernels.count(c.kernel))
                cache->kernels[c.kernel] = pxvk::ptx::compile_kernel(cache->module, c.kernel, options);
        } catch (const std::exception &e) {
            fail(std::string("compile: ") + e.what());
            continue;
        }
        // Launches whose shared memory exceeds the device limit use the
        // storage-backed variant.
        const bool storage = force_storage_shared ||
            cache->kernels[c.kernel].info.static_shared_bytes + c.shared > caps.max_shared_bytes;
        auto &kernels = storage ? cache->storage_kernels : cache->kernels;
        auto &shaders = storage ? cache->storage_shaders : cache->shaders;
        if (!kernels.count(c.kernel)) {
            pxvk::ptx::CompileOptions variant = options;
            variant.shared_in_storage = true;
            kernels[c.kernel] = pxvk::ptx::compile_kernel(cache->module, c.kernel, variant);
        }
        auto &kernel = kernels[c.kernel];
        if (!shaders.count(c.kernel) && !spirv_filter.empty()) {
            const fs::path in = fs::temp_directory_path() / "pxvk-filter-in.spv";
            const fs::path out = fs::temp_directory_path() / "pxvk-filter-out.spv";
            std::ofstream(in, std::ios::binary)
                .write(reinterpret_cast<const char *>(kernel.spirv.data()), kernel.spirv.size() * 4);
            fs::remove(out);
            const std::string command = spirv_filter + " \"" + in.string() + "\" -o \"" + out.string() + "\"";
            if (std::system(("\"" + command + "\"").c_str()) != 0 || !fs::exists(out)) { // cmd.exe strips outer quotes
                fail("spirv filter failed");
                continue;
            }
            std::ifstream file(out, std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            kernel.spirv.assign(bytes.size() / 4, 0);
            std::memcpy(kernel.spirv.data(), bytes.data(), bytes.size() / 4 * 4);
        }
        if (!shaders.count(c.kernel)) {
            shaders[c.kernel] = device.create_shader(kernel.spirv, kernel.info.launch_block_bytes,
                                                     kernel.info.warp_synchronous, storage, error);
            if (!shaders[c.kernel]) {
                fail(error);
                continue;
            }
        }
        pxvk::vk::Shader *shader = shaders[c.kernel];

        // Memory image.
        const std::vector<Region> before = read_sparse(base.string() + ".before");
        const std::vector<Region> after = read_sparse(base.string() + ".after");
        std::uint64_t lo = UINT64_MAX, hi = 0;
        auto extend = [&](std::uint64_t a, std::uint64_t size) {
            lo = std::min(lo, a);
            hi = std::max(hi, a + size);
        };
        for (const auto &a : c.allocations) extend(a.first, a.second);
        for (const auto &g : c.globals) extend(g.second.first, g.second.second);
        for (const auto &r : before) extend(r.address, r.bytes.size());
        for (const auto &r : after) extend(r.address, r.bytes.size());
        lo &= ~std::uint64_t(255);
        // Keep 64 KiB unused below the image, like the runtime's heap
        // reserve: the compiler's base+offset addressing relies on it.
        lo = lo >= 0x10000 ? lo - 0x10000 : 0;
        // 1 MiB probe area for the "pxvk_debug" symbol after the image.
        const std::uint64_t debug_address = (hi + 255) & ~std::uint64_t(255);
        hi = debug_address + (1u << 20);
        const std::uint64_t span = (hi - lo + 255) & ~std::uint64_t(255);
        device.set_heap_base(static_cast<std::uint32_t>(lo));
        if (!device.ensure_heap(span, error) || !device.fill(0, 0, static_cast<std::size_t>(device.heap_size()), error)) {
            fail(error);
            continue;
        }
        bool uploaded = true;
        for (const auto &r : before)
            if (!(uploaded = device.upload(r.address - lo, r.bytes.data(), r.bytes.size(), error))) break;
        if (!uploaded) {
            fail("upload: " + error);
            continue;
        }

        // Launch block: params then the texture table.
        std::vector<std::uint8_t> launch(kernel.info.launch_block_bytes, 0);
        std::memcpy(launch.data(), c.params.data(), std::min<std::size_t>(c.params.size(), kernel.info.param_bytes));
        if (c.params.size() != kernel.info.param_bytes) {
            fail("parameter size mismatch: captured " + std::to_string(c.params.size()) + ", PTX " +
                 std::to_string(kernel.info.param_bytes));
            continue;
        }
        for (std::size_t t = 0; t < kernel.info.textures.size(); ++t) {
            const auto found = c.textures.find(kernel.info.textures[t]);
            std::uint32_t entry[4] = {0, 0, 0, 0};
            if (found != c.textures.end() && found->second.components > 0) {
                entry[0] = found->second.base;
                entry[1] = found->second.bytes / (4u * static_cast<std::uint32_t>(found->second.components));
                entry[2] = static_cast<std::uint32_t>(found->second.components);
            }
            std::memcpy(launch.data() + kernel.info.texture_table_offset + t * 16, entry, 16);
        }
        bool missing = false;
        for (std::size_t i = 0; i < kernel.info.symbols.size(); ++i) {
            const auto &name = kernel.info.symbols[i];
            std::uint32_t address = 0;
            if (name == "pxvk_debug") {
                address = static_cast<std::uint32_t>(debug_address);
            } else {
                const auto found = c.globals.find(name);
                if (found == c.globals.end()) {
                    fail("no captured address for symbol " + name);
                    missing = true;
                    break;
                }
                address = found->second.first;
            }
            std::memcpy(launch.data() + kernel.info.symbol_table_offset + 4 * i, &address, 4);
        }
        if (missing) continue;

        pxvk::vk::DispatchShape shape;
        std::copy(std::begin(c.grid), std::end(c.grid), shape.grid);
        std::copy(std::begin(c.block), std::end(c.block), shape.block);
        shape.shared_bytes = kernel.info.static_shared_bytes + c.shared;
        if (!device.dispatch(shader, shape, launch.data(), launch.size(), error)) {
            fail(error);
            continue;
        }
        std::vector<std::uint8_t> result(static_cast<std::size_t>(span));
        if (!device.download(result.data(), 0, result.size(), error)) {
            fail(error);
            continue;
        }

        // Expected image: zero + after records; and the before image for
        // counting what the kernel actually changed natively.
        std::vector<std::uint8_t> expected(static_cast<std::size_t>(span), 0), original(static_cast<std::size_t>(span), 0);
        for (const auto &r : after) std::memcpy(expected.data() + (r.address - lo), r.bytes.data(), r.bytes.size());
        for (const auto &r : before) std::memcpy(original.data() + (r.address - lo), r.bytes.data(), r.bytes.size());
        std::size_t changed = 0, exact_mismatch = 0, close = 0, near = 0, wrong = 0, wrong_int = 0, stray = 0;
        double max_rel = 0;
        int max_ulp = 0;
        std::vector<std::uint32_t> wrong_addresses;
        // --mismatches: every differing word with its grade.
        auto report = [&](std::size_t w, std::uint32_t e, std::uint32_t g, std::uint32_t o, const char *grade) {
            if (!all_mismatches) return;
            float fe, fg;
            std::memcpy(&fe, &e, 4);
            std::memcpy(&fg, &g, 4);
            std::printf("    %08x %-9s native=%08x (%g) vulkan=%08x (%g) before=%08x\n",
                        static_cast<unsigned>(lo + w), grade, e, fe, g, fg, o);
        };
        for (std::size_t w = 0; w + 4 <= debug_address - lo; w += 4) {
            std::uint32_t e, g, o;
            std::memcpy(&e, expected.data() + w, 4);
            std::memcpy(&g, result.data() + w, 4);
            std::memcpy(&o, original.data() + w, 4);
            if (e != o) ++changed;
            if (e == g) continue;
            ++exact_mismatch;
            if (e == o) ++stray; // native left it untouched
            // Memory images are untyped, and tolerances only make sense for
            // floats. Words with a zero exponent field are integers below
            // 2^23 (counts, indices, flags) or denormals, which PhysX's .ftz
            // arithmetic never produces: they must match exactly, or an
            // integer that is off by one would pass as "1 ULP". Signed zeros
            // are the exception.
            const auto exponent = [](std::uint32_t v) { return (v >> 23) & 0xffu; };
            const auto is_zero = [](std::uint32_t v) { return (v & 0x7fffffffu) == 0; };
            const bool zeros = is_zero(e) && is_zero(g);
            if (!zeros && ((exponent(e) == 0 && !is_zero(e)) || (exponent(g) == 0 && !is_zero(g)))) {
                ++wrong;
                ++wrong_int;
                if (wrong_addresses.size() < 6) wrong_addresses.push_back(static_cast<std::uint32_t>(lo + w));
                report(w, e, g, o, "wrong-int");
                continue;
            }
            const int ulp = ulp_distance(e, g);
            float fe, fg;
            std::memcpy(&fe, &e, 4);
            std::memcpy(&fg, &g, 4);
            // Scale by the pre-launch value as well: solver outputs are often
            // small deltas of large positions (catastrophic cancellation).
            float fo;
            std::memcpy(&fo, &o, 4);
            double scale = std::max(std::fabs(double(fe)), std::fabs(double(fg)));
            if (std::isfinite(fo)) scale = std::max(scale, std::fabs(double(fo)));
            const double rel = scale > 0 ? std::fabs(double(fe) - double(fg)) / scale : 0.0;
            if (ulp <= 4) {
                ++close;
                max_ulp = std::max(max_ulp, ulp);
                report(w, e, g, o, "ulp4");
            } else if (std::isfinite(fe) && std::isfinite(fg) &&
                       (rel <= 1e-3 ||
                        // Absolute closeness for float-sized values, and for
                        // a tiny float against exact zero (solver values
                        // converging to 0). Not between two tiny nonzero
                        // values: 32-bit pointers read as such floats, and
                        // any difference between them is below 1e-4.
                        (std::fabs(double(fe) - double(fg)) <= 1e-4 &&
                         ((exponent(e) >= 0x60u && exponent(g) >= 0x60u) || is_zero(e) || is_zero(g))))) {
                ++near;
                max_rel = std::max(max_rel, rel);
                report(w, e, g, o, "near");
            } else {
                ++wrong;
                if (wrong_addresses.size() < 6) wrong_addresses.push_back(static_cast<std::uint32_t>(lo + w));
                report(w, e, g, o, "wrong");
            }
        }
        for (std::uint64_t a = debug_address; a < hi; a += 4) {
            std::uint32_t v;
            std::memcpy(&v, result.data() + (a - lo), 4);
            if (v != 0) {
                float f;
                std::memcpy(&f, &v, 4);
                std::printf("    debug[%u] = %08x (%u, %g)\n", static_cast<unsigned>((a - debug_address) / 4), v, v, f);
            }
        }
        std::string timing;
        if (timing_runs > 0) {
            double best = 1e30;
            for (int run = 0; run < timing_runs; ++run) {
                for (const auto &r : before) device.upload(r.address - lo, r.bytes.data(), r.bytes.size(), error);
                device.dispatch(shader, shape, launch.data(), launch.size(), error);
                device.synchronize(error);
                for (double t : device.last_dispatch_times()) best = std::min(best, t);
            }
            char text[64];
            std::snprintf(text, sizeof(text), " gpu=%.3fms", best);
            timing = text;
        }
        const bool ok = wrong == 0;
        if (!ok) ++failures;
        std::printf("%06u %-48s %s f=%u changed=%zu mismatched=%zu ulp4=%zu(max %d) near=%zu(rel %.1e) wrong=%zu wrong_int=%zu stray=%zu%s\n",
                    ordinal, short_name.c_str(), !ok ? "BAD  " : exact_mismatch == 0 ? "EXACT" : "ok   ",
                    c.frame, changed, exact_mismatch, close, max_ulp, near, max_rel, wrong, wrong_int, stray,
                    timing.c_str());
        if (!ok && verbose) {
            for (std::uint32_t a : wrong_addresses) {
                std::uint32_t e, g, o;
                std::memcpy(&e, expected.data() + (a - lo), 4);
                std::memcpy(&g, result.data() + (a - lo), 4);
                std::memcpy(&o, original.data() + (a - lo), 4);
                float fe, fg;
                std::memcpy(&fe, &e, 4);
                std::memcpy(&fg, &g, 4);
                std::printf("    %08x native=%08x (%g) vulkan=%08x (%g) before=%08x\n", a, e, fe, g, fg, o);
            }
        }
        std::fflush(stdout);
        } catch (const std::exception &e) {
            if (!counted) ++total;
            ++failures;
            std::printf("%06u %-48s FAIL %s\n", ordinal, short_name.c_str(), e.what());
            std::fflush(stdout);
        }
    }
    if (total == 0) {
        std::fprintf(stderr, "none of the requested capture ordinals exist in %s\n", dir.string().c_str());
        return 2;
    }
    std::printf("%d/%d captures failed\n", failures, total);
    return failures == 0 ? 0 : 1;
}
