// pxvk-ptxc: compile PTX kernel entries to Vulkan SPIR-V.
//   pxvk-ptxc [--native-ftz] [-o <dir>] [--entry <name>]... <module.ptx>
// Without --entry every .entry in the module is compiled. Prints one line per
// entry with its interface metadata; returns nonzero if any entry failed.
#include "compiler.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    std::string input, out_dir;
    std::vector<std::string> entries;
    pxvk::ptx::CompileOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--entry" && i + 1 < argc) entries.push_back(argv[++i]);
        else if (arg == "--native-ftz") options.native_ftz = true;
        else if (arg == "--emulate-warps") options.emulate_warps = true;
        else if (arg == "--float-atomics") options.float_atomic_add = true;
        else if (arg == "--shader-clock") options.shader_clock = true;
        else if (arg == "--split-heap") options.heap_window_shift = 27, options.heap_windows = 8; // as on a 128 MiB device
        else input = arg;
    }
    if (input.empty()) {
        std::fprintf(stderr, "usage: pxvk-ptxc [--native-ftz] [--emulate-warps] [--float-atomics] [--shader-clock] [--split-heap] [-o dir] [--entry name]... module.ptx\n");
        return 2;
    }
    std::ifstream file(input, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    pxvk::ptx::Module module;
    try {
        module = pxvk::ptx::parse(text.str());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: %s\n", input.c_str(), e.what());
        return 1;
    }
    if (entries.empty())
        for (const auto &f : module.functions)
            if (f.is_entry && f.has_body) entries.push_back(f.name);
    int failures = 0;
    int index = 0;
    for (const std::string &entry : entries) {
        try {
            const auto kernel = pxvk::ptx::compile_kernel(module, entry, options);
            const auto &k = kernel.info;
            if (!out_dir.empty()) {
                const std::string path = out_dir + "/" + std::to_string(index) + ".spv";
                std::ofstream out(path, std::ios::binary);
                out.write(reinterpret_cast<const char *>(kernel.spirv.data()),
                          static_cast<std::streamsize>(kernel.spirv.size() * 4));
                if (!out.flush()) throw std::runtime_error("cannot write " + path);
            }
            std::printf("ok %s words=%zu params=%u shared=%u%s local=%u symbols=%zu textures=%zu%s%s%s\n",
                        entry.c_str(), kernel.spirv.size(), k.param_bytes, k.static_shared_bytes,
                        k.uses_dynamic_shared ? "+dyn" : "", k.local_bytes, k.symbols.size(),
                        k.textures.size(), k.uses_barrier ? " barrier" : "",
                        k.warp_synchronous ? " warp-sync" : "", k.uses_int64 ? " int64" : "");
        } catch (const std::exception &e) {
            std::printf("FAIL %s: %s\n", entry.c_str(), e.what());
            ++failures;
        }
        ++index;
    }
    return failures == 0 ? 0 : 1;
}
