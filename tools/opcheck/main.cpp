// pxvk-opcheck: per-instruction differential testing, native CUDA vs pxvk.
//
//   pxvk-opcheck [--device N] [--count N] [--filter substring] [--no-native]
//
// For every instruction variant in the table below a small PTX kernel is
// generated. Each thread reads a 16-byte input record (a, b, c, d words),
// applies the instruction and writes a 16-byte result record. The same
// kernel runs through the 32-bit NVIDIA driver (PTX JIT) and through the pxvk
// compiler on Vulkan, and the outputs are compared bit for bit. Inputs mix
// random bit patterns, ordinary floats and special values.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "compiler.h"
#include "device.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// ---- minimal CUDA driver API --------------------------------------------
using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUdeviceptr = std::uint32_t;

struct Cuda {
    HMODULE dll = nullptr;
    CUresult(__stdcall *cuInit)(unsigned) = nullptr;
    CUresult(__stdcall *cuDeviceGet)(CUdevice *, int) = nullptr;
    CUresult(__stdcall *cuCtxCreate)(CUcontext *, unsigned, CUdevice) = nullptr;
    CUresult(__stdcall *cuModuleLoadDataEx)(CUmodule *, const void *, unsigned, void *, void **) = nullptr;
    CUresult(__stdcall *cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
    CUresult(__stdcall *cuModuleUnload)(CUmodule) = nullptr;
    CUresult(__stdcall *cuMemAlloc)(CUdeviceptr *, std::size_t) = nullptr;
    CUresult(__stdcall *cuMemcpyHtoD)(CUdeviceptr, const void *, std::size_t) = nullptr;
    CUresult(__stdcall *cuMemcpyDtoH)(void *, CUdeviceptr, std::size_t) = nullptr;
    CUresult(__stdcall *cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, void *, void **, void **) = nullptr;
    CUresult(__stdcall *cuCtxSynchronize)() = nullptr;
    CUcontext context = nullptr;

    bool load()
    {
        dll = LoadLibraryA("nvcuda.dll");
        if (!dll) return false;
        auto get = [&](auto &fn, const char *name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
                reinterpret_cast<void *>(GetProcAddress(dll, name)));
            return fn != nullptr;
        };
        if (!get(cuInit, "cuInit") || !get(cuDeviceGet, "cuDeviceGet") ||
            !get(cuCtxCreate, "cuCtxCreate_v2") || !get(cuModuleLoadDataEx, "cuModuleLoadDataEx") ||
            !get(cuModuleGetFunction, "cuModuleGetFunction") || !get(cuModuleUnload, "cuModuleUnload") ||
            !get(cuMemAlloc, "cuMemAlloc_v2") || !get(cuMemcpyHtoD, "cuMemcpyHtoD_v2") ||
            !get(cuMemcpyDtoH, "cuMemcpyDtoH_v2") || !get(cuLaunchKernel, "cuLaunchKernel") ||
            !get(cuCtxSynchronize, "cuCtxSynchronize"))
            return false;
        CUdevice device = 0;
        return cuInit(0) == 0 && cuDeviceGet(&device, 0) == 0 && cuCtxCreate(&context, 0, device) == 0;
    }
};

// ---- test generation ----------------------------------------------------------
// Operand classes: f = f32, i = 32-bit int, h = 16-bit int, b = 8-bit int,
// l = 64-bit int, p = predicate.
struct Variant {
    const char *text;    // instruction text with {d},{a},{b},{c} placeholders
    char dst;
    const char *srcs;    // operand classes of a, b, c
};

const Variant kVariants[] = {
    {"add.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"sub.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"mul.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"fma.rn.ftz.f32 {d}, {a}, {b}, {c};", 'f', "fff"},
    {"div.full.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"div.rn.f32 {d}, {a}, {b};", 'f', "ff"},
    {"rcp.approx.ftz.f32 {d}, {a};", 'f', "f"},
    {"rsqrt.approx.ftz.f32 {d}, {a};", 'f', "f"},
    {"sqrt.approx.ftz.f32 {d}, {a};", 'f', "f"},
    {"abs.ftz.f32 {d}, {a};", 'f', "f"},
    {"neg.ftz.f32 {d}, {a};", 'f', "f"},
    {"min.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"max.ftz.f32 {d}, {a}, {b};", 'f', "ff"},
    {"cvt.rmi.ftz.f32.f32 {d}, {a};", 'f', "f"},
    {"cvt.rzi.ftz.s32.f32 {d}, {a};", 'i', "f"},
    {"cvt.rn.f32.u32 {d}, {a};", 'f', "i"},
    {"setp.eq.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.lt.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.le.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.gt.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.ge.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.ltu.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.leu.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.gtu.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.geu.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"setp.neu.ftz.f32 {d}, {a}, {b};", 'p', "ff"},
    {"selp.f32 {d}, {a}, {b}, {c};", 'f', "ffp"},
    {"add.s32 {d}, {a}, {b};", 'i', "ii"},
    {"sub.s32 {d}, {a}, {b};", 'i', "ii"},
    {"mul.lo.s32 {d}, {a}, {b};", 'i', "ii"},
    {"mad.lo.s32 {d}, {a}, {b}, {c};", 'i', "iii"},
    {"mul.wide.s16 {d}, {a}, {b};", 'i', "hh"},
    {"neg.s32 {d}, {a};", 'i', "i"},
    {"min.s32 {d}, {a}, {b};", 'i', "ii"},
    {"max.s32 {d}, {a}, {b};", 'i', "ii"},
    {"min.u32 {d}, {a}, {b};", 'i', "ii"},
    {"and.b32 {d}, {a}, {b};", 'i', "ii"},
    {"or.b32 {d}, {a}, {b};", 'i', "ii"},
    {"shl.b32 {d}, {a}, {b};", 'i', "ii"},
    {"shr.u32 {d}, {a}, {b};", 'i', "ii"},
    {"shr.s32 {d}, {a}, {b};", 'i', "ii"},
    {"shl.b32 {d}, {a}, 3;", 'i', "i"},
    {"shr.s32 {d}, {a}, 7;", 'i', "i"},
    {"bfe.u32 {d}, {a}, {b}, {c};", 'i', "iii"},
    {"setp.eq.s32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.ne.s32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.lt.s32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.gt.s32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.ge.s32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.lt.u32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.le.u32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.gt.u32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.ge.u32 {d}, {a}, {b};", 'p', "ii"},
    {"setp.eq.s16 {d}, {a}, {b};", 'p', "hh"},
    {"setp.ne.s16 {d}, {a}, {b};", 'p', "hh"},
    {"setp.lt.s16 {d}, {a}, {b};", 'p', "hh"},
    {"setp.eq.b16 {d}, {a}, {b};", 'p', "hh"},
    {"selp.b32 {d}, {a}, {b}, {c};", 'i', "iip"},
    {"selp.u16 {d}, {a}, {b}, {c};", 'h', "hhp"},
    {"and.b16 {d}, {a}, {b};", 'h', "hh"},
    {"or.b16 {d}, {a}, {b};", 'h', "hh"},
    {"cvt.s32.s16 {d}, {a};", 'i', "h"},
    {"cvt.u32.u16 {d}, {a};", 'i', "h"},
    {"cvt.u16.u32 {d}, {a};", 'h', "i"},
    {"cvt.s32.s8 {d}, {a};", 'i', "h"},
    {"cvt.u32.u64 {d}, {a};", 'i', "l"},
    {"clz.b64 {d}, {a};", 'i', "l"},
    {"shl.b64 {d}, {a}, {b};", 'l', "li"},
    {"and.b64 {d}, {a}, {b};", 'l', "ll"},
    {"or.b64 {d}, {a}, {b};", 'l', "ll"},
    {"xor.b64 {d}, {a}, {b};", 'l', "ll"},
    {"add.s64 {d}, {a}, {b};", 'l', "ll"},
    {"setp.eq.s64 {d}, {a}, {b};", 'p', "ll"},
    {"setp.ne.s64 {d}, {a}, {b};", 'p', "ll"},
    {"selp.b64 {d}, {a}, {b}, {c};", 'l', "llp"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 {d}, %f7, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; sub.ftz.f32 {d}, {c}, %f7;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; sub.ftz.f32 {d}, %f7, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 %f6, %f7, {c}; sub.ftz.f32 %f5, %f7, {c}; add.ftz.f32 {d}, %f6, %f5;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 %f6, %f7, {c}; mul.ftz.f32 {d}, %f6, %f7;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; setp.ne.u32 %p7, %r13, 12345; @%p7 bra LX; mov.f32 %f7, 0f00000000; LX: add.ftz.f32 {d}, %f7, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; setp.ne.u32 %p7, %r13, 12345; @%p7 bra LX; mov.f32 %f6, 0f00000000; LX: add.ftz.f32 {d}, %f7, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; mul.ftz.f32 %f6, {c}, {a}; add.ftz.f32 {d}, %f7, %f6;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; mul.ftz.f32 %f6, {c}, {a}; sub.ftz.f32 {d}, %f7, %f6;", 'f', "fff"},
    {"mul.ftz.f32 %f6, {c}, {a}; mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 {d}, %f7, %f6;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; setp.gt.ftz.f32 %p6, %f7, 0f00000000; add.ftz.f32 %f6, %f7, {c}; selp.f32 {d}, %f6, %f7, %p6;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 %f6, %f7, {c}; max.ftz.f32 {d}, %f6, %f7;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 %f6, %f7, {c}; st.global.f32 [%r8+12], %f7; mov.f32 {d}, %f6;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; neg.ftz.f32 %f6, %f7; add.ftz.f32 {d}, %f6, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; add.ftz.f32 %f6, %f7, {c}; add.ftz.f32 {d}, %f6, %f7;", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {a}; fma.rn.ftz.f32 %f6, {b}, {b}, %f7; add.ftz.f32 {d}, %f6, {c};", 'f', "fff"},
    {"mul.ftz.f32 %f7, {a}, {b}; @{c} add.ftz.f32 %f7, %f7, {b}; mov.f32 {d}, %f7;", 'f', "ffp"},
    {"and.pred {d}, {a}, {b};", 'p', "pp"},
    {"or.pred {d}, {a}, {b};", 'p', "pp"},
    {"not.pred {d}, {a};", 'p', "p"},
};

std::string reg_for(char cls, int index)
{
    switch (cls) {
    case 'f': return "%f" + std::to_string(index);
    case 'h':
    case 'b': return "%h" + std::to_string(index);
    case 'l': return "%l" + std::to_string(index);
    case 'p': return "%p" + std::to_string(index);
    default: return "%i" + std::to_string(index);
    }
}

std::string make_ptx(const Variant &v)
{
    std::string body =
        ".version 4.1\n.target sm_52\n.address_size 32\n"
        ".visible .entry k(.param .u32 pin, .param .u32 pout)\n{\n"
        ".reg .u32 %r<16>;\n.reg .f32 %f<8>;\n.reg .u32 %i<8>;\n.reg .u16 %h<8>;\n"
        ".reg .u64 %l<8>;\n.reg .pred %p<8>;\n"
        "mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %ntid.x;\nmov.u32 %r3, %tid.x;\n"
        "mad.lo.s32 %r4, %r1, %r2, %r3;\nshl.b32 %r4, %r4, 4;\n"
        "ld.param.u32 %r5, [pin];\nld.param.u32 %r6, [pout];\n"
        "add.s32 %r7, %r5, %r4;\nadd.s32 %r8, %r6, %r4;\n"
        "ld.global.v4.u32 {%r10, %r11, %r12, %r13}, [%r7];\n";
    // Operand i takes input word i (64-bit operands take words 2i, 2i+1 mod 4).
    for (int i = 0; v.srcs[i]; ++i) {
        const std::string word = "%r1" + std::to_string(i);
        const std::string r = reg_for(v.srcs[i], i + 1);
        switch (v.srcs[i]) {
        case 'f': body += "mov.b32 " + r + ", " + word + ";\n"; break;
        case 'h': body += "cvt.u16.u32 " + r + ", " + word + ";\n"; break;
        case 'l':
            body += "mov.b64 " + r + ", {" + word + ", %r1" + std::to_string((i + 1) % 4) + "};\n";
            break;
        case 'p': body += "and.b32 %r9, " + word + ", 1;\nsetp.ne.u32 " + r + ", %r9, 0;\n"; break;
        default: body += "mov.u32 " + r + ", " + word + ";\n"; break;
        }
    }
    std::string text = v.text;
    auto replace = [&](const std::string &key, const std::string &value) {
        for (std::size_t at; (at = text.find(key)) != std::string::npos;) text.replace(at, key.size(), value);
    };
    replace("{d}", reg_for(v.dst, 0));
    for (int i = 0; v.srcs[i]; ++i) replace(std::string("{") + char('a' + i) + "}", reg_for(v.srcs[i], i + 1));
    body += text + "\n";
    switch (v.dst) {
    case 'f': body += "mov.b32 %r14, %f0;\nmov.u32 %r15, 0;\n"; break;
    case 'h': body += "cvt.u32.u16 %r14, %h0;\nmov.u32 %r15, 0;\n"; break;
    case 'l': body += "mov.b64 {%r14, %r15}, %l0;\n"; break;
    case 'p': body += "selp.u32 %r14, 1, 0, %p0;\nmov.u32 %r15, 0;\n"; break;
    default: body += "mov.u32 %r14, %i0;\nmov.u32 %r15, 0;\n"; break;
    }
    body += "st.global.v4.u32 [%r8], {%r14, %r15, %r10, %r11};\nret;\n}\n";
    return body;
}

std::vector<std::uint32_t> make_inputs(std::size_t records, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    static const std::uint32_t specials[] = {
        0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000,
        0x00000001, 0x007fffff, 0x80000001, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x4b000000,
        0x4effffff, 0x4f000000, 0xcf000000, 0x4f800000, 0x3f000000, 0x40490fdb, 0x0000001f,
        0x00000020, 0x00000021, 0xffffffff, 0x7fffffff, 0x0000ffff, 0x00008000, 0x00000080};
    std::uniform_real_distribution<float> uniform(-100.0f, 100.0f);
    std::vector<std::uint32_t> words(records * 4);
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint32_t pick = rng() % 8;
        std::uint32_t w;
        if (pick == 0) w = specials[rng() % (sizeof(specials) / 4)];
        else if (pick < 4) {
            const float f = uniform(rng) * std::ldexp(1.0f, static_cast<int>(rng() % 40) - 20);
            std::memcpy(&w, &f, 4);
        } else if (pick < 6) w = rng() % 64;
        else w = rng();
        words[i] = w;
    }
    return words;
}

} // namespace

int main(int argc, char **argv)
{
    std::string device_selector;
    std::size_t count = 65536;
    std::string filter;
    bool no_contraction = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device_selector = argv[++i];
        else if (arg == "--count" && i + 1 < argc) count = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (arg == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (arg == "--no-contract") no_contraction = true;
    }
    count = (count + 255) / 256 * 256;
    Cuda cuda;
    if (!cuda.load()) {
        std::fprintf(stderr, "native CUDA unavailable\n");
        return 2;
    }
    pxvk::vk::Device device;
    std::string error;
    if (!device.initialize(device_selector, error)) {
        std::fprintf(stderr, "vulkan: %s\n", error.c_str());
        return 2;
    }
    std::printf("device: %s\n", device.caps().name.c_str());
    pxvk::ptx::CompileOptions options;
    options.native_ftz = device.caps().native_ftz;
    options.preserve_nan = device.caps().preserve_nan;
    options.no_contraction = no_contraction;

    const std::size_t bytes = count * 16;
    CUdeviceptr native_in = 0, native_out = 0;
    cuda.cuMemAlloc(&native_in, bytes);
    cuda.cuMemAlloc(&native_out, bytes);
    // Vulkan layout: input at heap offset 0, output after it.
    const std::uint32_t base = 0x10000;
    device.set_heap_base(base);
    device.ensure_heap(bytes * 2, error);

    int failed_variants = 0;
    std::uint32_t seed = 1;
    for (const Variant &v : kVariants) {
        if (!filter.empty() && std::string(v.text).find(filter) == std::string::npos) continue;
        const std::string ptx = make_ptx(v);
        const std::vector<std::uint32_t> input = make_inputs(count, seed++);
        std::vector<std::uint32_t> native(count * 4), vulkan(count * 4);

        CUmodule module = nullptr;
        CUfunction function = nullptr;
        if (cuda.cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr) != 0 ||
            cuda.cuModuleGetFunction(&function, module, "k") != 0) {
            std::printf("%-40s native JIT failed\n", v.text);
            ++failed_variants;
            continue;
        }
        cuda.cuMemcpyHtoD(native_in, input.data(), bytes);
        void *params[] = {&native_in, &native_out};
        cuda.cuLaunchKernel(function, static_cast<unsigned>(count / 256), 1, 1, 256, 1, 1, 0, nullptr, params, nullptr);
        cuda.cuCtxSynchronize();
        cuda.cuMemcpyDtoH(native.data(), native_out, bytes);
        cuda.cuModuleUnload(module);

        try {
            const auto parsed = pxvk::ptx::parse(ptx);
            const auto kernel = pxvk::ptx::compile_kernel(parsed, "k", options);
            auto *shader = device.create_shader(kernel.spirv, kernel.info.launch_block_bytes,
                                                kernel.info.warp_synchronous, false, error);
            const std::uint32_t launch_params[2] = {base, static_cast<std::uint32_t>(base + bytes)};
            std::vector<std::uint8_t> launch(kernel.info.launch_block_bytes, 0);
            std::memcpy(launch.data(), launch_params, 8);
            pxvk::vk::DispatchShape shape;
            shape.grid[0] = static_cast<std::uint32_t>(count / 256);
            shape.block[0] = 256;
            if (!shader || !device.upload(0, input.data(), bytes, error) ||
                !device.dispatch(shader, shape, launch.data(), launch.size(), error) ||
                !device.download(vulkan.data(), bytes, bytes, error))
                throw std::runtime_error(error);
        } catch (const std::exception &e) {
            std::printf("%-40s pxvk failed: %s\n", v.text, e.what());
            ++failed_variants;
            continue;
        }

        std::size_t mismatches = 0;
        int max_ulp = 0;
        std::string example;
        for (std::size_t r = 0; r < count; ++r) {
            const std::uint32_t *n = &native[r * 4], *g = &vulkan[r * 4];
            if (n[0] == g[0] && n[1] == g[1]) continue;
            // NaN payloads are not portable (NVIDIA canonicalizes to
            // 0x7fffffff); any NaN matches any NaN.
            auto is_nan = [](std::uint32_t w) { return (w & 0x7f800000u) == 0x7f800000u && (w & 0x007fffffu); };
            if (v.dst == 'f' && is_nan(n[0]) && is_nan(g[0])) continue;
            ++mismatches;
            if (v.dst == 'f') {
                const std::int64_t a = (n[0] & 0x80000000u) ? -std::int64_t(n[0] & 0x7fffffff) : n[0];
                const std::int64_t b = (g[0] & 0x80000000u) ? -std::int64_t(g[0] & 0x7fffffff) : g[0];
                max_ulp = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(max_ulp, std::llabs(a - b)), 1 << 30));
            }
            if (example.size() < 400) {
                char text[160];
                const std::uint32_t *in = &input[r * 4];
                std::snprintf(text, sizeof(text), "\n      in=%08x %08x %08x native=%08x:%08x pxvk=%08x:%08x",
                              in[0], in[1], in[2], n[0], n[1], g[0], g[1]);
                example += text;
            }
        }
        if (mismatches == 0) {
            std::printf("%-40s exact\n", v.text);
        } else {
            ++failed_variants;
            std::printf("%-40s %zu/%zu mismatches%s%s\n", v.text, mismatches, count,
                        v.dst == 'f' ? (" max_ulp=" + std::to_string(max_ulp)).c_str() : "", example.c_str());
        }
        std::fflush(stdout);
    }
    std::printf("%d variant(s) differ\n", failed_variants);
    return failed_variants == 0 ? 0 : 1;
}
