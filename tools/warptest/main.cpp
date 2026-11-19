// pxvk-warptest: does the GPU keep a warp's lanes in step without barriers?
//
//   pxvk-warptest [--device N] [--blocks N] [--block N] [--rounds N] [--runs N]
//
// Warp-synchronous PTX (PhysX, APEX) exchanges values between the lanes of a
// warp through volatile shared memory without barriers, relying on lockstep
// execution, which neither CUDA nor Vulkan promises. pxvk compiles it with
// subgroup control barriers at reconvergence points under
// VK_KHR_shader_maximal_reconvergence (CompileOptions::WarpSync::maximal,
// skipped where the device lacks it), or else memory-only subgroup fences
// (fence). For comparison: control barriers without maximal reconvergence
// (control, where the driver decides which lanes a barrier joins) and
// nothing at all (none). The exit code is 1 if a mode pxvk uses (maximal,
// fence) mismatches. Each test kernel
// checks itself: every lane computes which values the other lanes of its warp
// must have stored under lockstep and counts the loads that differ. The
// patterns, each over many rounds that overwrite the same slots:
//
//   divergent  some lanes store under a branch (after a lane-dependent busy
//              loop that tries to put them out of step), then every lane reads
//              another lane's slot (APEX's BasicIOS_compactKernel)
//   scan       the classic warp-synchronous inclusive scan (padded volatile
//              buffer, s[i] += s[i - 1, 2, 4, 8, 16]), with lane-dependent
//              busy loops between the steps
//
// A mismatch means that GPU, driver and strategy can break warp-synchronous
// code in practice. Zero mismatches over many runs is evidence, not proof.
#include "compiler.h"
#include "device.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Shared prologue: lane, warp base, hash helpers live in each kernel.
// h(x) = murmur-like finaliser of x; key = f(seed, block, warp, round).
const char *kDivergent = R"(
.version 4.1
.target sm_52
.address_size 32
.visible .entry k(.param .u32 pout, .param .u32 seed, .param .u32 rounds, .param .u32 skew)
{
.reg .pred %p<8>;
.reg .u32 %r<64>;
.shared .align 4 .u32 sbuf[1024];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
and.b32 %r4, %r1, 31;
and.b32 %r5, %r1, 0xffffffe0;
ld.param.u32 %r6, [seed];
ld.param.u32 %r7, [rounds];
ld.param.u32 %r8, [skew];
mov.u32 %r9, sbuf;
shl.b32 %r10, %r1, 2;
add.u32 %r11, %r9, %r10;
shl.b32 %r12, %r5, 2;
add.u32 %r13, %r9, %r12;
mov.u32 %r20, 0;
mov.u32 %r21, 0;
mov.u32 %r22, 1;
mov.u32 %r23, 0xffffffff;
LOOP:
mad.lo.u32 %r30, %r2, 7919, %r6;
mad.lo.u32 %r30, %r20, 104729, %r30;
mad.lo.u32 %r30, %r5, 31337, %r30;
mad.lo.u32 %r31, %r4, 0x9e3779b1, %r30;
shr.u32 %r32, %r31, 15;
xor.b32 %r31, %r31, %r32;
mul.lo.u32 %r31, %r31, 0x85ebca6b;
shr.u32 %r32, %r31, 13;
xor.b32 %r31, %r31, %r32;
st.volatile.shared.u32 [%r11], %r23;
and.b32 %r33, %r31, 3;
setp.ne.u32 %p1, %r33, 0;
@!%p1 bra SKIP;
shr.u32 %r34, %r31, 8;
and.b32 %r34, %r34, %r8;
SPIN:
setp.eq.u32 %p2, %r34, 0;
@%p2 bra SPUN;
mad.lo.u32 %r22, %r22, 1664525, 1013904223;
sub.u32 %r34, %r34, 1;
bra SPIN;
SPUN:
xor.b32 %r35, %r31, %r20;
st.volatile.shared.u32 [%r11], %r35;
SKIP:
and.b32 %r36, %r30, 15;
add.u32 %r36, %r36, %r4;
add.u32 %r36, %r36, 1;
and.b32 %r36, %r36, 31;
shl.b32 %r37, %r36, 2;
add.u32 %r37, %r13, %r37;
ld.volatile.shared.u32 %r38, [%r37];
mad.lo.u32 %r40, %r36, 0x9e3779b1, %r30;
shr.u32 %r41, %r40, 15;
xor.b32 %r40, %r40, %r41;
mul.lo.u32 %r40, %r40, 0x85ebca6b;
shr.u32 %r41, %r40, 13;
xor.b32 %r40, %r40, %r41;
and.b32 %r42, %r40, 3;
setp.ne.u32 %p3, %r42, 0;
xor.b32 %r43, %r40, %r20;
selp.b32 %r44, %r43, %r23, %p3;
setp.ne.u32 %p4, %r38, %r44;
selp.u32 %r45, 1, 0, %p4;
add.u32 %r21, %r21, %r45;
add.u32 %r20, %r20, 1;
setp.lt.u32 %p5, %r20, %r7;
@%p5 bra LOOP;
ld.param.u32 %r50, [pout];
mad.lo.u32 %r51, %r2, %r3, %r1;
shl.b32 %r51, %r51, 3;
add.u32 %r52, %r50, %r51;
st.global.u32 [%r52], %r21;
st.global.u32 [%r52+4], %r22;
ret;
}
)";

// Inclusive warp scan over a padded buffer of 64 words per warp (the first
// 32 stay 0), values v(lane) = h(lane) & 255; expected = sum of v(0..lane).
const char *kScan = R"(
.version 4.1
.target sm_52
.address_size 32
.visible .entry k(.param .u32 pout, .param .u32 seed, .param .u32 rounds, .param .u32 skew)
{
.reg .pred %p<8>;
.reg .u32 %r<64>;
.shared .align 4 .u32 sbuf[2048];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
and.b32 %r4, %r1, 31;
and.b32 %r5, %r1, 0xffffffe0;
ld.param.u32 %r6, [seed];
ld.param.u32 %r7, [rounds];
ld.param.u32 %r8, [skew];
mov.u32 %r9, sbuf;
shl.b32 %r10, %r5, 3;
add.u32 %r10, %r9, %r10;
shl.b32 %r11, %r4, 2;
add.u32 %r12, %r10, %r11;
add.u32 %r13, %r12, 128;
mov.u32 %r20, 0;
mov.u32 %r21, 0;
mov.u32 %r22, 1;
mov.u32 %r23, 0;
LOOP:
mad.lo.u32 %r30, %r2, 7919, %r6;
mad.lo.u32 %r30, %r20, 104729, %r30;
mad.lo.u32 %r30, %r5, 31337, %r30;
mad.lo.u32 %r31, %r4, 0x9e3779b1, %r30;
shr.u32 %r32, %r31, 15;
xor.b32 %r31, %r31, %r32;
mul.lo.u32 %r31, %r31, 0x85ebca6b;
shr.u32 %r32, %r31, 13;
xor.b32 %r31, %r31, %r32;
and.b32 %r33, %r31, 255;
st.volatile.shared.u32 [%r12], %r23;
st.volatile.shared.u32 [%r13], %r33;
mov.u32 %r24, 1;
STEP:
shr.u32 %r34, %r31, %r24;
and.b32 %r34, %r34, %r8;
SPIN:
setp.eq.u32 %p2, %r34, 0;
@%p2 bra SPUN;
mad.lo.u32 %r22, %r22, 1664525, 1013904223;
sub.u32 %r34, %r34, 1;
bra SPIN;
SPUN:
shl.b32 %r35, %r24, 2;
sub.u32 %r36, %r13, %r35;
ld.volatile.shared.u32 %r37, [%r36];
add.u32 %r33, %r33, %r37;
st.volatile.shared.u32 [%r13], %r33;
shl.b32 %r24, %r24, 1;
setp.le.u32 %p3, %r24, 16;
@%p3 bra STEP;
mov.u32 %r40, 0;
mov.u32 %r41, 0;
SUM:
mad.lo.u32 %r42, %r41, 0x9e3779b1, %r30;
shr.u32 %r43, %r42, 15;
xor.b32 %r42, %r42, %r43;
mul.lo.u32 %r42, %r42, 0x85ebca6b;
shr.u32 %r43, %r42, 13;
xor.b32 %r42, %r42, %r43;
and.b32 %r42, %r42, 255;
add.u32 %r40, %r40, %r42;
add.u32 %r41, %r41, 1;
setp.le.u32 %p4, %r41, %r4;
@%p4 bra SUM;
setp.ne.u32 %p5, %r33, %r40;
selp.u32 %r45, 1, 0, %p5;
add.u32 %r21, %r21, %r45;
add.u32 %r20, %r20, 1;
setp.lt.u32 %p6, %r20, %r7;
@%p6 bra LOOP;
ld.param.u32 %r50, [pout];
mad.lo.u32 %r51, %r2, %r3, %r1;
shl.b32 %r51, %r51, 3;
add.u32 %r52, %r50, %r51;
st.global.u32 [%r52], %r21;
st.global.u32 [%r52+4], %r22;
ret;
}
)";

struct Test {
    const char *name;
    const char *ptx;
};
const Test kTests[] = {{"divergent", kDivergent}, {"scan", kScan}};

struct Mode {
    const char *name;
    pxvk::ptx::CompileOptions::WarpSync sync;
};
const Mode kModes[] = {{"none", pxvk::ptx::CompileOptions::WarpSync::none},
                       {"fence", pxvk::ptx::CompileOptions::WarpSync::fence},
                       {"maximal", pxvk::ptx::CompileOptions::WarpSync::maximal},
                       {"control", pxvk::ptx::CompileOptions::WarpSync::control}};

// Modes pxvk compiles with; the others are only for comparison.
bool used_by_pxvk(pxvk::ptx::CompileOptions::WarpSync sync)
{
    return sync == pxvk::ptx::CompileOptions::WarpSync::fence ||
           sync == pxvk::ptx::CompileOptions::WarpSync::maximal;
}

} // namespace

int main(int argc, char **argv)
{
    std::string selector;
    std::uint32_t blocks = 512, block = 256, rounds = 256, runs = 20;
    std::vector<std::uint32_t> skews = {0, 3, 31};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() { return static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10)); };
        if (arg == "--device" && i + 1 < argc) selector = argv[++i];
        else if (arg == "--blocks" && i + 1 < argc) blocks = next();
        else if (arg == "--block" && i + 1 < argc) block = next();
        else if (arg == "--rounds" && i + 1 < argc) rounds = next();
        else if (arg == "--runs" && i + 1 < argc) runs = next();
    }
    if (block == 0 || block % 32 != 0 || block > 1024) {
        std::fprintf(stderr, "--block must be a multiple of 32 up to 1024\n");
        return 2;
    }
    pxvk::vk::Device device;
    std::string error;
    if (!device.initialize(selector, error)) {
        std::fprintf(stderr, "vulkan: %s\n", error.c_str());
        return 2;
    }
    const auto &caps = device.caps();
    std::printf("device: %s (subgroup %u, warp subgroup %u, subgroup control %d, maximal reconvergence %d)\n",
                caps.name.c_str(), caps.subgroup_size, caps.warp_subgroup_size, caps.subgroup_control,
                caps.maximal_reconvergence);
    std::printf("%u blocks x %u threads, %u rounds per launch, %u launches per case\n", blocks, block, rounds, runs);
    const std::uint32_t threads = blocks * block;
    const std::size_t bytes = static_cast<std::size_t>(threads) * 8;
    const std::uint32_t base = 0x10000;
    device.set_heap_base(base);
    if (!device.ensure_heap(bytes, error)) {
        std::fprintf(stderr, "heap: %s\n", error.c_str());
        return 2;
    }
    bool any = false;
    for (const Test &test : kTests) {
        for (const Mode &mode : kModes) {
            if (mode.sync == pxvk::ptx::CompileOptions::WarpSync::maximal && !caps.maximal_reconvergence) {
                std::printf("%-9s %-7s skipped: no VK_KHR_shader_maximal_reconvergence\n", test.name, mode.name);
                continue;
            }
            pxvk::ptx::CompileOptions options;
            options.native_ftz = caps.native_ftz;
            options.preserve_denorm = caps.preserve_denorm;
            options.preserve_nan = caps.preserve_nan;
            options.subgroup_shuffle = caps.subgroup_shuffle;
            options.subgroup_ballot = caps.subgroup_ballot;
            options.emulate_warps = caps.narrow_subgroups();
            options.warp_sync = mode.sync;
            pxvk::vk::Shader *shader = nullptr;
            pxvk::ptx::CompiledKernel kernel;
            try {
                kernel = pxvk::ptx::compile_kernel(pxvk::ptx::parse(test.ptx), "k", options);
                shader = device.create_shader(kernel.spirv, kernel.info.launch_block_bytes,
                                              kernel.info.warp_synchronous, false, error);
            } catch (const std::exception &e) {
                error = e.what();
            }
            if (!shader) {
                std::printf("%-9s %-7s compile failed: %s\n", test.name, mode.name, error.c_str());
                any = true;
                continue;
            }
            for (std::uint32_t skew : skews) {
                std::uint64_t mismatches = 0, loads = 0;
                double ms = 0;
                for (std::uint32_t run = 0; run < runs; ++run) {
                    const std::uint32_t params[4] = {base, 0x9e3779b9u * (run + 1) + skew, rounds, skew};
                    std::vector<std::uint8_t> launch(kernel.info.launch_block_bytes, 0);
                    std::memcpy(launch.data(), params, sizeof(params));
                    pxvk::vk::DispatchShape shape;
                    shape.grid[0] = blocks;
                    shape.block[0] = block;
                    shape.shared_bytes = kernel.info.static_shared_bytes; // sizes the pipeline's shared memory
                    std::vector<std::uint32_t> out(threads * 2);
                    const auto start = std::chrono::steady_clock::now();
                    if (!device.dispatch(shader, shape, launch.data(), launch.size(), error) ||
                        !device.download(out.data(), 0, bytes, error)) {
                        std::printf("%-9s %-7s skew %-2u launch failed: %s\n", test.name, mode.name, skew, error.c_str());
                        any = true;
                        break;
                    }
                    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    for (std::uint32_t t = 0; t < threads; ++t) mismatches += out[t * 2];
                    loads += static_cast<std::uint64_t>(threads) * rounds;
                }
                std::printf("%-9s %-7s skew %-2u %12llu mismatches in %llu checks  (%.2f ms per launch)\n", test.name,
                            mode.name, skew, static_cast<unsigned long long>(mismatches),
                            static_cast<unsigned long long>(loads), ms / runs);
                if (mismatches != 0 && used_by_pxvk(mode.sync)) any = true;
            }
        }
    }
    return any ? 1 : 0;
}
