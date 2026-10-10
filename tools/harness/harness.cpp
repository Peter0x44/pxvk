// pxvk-harness.dll: deterministic benchmark/capture harness for GLUT PhysX
// samples. Injected by pxvk-run.exe. Configured through environment
// variables so the same binary drives native-CUDA and pxvk runs:
//
//   PXVK_HARNESS_OUT      output directory (frames.csv, summary.txt, *.bmp)
//   PXVK_HARNESS_FRAMES   exit after this many presented frames (0 = never)
//   PXVK_HARNESS_DT_US    virtual microseconds per frame (0 = real time)
//   PXVK_HARNESS_KEYS     "frame:key,..." keys sent to the GLUT keyboard
//                         callback before the given frame is drawn
//   PXVK_HARNESS_CAPTURE  "frame,..." frames whose back buffer is saved
//
// Virtual time makes the PhysX step sequence independent of wall-clock
// speed, so a native run and a pxvk run simulate identical frames.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static_assert(sizeof(void *) == 4, "harness requires a 32-bit build");

namespace {

using SwapFn = void(__stdcall *)();
using KeyboardCallback = void(__cdecl *)(unsigned char, int, int);
using KeyboardFuncFn = void(__stdcall *)(KeyboardCallback);
using GlutGetFn = int(__stdcall *)(unsigned int);
using QpcFn = BOOL(WINAPI *)(LARGE_INTEGER *);
using TickFn = DWORD(WINAPI *)();
using SleepFn = void(WINAPI *)(DWORD);
using ReadPixelsFn = void(__stdcall *)(int, int, int, int, unsigned, unsigned, void *);
using GetIntegervFn = void(__stdcall *)(unsigned, int *);
using ReadBufferFn = void(__stdcall *)(unsigned);
using FinishFn = void(__stdcall *)();

SwapFn g_swap = nullptr;
KeyboardFuncFn g_keyboard_func = nullptr;
GlutGetFn g_glut_get = nullptr;
QpcFn g_qpc = nullptr;
TickFn g_time_get_time = nullptr;
SleepFn g_sleep = nullptr;
KeyboardCallback g_app_keyboard = nullptr;

LARGE_INTEGER g_freq{};
LONGLONG g_real_start = 0;
LONGLONG g_last_real = 0;
std::uint64_t g_dt_us = 0;
std::uint64_t g_virtual_us = 0;
unsigned g_frame = 0;
unsigned g_exit_frame = 0;
std::map<unsigned, std::string> g_keys;
std::vector<unsigned> g_captures;
std::string g_out;
FILE *g_csv = nullptr;
std::vector<double> g_frame_ms;

std::string env(const char *name)
{
    char buffer[4096];
    const DWORD length = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    return length == 0 || length >= sizeof(buffer) ? std::string() : std::string(buffer, length);
}

// A switch is on when set to anything but empty, 0, off, false or no
// (docs/environment.md).
bool env_flag(const char *name)
{
    const std::string value = env(name);
    for (const char *off : {"", "0", "off", "false", "no"})
        if (_stricmp(value.c_str(), off) == 0) return false;
    return true;
}

LONGLONG real_counter()
{
    LARGE_INTEGER now{};
    g_qpc(&now);
    return now.QuadPart;
}

// Virtual clock in microseconds. Sleep() advances it as well so wait loops
// in the application terminate without consulting wall-clock time.
unsigned g_queries_since_swap = 0;
unsigned g_stall_breaks = 0;
volatile LONG g_calls_qpc = 0, g_calls_tick = 0, g_calls_sleep = 0, g_calls_glutget = 0;

std::uint64_t now_us()
{
    if (g_dt_us == 0) {
        return static_cast<std::uint64_t>((real_counter() - g_real_start) * 1000000 /
                                          g_freq.QuadPart);
    }
    // Applications that skip drawing until time advances would otherwise
    // deadlock on a frozen clock. The poll count is a deterministic function
    // of application control flow, so this keeps runs reproducible.
    if (++g_queries_since_swap >= 256u) {
        g_queries_since_swap = 0;
        g_virtual_us += g_dt_us;
        ++g_stall_breaks;
    }
    return g_virtual_us;
}

BOOL WINAPI hook_qpc(LARGE_INTEGER *value)
{
    InterlockedIncrement(&g_calls_qpc);
    if (value == nullptr) return FALSE;
    value->QuadPart = static_cast<LONGLONG>(now_us() * g_freq.QuadPart / 1000000u) + g_real_start;
    return TRUE;
}

DWORD WINAPI hook_tick() { InterlockedIncrement(&g_calls_tick); return static_cast<DWORD>(now_us() / 1000u) + 100000u; }

void WINAPI hook_sleep(DWORD ms)
{
    InterlockedIncrement(&g_calls_sleep);
    if (g_dt_us != 0) {
        g_virtual_us += static_cast<std::uint64_t>(ms) * 1000u;
        g_sleep(0);
        return;
    }
    g_sleep(ms);
}

int __stdcall hook_glut_get(unsigned int what)
{
    InterlockedIncrement(&g_calls_glutget);
    constexpr unsigned kElapsedTime = 700;
    if (what == kElapsedTime) return static_cast<int>(now_us() / 1000u);
    return g_glut_get(what);
}

void __stdcall hook_keyboard_func(KeyboardCallback callback)
{
    g_app_keyboard = callback;
    g_keyboard_func(callback);
}

void write_bmp(const std::string &path, int width, int height, const std::vector<unsigned char> &bgr)
{
    FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return;
    const int row = (width * 3 + 3) & ~3;
    const std::uint32_t image = static_cast<std::uint32_t>(row * height);
    unsigned char header[54]{};
    header[0] = 'B';
    header[1] = 'M';
    const std::uint32_t file_size = 54u + image;
    std::memcpy(header + 2, &file_size, 4);
    const std::uint32_t offset = 54, info = 40;
    std::memcpy(header + 10, &offset, 4);
    std::memcpy(header + 14, &info, 4);
    std::memcpy(header + 18, &width, 4);
    std::memcpy(header + 22, &height, 4);
    header[26] = 1;
    header[28] = 24;
    std::memcpy(header + 34, &image, 4);
    std::fwrite(header, 1, sizeof(header), file);
    std::vector<unsigned char> padded(static_cast<std::size_t>(row), 0);
    for (int y = 0; y < height; ++y) {
        std::memcpy(padded.data(), bgr.data() + static_cast<std::size_t>(y) * width * 3, width * 3);
        std::fwrite(padded.data(), 1, padded.size(), file);
    }
    std::fclose(file);
}

void capture(unsigned frame)
{
    HMODULE gl = GetModuleHandleA("opengl32.dll");
    if (gl == nullptr) return;
    auto read_pixels = reinterpret_cast<ReadPixelsFn>(GetProcAddress(gl, "glReadPixels"));
    auto get_integerv = reinterpret_cast<GetIntegervFn>(GetProcAddress(gl, "glGetIntegerv"));
    auto read_buffer = reinterpret_cast<ReadBufferFn>(GetProcAddress(gl, "glReadBuffer"));
    auto finish = reinterpret_cast<FinishFn>(GetProcAddress(gl, "glFinish"));
    if (!read_pixels || !get_integerv || !read_buffer || !finish) return;
    int viewport[4]{};
    constexpr unsigned kViewport = 0x0BA2, kBack = 0x0405, kBgr = 0x80E0,
                       kUnsignedByte = 0x1401, kPackAlignment = 0x0D05;
    get_integerv(kViewport, viewport);
    if (viewport[2] <= 0 || viewport[3] <= 0) return;
    auto pixel_store = reinterpret_cast<void(__stdcall *)(unsigned, int)>(
        GetProcAddress(gl, "glPixelStorei"));
    if (pixel_store) pixel_store(kPackAlignment, 1);
    finish();
    read_buffer(kBack);
    std::vector<unsigned char> pixels(static_cast<std::size_t>(viewport[2]) * viewport[3] * 3);
    read_pixels(viewport[0], viewport[1], viewport[2], viewport[3], kBgr, kUnsignedByte,
                pixels.data());
    char name[64];
    std::snprintf(name, sizeof(name), "\\frame_%05u.bmp", frame);
    write_bmp(g_out + name, viewport[2], viewport[3], pixels);
}

void finish_run()
{
    if (g_csv) std::fclose(g_csv);
    g_csv = nullptr;
    FILE *summary = std::fopen((g_out + "\\summary.txt").c_str(), "w");
    if (summary) {
        // Skip the first frames of every scene segment would be nicer; keep
        // it simple: report all frames and the steady-state half.
        double total = 0, tail = 0;
        std::size_t tail_count = 0;
        for (std::size_t i = 0; i < g_frame_ms.size(); ++i) {
            total += g_frame_ms[i];
            if (i >= g_frame_ms.size() / 2) {
                tail += g_frame_ms[i];
                ++tail_count;
            }
        }
        std::fprintf(summary, "stall_breaks=%u\nframes=%u\nmean_ms=%.3f\ntail_mean_ms=%.3f\ntail_fps=%.2f\n",
                     g_stall_breaks, static_cast<unsigned>(g_frame_ms.size()),
                     g_frame_ms.empty() ? 0.0 : total / g_frame_ms.size(),
                     tail_count ? tail / tail_count : 0.0,
                     tail_count && tail > 0 ? 1000.0 * tail_count / tail : 0.0);
        std::fclose(summary);
    }
}

void log_renderer()
{
    HMODULE gl = GetModuleHandleA("opengl32.dll");
    auto get_string = gl ? reinterpret_cast<const char *(__stdcall *)(unsigned)>(GetProcAddress(gl, "glGetString")) : nullptr;
    if (!get_string) return;
    constexpr unsigned kVendor = 0x1F00, kRenderer = 0x1F01;
    if (FILE *log = std::fopen((g_out + "\\harness.log").c_str(), "a")) {
        std::fprintf(log, "gl_vendor=%s gl_renderer=%s\n", get_string(kVendor), get_string(kRenderer));
        std::fclose(log);
    }
}

void __stdcall hook_swap()
{
    if (g_frame == 0) {
        log_renderer();
        // Measure throughput, not the display refresh: disable vsync.
        HMODULE gl = GetModuleHandleA("opengl32.dll");
        auto get_proc = gl ? reinterpret_cast<void *(__stdcall *)(const char *)>(GetProcAddress(gl, "wglGetProcAddress")) : nullptr;
        if (get_proc && !env_flag("PXVK_HARNESS_VSYNC")) {
            if (auto swap_interval = reinterpret_cast<int(__stdcall *)(int)>(get_proc("wglSwapIntervalEXT")))
                swap_interval(0);
        }
    }
    if (std::find(g_captures.begin(), g_captures.end(), g_frame) != g_captures.end())
        capture(g_frame);
    g_swap();
    const LONGLONG now = real_counter();
    const double ms = static_cast<double>(now - g_last_real) * 1000.0 / g_freq.QuadPart;
    g_last_real = now;
    if (g_csv && (g_frame % 64u) == 0u) std::fflush(g_csv);
    if (g_csv) std::fprintf(g_csv, "%u,%.4f\n", g_frame, ms);
    g_frame_ms.push_back(ms);
    ++g_frame;
    {
        // Published for the capture tracer, which selects launches by frame.
        char text[16];
        std::snprintf(text, sizeof(text), "%u", g_frame);
        SetEnvironmentVariableA("PXVK_FRAME", text);
    }
    g_virtual_us += g_dt_us;
    g_queries_since_swap = 0;
    const auto key = g_keys.find(g_frame);
    if (key != g_keys.end() && g_app_keyboard != nullptr) {
        for (char c : key->second) g_app_keyboard(static_cast<unsigned char>(c), 0, 0);
    }
    if (g_exit_frame != 0 && g_frame >= g_exit_frame) {
        finish_run();
        TerminateProcess(GetCurrentProcess(), 0);
    }
}

bool patch(HMODULE module, const char *library, const char *function, void *replacement,
           void **original)
{
    auto *base = reinterpret_cast<unsigned char *>(module);
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS32 *>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dir.VirtualAddress == 0) return false;
    for (auto *desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress);
         desc->Name != 0; ++desc) {
        if (_stricmp(reinterpret_cast<const char *>(base + desc->Name), library) != 0) continue;
        auto *names = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + desc->OriginalFirstThunk);
        auto *slots = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + desc->FirstThunk);
        for (; names->u1.AddressOfData != 0; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL32(names->u1.Ordinal)) continue;
            const auto *import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME *>(
                base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char *>(import->Name), function) != 0)
                continue;
            void **slot = reinterpret_cast<void **>(&slots->u1.Function);
            DWORD old = 0;
            if (!VirtualProtect(slot, sizeof(void *), PAGE_READWRITE, &old)) return false;
            if (original && *original == nullptr) *original = *slot;
            *slot = replacement;
            VirtualProtect(slot, sizeof(void *), old, &old);
            return true;
        }
    }
    return false;
}

void parse_config()
{
    g_out = env("PXVK_HARNESS_OUT");
    if (g_out.empty()) g_out = ".";
    CreateDirectoryA(g_out.c_str(), nullptr);
    g_exit_frame = static_cast<unsigned>(std::strtoul(env("PXVK_HARNESS_FRAMES").c_str(), nullptr, 10));
    g_dt_us = std::strtoull(env("PXVK_HARNESS_DT_US").c_str(), nullptr, 10);
    std::string keys = env("PXVK_HARNESS_KEYS");
    for (std::size_t start = 0; start < keys.size();) {
        std::size_t end = keys.find(',', start);
        if (end == std::string::npos) end = keys.size();
        const std::string item = keys.substr(start, end - start);
        const std::size_t colon = item.find(':');
        if (colon != std::string::npos)
            g_keys[static_cast<unsigned>(std::strtoul(item.c_str(), nullptr, 10))] += item.substr(colon + 1);
        start = end + 1;
    }
    std::string captures = env("PXVK_HARNESS_CAPTURE");
    for (std::size_t start = 0; start < captures.size();) {
        std::size_t end = captures.find(',', start);
        if (end == std::string::npos) end = captures.size();
        g_captures.push_back(static_cast<unsigned>(std::strtoul(captures.substr(start, end - start).c_str(), nullptr, 10)));
        start = end + 1;
    }
}

DWORD WINAPI watchdog(void *)
{
    for (int second = 0;; ++second) {
        g_sleep(1000);
        if (FILE *log = std::fopen((g_out + "\\watchdog.log").c_str(), "a")) {
            std::fprintf(log, "t=%d frame=%u qpc=%ld tick=%ld sleep=%ld glutget=%ld vus=%llu breaks=%u\n",
                         second, g_frame, g_calls_qpc, g_calls_tick, g_calls_sleep, g_calls_glutget,
                         static_cast<unsigned long long>(g_virtual_us), g_stall_breaks);
            std::fclose(log);
        }
    }
}

// PhysX samples try to reach the PhysX Visual Debugger on 127.0.0.1:5425.
// On this machine the SYN is dropped rather than refused, so connect()
// blocks indefinitely. Accept and immediately close such connections.
DWORD WINAPI pvd_sink(void *)
{
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 0;
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(5425);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listener, 4) != 0)
        return 0;
    for (;;) {
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client == INVALID_SOCKET) return 0;
        closesocket(client);
    }
}

void install()
{
    CreateThread(nullptr, 0, pvd_sink, nullptr, 0, nullptr);
    parse_config();
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    g_qpc = reinterpret_cast<QpcFn>(GetProcAddress(kernel, "QueryPerformanceCounter"));
    g_sleep = reinterpret_cast<SleepFn>(GetProcAddress(kernel, "Sleep"));
    QueryPerformanceFrequency(&g_freq);
    g_real_start = g_last_real = real_counter();
    g_csv = std::fopen((g_out + "\\frames.csv").c_str(), "w");
    if (g_csv) std::fprintf(g_csv, "frame,ms\n");

    CreateThread(nullptr, 0, watchdog, nullptr, 0, nullptr);
    HMODULE exe = GetModuleHandleA(nullptr);
    const bool swap_ok = patch(exe, "glut32.dll", "glutSwapBuffers",
                               reinterpret_cast<void *>(&hook_swap),
                               reinterpret_cast<void **>(&g_swap));
    if (FILE *log = std::fopen((g_out + "\\harness.log").c_str(), "w")) {
        HMODULE glut = GetModuleHandleA("glut32.dll");
        std::fprintf(log, "swap_patched=%d original=%p glut=%p expected=%p\n", swap_ok,
                     reinterpret_cast<void *>(g_swap), reinterpret_cast<void *>(glut),
                     glut ? reinterpret_cast<void *>(GetProcAddress(glut, "glutSwapBuffers")) : nullptr);
        std::fclose(log);
    }
    patch(exe, "glut32.dll", "glutKeyboardFunc", reinterpret_cast<void *>(&hook_keyboard_func),
          reinterpret_cast<void **>(&g_keyboard_func));
    patch(exe, "glut32.dll", "glutGet", reinterpret_cast<void *>(&hook_glut_get),
          reinterpret_cast<void **>(&g_glut_get));
    if (g_dt_us != 0) {
        void *ignored = nullptr;
        patch(exe, "KERNEL32.dll", "QueryPerformanceCounter", reinterpret_cast<void *>(&hook_qpc), &ignored);
        ignored = nullptr;
        patch(exe, "KERNEL32.dll", "GetTickCount", reinterpret_cast<void *>(&hook_tick), &ignored);
        ignored = nullptr;
        patch(exe, "KERNEL32.dll", "Sleep", reinterpret_cast<void *>(&hook_sleep), &ignored);
        patch(exe, "WINMM.dll", "timeGetTime", reinterpret_cast<void *>(&hook_tick),
              reinterpret_cast<void **>(&g_time_get_time));
    }
}

} // namespace

extern "C" BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        install();
    }
    return TRUE;
}
