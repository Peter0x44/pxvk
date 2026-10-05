// pxvk-nativeinfo: print native CUDA device attributes and the function
// attributes of every entry in the given PTX modules (32-bit driver).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "ptx.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char **argv)
{
    HMODULE dll = LoadLibraryA("nvcuda.dll");
    if (!dll) return 2;
    auto get = [&](const char *name) { return reinterpret_cast<void *>(GetProcAddress(dll, name)); };
    auto cuInit = reinterpret_cast<int(__stdcall *)(unsigned)>(get("cuInit"));
    auto cuDeviceGet = reinterpret_cast<int(__stdcall *)(int *, int)>(get("cuDeviceGet"));
    auto cuCtxCreate = reinterpret_cast<int(__stdcall *)(void **, unsigned, int)>(get("cuCtxCreate_v2"));
    auto cuDeviceGetAttribute = reinterpret_cast<int(__stdcall *)(int *, int, int)>(get("cuDeviceGetAttribute"));
    auto cuModuleLoadData = reinterpret_cast<int(__stdcall *)(void **, const void *)>(get("cuModuleLoadData"));
    auto cuModuleGetFunction = reinterpret_cast<int(__stdcall *)(void **, void *, const char *)>(get("cuModuleGetFunction"));
    auto cuFuncGetAttribute = reinterpret_cast<int(__stdcall *)(int *, int, void *)>(get("cuFuncGetAttribute"));
    int device = 0;
    void *context = nullptr;
    if (cuInit(0) || cuDeviceGet(&device, 0) || cuCtxCreate(&context, 0, device)) return 3;
    std::printf("device attributes:");
    for (int a = 1; a <= 90; ++a) {
        int v = 0;
        if (cuDeviceGetAttribute(&v, a, device) == 0) std::printf(" %d=%d", a, v);
    }
    std::printf("\n");
    for (int i = 1; i < argc; ++i) {
        std::ifstream in(argv[i], std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        const std::string ptx = text.str();
        void *module = nullptr;
        if (cuModuleLoadData(&module, ptx.c_str()) != 0) {
            std::printf("%s: load failed\n", argv[i]);
            continue;
        }
        const auto parsed = pxvk::ptx::parse(ptx);
        for (const auto &f : parsed.functions) {
            if (!f.is_entry) continue;
            void *function = nullptr;
            if (cuModuleGetFunction(&function, module, f.name.c_str()) != 0) continue;
            int threads = 0, shared = 0, constant = 0, local = 0, regs = 0, ptxv = 0, binv = 0;
            cuFuncGetAttribute(&threads, 0, function);
            cuFuncGetAttribute(&shared, 1, function);
            cuFuncGetAttribute(&constant, 2, function);
            cuFuncGetAttribute(&local, 3, function);
            cuFuncGetAttribute(&regs, 4, function);
            cuFuncGetAttribute(&ptxv, 5, function);
            cuFuncGetAttribute(&binv, 6, function);
            std::printf("%-70.70s threads=%d shared=%d const=%d local=%d regs=%d ptx=%d bin=%d\n", f.name.c_str(),
                        threads, shared, constant, local, regs, ptxv, binv);
        }
    }
    return 0;
}
