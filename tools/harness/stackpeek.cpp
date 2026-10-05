// pxvk-stackpeek.exe <pid>: print each thread's EIP and the code addresses
// found on its stack, symbolized as module+offset. A poor man's debugger for
// diagnosing hangs inside 32-bit samples.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct Module {
    std::uintptr_t base, end;
    std::string name;
};

static std::string where(const std::vector<Module> &modules, std::uintptr_t address)
{
    for (const Module &m : modules)
        if (address >= m.base && address < m.end) {
            char text[300];
            std::snprintf(text, sizeof(text), "%s+0x%lx", m.name.c_str(),
                          static_cast<unsigned long>(address - m.base));
            return text;
        }
    return {};
}

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    const DWORD pid = static_cast<DWORD>(std::strtoul(argv[1], nullptr, 10));
    HANDLE process = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!process) return 3;
    std::vector<Module> modules;
    HMODULE handles[1024];
    DWORD needed = 0;
    if (EnumProcessModules(process, handles, sizeof(handles), &needed)) {
        for (DWORD i = 0; i < needed / sizeof(HMODULE); ++i) {
            MODULEINFO info{};
            char name[MAX_PATH]{};
            GetModuleInformation(process, handles[i], &info, sizeof(info));
            GetModuleBaseNameA(process, handles[i], name, MAX_PATH);
            modules.push_back({reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll),
                               reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll) + info.SizeOfImage,
                               name});
        }
    }
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 entry{sizeof(entry)};
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid) continue;
        HANDLE thread = OpenThread(THREAD_ALL_ACCESS, FALSE, entry.th32ThreadID);
        if (!thread) continue;
        SuspendThread(thread);
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(thread, &context)) {
            std::printf("thread %lu eip=%08lx %s\n", entry.th32ThreadID, context.Eip,
                        where(modules, context.Eip).c_str());
            std::vector<std::uint32_t> stack(2048);
            SIZE_T read = 0;
            ReadProcessMemory(process, reinterpret_cast<void *>(context.Esp), stack.data(),
                              stack.size() * 4, &read);
            int shown = 0;
            for (SIZE_T i = 0; i < read / 4 && shown < 14; ++i) {
                const std::string text = where(modules, stack[i]);
                if (!text.empty()) {
                    std::printf("    [esp+%03lx] %08x %s\n", static_cast<unsigned long>(i * 4),
                                stack[i], text.c_str());
                    ++shown;
                }
            }
        }
        ResumeThread(thread);
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
    return 0;
}
