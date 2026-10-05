// pxvk-run.exe: console launcher that starts a 32-bit process suspended,
// loads each --inject DLL in order, then resumes it and returns its exit code.
//   pxvk-run.exe [--inject a.dll]... [--timeout seconds] -- app.exe [args]
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::wstring full_path(const wchar_t *path)
{
    wchar_t buffer[32768];
    const DWORD length = GetFullPathNameW(path, 32768, buffer, nullptr);
    return length == 0 ? std::wstring(path) : std::wstring(buffer, length);
}

std::wstring quote(const std::wstring &argument)
{
    std::wstring output = L"\"";
    unsigned slashes = 0;
    for (wchar_t c : argument) {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'"') { output.append(slashes * 2 + 1, L'\\'); output.push_back(c); slashes = 0; continue; }
        output.append(slashes, L'\\');
        slashes = 0;
        output.push_back(c);
    }
    output.append(slashes * 2, L'\\');
    output.push_back(L'"');
    return output;
}

bool inject(HANDLE process, const std::wstring &dll)
{
    const SIZE_T bytes = (dll.size() + 1) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(process, nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (remote == nullptr) return false;
    WriteProcessMemory(process, remote, dll.c_str(), bytes, nullptr);
    auto load = reinterpret_cast<LPTHREAD_START_ROUTINE>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"));
    HANDLE thread = CreateRemoteThread(process, nullptr, 0, load, remote, 0, nullptr);
    if (thread == nullptr) return false;
    WaitForSingleObject(thread, INFINITE);
    DWORD module = 0;
    GetExitCodeThread(thread, &module);
    CloseHandle(thread);
    VirtualFreeEx(process, remote, 0, MEM_RELEASE);
    return module != 0;
}

} // namespace

int wmain(int argc, wchar_t **argv)
{
    std::vector<std::wstring> dlls;
    DWORD timeout_ms = INFINITE;
    int i = 1;
    for (; i < argc; ++i) {
        const std::wstring arg = argv[i];
        if (arg == L"--inject" && i + 1 < argc) dlls.push_back(full_path(argv[++i]));
        else if (arg == L"--timeout" && i + 1 < argc) timeout_ms = static_cast<DWORD>(_wtoi(argv[++i])) * 1000u;
        else if (arg == L"--") { ++i; break; }
        else break;
    }
    if (i >= argc) {
        std::fwprintf(stderr, L"usage: pxvk-run [--inject dll]... [--timeout s] -- app [args]\n");
        return 2;
    }
    const std::wstring application = full_path(argv[i]);
    std::wstring command;
    for (int j = i; j < argc; ++j) {
        if (!command.empty()) command.push_back(L' ');
        command += quote(j == i ? application : std::wstring(argv[j]));
    }
    std::wstring directory = application.substr(0, application.find_last_of(L"\\/"));
    STARTUPINFOW startup{sizeof(startup)};
    PROCESS_INFORMATION info{};
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(0);
    if (!CreateProcessW(application.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
                        CREATE_SUSPENDED, nullptr, directory.c_str(), &startup, &info)) {
        std::fwprintf(stderr, L"CreateProcess failed: %lu\n", GetLastError());
        return 2;
    }
    for (const std::wstring &dll : dlls) {
        if (!inject(info.hProcess, dll)) {
            std::fwprintf(stderr, L"injection failed: %ls\n", dll.c_str());
            TerminateProcess(info.hProcess, 2);
            return 2;
        }
    }
    ResumeThread(info.hThread);
    if (WaitForSingleObject(info.hProcess, timeout_ms) == WAIT_TIMEOUT) {
        std::fwprintf(stderr, L"timeout: terminating target\n");
        TerminateProcess(info.hProcess, 3);
        WaitForSingleObject(info.hProcess, INFINITE);
    }
    DWORD code = 2;
    GetExitCodeProcess(info.hProcess, &code);
    CloseHandle(info.hThread);
    CloseHandle(info.hProcess);
    return static_cast<int>(code);
}
