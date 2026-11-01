// Loads a DLL into a suspended process (pxvk-launch, and pxvk's own child
// processes; see loader.cpp): writes its path into the process and runs
// LoadLibraryW there on a remote thread. kernel32.dll is mapped at the same
// address in every 32-bit process, so this process's LoadLibraryW address is
// valid in the target.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cwchar>

namespace pxvk::runtime {

// Returns 0 on success, otherwise a Win32 error code (ERROR_DLL_INIT_FAILED
// when LoadLibraryW ran but returned NULL in the target).
inline DWORD inject_library(HANDLE process, const wchar_t *path)
{
    const SIZE_T bytes = (std::wcslen(path) + 1u) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(process, nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (remote == nullptr) return GetLastError();
    DWORD error = 0;
    SIZE_T written = 0;
    if (WriteProcessMemory(process, remote, path, bytes, &written) == FALSE || written != bytes) {
        error = GetLastError() != 0 ? GetLastError() : ERROR_PARTIAL_COPY;
    } else {
        const auto load_library = reinterpret_cast<LPTHREAD_START_ROUTINE>(
            reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW")));
        HANDLE thread = CreateRemoteThread(process, nullptr, 0, load_library, remote, 0, nullptr);
        if (thread == nullptr) {
            error = GetLastError();
        } else {
            WaitForSingleObject(thread, INFINITE);
            DWORD module = 0;
            GetExitCodeThread(thread, &module);
            CloseHandle(thread);
            if (module == 0u) error = ERROR_DLL_INIT_FAILED;
        }
    }
    VirtualFreeEx(process, remote, 0, MEM_RELEASE);
    return error;
}

} // namespace pxvk::runtime
