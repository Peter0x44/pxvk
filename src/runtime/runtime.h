#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace pxvk::runtime {

void install_driver_redirect(HMODULE self);
void remove_driver_redirect();
// How PhysX's NVAPI loads are served, for the log: "pxvk" (pxvk-nvapi.dll),
// "system" (PXVK_NVAPI=system) or "missing" (no pxvk-nvapi.dll found).
const char *nvapi_mode();

} // namespace pxvk::runtime
