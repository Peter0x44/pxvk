#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace pxvk::runtime {

void install_driver_redirect(HMODULE self);
void remove_driver_redirect();

} // namespace pxvk::runtime
