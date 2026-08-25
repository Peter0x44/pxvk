#include "launch_params.h"

#include <stdint.h>

#define PXVK_LAUNCH_PARAM_END ((uintptr_t)0)
#define PXVK_LAUNCH_PARAM_BUFFER_POINTER ((uintptr_t)1)
#define PXVK_LAUNCH_PARAM_BUFFER_SIZE ((uintptr_t)2)
#define PXVK_MAX_EXTRA_ENTRIES 32

static bool read_extra_pointer(
    void *const *extra,
    size_t index,
    pxvk_read_memory_fn read_memory,
    void *read_context,
    void **value)
{
    uintptr_t base = (uintptr_t)extra;
    uintptr_t offset = index * sizeof(*extra);
    if (base > UINTPTR_MAX - offset) {
        return false;
    }
    return read_memory(read_context, (const void *)(base + offset), value, sizeof(*value));
}

pxvk_launch_params_status pxvk_parse_launch_extra(
    void *const *extra,
    pxvk_read_memory_fn read_memory,
    void *read_context,
    pxvk_launch_params *result)
{
    const void *buffer = NULL;
    size_t buffer_size = 0;
    bool have_buffer = false;
    bool have_size = false;
    size_t index = 0;

    if (result == NULL || read_memory == NULL) {
        return PXVK_LAUNCH_PARAMS_MALFORMED;
    }
    result->buffer = NULL;
    result->size = 0;
    if (extra == NULL) {
        return PXVK_LAUNCH_PARAMS_NONE;
    }

    while (index < PXVK_MAX_EXTRA_ENTRIES) {
        void *token_pointer = NULL;
        uintptr_t token;
        if (!read_extra_pointer(extra, index++, read_memory, read_context, &token_pointer)) {
            return PXVK_LAUNCH_PARAMS_UNREADABLE;
        }
        token = (uintptr_t)token_pointer;
        if (token == PXVK_LAUNCH_PARAM_END) {
            if (!have_buffer && !have_size) {
                return PXVK_LAUNCH_PARAMS_NONE;
            }
            if (!have_buffer || !have_size) {
                return PXVK_LAUNCH_PARAMS_MALFORMED;
            }
            result->buffer = buffer;
            result->size = buffer_size;
            return PXVK_LAUNCH_PARAMS_OK;
        }

        if (index >= PXVK_MAX_EXTRA_ENTRIES) {
            return PXVK_LAUNCH_PARAMS_MALFORMED;
        }

        if (token == PXVK_LAUNCH_PARAM_BUFFER_POINTER) {
            void *value = NULL;
            if (have_buffer) {
                return PXVK_LAUNCH_PARAMS_MALFORMED;
            }
            if (!read_extra_pointer(extra, index++, read_memory, read_context, &value)) {
                return PXVK_LAUNCH_PARAMS_UNREADABLE;
            }
            buffer = value;
            have_buffer = true;
        } else if (token == PXVK_LAUNCH_PARAM_BUFFER_SIZE) {
            void *size_pointer = NULL;
            if (have_size) {
                return PXVK_LAUNCH_PARAMS_MALFORMED;
            }
            if (!read_extra_pointer(extra, index++, read_memory, read_context, &size_pointer)) {
                return PXVK_LAUNCH_PARAMS_UNREADABLE;
            }
            if (size_pointer == NULL ||
                !read_memory(read_context, size_pointer, &buffer_size, sizeof(buffer_size))) {
                return PXVK_LAUNCH_PARAMS_UNREADABLE;
            }
            have_size = true;
        } else {
            return PXVK_LAUNCH_PARAMS_MALFORMED;
        }
    }

    return PXVK_LAUNCH_PARAMS_MALFORMED;
}

const char *pxvk_launch_params_status_name(pxvk_launch_params_status status)
{
    switch (status) {
    case PXVK_LAUNCH_PARAMS_NONE:
        return "none";
    case PXVK_LAUNCH_PARAMS_OK:
        return "ok";
    case PXVK_LAUNCH_PARAMS_MALFORMED:
        return "malformed";
    case PXVK_LAUNCH_PARAMS_UNREADABLE:
        return "unreadable";
    default:
        return "unknown";
    }
}
