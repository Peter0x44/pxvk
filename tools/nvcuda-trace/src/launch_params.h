#ifndef PXVK_LAUNCH_PARAMS_H
#define PXVK_LAUNCH_PARAMS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*pxvk_read_memory_fn)(void *context, const void *address, void *output, size_t size);

typedef enum pxvk_launch_params_status {
    PXVK_LAUNCH_PARAMS_NONE = 0,
    PXVK_LAUNCH_PARAMS_OK,
    PXVK_LAUNCH_PARAMS_MALFORMED,
    PXVK_LAUNCH_PARAMS_UNREADABLE
} pxvk_launch_params_status;

typedef struct pxvk_launch_params {
    const void *buffer;
    size_t size;
} pxvk_launch_params;

pxvk_launch_params_status pxvk_parse_launch_extra(
    void *const *extra,
    pxvk_read_memory_fn read_memory,
    void *read_context,
    pxvk_launch_params *result);

const char *pxvk_launch_params_status_name(pxvk_launch_params_status status);

#ifdef __cplusplus
}
#endif

#endif
