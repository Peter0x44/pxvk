#include <volk.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *independence_name(VkShaderFloatControlsIndependence value)
{
    switch (value) {
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY:
        return "32_BIT_ONLY";
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL:
        return "ALL";
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE:
        return "NONE";
    default:
        return "UNKNOWN";
    }
}

static bool has_extension(
    const VkExtensionProperties *extensions,
    uint32_t count,
    const char *name)
{
    uint32_t index;
    for (index = 0; index < count; ++index) {
        if (strcmp(extensions[index].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static void print_bool(FILE *output, const char *name, VkBool32 value, bool comma)
{
    fprintf(output, "        \"%s\": %s%s\n", name, value ? "true" : "false", comma ? "," : "");
}

static int compare_extension(const void *left, const void *right)
{
    const VkExtensionProperties *a = left;
    const VkExtensionProperties *b = right;
    return strcmp(a->extensionName, b->extensionName);
}

int main(int argc, char **argv)
{
    uint32_t loader_version = VK_API_VERSION_1_0;
    VkApplicationInfo application = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "pxvk-vulkan-probe",
        .applicationVersion = 1,
        .pEngineName = "none",
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice *devices = NULL;
    uint32_t device_count = 0;
    FILE *output = NULL;
    VkResult result;
    uint32_t device_index;

    if (argc != 2) {
        fprintf(stderr, "usage: vulkan-probe <output.json>\n");
        return 2;
    }
    if (volkInitialize() != VK_SUCCESS) {
        fputs("volkInitialize failed\n", stderr);
        return 3;
    }
    if (vkEnumerateInstanceVersion != NULL) {
        vkEnumerateInstanceVersion(&loader_version);
    }
    if (loader_version < application.apiVersion) {
        application.apiVersion = loader_version;
    }
    result = vkCreateInstance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", result);
        return 4;
    }
    volkLoadInstance(instance);
    result = vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (result != VK_SUCCESS || device_count == 0) {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed: %d count=%u\n", result, device_count);
        vkDestroyInstance(instance, NULL);
        return 5;
    }
    devices = calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
        vkDestroyInstance(instance, NULL);
        return 6;
    }
    result = vkEnumeratePhysicalDevices(instance, &device_count, devices);
    if (result != VK_SUCCESS) {
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 7;
    }
    output = fopen(argv[1], "wb");
    if (output == NULL) {
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 8;
    }

    fprintf(output,
            "{\n  \"pointer_bits\": %u,\n"
            "  \"vulkan_header_version\": \"%u.%u.%u\",\n"
            "  \"loader_api_version\": \"%u.%u.%u\",\n  \"devices\": [\n",
            (unsigned int)(sizeof(void *) * 8),
            VK_API_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE),
            VK_API_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE),
            VK_API_VERSION_PATCH(VK_HEADER_VERSION_COMPLETE),
            VK_API_VERSION_MAJOR(loader_version),
            VK_API_VERSION_MINOR(loader_version),
            VK_API_VERSION_PATCH(loader_version));

    for (device_index = 0; device_index < device_count; ++device_index) {
        VkPhysicalDevice device = devices[device_index];
        VkPhysicalDeviceSubgroupProperties subgroup = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        };
        VkPhysicalDeviceFloatControlsProperties float_controls = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
            .pNext = &subgroup,
        };
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup_control_properties = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
            .pNext = &float_controls,
        };
        VkPhysicalDeviceMaintenance3Properties maintenance3 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
            .pNext = &subgroup_control_properties,
        };
        VkPhysicalDeviceMaintenance4Properties maintenance4 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
            .pNext = &maintenance3,
        };
        VkPhysicalDeviceProperties2 properties2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &maintenance4,
        };
        VkPhysicalDeviceShaderFloatControls2Features float_controls2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
        };
        VkPhysicalDeviceRobustness2FeaturesEXT robustness2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
            .pNext = &float_controls2,
        };
        VkPhysicalDeviceVulkanMemoryModelFeatures memory_model = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
            .pNext = &robustness2,
        };
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
            .pNext = &memory_model,
        };
        VkPhysicalDeviceShaderFloat16Int8Features float16_int8 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            .pNext = &subgroup_control,
        };
        VkPhysicalDevice8BitStorageFeatures storage8 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
            .pNext = &float16_int8,
        };
        VkPhysicalDevice16BitStorageFeatures storage16 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            .pNext = &storage8,
        };
        VkPhysicalDeviceFeatures2 features2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &storage16,
        };
        VkQueueFamilyProperties *queue_families = NULL;
        uint32_t queue_family_count = 0;
        uint32_t compute_queue_count = 0;
        VkExtensionProperties *extensions = NULL;
        uint32_t extension_count = 0;
        uint32_t extension_index;
        uint32_t queue_index;

        vkGetPhysicalDeviceProperties2(device, &properties2);
        vkGetPhysicalDeviceFeatures2(device, &features2);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, NULL);
        queue_families = calloc(queue_family_count, sizeof(*queue_families));
        if (queue_families != NULL) {
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families);
            for (queue_index = 0; queue_index < queue_family_count; ++queue_index) {
                if ((queue_families[queue_index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    compute_queue_count += queue_families[queue_index].queueCount;
                }
            }
        }
        vkEnumerateDeviceExtensionProperties(device, NULL, &extension_count, NULL);
        extensions = calloc(extension_count, sizeof(*extensions));
        if (extensions != NULL) {
            vkEnumerateDeviceExtensionProperties(device, NULL, &extension_count, extensions);
            qsort(extensions, extension_count, sizeof(*extensions), compare_extension);
        }

        fprintf(output,
                "    {\n      \"name\": \"%s\",\n      \"vendor_id\": \"0x%04x\",\n"
                "      \"device_id\": \"0x%04x\",\n      \"api_version\": \"%u.%u.%u\",\n"
                "      \"driver_version_raw\": %u,\n      \"compute_queue_count\": %u,\n",
                properties2.properties.deviceName,
                properties2.properties.vendorID,
                properties2.properties.deviceID,
                VK_API_VERSION_MAJOR(properties2.properties.apiVersion),
                VK_API_VERSION_MINOR(properties2.properties.apiVersion),
                VK_API_VERSION_PATCH(properties2.properties.apiVersion),
                properties2.properties.driverVersion,
                compute_queue_count);

        fprintf(output,
                "      \"limits\": {\n"
                "        \"max_compute_shared_memory_size\": %u,\n"
                "        \"max_compute_work_group_invocations\": %u,\n"
                "        \"max_compute_work_group_size\": [%u, %u, %u],\n"
                "        \"max_compute_work_group_count\": [%u, %u, %u],\n"
                "        \"max_storage_buffer_range\": %u,\n"
                "        \"min_storage_buffer_offset_alignment\": %llu,\n"
                "        \"max_push_constants_size\": %u,\n"
                "        \"max_per_stage_storage_buffers\": %u,\n"
                "        \"max_descriptor_set_storage_buffers\": %u,\n"
                "        \"max_texel_buffer_elements\": %u,\n"
                "        \"max_memory_allocation_size\": %llu,\n"
                "        \"max_buffer_size\": %llu\n"
                "      },\n",
                properties2.properties.limits.maxComputeSharedMemorySize,
                properties2.properties.limits.maxComputeWorkGroupInvocations,
                properties2.properties.limits.maxComputeWorkGroupSize[0],
                properties2.properties.limits.maxComputeWorkGroupSize[1],
                properties2.properties.limits.maxComputeWorkGroupSize[2],
                properties2.properties.limits.maxComputeWorkGroupCount[0],
                properties2.properties.limits.maxComputeWorkGroupCount[1],
                properties2.properties.limits.maxComputeWorkGroupCount[2],
                properties2.properties.limits.maxStorageBufferRange,
                (unsigned long long)properties2.properties.limits.minStorageBufferOffsetAlignment,
                properties2.properties.limits.maxPushConstantsSize,
                properties2.properties.limits.maxPerStageDescriptorStorageBuffers,
                properties2.properties.limits.maxDescriptorSetStorageBuffers,
                properties2.properties.limits.maxTexelBufferElements,
                (unsigned long long)maintenance3.maxMemoryAllocationSize,
                (unsigned long long)maintenance4.maxBufferSize);

        fprintf(output,
                "      \"subgroup\": {\n"
                "        \"default_size\": %u,\n"
                "        \"supported_stages\": \"0x%x\",\n"
                "        \"supported_operations\": \"0x%x\",\n"
                "        \"min_size\": %u,\n"
                "        \"max_size\": %u,\n"
                "        \"max_compute_workgroup_subgroups\": %u,\n"
                "        \"required_size_stages\": \"0x%x\"\n"
                "      },\n",
                subgroup.subgroupSize,
                subgroup.supportedStages,
                subgroup.supportedOperations,
                subgroup_control_properties.minSubgroupSize,
                subgroup_control_properties.maxSubgroupSize,
                subgroup_control_properties.maxComputeWorkgroupSubgroups,
                subgroup_control_properties.requiredSubgroupSizeStages);

        fprintf(output,
                "      \"float_controls\": {\n"
                "        \"denorm_behavior_independence\": \"%s\",\n"
                "        \"rounding_mode_independence\": \"%s\",\n",
                independence_name(float_controls.denormBehaviorIndependence),
                independence_name(float_controls.roundingModeIndependence));
        print_bool(output, "signed_zero_inf_nan_preserve_float32", float_controls.shaderSignedZeroInfNanPreserveFloat32, true);
        print_bool(output, "denorm_preserve_float32", float_controls.shaderDenormPreserveFloat32, true);
        print_bool(output, "denorm_flush_to_zero_float32", float_controls.shaderDenormFlushToZeroFloat32, true);
        print_bool(output, "rounding_mode_rte_float32", float_controls.shaderRoundingModeRTEFloat32, true);
        print_bool(output, "rounding_mode_rtz_float32", float_controls.shaderRoundingModeRTZFloat32, true);
        print_bool(output, "shader_float_controls_2", float_controls2.shaderFloatControls2, false);
        fprintf(output, "      },\n");

        fprintf(output, "      \"features\": {\n");
        print_bool(output, "robust_buffer_access", features2.features.robustBufferAccess, true);
        print_bool(output, "shader_int64", features2.features.shaderInt64, true);
        print_bool(output, "shader_float64", features2.features.shaderFloat64, true);
        print_bool(output, "storage_buffer_16_bit_access", storage16.storageBuffer16BitAccess, true);
        print_bool(output, "storage_buffer_8_bit_access", storage8.storageBuffer8BitAccess, true);
        print_bool(output, "uniform_and_storage_buffer_8_bit_access", storage8.uniformAndStorageBuffer8BitAccess, true);
        print_bool(output, "shader_int8", float16_int8.shaderInt8, true);
        print_bool(output, "subgroup_size_control", subgroup_control.subgroupSizeControl, true);
        print_bool(output, "compute_full_subgroups", subgroup_control.computeFullSubgroups, true);
        print_bool(output, "vulkan_memory_model", memory_model.vulkanMemoryModel, true);
        print_bool(output, "robust_buffer_access_2", robustness2.robustBufferAccess2, false);
        fprintf(output, "      },\n");

        fprintf(output,
                "      \"extension_support\": {\n"
                "        \"VK_KHR_shader_float_controls\": %s,\n"
                "        \"VK_KHR_shader_float_controls2\": %s,\n"
                "        \"VK_EXT_subgroup_size_control\": %s,\n"
                "        \"VK_KHR_8bit_storage\": %s,\n"
                "        \"VK_KHR_16bit_storage\": %s,\n"
                "        \"VK_KHR_vulkan_memory_model\": %s,\n"
                "        \"VK_EXT_robustness2\": %s,\n"
                "        \"VK_KHR_buffer_device_address\": %s\n"
                "      },\n",
                has_extension(extensions, extension_count, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_KHR_8BIT_STORAGE_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) ? "true" : "false",
                has_extension(extensions, extension_count, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) ? "true" : "false");

        fprintf(output, "      \"extensions\": [");
        for (extension_index = 0; extension_index < extension_count; ++extension_index) {
            fprintf(output,
                    "%s\"%s\"",
                    extension_index == 0 ? "" : ", ",
                    extensions[extension_index].extensionName);
        }
        fprintf(output, "]\n    }%s\n", device_index + 1 == device_count ? "" : ",");

        free(extensions);
        free(queue_families);
    }

    fprintf(output, "  ]\n}\n");
    fclose(output);
    free(devices);
    vkDestroyInstance(instance, NULL);
    return 0;
}
