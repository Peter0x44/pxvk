// Vulkan compute backend for pxvk.
//
// CUDA device memory is one device-local storage buffer ("heap"); a CUDA
// device pointer is heap_base + byte offset. Every operation (uploads,
// fills, copies, dispatches) is recorded into a single command batch in
// issue order with full barriers between commands, which preserves CUDA
// default-stream semantics. The batch is submitted only when the host needs
// results (download/synchronize) or a staging ring fills up.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pxvk::vk {

struct Capabilities {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t api_version = 0;
    bool discrete = false;
    bool int64 = false;
    bool native_ftz = false;       // fp32 denormFlushToZero execution mode
    bool preserve_nan = false;     // fp32 SignedZeroInfNanPreserve execution mode
    bool subgroup_control = false; // can require a subgroup size of 32
    std::uint32_t subgroup_size = 0;
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_workgroup_size[3] = {0, 0, 0};
    std::uint32_t max_workgroup_count[3] = {0, 0, 0};
    std::uint32_t max_shared_bytes = 0;
    std::uint64_t max_heap_bytes = 0;
    std::uint64_t device_local_bytes = 0;
    std::string pipeline_cache_uuid; // hex; identifies compatible pipeline caches
};

struct Shader;
struct HostBuffer;

struct DispatchShape {
    std::uint32_t grid[3] = {1, 1, 1};
    std::uint32_t block[3] = {1, 1, 1};
    std::uint32_t shared_bytes = 0; // static + dynamic
};

struct Stats {
    std::uint64_t submits = 0, dispatches = 0, uploads = 0, downloads = 0, pipelines = 0;
    std::uint64_t upload_bytes = 0, download_bytes = 0;
    double wait_ms = 0;           // host time blocked on fences
    double pipeline_ms = 0;       // host time creating pipelines
};

// GPU time per dispatch label, accumulated while timing is enabled.
struct KernelTime {
    std::uint64_t count = 0;
    double total_ms = 0;
};

class Device {
public:
    Device();
    ~Device();
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // Lists physical devices as "index: name".
    static std::vector<std::string> enumerate(std::string &error);

    // index < 0 picks the first discrete GPU, else the first GPU.
    bool initialize(int index, std::string &error);
    // Selects by case-insensitive device-name substring, or by index when the
    // text is a number; empty text means the default choice. Device order is
    // not stable across processes (the loader applies OS GPU preferences), so
    // names are the reliable way to pick a GPU.
    bool initialize(const std::string &selector, std::string &error);
    void shutdown();
    const Capabilities &caps() const;

    // Heap management. Growing preserves contents.
    bool ensure_heap(std::uint64_t bytes, std::string &error);
    std::uint64_t heap_size() const;
    void set_heap_base(std::uint32_t base);
    std::uint32_t heap_base() const;

    // Recorded transfers on device addresses (heap_base relative handled
    // by the caller: offsets here are heap byte offsets).
    bool upload(std::uint64_t offset, const void *data, std::size_t bytes, std::string &error);
    bool fill(std::uint64_t offset, std::uint8_t value, std::size_t bytes, std::string &error);
    bool copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes, std::string &error);
    // Submits outstanding work, waits, and reads memory back.
    bool download(void *data, std::uint64_t offset, std::size_t bytes, std::string &error);

    // Host-visible, cached memory used for CUDA pinned allocations. Copies
    // between it and the heap are recorded like any other command, which
    // gives cuMemcpy*Async semantics without an intermediate memcpy.
    HostBuffer *allocate_host(std::size_t bytes, void **mapped, std::string &error);
    void free_host(HostBuffer *buffer); // synchronizes first
    bool copy_from_host(HostBuffer *buffer, std::uint64_t source, std::uint64_t heap_offset, std::size_t bytes,
                        std::string &error);
    bool copy_to_host(HostBuffer *buffer, std::uint64_t destination, std::uint64_t heap_offset, std::size_t bytes,
                      std::string &error);
    // True while recorded commands have not been submitted and completed.
    bool busy() const;

    Shader *create_shader(const std::vector<std::uint32_t> &spirv, std::uint32_t launch_block_bytes,
                          bool warp_synchronous, bool shared_in_storage, std::string &error);
    // `label` (kept by pointer, must outlive the device) names the dispatch
    // in GPU timing profiles.
    bool dispatch(Shader *shader, const DispatchShape &shape, const void *launch_block,
                  std::size_t launch_block_bytes, std::string &error, const char *label = nullptr);

    bool synchronize(std::string &error);

    // Persistent VkPipelineCache. load() must run before the first pipeline
    // is created; incompatible or corrupt data is ignored by the driver.
    void load_pipeline_cache(const std::string &path);
    bool save_pipeline_cache(const std::string &path);
    const Stats &stats() const;

    // Optional GPU timing of each dispatch (for profiling tools).
    void set_timing(bool enabled);
    // Durations in milliseconds of the dispatches of the last completed
    // batch, in issue order.
    const std::vector<double> &last_dispatch_times() const;
    // Accumulated GPU time per label since the last call (timing mode).
    std::vector<std::pair<std::string, KernelTime>> take_profile();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pxvk::vk
