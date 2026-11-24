#ifndef __SD_CONDITIONING_CONDITIONING_CACHE_H__
#define __SD_CONDITIONING_CONDITIONING_CACHE_H__

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <string>
#include <tuple>

#include "conditioning/conditioner.hpp"

class ConditioningCache {
    struct Entry {
        ConditionerParams params;
        std::vector<sd::Tensor<float>> ref_images;
        std::vector<MiniMaxH3PresentationItem> references;
        SDCondition condition;

        Entry(const ConditionerParams& input, const SDCondition& output)
            : params(input), condition(output) {
            // Request-owned reference pointers must not outlive the request.
            if (input.ref_images != nullptr) {
                ref_images        = *input.ref_images;
                params.ref_images = &ref_images;
            }
            if (input.minimax_h3_references != nullptr) {
                references                   = *input.minimax_h3_references;
                params.minimax_h3_references = &references;
            }
        }

        Entry(const Entry&)            = delete;
        Entry& operator=(const Entry&) = delete;
    };

    size_t capacity_ = 4;
    std::list<Entry> entries_;

    static bool same_images(const std::vector<sd::Tensor<float>>& a,
                            const std::vector<sd::Tensor<float>>& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                          [](const sd::Tensor<float>& x, const sd::Tensor<float>& y) {
                              return x.shape() == y.shape() && x.values() == y.values();
                          });
    }

    static bool same_params(const ConditionerParams& a, const ConditionerParams& b) {
        const auto fields = [](const ConditionerParams& p) {
            const auto& r = p.ref_image_params;
            return std::tie(p.text, p.clip_skip, p.width, p.height, p.zero_out_masked,
                            r.pass_to_vlm, r.pass_to_dit, r.ref_index_mode,
                            r.force_ref_timestep_zero, r.resize_before_vae, r.vae_input_max_pixels,
                            r.vlm_resize_mode, r.vlm_min_size, r.vlm_max_size, r.resize_vae_to_target,
                            r.ref_image_size);
        };
        if (fields(a) != fields(b) ||
            (a.ref_images == nullptr) != (b.ref_images == nullptr) ||
            (a.minimax_h3_references == nullptr) != (b.minimax_h3_references == nullptr)) {
            return false;
        }
        if (a.ref_images != nullptr && !same_images(*a.ref_images, *b.ref_images)) {
            return false;
        }
        if (a.minimax_h3_references != nullptr &&
            !std::equal(a.minimax_h3_references->begin(), a.minimax_h3_references->end(),
                        b.minimax_h3_references->begin(), b.minimax_h3_references->end(),
                        [](const MiniMaxH3PresentationItem& x, const MiniMaxH3PresentationItem& y) {
                            return x.kind == y.kind && x.timestamps == y.timestamps && same_images(x.frames, y.frames);
                        })) {
            return false;
        }
        return true;
    }

    // Optional persistent cache (SD_COND_CACHE_DIR): text-only conditions that consist of cross-attention tensors only, e.g. the
    // LTX prompt embeddings. The directory must belong to one text encoder; the file name is a hash of the prompt parameters.
    static std::string disk_path(const ConditionerParams& p) {
        const char* dir = std::getenv("SD_COND_CACHE_DIR");
        if (dir == nullptr || *dir == '\0' || (p.ref_images != nullptr && !p.ref_images->empty()) ||
            (p.minimax_h3_references != nullptr && !p.minimax_h3_references->empty())) {
            return "";
        }
        uint64_t h = 1469598103934665603ULL;
        auto mix   = [&h](const void* data, size_t n) {
            for (size_t i = 0; i < n; i++) {
                h = (h ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ULL;
            }
        };
        mix(p.text.data(), p.text.size());
        mix(&p.clip_skip, sizeof(p.clip_skip));
        mix(&p.width, sizeof(p.width));
        mix(&p.height, sizeof(p.height));
        mix(&p.zero_out_masked, sizeof(p.zero_out_masked));
        char name[40];
        snprintf(name, sizeof(name), "/%016llx.cond", static_cast<unsigned long long>(h));
        return std::string(dir) + name;
    }

    static bool crossattn_only(const SDCondition& c) {
        SDCondition rest = c;
        rest.c_crossattn = sd::Tensor<float>();
        rest.extra_c_crossattns.clear();
        return rest.empty() && rest.c_ref_images.empty() && rest.c_ref_audios.empty() && rest.c_image_embeds.empty();
    }

    static void write_tensor(std::ofstream& f, const sd::Tensor<float>& t) {
        int64_t nd = static_cast<int64_t>(t.shape().size());
        f.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
        f.write(reinterpret_cast<const char*>(t.shape().data()), sizeof(int64_t) * nd);
        f.write(reinterpret_cast<const char*>(t.data()), sizeof(float) * t.values().size());
    }

    static bool read_tensor(std::ifstream& f, sd::Tensor<float>& out) {
        int64_t nd = 0;
        if (!f.read(reinterpret_cast<char*>(&nd), sizeof(nd)) || nd < 0 || nd > 8) {
            return false;
        }
        std::vector<int64_t> shape(nd);
        int64_t n = 1;
        if (nd > 0 && !f.read(reinterpret_cast<char*>(shape.data()), sizeof(int64_t) * nd)) {
            return false;
        }
        for (auto d : shape) {
            n *= d;
        }
        std::vector<float> data(static_cast<size_t>(n));
        if (!f.read(reinterpret_cast<char*>(data.data()), sizeof(float) * data.size())) {
            return false;
        }
        out = sd::Tensor<float>(shape, std::move(data));
        return true;
    }

    static bool load_from_disk(const std::string& path, SDCondition& out) {
        std::ifstream f(path, std::ios::binary);
        int64_t magic = 0, extra = 0;
        if (!f || !f.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != 0x31434453 ||
            !f.read(reinterpret_cast<char*>(&extra), sizeof(extra)) || extra < 0 || extra > 16 || !read_tensor(f, out.c_crossattn)) {
            return false;
        }
        for (int64_t i = 0; i < extra; i++) {
            sd::Tensor<float> t;
            if (!read_tensor(f, t)) {
                return false;
            }
            out.extra_c_crossattns.push_back(std::move(t));
        }
        return true;
    }

    static void save_to_disk(const std::string& path, const SDCondition& c) {
        if (!crossattn_only(c) || c.c_crossattn.empty()) {
            LOG_INFO("conditioning disk cache skipped: not a plain cross-attention condition");
            return;
        }
        std::ofstream f(path + ".tmp", std::ios::binary);
        int64_t magic = 0x31434453, extra = static_cast<int64_t>(c.extra_c_crossattns.size());
        f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<const char*>(&extra), sizeof(extra));
        write_tensor(f, c.c_crossattn);
        for (const auto& t : c.extra_c_crossattns) {
            write_tensor(f, t);
        }
        f.close();
        if (f.fail() || std::rename((path + ".tmp").c_str(), path.c_str()) != 0) {
            std::remove((path + ".tmp").c_str());
        }
    }

public:
    void set_capacity(size_t capacity) {
        capacity_ = capacity;
        while (entries_.size() > capacity_) {
            entries_.pop_back();
        }
    }

    void clear() {
        entries_.clear();
    }

    SDCondition get(Conditioner& conditioner, int n_threads, const ConditionerParams& params) {
        if (capacity_ == 0) {
            const std::string disk = disk_path(params);
            SDCondition condition;
            if (!disk.empty() && load_from_disk(disk, condition) && !condition.empty()) {
                LOG_INFO("conditioning disk cache hit");
                return condition;
            }
            condition = conditioner.get_learned_condition(n_threads, params);
            if (!condition.empty() && !disk.empty()) {
                save_to_disk(disk, condition);
            }
            return condition;
        }
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (same_params(it->params, params)) {
                entries_.splice(entries_.begin(), entries_, it);
                LOG_INFO("conditioning cache hit");
                return entries_.front().condition;
            }
        }
        const std::string disk = disk_path(params);
        SDCondition condition;
        if (!disk.empty() && load_from_disk(disk, condition) && !condition.empty()) {
            LOG_INFO("conditioning disk cache hit");
        } else {
            condition = conditioner.get_learned_condition(n_threads, params);
            if (!condition.empty() && !disk.empty()) {
                save_to_disk(disk, condition);
            }
        }
        if (!condition.empty()) {
            if (entries_.size() == capacity_) {
                entries_.pop_back();
            }
            entries_.emplace_front(params, condition);
            LOG_VERBOSE("conditioning cache stored (%zu/%zu)", entries_.size(), capacity_);
        }
        return condition;
    }
};

#endif  // __SD_CONDITIONING_CONDITIONING_CACHE_H__
