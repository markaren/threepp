#include "threepp/renderers/vulkan/RenderGraph.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace threepp::vulkan::rg {

    namespace {

        constexpr VkAccessFlags2 kWriteBits =
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

        constexpr VkPipelineStageFlags2 kAll = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        constexpr VkAccessFlags2 kAllAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

        bool covers(const std::vector<std::pair<VkPipelineStageFlags2, VkAccessFlags2>>& visible,
                    VkPipelineStageFlags2 stages, VkAccessFlags2 access) {
            for (const auto& [s, a] : visible) {
                if ((s & stages) == stages && (a & access) == access) return true;
            }
            return false;
        }

        const char* layoutName(VkImageLayout l) {
            switch (l) {
                case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
                case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "COLOR_ATT";
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DS_ATT";
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DS_RO";
                case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL: return "DEPTH_ATT";
                case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL: return "DEPTH_RO";
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_RO";
                case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "XFER_SRC";
                case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "XFER_DST";
                case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "PRESENT";
                default: return "?";
            }
        }

        std::string stageNames(VkPipelineStageFlags2 s) {
            if (s == 0) return "NONE";
            if (s == kAll) return "ALL";
            struct N { VkPipelineStageFlags2 bit; const char* name; };
            static const N names[] = {
                    {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, "INDIRECT"},
                    {VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, "VTX_IN"},
                    {VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, "VS"},
                    {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, "FS"},
                    {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, "EFT"},
                    {VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, "LFT"},
                    {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, "COLOR"},
                    {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, "CS"},
                    {VK_PIPELINE_STAGE_2_TRANSFER_BIT, "XFER"},
                    {VK_PIPELINE_STAGE_2_COPY_BIT, "COPY"},
                    {VK_PIPELINE_STAGE_2_CLEAR_BIT, "CLEAR"},
                    {VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, "VTX_ATTR"},
                    {VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, "IDX_IN"},
                    {VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, "AS_BUILD"},
                    {VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, "RT"},
                    {VK_PIPELINE_STAGE_2_HOST_BIT, "HOST"},
            };
            std::string out;
            for (const auto& n : names) {
                if (s & n.bit) {
                    if (!out.empty()) out += '|';
                    out += n.name;
                    s &= ~n.bit;
                }
            }
            if (s) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%s0x%llx", out.empty() ? "" : "|",
                              static_cast<unsigned long long>(s));
                out += buf;
            }
            return out;
        }

    }// namespace

    // ── Access helpers ───────────────────────────────────────────────────────

    Access sampled(VkImageLayout layout, VkPipelineStageFlags2 stages) {
        return {stages, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, layout, false};
    }
    Access storageRead(VkPipelineStageFlags2 stages, VkImageLayout layout) {
        return {stages, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, layout, false};
    }
    Access storageWrite(VkPipelineStageFlags2 stages, VkImageLayout layout) {
        return {stages, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, layout, true};
    }
    Access storageReadWrite(VkPipelineStageFlags2 stages, VkImageLayout layout) {
        return {stages, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, layout, true};
    }
    Access generalRead(VkPipelineStageFlags2 stages) {
        return {stages, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, false};
    }
    Access uniformRead(VkPipelineStageFlags2 stages) {
        return {stages, VK_ACCESS_2_UNIFORM_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED, false};
    }
    Access indirectRead() {
        return {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, false};
    }
    Access transferSrc(VkImageLayout layout) {
        return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, layout, false};
    }
    Access transferDst(VkImageLayout layout) {
        return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, layout, true};
    }
    Access colorAttachment(VkImageLayout layout, bool loadOrBlend) {
        VkAccessFlags2 a = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        if (loadOrBlend) a |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, a, layout, true};
    }
    Access depthAttachment(VkImageLayout layout, bool write) {
        VkAccessFlags2 a = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        if (write) a |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        return {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                a, layout, write};
    }
    Access accelRead(VkPipelineStageFlags2 stages) {
        return {stages, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, VK_IMAGE_LAYOUT_UNDEFINED, false};
    }

    // ── Builder ──────────────────────────────────────────────────────────────

    PassBuilder& PassBuilder::use(ImageHandle image, const Access& access, uint32_t baseMip,
                                  uint32_t mipCount) {
        if (!image.valid()) return *this;
        graph_.compiled_ = false;
        // One access per subresource per pass: a pass's uses run concurrently,
        // so a second declaration of the same range widens the first.
        for (auto& u : graph_.passes_[pass_].uses) {
            if (u.isImage && u.resource == image.index && u.baseMip == baseMip && u.mipCount == mipCount) {
                if (u.access.layout != access.layout) {
                    std::fprintf(stderr, "[RenderGraph] pass '%s' uses image '%s' in two layouts\n",
                                 graph_.passes_[pass_].name, graph_.images_[image.index].name);
                }
                u.access.stages |= access.stages;
                u.access.access |= access.access;
                u.access.write = u.access.write || access.write;
                if (access.finalLayout != VK_IMAGE_LAYOUT_UNDEFINED) u.access.finalLayout = access.finalLayout;
                return *this;
            }
        }
        graph_.passes_[pass_].uses.push_back({true, image.index, baseMip, mipCount, access});
        return *this;
    }

    PassBuilder& PassBuilder::use(BufferHandle buffer, const Access& access) {
        if (!buffer.valid()) return *this;
        graph_.compiled_ = false;
        for (auto& u : graph_.passes_[pass_].uses) {
            if (!u.isImage && u.resource == buffer.index) {
                u.access.stages |= access.stages;
                u.access.access |= access.access;
                u.access.write = u.access.write || access.write;
                return *this;
            }
        }
        graph_.passes_[pass_].uses.push_back({false, buffer.index, 0, 1, access});
        return *this;
    }

    // ── Graph ────────────────────────────────────────────────────────────────

    void RenderGraph::reset() {
        for (auto& p : passes_) {
            p.uses.clear();
            p.barriers.clear();
            p.execute = nullptr;
            freePasses_.push_back(std::move(p));
        }
        passes_.clear();
        for (auto& img : images_) {
            img.mips.clear();
            freeImages_.push_back(std::move(img));
        }
        images_.clear();
        buffers_.clear();
        imageIndex_.clear();
        bufferIndex_.clear();
        memoryIndex_.clear();
        entry_.clear();
        exit_.clear();
        entryMemory_ = exitMemory_ = false;
        compiled_ = false;
    }

    ImageHandle RenderGraph::importImage(const char* name, VkImage image, VkImageAspectFlags aspect,
                                         uint32_t mipLevels, VkImageLayout entryLayout,
                                         VkImageLayout exitLayout) {
        if (image == VK_NULL_HANDLE) return {};
        const auto [it, inserted] = imageIndex_.try_emplace(image, static_cast<uint32_t>(images_.size()));
        if (!inserted) return {it->second};
        Image img;
        if (!freeImages_.empty()) {
            img = std::move(freeImages_.back());
            freeImages_.pop_back();
        }
        img.name        = name ? name : "";
        img.image       = image;
        img.heap        = nullptr;
        img.memOffset   = img.memSize = 0;
        img.aspect      = aspect;
        img.mipLevels   = std::max(mipLevels, 1u);
        img.entryLayout = entryLayout;
        img.exitLayout  = exitLayout == VK_IMAGE_LAYOUT_UNDEFINED ? entryLayout : exitLayout;
        images_.push_back(std::move(img));
        compiled_ = false;
        return {static_cast<uint32_t>(images_.size() - 1)};
    }

    BufferHandle RenderGraph::importBuffer(const char* name, VkBuffer buffer) {
        if (buffer == VK_NULL_HANDLE) return {};
        const auto [it, inserted] = bufferIndex_.try_emplace(buffer, static_cast<uint32_t>(buffers_.size()));
        if (!inserted) return {it->second};
        buffers_.push_back({name ? name : "", buffer, {}});
        compiled_ = false;
        return {static_cast<uint32_t>(buffers_.size() - 1)};
    }

    BufferHandle RenderGraph::importMemory(const char* name) {
        if (!name) name = "";
        for (const uint32_t i : memoryIndex_) {
            if (std::strcmp(buffers_[i].name, name) == 0) return {i};
        }
        memoryIndex_.push_back(static_cast<uint32_t>(buffers_.size()));
        buffers_.push_back({name, VK_NULL_HANDLE, {}});
        compiled_ = false;
        return {static_cast<uint32_t>(buffers_.size() - 1)};
    }

    void RenderGraph::setMemoryRange(VkImage image, const void* heap, VkDeviceSize offset, VkDeviceSize size) {
        const auto it = imageIndex_.find(image);
        if (it == imageIndex_.end() || heap == nullptr) return;
        auto& img       = images_[it->second];
        img.heap        = heap;
        img.memOffset   = offset;
        img.memSize     = size;
        // Another image may have written the memory since this one last held
        // it: its contents are dead at the start of the graph and at the end.
        img.entryLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img.exitLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        compiled_       = false;
    }

    PassBuilder RenderGraph::addPass(const char* name, ExecuteFn execute) {
        if (!freePasses_.empty()) {
            passes_.push_back(std::move(freePasses_.back()));
            freePasses_.pop_back();
        } else {
            passes_.emplace_back();
        }
        passes_.back().name    = name ? name : "";
        passes_.back().execute = std::move(execute);
        compiled_ = false;
        return {*this, static_cast<uint32_t>(passes_.size() - 1)};
    }

    // One access against one resource state. Appends at most one barrier.
    void RenderGraph::plan(State& s, const Access& a, bool isImage, uint32_t resource, uint32_t mip,
                           std::vector<PlannedBarrier>& out) {
        // UNDEFINED on an image: the pass transitions it itself (see Access).
        const bool anyLayout    = isImage && a.layout == VK_IMAGE_LAYOUT_UNDEFINED;
        const bool layoutChange = isImage && !anyLayout && a.layout != s.layout;
        const VkImageLayout inLayout = anyLayout ? s.layout : a.layout;
        // A layout change the pass makes itself is a write for ordering.
        const bool passTransitions = isImage && a.finalLayout != VK_IMAGE_LAYOUT_UNDEFINED &&
                                     (anyLayout || a.finalLayout != inLayout);
        const bool write = a.write || passTransitions;

        if (write || layoutChange) {
            const VkPipelineStageFlags2 src = s.writeStages | s.readers;
            if (src != 0 || layoutChange) {
                PlannedBarrier b;
                b.isImage   = isImage;
                b.resource  = resource;
                b.baseMip   = mip;
                b.mipCount  = 1;
                b.srcStages = src != 0 ? src : VK_PIPELINE_STAGE_2_NONE;
                b.srcAccess = s.writeAccess;
                b.dstStages = a.stages;
                b.dstAccess = a.access;
                b.oldLayout = isImage ? s.layout : VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = isImage ? inLayout : VK_IMAGE_LAYOUT_UNDEFINED;
                out.push_back(b);
            }
            if (isImage) s.layout = inLayout;
            s.visible.clear();
            if (write) {
                s.writeStages = a.stages;
                s.writeAccess = a.access & kWriteBits;
                s.readers     = 0;
                // A read-write pass reads its own writes through its own
                // internal barriers; the graph only tracks the pass boundary.
            } else {
                // A layout transition is a write the barrier performs. It is
                // visible to this access; later accesses chain on its stages.
                s.writeStages = a.stages;
                s.writeAccess = 0;
                s.readers     = a.stages;
                s.visible.emplace_back(a.stages, a.access);
            }
            if (passTransitions) s.layout = a.finalLayout;
            return;
        }

        // A read in the current layout.
        s.readers |= a.stages;
        if (s.writeStages == 0) return;// nothing written since the last full sync
        if (covers(s.visible, a.stages, a.access)) return;
        PlannedBarrier b;
        b.isImage   = isImage;
        b.resource  = resource;
        b.baseMip   = mip;
        b.mipCount  = 1;
        b.srcStages = s.writeStages;
        b.srcAccess = s.writeAccess;
        b.dstStages = a.stages;
        b.dstAccess = a.access;
        b.oldLayout = b.newLayout = isImage ? s.layout : VK_IMAGE_LAYOUT_UNDEFINED;
        out.push_back(b);
        s.visible.emplace_back(a.stages, a.access);
    }

    void RenderGraph::compile() {
        if (compiled_) return;
        entry_.clear();
        exit_.clear();
        for (auto& img : images_) {
            img.mips.resize(img.mipLevels);
            for (auto& m : img.mips) {
                m.touched     = false;
                m.layout      = img.entryLayout;
                m.writeStages = 0;
                m.writeAccess = 0;
                m.readers     = 0;
                m.visible.clear();
            }
        }
        for (auto& buf : buffers_) buf.state = State{};

        // ── Aliasing: who shares memory with whom, and when each is live ────
        aliasErrors_.clear();
        bool anyAlias = false;
        for (auto& img : images_) {
            img.firstPass = UINT32_MAX;
            img.lastPass  = 0;
            img.aliases.clear();
            img.aliasPlanned  = false;
            img.aliasSrcStages = 0;
            img.aliasSrcAccess = 0;
            anyAlias = anyAlias || img.heap != nullptr;
        }
        uint32_t aliasMemory = UINT32_MAX;
        if (anyAlias) {
            for (uint32_t p = 0; p < passes_.size(); ++p) {
                for (const auto& u : passes_[p].uses) {
                    if (!u.isImage) continue;
                    auto& img     = images_[u.resource];
                    img.firstPass = std::min(img.firstPass, p);
                    img.lastPass  = std::max(img.lastPass, p);
                }
            }
            for (uint32_t i = 0; i < images_.size(); ++i) {
                auto& a = images_[i];
                if (!a.heap || a.firstPass == UINT32_MAX) continue;
                for (uint32_t j = i + 1; j < images_.size(); ++j) {
                    auto& b = images_[j];
                    if (b.heap != a.heap || b.firstPass == UINT32_MAX) continue;
                    if (a.memOffset >= b.memOffset + b.memSize || b.memOffset >= a.memOffset + a.memSize) continue;
                    a.aliases.push_back(j);
                    b.aliases.push_back(i);
                    if (a.firstPass <= b.lastPass && b.firstPass <= a.lastPass) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                                      "images '%s' (passes %u..%u) and '%s' (passes %u..%u) share memory",
                                      a.name, a.firstPass, a.lastPass, b.name, b.firstPass, b.lastPass);
                        aliasErrors_.emplace_back(buf);
                    }
                }
            }
            // The barrier that hands memory from one image to the next is a
            // global one: the writes it waits for were made to another image.
            aliasMemory = importMemory("rg.aliasing").index;
            buffers_[aliasMemory].state = State{};
        }

        // First-use layouts go into the entry barrier: nothing in the graph has
        // touched the image yet, so there is no in-graph source to wait for —
        // unless the image shares memory with one used earlier in the graph,
        // in which case the first use waits for that one's last.

        // The boundary barriers are full on the OUTSIDE (what came before the
        // graph and what comes after are unknown to it) and exact on the
        // inside: the stages the graph's own passes declared.
        usedStages_ = 0;
        for (const auto& pass : passes_) {
            for (const auto& u : pass.uses) usedStages_ |= u.access.stages;
        }
        if (usedStages_ == 0) usedStages_ = kAll;

        for (auto& pass : passes_) {
            pass.barriers.clear();
            for (const auto& u : pass.uses) {
                if (u.isImage && !images_[u.resource].aliases.empty() && !images_[u.resource].aliasPlanned) {
                    auto& img        = images_[u.resource];
                    img.aliasPlanned = true;
                    for (const uint32_t o : img.aliases) {
                        const auto& other = images_[o];
                        if (other.firstPass >= img.firstPass) continue;// not used yet
                        for (const auto& m : other.mips) {
                            img.aliasSrcStages |= m.writeStages | m.readers;
                            img.aliasSrcAccess |= m.writeAccess;
                        }
                    }
                    if (img.aliasSrcStages != 0) {
                        PlannedBarrier b;
                        b.isImage   = false;
                        b.resource  = aliasMemory;
                        b.srcStages = img.aliasSrcStages;
                        b.srcAccess = img.aliasSrcAccess;
                        b.dstStages = u.access.stages;
                        b.dstAccess = u.access.access;
                        pass.barriers.push_back(b);
                    }
                }
                if (!u.isImage) {
                    plan(buffers_[u.resource].state, u.access, false, u.resource, 0, pass.barriers);
                    continue;
                }
                auto& img = images_[u.resource];
                const uint32_t first = std::min(u.baseMip, img.mipLevels - 1);
                const uint32_t count = u.mipCount == 0 ? img.mipLevels - first
                                                       : std::min(u.mipCount, img.mipLevels - first);
                for (uint32_t m = first; m < first + count; ++m) {
                    State& s = img.mips[m];
                    if (!s.touched && img.aliasSrcStages != 0) {
                        // Memory handed over from an alias: discard behind the
                        // barrier above rather than in the entry barrier.
                        s.touched = true;
                        if (u.access.layout != VK_IMAGE_LAYOUT_UNDEFINED) {
                            PlannedBarrier b;
                            b.resource  = u.resource;
                            b.baseMip   = m;
                            b.srcStages = img.aliasSrcStages;
                            b.srcAccess = 0;
                            b.dstStages = u.access.stages;
                            b.dstAccess = u.access.access;
                            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                            b.newLayout = u.access.layout;
                            pass.barriers.push_back(b);
                            s.layout = u.access.layout;
                        }
                    } else if (!s.touched) {
                        s.touched = true;
                        if (u.access.layout != VK_IMAGE_LAYOUT_UNDEFINED && u.access.layout != s.layout) {
                            PlannedBarrier b;
                            b.resource  = u.resource;
                            b.baseMip   = m;
                            b.srcStages = kAll;
                            b.srcAccess = VK_ACCESS_2_MEMORY_WRITE_BIT;
                            b.dstStages = usedStages_;
                            b.dstAccess = kAllAccess;
                            b.oldLayout = s.layout;
                            b.newLayout = u.access.layout;
                            entry_.push_back(b);
                            s.layout = u.access.layout;
                        }
                    }
                    plan(s, u.access, true, u.resource, m, pass.barriers);
                }
            }
            // Merge consecutive mips of one image with identical parameters.
            std::vector<PlannedBarrier> merged;
            for (const auto& b : pass.barriers) {
                if (!merged.empty()) {
                    auto& m = merged.back();
                    if (m.isImage && b.isImage && m.resource == b.resource &&
                        m.baseMip + m.mipCount == b.baseMip && m.srcStages == b.srcStages &&
                        m.dstStages == b.dstStages && m.srcAccess == b.srcAccess &&
                        m.dstAccess == b.dstAccess && m.oldLayout == b.oldLayout &&
                        m.newLayout == b.newLayout) {
                        ++m.mipCount;
                        continue;
                    }
                }
                merged.push_back(b);
            }
            pass.barriers = std::move(merged);
        }

        for (uint32_t i = 0; i < images_.size(); ++i) {
            const auto& img = images_[i];
            for (uint32_t m = 0; m < img.mipLevels; ++m) {
                if (!img.mips[m].touched || img.mips[m].layout == img.exitLayout ||
                    img.exitLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                    continue;
                PlannedBarrier b;
                b.resource  = i;
                b.baseMip   = m;
                b.srcStages = usedStages_;
                b.srcAccess = VK_ACCESS_2_MEMORY_WRITE_BIT;
                b.dstStages = kAll;
                b.dstAccess = kAllAccess;
                b.oldLayout = img.mips[m].layout;
                b.newLayout = img.exitLayout;
                if (!exit_.empty() && exit_.back().resource == i &&
                    exit_.back().baseMip + exit_.back().mipCount == m &&
                    exit_.back().oldLayout == b.oldLayout) {
                    ++exit_.back().mipCount;
                } else {
                    exit_.push_back(b);
                }
            }
        }
        // Recording after the graph was written against the old per-stage
        // barriers, which may include one this graph no longer issues (a write
        // outside the graph after a read inside it), so the exit is a full
        // barrier whenever the graph recorded anything.
        entryMemory_ = !passes_.empty();
        exitMemory_  = !passes_.empty();
        compiled_    = true;
    }

    void RenderGraph::record(VkCommandBuffer cb, const std::vector<PlannedBarrier>& barriers,
                             bool globalMemory, VkPipelineStageFlags2 memSrc,
                             VkPipelineStageFlags2 memDst) const {
        if (barriers.empty() && !globalMemory) return;
        auto& ib = imageScratch_;
        auto& bb = bufferScratch_;
        auto& mb = memoryScratch_;
        ib.clear();
        bb.clear();
        mb.clear();
        if (globalMemory) {
            VkMemoryBarrier2 m{};
            m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            m.srcStageMask  = memSrc;
            m.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            m.dstStageMask  = memDst;
            m.dstAccessMask = kAllAccess;
            mb.push_back(m);
        }
        for (const auto& p : barriers) {
            if (p.isImage) {
                const auto& img = images_[p.resource];
                VkImageMemoryBarrier2 b{};
                b.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask                    = p.srcStages;
                b.srcAccessMask                   = p.srcAccess;
                b.dstStageMask                    = p.dstStages;
                b.dstAccessMask                   = p.dstAccess;
                b.oldLayout                       = p.oldLayout;
                b.newLayout                       = p.newLayout;
                b.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                b.image                           = img.image;
                b.subresourceRange.aspectMask     = img.aspect;
                b.subresourceRange.baseMipLevel   = p.baseMip;
                b.subresourceRange.levelCount     = p.mipCount;
                b.subresourceRange.baseArrayLayer = 0;
                b.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
                ib.push_back(b);
            } else if (buffers_[p.resource].buffer == VK_NULL_HANDLE) {
                VkMemoryBarrier2 m{};
                m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                m.srcStageMask  = p.srcStages;
                m.srcAccessMask = p.srcAccess;
                m.dstStageMask  = p.dstStages;
                m.dstAccessMask = p.dstAccess;
                mb.push_back(m);
            } else {
                VkBufferMemoryBarrier2 b{};
                b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                b.srcStageMask        = p.srcStages;
                b.srcAccessMask       = p.srcAccess;
                b.dstStageMask        = p.dstStages;
                b.dstAccessMask       = p.dstAccess;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.buffer              = buffers_[p.resource].buffer;
                b.offset              = 0;
                b.size                = VK_WHOLE_SIZE;
                bb.push_back(b);
            }
        }
        VkDependencyInfo di{};
        di.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount       = static_cast<uint32_t>(mb.size());
        di.pMemoryBarriers          = mb.data();
        di.imageMemoryBarrierCount  = static_cast<uint32_t>(ib.size());
        di.pImageMemoryBarriers     = ib.data();
        di.bufferMemoryBarrierCount = static_cast<uint32_t>(bb.size());
        di.pBufferMemoryBarriers    = bb.data();
        vkCmdPipelineBarrier2(cb, &di);
    }

    void RenderGraph::execute(VkCommandBuffer cb) {
        compile();
        if (passes_.empty()) return;
        record(cb, entry_, entryMemory_, kAll, usedStages_);
        for (const auto& pass : passes_) {
            record(cb, pass.barriers, false, 0, 0);
            if (pass.execute) pass.execute(cb);
        }
        record(cb, exit_, exitMemory_, usedStages_, kAll);
    }

    std::string RenderGraph::dump() const {
        std::ostringstream os;
        auto line = [&](const PlannedBarrier& b) {
            os << "    " << (b.isImage ? images_[b.resource].name : buffers_[b.resource].name);
            if (!b.isImage && buffers_[b.resource].buffer == VK_NULL_HANDLE) os << " (memory)";
            if (b.isImage && images_[b.resource].mipLevels > 1)
                os << " mip " << b.baseMip << "+" << b.mipCount;
            os << ": " << stageNames(b.srcStages) << " -> " << stageNames(b.dstStages);
            if (b.isImage && b.oldLayout != b.newLayout)
                os << "  " << layoutName(b.oldLayout) << " -> " << layoutName(b.newLayout);
            os << "\n";
        };
        os << "entry" << (entryMemory_ ? " (+global)" : "") << "\n";
        for (const auto& b : entry_) line(b);
        for (const auto& p : passes_) {
            os << "pass " << p.name << "\n";
            for (const auto& b : p.barriers) line(b);
        }
        os << "exit" << (exitMemory_ ? " (+global)" : "") << "\n";
        for (const auto& b : exit_) line(b);
        return os.str();
    }

}// namespace threepp::vulkan::rg
