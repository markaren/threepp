
#include "threepp/loaders/AsyncGroup.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace threepp;

namespace {

    struct LoadContext {
        std::weak_ptr<AsyncGroup> weak;
        std::function<std::shared_ptr<Group>()> fn;
    };

    void executeLoad(LoadContext* ctx) {
        std::shared_ptr<Group> result;
        try {
            result = ctx->fn();
        } catch (...) {}

        if (auto ag = ctx->weak.lock()) {
            ag->deliverResult(std::move(result));
        }
    }

}// namespace

struct AsyncGroup::Impl {
    std::mutex mutex;
    std::vector<std::shared_ptr<Object3D>> pendingChildren;
    std::vector<LoadedCallback> callbacks;
    std::atomic<bool> hasPending{false};
    std::atomic<bool> loaded{false};
    std::atomic<bool> loading{false};
};

AsyncGroup::AsyncGroup(): pimpl_(std::make_unique<Impl>()) {}

AsyncGroup::~AsyncGroup() = default;

std::string AsyncGroup::type() const {
    return "AsyncGroup";
}

bool AsyncGroup::isLoaded() const {
    return pimpl_->loaded.load(std::memory_order_acquire);
}

bool AsyncGroup::isLoading() const {
    return pimpl_->loading.load(std::memory_order_acquire);
}

void AsyncGroup::deliverResult(std::shared_ptr<Group> result) {
    std::lock_guard lock(pimpl_->mutex);
    if (result) {
        pimpl_->pendingChildren.push_back(std::move(result));
        pimpl_->hasPending.store(true, std::memory_order_release);
    }
    pimpl_->loading = false;
}

void AsyncGroup::setLoading(bool value) {
    pimpl_->loading = value;
}

void AsyncGroup::onLoaded(LoadedCallback cb) {
    if (pimpl_->loaded) {
        cb(*this);
    } else {
        pimpl_->callbacks.push_back(std::move(cb));
    }
}

void AsyncGroup::updateMatrixWorld(bool force) {
    if (pimpl_->hasPending.load(std::memory_order_acquire)) {
        std::lock_guard lock(pimpl_->mutex);
        for (auto& child : pimpl_->pendingChildren) {
            for (auto& anim : child->animations) {
                this->animations.push_back(anim);
            }
            this->add(child);
        }
        pimpl_->pendingChildren.clear();
        pimpl_->loaded = true;
        pimpl_->hasPending.store(false, std::memory_order_release);

        for (auto& cb : pimpl_->callbacks) {
            cb(*this);
        }
        pimpl_->callbacks.clear();
    }
    Group::updateMatrixWorld(force);
}

std::shared_ptr<AsyncGroup> AsyncGroup::create() {
    return std::shared_ptr<AsyncGroup>(new AsyncGroup());
}

std::shared_ptr<Object3D> AsyncGroup::createDefault() {
    return create();
}

std::shared_ptr<AsyncGroup> threepp::loadAsync(std::function<std::shared_ptr<Group>()> loadFn) {
    auto group = AsyncGroup::create();
    group->setLoading(true);
    auto* ctx = new LoadContext{group, std::move(loadFn)};

#ifdef __EMSCRIPTEN__
    emscripten_async_call(
            [](void* arg) {
                auto* c = static_cast<LoadContext*>(arg);
                executeLoad(c);
                delete c;
            },
            ctx, 0);
#else
    std::thread([ctx]() {
        executeLoad(ctx);
        delete ctx;
    }).detach();
#endif

    return group;
}
