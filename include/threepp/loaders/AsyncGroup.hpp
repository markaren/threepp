
#ifndef THREEPP_ASYNCGROUP_HPP
#define THREEPP_ASYNCGROUP_HPP

#include "threepp/loaders/Loader.hpp"
#include "threepp/objects/Group.hpp"

#include <filesystem>
#include <functional>

namespace threepp {

    class AsyncGroup: public Group {

    public:
        using LoadedCallback = std::function<void(AsyncGroup&)>;

        [[nodiscard]] std::string type() const override;

        [[nodiscard]] bool isLoaded() const;

        [[nodiscard]] bool isLoading() const;

        void onLoaded(LoadedCallback cb);

        void updateMatrixWorld(bool force = false) override;

        static std::shared_ptr<AsyncGroup> create();

        void deliverResult(std::shared_ptr<Group> result);
        void setLoading(bool value);

        ~AsyncGroup() override;

    protected:
        std::shared_ptr<Object3D> createDefault() override;

    private:
        AsyncGroup();

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

    std::shared_ptr<AsyncGroup> loadAsync(std::function<std::shared_ptr<Group>()> loadFn);

    template<typename LoaderT, typename... Args>
    std::shared_ptr<AsyncGroup> loadAsync(LoaderT& loader, Args&&... args) {
        return loadAsync([&loader, ...args = std::forward<Args>(args)]() -> std::shared_ptr<Group> {
            return loader.load(args...);
        });
    }

}// namespace threepp

#endif//THREEPP_ASYNCGROUP_HPP
