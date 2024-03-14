
#ifndef THREEPP_UTILS_SCOPE_HPP
#define THREEPP_UTILS_SCOPE_HPP

#include <memory>

namespace threepp::utils {

    using ScopeExit = std::shared_ptr<void>;

    /// Execute the function when the last copy of the returned value goes out of scope
    template<typename TFn>
    [[nodiscard]] ScopeExit at_scope_exit(TFn&& fn) {
        return std::shared_ptr<void>((void*) (nullptr), [fn = std::forward<TFn>(fn)](auto) {
            fn();
        });
    }

    /// Set @p var with @p value when the last copy of the returned value goes out of scope
    template<typename T>
    [[nodiscard]] ScopeExit set_at_scope_exit(T& var, T&& value) {
        return at_scope_exit([&var, value = std::forward<T>(value)]() {
            var = value;
        });
    }

    /// Set @p var with @newValue immediately and then reset @p var with it's original value with the
    /// last copy of the returned value goes out of scope.
    template<typename T>
    [[nodiscard]] ScopeExit reset_at_scope_exit(T& var, T&& newValue) {
        T oldValue = std::move(var);
        var = std::forward<T>(newValue);
        return set_at_scope_exit(var, std::move(oldValue));
    }

}// namespace threepp::utils

#endif//THREEPP_UTILS_SCOPE_HPP
