// Property bindings with macro-call frames.
// One flat binding map plus a stack of frames; every write inside a frame records the
// binding it displaced, and popping the frame replays those records in reverse.

#ifndef THREEPP_XACRO_SCOPE_HPP
#define THREEPP_XACRO_SCOPE_HPP

#include "threepp/loaders/xacro/Value.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace threepp::xacro {

    class Scope {

    public:
        void pushFrame();

        void popFrame();

        [[nodiscard]] std::size_t depth() const { return frames_.size(); }

        [[nodiscard]] bool has(const std::string& name) const;

        [[nodiscard]] const Value* find(const std::string& name) const;

        [[nodiscard]] Value get(const std::string& name) const;

        // write in the current frame
        void set(const std::string& name, Value value);

        // write in the caller's frame: survives the current frame's pop
        void setParent(const std::string& name, Value value);

        // write in the root frame: survives every pop
        void setGlobal(const std::string& name, Value value);

        [[nodiscard]] const std::map<std::string, Value>& bindings() const { return bindings_; }

    private:
        struct Saved {
            std::string name;
            bool existed{};
            Value previous;
        };

        [[nodiscard]] Saved capture(const std::string& name) const;

        static std::optional<Saved> take(std::vector<Saved>& frame, const std::string& name);

        static bool records(const std::vector<Saved>& frame, const std::string& name);

        std::map<std::string, Value> bindings_;
        std::vector<std::vector<Saved>> frames_;
    };

}// namespace threepp::xacro

#endif
