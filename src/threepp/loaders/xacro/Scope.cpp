
#include "threepp/loaders/xacro/Scope.hpp"

#include "threepp/loaders/xacro/Diagnostics.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::xacro;

void Scope::pushFrame() {

    frames_.emplace_back();
}

void Scope::popFrame() {

    if (frames_.empty()) throw XacroError("scope frame underflow");

    auto frame = std::move(frames_.back());
    frames_.pop_back();

    for (auto it = frame.rbegin(); it != frame.rend(); ++it) {
        if (it->existed) {
            bindings_[it->name] = it->previous;
        } else {
            bindings_.erase(it->name);
        }
    }
}

bool Scope::has(const std::string& name) const {

    return bindings_.count(name) != 0;
}

const Value* Scope::find(const std::string& name) const {

    const auto it = bindings_.find(name);
    return it == bindings_.end() ? nullptr : &it->second;
}

Value Scope::get(const std::string& name) const {

    const auto it = bindings_.find(name);
    if (it == bindings_.end()) throw XacroError("undefined property '" + name + "'");
    return it->second;
}

Scope::Saved Scope::capture(const std::string& name) const {

    Saved s;
    s.name = name;
    const auto it = bindings_.find(name);
    s.existed = it != bindings_.end();
    if (s.existed) s.previous = it->second;
    return s;
}

std::optional<Scope::Saved> Scope::take(std::vector<Saved>& frame, const std::string& name) {

    const auto it = std::find_if(frame.begin(), frame.end(),
                                 [&](const Saved& s) { return s.name == name; });
    if (it == frame.end()) return std::nullopt;

    Saved s = *it;
    frame.erase(it);
    return s;
}

bool Scope::records(const std::vector<Saved>& frame, const std::string& name) {

    return std::any_of(frame.begin(), frame.end(), [&](const Saved& s) { return s.name == name; });
}

void Scope::set(const std::string& name, Value value) {

    if (!frames_.empty() && !records(frames_.back(), name)) {
        frames_.back().push_back(capture(name));
    }
    bindings_[name] = std::move(value);
}

void Scope::setParent(const std::string& name, Value value) {

    if (frames_.empty()) {
        bindings_[name] = std::move(value);
        return;
    }

    // whatever the current frame would have restored is exactly what the caller's frame
    // should restore instead, so hand the record over rather than dropping it
    auto moved = take(frames_.back(), name);

    if (frames_.size() >= 2) {
        auto& parent = frames_[frames_.size() - 2];
        if (!records(parent, name)) {
            parent.push_back(moved ? *moved : capture(name));
        }
    }

    bindings_[name] = std::move(value);
}

void Scope::setGlobal(const std::string& name, Value value) {

    for (auto& frame : frames_) {
        take(frame, name);
    }
    bindings_[name] = std::move(value);
}
