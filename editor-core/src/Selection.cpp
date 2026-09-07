
#include "threepp/extras/editor/Selection.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>

using namespace threepp;
using namespace threepp::editor;


std::string Selection::uuid() const {

    return uuid_;
}

void Selection::set(Object3D* object) {

    if (object == current_) return;

    current_ = object;
    uuid_ = object ? object->uuid : std::string{};

    // Copy first: a listener is allowed to change the selection again (the
    // hierarchy scrolls, the inspector resets its drag state), which would
    // otherwise invalidate the iterator we are walking.
    const auto snapshot = listeners_;
    for (const auto& entry : snapshot) entry.fn(current_);
}

int Selection::onChange(std::function<void(Object3D*)> listener) {

    if (!listener) return 0;
    const int id = nextId_++;
    listeners_.push_back({id, std::move(listener)});
    return id;
}

void Selection::removeListener(int id) {

    listeners_.erase(std::remove_if(listeners_.begin(), listeners_.end(),
                                    [id](const Entry& e) { return e.id == id; }),
                     listeners_.end());
}
