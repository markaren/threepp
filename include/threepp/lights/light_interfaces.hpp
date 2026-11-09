
#ifndef THREEPP_LIGHT_INTERFACES_HPP
#define THREEPP_LIGHT_INTERFACES_HPP

#include "threepp/lights/LightShadow.hpp"

#include <algorithm>
#include <vector>

namespace threepp {

    class LightWithShadow {

    public:
        std::shared_ptr<LightShadow> shadow;

        virtual ~LightWithShadow() = default;

    protected:
        explicit LightWithShadow(const std::shared_ptr<LightShadow>& shadow): shadow(shadow) {}
    };

    class LightWithTarget {

    public:
        [[nodiscard]] const Object3D& target() const {

            return target_ ? *target_ : defaultTarget;
        }

        void setTarget(Object3D& target) {

            this->target_ = &target;
        }

        virtual ~LightWithTarget() = default;

    protected:
        // Called by a light's copy() after Object3D::copy has copied the
        // children. A target inside the source's own subtree (a loaded glTF
        // light's "target" child) was copied along with it, so the copy aims at
        // that copy, found by the same child-index path; without the children
        // (recursive false) it falls back to the default target, since the
        // source's child dies with the source. Any other target belongs to the
        // application's scene and is shared, as before. The default target
        // cannot be moved through this interface, so there is nothing to copy
        // when the source has none.
        void copyTarget(const LightWithTarget& source, const Object3D& sourceObject, Object3D& self, bool recursive) {

            this->target_ = nullptr;
            Object3D* target = source.target_;
            if (!target) return;

            // The child-index path from the source down to its target, if the
            // target is in its subtree.
            std::vector<size_t> path;
            const Object3D* node = target;
            while (node && node != &sourceObject) {
                const Object3D* parent = node->parent;
                if (!parent) break;
                const auto& siblings = parent->children;
                path.push_back(static_cast<size_t>(std::find(siblings.begin(), siblings.end(), node) - siblings.begin()));
                node = parent;
            }
            if (node != &sourceObject) {
                this->target_ = target;// outside the source: shared
                return;
            }
            if (!recursive && !path.empty()) return;// not copied: the default target

            // Object3D::copy appends the copied children after any the copy
            // already had, so the i-th source child is i places into the last
            // sourceChildren.size() of the copy's.
            const Object3D* from = &sourceObject;
            Object3D* to = &self;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                const size_t skip = to->children.size() - std::min(to->children.size(), from->children.size());
                if (*it >= from->children.size() || skip + *it >= to->children.size()) return;
                from = from->children[*it];
                to = to->children[skip + *it];
            }
            this->target_ = to;
        }

    private:
        Object3D* target_ = nullptr;
        Object3D defaultTarget;
    };

}// namespace threepp

#endif//THREEPP_LIGHT_INTERFACES_HPP
