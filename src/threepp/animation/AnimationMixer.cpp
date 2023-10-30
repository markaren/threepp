
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/animation/AnimationAction.hpp"
#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/PropertyMixer.hpp"
#include "threepp/constants.hpp"

using namespace threepp;

namespace {

    struct ClipAction {
        std::vector<AnimationAction*> knownActions;
        std::unordered_map<std::string, AnimationAction*> actionByRoot;
    };

}// namespace

struct AnimationMixer::Impl {

    Object3D* _root;
    int _accuIndex{0};

    int _nActiveActions;
    int _nActiveBindings;
    int _nActiveControlInterpolants;

    std::vector<PropertyBinding*> _bindings;
    std::vector<std::shared_ptr<AnimationAction>> _actions;
    std::unordered_map<std::string, ClipAction> _actionsByClip;
    std::unordered_map<std::string, std::unordered_map<std::string, PropertyMixer*>> _bindingsByRootAndName;

    AnimationMixer& scope;

    explicit Impl(AnimationMixer& scope, Object3D& root): scope(scope), _root(&root) {

        _initMemoryManager();
    }

    void _bindAction(const std::shared_ptr<AnimationAction>& action, AnimationAction* prototypeAction) {

        const auto root = action->_localRoot ? action->_localRoot : this->_root;
        auto& tracks = action->_clip->tracks;
        const auto nTracks = tracks.size();
        auto& bindings = action->_propertyBindings;
        auto& interpolants = action->_interpolants;
        const auto rootUuid = root->uuid;
        auto& bindingsByRoot = this->_bindingsByRootAndName;
        //
        //        //        auto& bindingsByName = bindingsByRoot[ rootUuid ];
        //
        if (!bindingsByRoot.count(rootUuid)) {

            std::unordered_map<std::string, PropertyMixer*> bindingsByName;
            bindingsByRoot[rootUuid] = bindingsByName;
        }

        for (auto i = 0; i != nTracks; ++i) {

            //            const auto& track = tracks[i];
            //            const auto trackName = track->getName();
            //
            //            auto& binding = bindingsByName[trackName];
            //
            //            if (binding != undefined) {
            //
            //                bindings[i] = binding;
            //
            //            } else {
            //
            //                binding = bindings[i];
            //
            //                if (binding != undefined) {
            //
            //                    // existing binding, make sure the cache knows
            //
            //                    if (!binding._cacheIndex) {
            //
            //                        ++binding.referenceCount;
            //                        this->_addInactiveBinding(binding, rootUuid, trackName);
            //                    }
            //
            //                    continue;
            //                }
            //
            //                const auto path = prototypeAction && prototypeAction._propertyBindings[i].binding.parsedPath;
            //
            //                binding = std::make_unique<PropertyMixer>(
            //                        PropertyBinding::create(root, trackName, path),
            //                        track.ValueTypeName, track->getValueSize());
            //
            //                ++binding.referenceCount;
            //                this->_addInactiveBinding(binding, rootUuid, trackName);
            //
            //                bindings[i] = binding;
            //            }

            //            interpolants[i].resultBuffer = binding.buffer;
        }
    }

    void _initMemoryManager() {
        this->_actions.clear();// 'nActiveActions' followed by inactive ones
        this->_nActiveActions = 0;

        this->_actionsByClip = {};
        // inside:
        // {
        // 	knownActions: Array< AnimationAction > - used as prototypes
        // 	actionByRoot: AnimationAction - lookup
        // }


        this->_bindings = {};// 'nActiveBindings' followed by inactive ones
        this->_nActiveBindings = 0;

        this->_bindingsByRootAndName = {};// inside: Map< name, PropertyMixer >


        //        this->_controlInterpolants = {};// same game as above
        this->_nActiveControlInterpolants = 0;
    }

    AnimationAction* clipAction(const std::string& clipName, Object3D* optionalRoot, std::optional<AnimationBlendMode> blendMode) {

        auto root = optionalRoot ? optionalRoot : this->_root;
        auto clip = AnimationClip::findByName(root, clipName);

        return clipAction(clip, root, blendMode);
    }

    // return an action for a clip optionally using a custom root target
    // object (this method allocates a lot of dynamic memory in case a
    // previously unknown clip/root combination is specified)
    AnimationAction* clipAction(std::shared_ptr<AnimationClip> clipObject, Object3D* optionalRoot, std::optional<AnimationBlendMode> blendMode) {

        auto root = optionalRoot ? optionalRoot : this->_root;
        auto rootUuid = root->uuid;

        auto clipUuid = clipObject->uuid();

        AnimationAction* prototypeAction = nullptr;

        if (!blendMode) {

            if (clipObject != nullptr) {

                blendMode = clipObject->blendMode;

            } else {

                blendMode = AnimationBlendMode::Normal;
            }
        }

        if (this->_actionsByClip.count(clipUuid)) {

            auto actionsForClip = this->_actionsByClip[clipUuid];

            if (actionsForClip.actionByRoot.count(rootUuid)) {

                const auto existingAction = actionsForClip.actionByRoot.at(rootUuid);
                if (existingAction->blendMode == blendMode) {

                    return existingAction;
                }
            }

            // we know the clip, so we don't have to parse all
            // the bindings again but can just copy
            prototypeAction = actionsForClip.knownActions[0];

            // also, take the clip from the prototype action
            if (clipObject == nullptr)
                clipObject = prototypeAction->_clip;
        }

        // clip must be known when specified via string
        if (clipObject == nullptr) return nullptr;

        // allocate all resources required to run it
        auto newAction = std::make_shared<AnimationAction>(scope, clipObject, optionalRoot, blendMode);

        this->_bindAction(newAction, prototypeAction);

        // and make the action known to the memory manager
        this->_addInactiveAction(newAction, clipUuid, rootUuid);

        return newAction.get();
    }

    void _addInactiveAction(
            const std::shared_ptr<AnimationAction>& action,
            const std::string& clipUuid, const std::string& rootUuid) {

        auto& actions = this->_actions;
        auto& actionsByClip = this->_actionsByClip;

        ClipAction actionsForClip;

        if (!actionsByClip.count(clipUuid)) {

            actionsForClip = ClipAction{
                    {action.get()},
                    {}};

            action->_byClipCacheIndex = 0;

            actionsByClip[clipUuid] = actionsForClip;

        } else {

            auto& actionsForClip = actionsByClip.at(clipUuid);

            auto& knownActions = actionsForClip.knownActions;

            action->_byClipCacheIndex = knownActions.size();
            knownActions.emplace_back(action.get());
        }

        action->_cacheIndex = actions.size();
        actions.emplace_back(action);

        actionsForClip.actionByRoot[rootUuid] = action.get();
    }

    void stopAllAction() {

        auto& actions = this->_actions;
        const auto nActions = this->_nActiveActions;

        for (auto i = nActions - 1; i >= 0; --i) {

            actions[i]->stop();
        }
    }

    void update(float deltaTime) {

        deltaTime *= this->scope.timeScale;

        auto& actions = this->_actions;
        const auto nActions = this->_nActiveActions;

        const auto time = this->scope.time += deltaTime;
        const auto timeDirection = math::sgn(deltaTime);

        const auto accuIndex = this->_accuIndex ^= 1;

        // run active actions

        for (auto i = 0; i != nActions; ++i) {

            auto& action = actions[i];

            action->_update(time, deltaTime, timeDirection, accuIndex);
        }

        // update scene graph

        auto& bindings = this->_bindings;
        auto nBindings = this->_nActiveBindings;

        for (auto i = 0; i != nBindings; ++i) {

            //            bindings[i]->apply(accuIndex);
        }
    }

    void _activateAction(const std::shared_ptr<AnimationAction> action) {

        if (!this->_isActiveAction(*action)) {

            if (!action->_cacheIndex) {

                // this action has been forgotten by the cache, but the user
                // appears to be still using it -> rebind

                const auto rootUuid = (action->_localRoot ? action->_localRoot : _root)->uuid;
                const auto clipUuid = action->_clip->uuid();

                AnimationAction* prototypeAction = nullptr;
                if (this->_actionsByClip.count(clipUuid)) {
                    prototypeAction = this->_actionsByClip[clipUuid].knownActions[0];
                }
                this->_bindAction(action,
                                  prototypeAction);

                this->_addInactiveAction(action, clipUuid, rootUuid);
            }

            const auto& bindings = action->_propertyBindings;

            // increment reference counts / sort out state
            for (unsigned i = 0, n = bindings.size(); i != n; ++i) {

                auto& binding = bindings[i];

                //                if (binding.useCount++ == 0) {
                //
                //                    this->_lendBinding(binding);
                //                    binding.saveOriginalState();
                //                }
            }

            this->_lendAction(action);
        }
    }

    void _deactivateAction(const std::shared_ptr<AnimationAction>& action) {

        if (this->_isActiveAction(*action)) {

            const auto& bindings = action->_propertyBindings;

            // decrement reference counts / sort out state
            for (unsigned i = 0, n = bindings.size(); i != n; ++i) {

                const auto& binding = bindings[i];

                //                if ( -- binding.useCount === 0 ) {
                //
                //                    binding.restoreOriginalState();
                //                    this->_takeBackBinding( binding );
                //
                //                }
            }

            this->_takeBackAction(action);
        }
    }

    void _lendAction(const std::shared_ptr<AnimationAction>& action) {

        // [ active actions |  inactive actions  ]
        // [  active actions >| inactive actions ]
        //                 s        a
        //                  <-swap->
        //                 a        s

        auto& actions = this->_actions;
        const auto prevIndex = *action->_cacheIndex;

        const auto lastActiveIndex = this->_nActiveActions++;

        auto& firstInactiveAction = actions[lastActiveIndex];

        action->_cacheIndex = lastActiveIndex;
        actions[lastActiveIndex] = action;

        firstInactiveAction->_cacheIndex = prevIndex;
        actions[prevIndex] = firstInactiveAction;
    }

    void _takeBackAction(const std::shared_ptr<AnimationAction>& action) {

        // [  active actions  | inactive actions ]
        // [ active actions |< inactive actions  ]
        //        a        s
        //         <-swap->
        //        s        a

        auto& actions = this->_actions;
        const auto prevIndex = *action->_cacheIndex;

        const auto firstInactiveIndex = --this->_nActiveActions;

        const auto& lastActiveAction = actions[firstInactiveIndex];

        action->_cacheIndex = firstInactiveIndex;
        actions[firstInactiveIndex] = action;

        lastActiveAction->_cacheIndex = prevIndex;
        actions[prevIndex] = lastActiveAction;
    }

    bool _isActiveAction(AnimationAction& action) const {

        const auto index = action._cacheIndex;
        return index && index < _nActiveActions;
    }

    // Allows you to seek to a specific time in an animation.
    void setTime(float timeInSeconds) {

        scope.time = 0;// Zero out time attribute for AnimationMixer object;
        for (const auto& _action : this->_actions) {

            _action->time = 0;// Zero out time attribute for all associated AnimationAction objects.
        }

        this->update(timeInSeconds);// Update used to set exact time. Returns "this" AnimationMixer object.
    }
};

AnimationMixer::AnimationMixer(Object3D& root)
    : pimpl_(std::make_unique<Impl>(*this, root)) {}

AnimationAction* AnimationMixer::clipAction(const std::shared_ptr<AnimationClip>& clip,
                                            Object3D* optionalRoot,
                                            std::optional<AnimationBlendMode> blendMode) {

    return pimpl_->clipAction(clip, optionalRoot, blendMode);
}

void AnimationMixer::stopAllAction() {

    pimpl_->stopAllAction();
}


void AnimationMixer::update(float dt) {

    pimpl_->update(dt);
}

bool AnimationMixer::_isActiveAction(AnimationAction& action) {

    return pimpl_->_isActiveAction(action);
}

void AnimationMixer::_activateAction(const std::shared_ptr<AnimationAction>& action) {

    pimpl_->_activateAction(action);
}

void AnimationMixer::_deactivateAction(const std::shared_ptr<AnimationAction>& action) {

    pimpl_->_deactivateAction(action);
}

AnimationMixer::~AnimationMixer() = default;
