
#include "threepp/extras/editor/Command.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/objects/SplatCloud.hpp"

#include <algorithm>
#include <unordered_map>

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

    // One detached subtree the history is keeping alive, and the NEWEST position
    // in each stack that holds it.
    struct Held {
        std::size_t bytes = 0;
        std::size_t newestUndo = kNoIndex;
        std::size_t newestRedo = kNoIndex;
    };

    // Everything the history retains, in one pass and weighed once per subtree.
    //
    // The two indices are what let the prune price a candidate prefix without
    // walking the stacks again: a subtree survives dropping the first `u` undo
    // entries and the first `r` redo entries exactly when one of its holders sits
    // at or past that position. Without them, pricing 200 candidates means 200
    // traversals of every retained subtree — and being over budget is a legal
    // steady state here, so that cost would be paid on every push.
    std::vector<Held> survey(const std::vector<std::unique_ptr<Command>>& undo,
                             const std::vector<std::unique_ptr<Command>>& redo) {

        std::vector<Held> held;
        std::unordered_map<const Object3D*, std::size_t> seen;
        std::vector<Object3D*> roots;

        const auto note = [&](const Command& command, std::size_t position, bool isUndo) {
            roots.clear();
            command.retainedRoots(roots);
            for (auto* root : roots) {

                // Parented means the scene co-owns it and dropping the command
                // frees nothing. What survives this test is a set of DISJOINT
                // subtrees — anything nested inside another reported root has a
                // parent — so no splat below is weighed twice.
                if (!root || root->parent) continue;

                const auto [it, fresh] = seen.try_emplace(root, held.size());
                if (fresh) held.push_back(Held{retainedSubtreeBytes(*root), kNoIndex, kNoIndex});

                // Positions arrive in ascending order, so the last write wins and
                // is the newest holder.
                if (isUndo) held[it->second].newestUndo = position;
                else held[it->second].newestRedo = position;
            }
        };

        for (std::size_t i = 0; i < undo.size(); ++i) note(*undo[i], i, true);
        for (std::size_t i = 0; i < redo.size(); ++i) note(*redo[i], i, false);

        return held;
    }

    // What the history would still retain with the first `undoDrop` undo entries
    // and `redoDrop` redo entries gone.
    std::size_t bytesAfter(const std::vector<Held>& held, std::size_t undoDrop, std::size_t redoDrop) {

        std::size_t total = 0;
        for (const auto& subtree : held) {

            const bool heldByUndo = subtree.newestUndo != kNoIndex && subtree.newestUndo >= undoDrop;
            const bool heldByRedo = subtree.newestRedo != kNoIndex && subtree.newestRedo >= redoDrop;
            if (heldByUndo || heldByRedo) total += subtree.bytes;
        }
        return total;
    }

    // How many entries to drop off the front of a stack, at most `maxDrop`, given
    // `after(k)` = what the whole history would still retain with the first k
    // gone. The smallest k that fits the budget; if no k fits, the smallest one
    // that frees the most, because dropping history that frees nothing is pure
    // loss. Zero when nothing can help.
    template<class After>
    std::size_t chooseDrop(std::size_t maxDrop, std::size_t limit, const After& after) {

        if (maxDrop == 0) return 0;

        std::size_t best = 0;
        std::size_t bestBytes = after(0);

        for (std::size_t k = 1; k <= maxDrop; ++k) {

            const std::size_t bytes = after(k);
            if (bytes <= limit) return k;
            if (bytes < bestBytes) {
                bestBytes = bytes;
                best = k;
            }
        }
        return best;
    }

}// namespace


std::size_t threepp::editor::retainedSubtreeBytes(const Object3D& root) {

    std::size_t bytes = 0;
    // traverse() is non-const, and weighing a subtree does not modify it.
    const_cast<Object3D&>(root).traverse([&bytes](Object3D& node) {
        if (const auto* cloud = dynamic_cast<const SplatCloud*>(&node)) bytes += cloud->cpuBytes();
    });
    return bytes;
}


CommandStack::CommandStack(std::size_t limit, std::size_t byteLimit)
    : limit_(limit == 0 ? 1 : limit), byteLimit_(byteLimit) {}

void CommandStack::execute(std::unique_ptr<Command> command) {

    if (!command) return;

    command->redo();
    push(std::move(command));
}

void CommandStack::push(std::unique_ptr<Command> command) {

    if (!command) return;

    // Any new edit invalidates the redo branch.
    redo_.clear();

    // Coalesce into the previous command when a transaction is open and both
    // describe the same field of the same object.
    if (transactionDepth_ > 0 && undo_.size() > transactionStart_) {
        auto& top = undo_.back();
        const auto key = command->mergeKey();
        if (!key.empty() && key == top->mergeKey() && top->mergeWith(*command)) {
            notify();
            return;
        }
    }

    undo_.push_back(std::move(command));
    trim();
    pruneToByteLimit();
    notify();
}

bool CommandStack::undo() {

    if (undo_.empty()) return false;

    auto command = std::move(undo_.back());
    undo_.pop_back();
    command->undo();
    redo_.push_back(std::move(command));

    // The transaction watermark refers to positions in undo_; keep it valid.
    if (transactionStart_ > undo_.size()) transactionStart_ = undo_.size();

    // Undoing MOVES retention: an undone "Add" detaches the object it created,
    // so the command that held it harmlessly a moment ago now holds it alive.
    pruneToByteLimit();

    notify();
    return true;
}

bool CommandStack::redo() {

    if (redo_.empty()) return false;

    auto command = std::move(redo_.back());
    redo_.pop_back();
    command->redo();
    undo_.push_back(std::move(command));

    // The mirror of undo(): a redone "Delete" takes its subtree back out of the
    // scene, and the history is what holds it now.
    pruneToByteLimit();

    notify();
    return true;
}

bool CommandStack::canUndo() const {

    return !undo_.empty();
}

bool CommandStack::canRedo() const {

    return !redo_.empty();
}

std::string CommandStack::undoName() const {

    return undo_.empty() ? std::string{} : undo_.back()->name();
}

std::string CommandStack::redoName() const {

    return redo_.empty() ? std::string{} : redo_.back()->name();
}

void CommandStack::rebind(Object3D& root) {

    const auto prune = [&root](std::vector<std::unique_ptr<Command>>& stack) {
        std::erase_if(stack, [&root](const std::unique_ptr<Command>& command) {
            return !command->rebind(root);
        });
    };
    prune(undo_);
    prune(redo_);

    // The watermark indexes into undo_, which may just have shrunk.
    if (transactionStart_ > undo_.size()) transactionStart_ = undo_.size();

    // A swap moves retention in both directions: a command whose target came
    // back in the new graph re-resolves onto the live node and holds nothing,
    // while one whose target did not is left holding the dead graph's instance,
    // which ~Object3D detached on the way out. Re-weigh either way.
    pruneToByteLimit();

    notify();
}

void CommandStack::clear() {

    undo_.clear();
    redo_.clear();
    transactionDepth_ = 0;
    transactionStart_ = 0;
    notify();
}

void CommandStack::beginTransaction() {

    if (transactionDepth_++ == 0) {
        // Only commands pushed from here on may merge with each other; the
        // command already on top belongs to a finished edit.
        transactionStart_ = undo_.size();
    }
}

void CommandStack::endTransaction() {

    if (transactionDepth_ > 0) --transactionDepth_;
    if (transactionDepth_ == 0) transactionStart_ = undo_.size();
}

void CommandStack::onChange(std::function<void()> listener) {

    if (listener) listeners_.push_back(std::move(listener));
}

void CommandStack::onPrune(PruneListener listener) {

    if (listener) pruneListeners_.push_back(std::move(listener));
}

void CommandStack::setByteLimit(std::size_t bytes) {

    byteLimit_ = bytes;
    pruneToByteLimit();
    notify();
}

std::size_t CommandStack::retainedBytes() const {

    const auto held = survey(undo_, redo_);
    return bytesAfter(held, 0, 0);
}

void CommandStack::pruneToByteLimit() {

    const auto held = survey(undo_, redo_);
    const std::size_t before = bytesAfter(held, 0, 0);
    if (before <= byteLimit_) return;

    // The redo branch goes first, and all of it may go. It is speculative future
    // that the next edit would discard wholesale anyway, which makes it the
    // cheapest history in the stack to lose — and dropping from its FRONT costs
    // the most distant redo step while leaving every nearer one reachable in
    // order.
    const std::size_t redoDrop = chooseDrop(redo_.size(), byteLimit_, [&held](std::size_t k) {
        return bytesAfter(held, 0, k);
    });

    // On the undo side the newest entry that is HOLDING something survives, along
    // with everything newer than it. That is the guarantee worth breaking the
    // bound for: whatever you just deleted can be taken back, even if it was a
    // scan bigger than the whole budget.
    //
    // Protecting only undo_.back() would not deliver it — one unrelated edit
    // pushed after the deletion would leave the deletion droppable and the undo
    // would vanish while the user was looking elsewhere. Protecting the newest
    // RETAINER bounds the overshoot at one held subtree, which is the same trade,
    // honestly kept.
    //
    // The asymmetry with redo is deliberate: "what I just did can be taken back"
    // is a promise, "what I just took back can be reinstated" is a convenience,
    // and where the two collide — undoing a huge import — the memory is worth
    // more than the convenience.
    std::size_t newestRetainer = 0;
    for (const auto& subtree : held) {
        if (subtree.newestUndo != kNoIndex) newestRetainer = std::max(newestRetainer, subtree.newestUndo);
    }

    std::size_t undoDrop = 0;
    if (bytesAfter(held, 0, redoDrop) > byteLimit_) {
        undoDrop = chooseDrop(newestRetainer, byteLimit_, [&held, redoDrop](std::size_t k) {
            return bytesAfter(held, k, redoDrop);
        });
    }

    if (redoDrop == 0 && undoDrop == 0) return;

    std::vector<std::string> dropped;
    dropped.reserve(redoDrop + undoDrop);
    for (std::size_t i = 0; i < redoDrop; ++i) dropped.push_back(redo_[i]->name());
    for (std::size_t i = 0; i < undoDrop; ++i) dropped.push_back(undo_[i]->name());

    redo_.erase(redo_.begin(), redo_.begin() + static_cast<std::ptrdiff_t>(redoDrop));
    undo_.erase(undo_.begin(), undo_.begin() + static_cast<std::ptrdiff_t>(undoDrop));

    // Same bookkeeping trim() does: the watermark indexes into undo_.
    transactionStart_ = transactionStart_ > undoDrop ? transactionStart_ - undoDrop : 0;

    reportPrune(dropped, before - bytesAfter(held, undoDrop, redoDrop));
}

void CommandStack::reportPrune(const std::vector<std::string>& dropped, std::size_t bytesFreed) const {

    for (const auto& listener : pruneListeners_) listener(dropped, bytesFreed);
}

void CommandStack::notify() const {

    for (const auto& listener : listeners_) listener();
}

void CommandStack::trim() {

    if (undo_.size() <= limit_) return;

    const std::size_t excess = undo_.size() - limit_;

    // The count cap can drop a held subtree exactly as the byte budget can, and
    // then it owes the same notice — a 200-entry session whose oldest entry is a
    // deleted scan should not lose a multi-gigabyte undo in silence. Measured
    // only when something was actually holding memory, so an ordinary session
    // pays one virtual call per erased entry and nothing else.
    std::vector<std::string> dropped;
    std::vector<Object3D*> roots;
    for (std::size_t i = 0; i < excess; ++i) {

        dropped.push_back(undo_[i]->name());
        undo_[i]->retainedRoots(roots);
    }
    const bool holdsAnything = std::any_of(roots.begin(), roots.end(),
                                           [](const Object3D* root) { return root && !root->parent; });
    const std::size_t before = holdsAnything ? retainedBytes() : 0;

    undo_.erase(undo_.begin(), undo_.begin() + static_cast<std::ptrdiff_t>(excess));
    transactionStart_ = transactionStart_ > excess ? transactionStart_ - excess : 0;

    if (!holdsAnything) return;

    const std::size_t after = retainedBytes();
    if (after < before) reportPrune(dropped, before - after);
}
