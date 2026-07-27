
#include "threepp/extras/editor/Command.hpp"

using namespace threepp;
using namespace threepp::editor;


CommandStack::CommandStack(std::size_t limit)
    : limit_(limit == 0 ? 1 : limit) {}

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

    notify();
    return true;
}

bool CommandStack::redo() {

    if (redo_.empty()) return false;

    auto command = std::move(redo_.back());
    redo_.pop_back();
    command->redo();
    undo_.push_back(std::move(command));

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

void CommandStack::notify() const {

    for (const auto& listener : listeners_) listener();
}

void CommandStack::trim() {

    if (undo_.size() <= limit_) return;

    const std::size_t excess = undo_.size() - limit_;
    undo_.erase(undo_.begin(), undo_.begin() + static_cast<std::ptrdiff_t>(excess));
    transactionStart_ = transactionStart_ > excess ? transactionStart_ - excess : 0;
}
