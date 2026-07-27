// Undo/redo primitives for the scene editor.
//
// A Command owns everything needed to apply a change and to put the world back
// the way it was. The stack never re-derives state from the scene, so a command
// stays correct even after unrelated edits.
//
// UI-agnostic on purpose: nothing here knows about ImGui, windows or input.
// The editor application builds commands and hands them to the stack; the
// library part is equally usable from a script, a test, or another front end.

#ifndef THREEPP_EDITOR_COMMAND_HPP
#define THREEPP_EDITOR_COMMAND_HPP

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

}// namespace threepp

namespace threepp::editor {

    class Command {

    public:
        virtual ~Command() = default;

        // Apply the change. Called once by CommandStack::execute() and again on
        // every redo(), so it must be idempotent with respect to its own stored
        // "after" state (never read the current scene value here).
        virtual void redo() = 0;

        // Put back the "before" state.
        virtual void undo() = 0;

        // Shown in the UI ("Undo Move Box"). Present tense, no "Undo" prefix.
        [[nodiscard]] virtual std::string name() const = 0;

        // Non-empty means "this command can absorb a newer one carrying the same
        // key while a transaction is open". A widget drag emits one command per
        // frame; without this the undo stack fills with hundreds of one-pixel
        // steps. Key convention: "<what>:<object-uuid>:<field>".
        [[nodiscard]] virtual std::string mergeKey() const { return {}; }

        // Absorb `newer` (guaranteed to have the same non-empty mergeKey).
        // Keep this command's "before" state and adopt `newer`'s "after" state.
        // Returning false makes the stack push `newer` separately.
        virtual bool mergeWith(const Command& newer) = 0;

        // The scene graph this command was recorded against has been swapped
        // for an equivalent one (play-stop snapshot restore) and every pointer
        // into the old graph is dangling. Re-resolve the targets by uuid
        // against `root` and return true; return false if that is impossible
        // and the stack must drop this command instead. Dropping is the
        // default: a command type that cannot name its targets by uuid must
        // not survive the swap.
        [[nodiscard]] virtual bool rebind(Object3D& root) { return false; }
    };


    // Bounded undo/redo stack with drag coalescing.
    //
    // Transactions exist for one reason: an ImGui drag reports a new value every
    // frame. Open a transaction on ImGui::IsItemActivated(), push freely, close
    // it on ImGui::IsItemDeactivatedAfterEdit(), and the whole drag lands as a
    // single undo step. Outside a transaction every push is its own step.
    class CommandStack {

    public:
        // Default cap. Commands hold shared_ptrs to removed subtrees, so an
        // unbounded stack is also an unbounded memory leak.
        static constexpr std::size_t defaultLimit = 200;

        explicit CommandStack(std::size_t limit = defaultLimit);

        // Apply and record. The usual entry point.
        void execute(std::unique_ptr<Command> command);

        // Record a change that the caller already applied (the gizmo moved the
        // object before we could see it). Same coalescing rules as execute().
        void push(std::unique_ptr<Command> command);

        bool undo();
        bool redo();

        [[nodiscard]] bool canUndo() const;
        [[nodiscard]] bool canRedo() const;

        // "" when there is nothing to undo/redo.
        [[nodiscard]] std::string undoName() const;
        [[nodiscard]] std::string redoName() const;

        void clear();

        // The document's scene was replaced by an equivalent graph (play-stop
        // restore). Ask every recorded command to re-resolve itself against
        // the new root and drop the ones that cannot — the alternative is an
        // undo that writes through dangling pointers.
        void rebind(Object3D& root);

        [[nodiscard]] std::size_t undoCount() const { return undo_.size(); }
        [[nodiscard]] std::size_t redoCount() const { return redo_.size(); }

        // Nesting is counted, so a panel may open a transaction around a group
        // of widgets that each open their own.
        void beginTransaction();
        void endTransaction();
        [[nodiscard]] bool inTransaction() const { return transactionDepth_ > 0; }

        // Fired after every change to the stack (execute/undo/redo/clear).
        void onChange(std::function<void()> listener);

    private:
        void notify() const;
        void trim();

        std::vector<std::unique_ptr<Command>> undo_;
        std::vector<std::unique_ptr<Command>> redo_;
        std::vector<std::function<void()>> listeners_;
        std::size_t limit_;
        int transactionDepth_ = 0;
        // Index into undo_ of the first command pushed inside the current
        // transaction — merging never reaches back past it.
        std::size_t transactionStart_ = 0;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_COMMAND_HPP
