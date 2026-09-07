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

        // Every scene subtree this command holds a strong reference to, appended
        // to `out`. Purely a statement of ownership: report what you hold and
        // let CommandStack decide what it costs.
        //
        // The stack ignores a reported root that is still parented — the scene
        // co-owns that one, and dropping the command would free nothing. What is
        // left, the DETACHED roots, are alive only because the history holds
        // them. Two useful consequences: a node inside another reported subtree
        // always has a parent, so the detached roots are disjoint and their
        // bytes cannot double count; and two commands that hold the SAME
        // detached subtree (add X, then delete X) report the same pointer, which
        // the stack counts once.
        virtual void retainedRoots(std::vector<Object3D*>& out) const {}

        // The scene graph this command was recorded against has been swapped
        // for an equivalent one (play-stop snapshot restore) and every pointer
        // into the old graph is dangling. Re-resolve the targets by uuid
        // against `root` and return true; return false if that is impossible
        // and the stack must drop this command instead. Dropping is the
        // default: a command type that cannot name its targets by uuid must
        // not survive the swap.
        [[nodiscard]] virtual bool rebind(Object3D& root) { return false; }
    };


    // What keeping this subtree in memory costs, in bytes.
    //
    // Gaussian-splat clouds are the only heavyweight it weighs, because they are
    // the only kind of scene node whose payload is measured in gigabytes: a
    // 6M-splat scan measures 2.5 GB of host memory, 3.6 GB once a GL frame has
    // built its data textures — three orders of magnitude past every mesh in the
    // same document. Geometry and material payloads are deliberately NOT counted:
    // they sit behind shared_ptrs the live scene usually co-owns, so charging them
    // to the history would report memory that dropping a command cannot free.
    //
    // A floor on what is held, then, not a total, and it errs the safe way in the
    // other direction too — an app holding its own shared_ptr to a detached
    // subtree makes this over-report, so a prune frees less than it predicted
    // rather than more. Refining that with use_count() would be a mistake: an
    // under-report leaves the stack sole owner of memory it never budgeted.
    [[nodiscard]] std::size_t retainedSubtreeBytes(const Object3D& root);


    // Bounded undo/redo stack with drag coalescing.
    //
    // Transactions exist for one reason: an ImGui drag reports a new value every
    // frame. Open a transaction on ImGui::IsItemActivated(), push freely, close
    // it on ImGui::IsItemDeactivatedAfterEdit(), and the whole drag lands as a
    // single undo step. Outside a transaction every push is its own step.
    //
    // Bounded twice over, by COUNT and by BYTES, because the two limits guard
    // different things. 200 entries keeps an editing session's history from
    // growing without end; it says nothing about size, and 200 entries that hold
    // three deleted splat scans are 5 GB of retained memory that looks, from the
    // outside, like a stack of ordinary undo steps.
    class CommandStack {

    public:
        // Default cap. Commands hold shared_ptrs to removed subtrees, so an
        // unbounded stack is also an unbounded memory leak.
        static constexpr std::size_t defaultLimit = 200;

        // Default byte budget: room for one deleted splat scan and not much more.
        // Measured per splat at SH degree 3 — 348 B of splat data, 64 B of the
        // identity instanceMatrix, 12 B of sorted index, and 182 B more once a GL
        // frame has built the data textures — a 6M-splat scan is 2.4 GiB detached
        // on Vulkan and 3.4 GiB after GL has drawn it, so the largest scans exceed
        // this on their own. That is deliberate and bounded: a held subtree is
        // never evicted while it is the newest one held, so the most recent
        // deletion stays undoable at whatever it costs, and a second deletion
        // evicts the first. Below the largest single scan the budget means
        // "exceeded", never "truncated".
        static constexpr std::size_t defaultByteLimit = static_cast<std::size_t>(3) << 30;

        explicit CommandStack(std::size_t limit = defaultLimit,
                              std::size_t byteLimit = defaultByteLimit);

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

        // Bytes the history is keeping alive right now: the sum over the DETACHED
        // subtrees its commands hold, each counted once (see retainedRoots and
        // retainedSubtreeBytes). Zero for an ordinary editing session — a command
        // that changed a value holds nothing.
        [[nodiscard]] std::size_t retainedBytes() const;

        [[nodiscard]] std::size_t byteLimit() const { return byteLimit_; }

        // Applies immediately: lowering the budget prunes on the spot. A budget
        // below the largest single retained subtree is legal and means "the bound
        // is exceeded", not "the history is truncated" — the newest held entry
        // survives any budget.
        void setByteLimit(std::size_t bytes);

        // Fired when history is dropped for holding memory — by the byte budget or
        // by the count cap — with the NAMES of the dropped entries and the bytes
        // that freed. Names rather than a count, because what the user needs to
        // hear is which undo is no longer there ("dropped Delete Sanctuaire").
        // Worth surfacing rather than doing silently: it is a promise withdrawn.
        using PruneListener = std::function<void(const std::vector<std::string>& droppedNames,
                                                 std::size_t bytesFreed)>;
        void onPrune(PruneListener listener);

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

        // Drop history from the far ends until retainedBytes() fits the budget.
        // "Far ends" means the oldest undo entries and the most DISTANT redo steps,
        // which is what keeps both stacks contiguous around the present. Drops
        // exactly as much as the budget needs and no more.
        void pruneToByteLimit();

        void reportPrune(const std::vector<std::string>& dropped, std::size_t bytesFreed) const;

        std::vector<std::unique_ptr<Command>> undo_;
        std::vector<std::unique_ptr<Command>> redo_;
        std::vector<std::function<void()>> listeners_;
        std::vector<PruneListener> pruneListeners_;
        std::size_t limit_;
        std::size_t byteLimit_;
        int transactionDepth_ = 0;
        // Index into undo_ of the first command pushed inside the current
        // transaction — merging never reaches back past it.
        std::size_t transactionStart_ = 0;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_COMMAND_HPP
