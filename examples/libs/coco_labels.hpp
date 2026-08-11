// coco_labels.hpp — the COCO class-name tables the detector demos print.
//
// There are genuinely TWO tables, and conflating them mislabels every box:
//
//   • kCoco80 — the 80 contiguous "thing" classes, in the darknet/YOLO order.
//     Model outputs index it directly (0 = person). Used by YOLOv8 and RT-DETR.
//     Note the darknet spellings ("motorbike", "aeroplane", "sofa", "tvmonitor")
//     — these are what the weights were trained/published against, so they are
//     preserved verbatim rather than normalised to the COCO wording.
//
//   • kCoco91 — the ORIGINAL COCO category ids, 1..90, which have gaps where
//     categories were dropped from the released dataset. Index 0 and every gap
//     read "N/A". RF-DETR predicts these raw ids, so its logits are 91 wide and
//     ~11 of those columns can never fire.
//
// Both were copy-pasted into four demos (two copies each). A mislabelled table
// is a silent wrong answer, not a crash, so the size contract is pinned with a
// static_assert here and again at each use site against that model's class count.
#pragma once

#include <array>
#include <string_view>

namespace coco {

    // ── COCO-80: contiguous, darknet order (YOLOv8, RT-DETR) ─────────────────
    inline constexpr std::array<std::string_view, 80> kCoco80{
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    // ── COCO-91: original category ids 1..90, gaps as "N/A" (RF-DETR) ─────────
    inline constexpr std::array<std::string_view, 91> kCoco91{
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
            "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"};

    // The initialiser lists above are long enough that a dropped or duplicated
    // entry is easy to miss and impossible to see in the output — it just shifts
    // every later label by one. std::array does not pad, so these sizes are the
    // guard: they fail the build instead of mislabelling a detection.
    static_assert(kCoco80.size() == 80, "COCO-80 table is not 80 entries");
    static_assert(kCoco91.size() == 91, "COCO-91 table is not 91 entries (ids 0..90)");

    // Bounds-checked lookup. A detector fed mismatched weights can emit an
    // out-of-range class index; the demos indexed the raw array with it, which
    // reads out of bounds. "?" makes that visible in the label instead.
    [[nodiscard]] inline constexpr std::string_view name80(int cls) {
        return cls >= 0 && cls < int(kCoco80.size()) ? kCoco80[std::size_t(cls)] : "?";
    }

    [[nodiscard]] inline constexpr std::string_view name91(int id) {
        return id >= 0 && id < int(kCoco91.size()) ? kCoco91[std::size_t(id)] : "?";
    }

}// namespace coco
