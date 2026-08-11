// A STORED-only ZIP writer: the counterpart of ZipReader, and the container
// behind threepp's single-file scene archive (.tpz, see ObjectExporter).
//
// Clean-room from the PKWARE .ZIP File Format Specification, same as the reader
// and for the same reason: nothing here compresses anything, so the whole
// format reduces to a local header, the payload verbatim, a central directory
// and an end-of-central-directory record. A scene archive holds already-
// compressed PNG/JPEG bytes and raw vertex data — deflating the first buys
// nothing and deflating the second would cost more time than the whole export.
//
// BYTE-IDENTICAL OUTPUT IS A GUARANTEE, not an accident: ObjectExporter
// advertises deterministic documents so autosaves and version control diff on
// what actually changed, and an archive that re-shuffled or re-timestamped
// itself would defeat that. Four rules buy it, and all four are load-bearing:
//
//   - General-purpose bit 3 is NEVER set. Real CRC-32 and real sizes go in the
//     local header, because we hold the whole entry in memory and know them.
//     That flag is exactly the trap ZipReader's header documents — a reader
//     that trusts such a local header sees an archive of empty files and no
//     error — and this writer must not be the thing that lays it.
//   - Every entry is stamped 1980-01-01 00:00, the earliest a DOS timestamp can
//     express. The mtime of a scene is the mtime of its file; putting the clock
//     inside the archive would make two identical exports differ.
//   - Entries are emitted in a fixed order regardless of add() order: fewer
//     path components first, then lexicographically. For the scene layout that
//     is scene.json, then buffers/*, then images/*.
//   - No extra fields, no comments, fixed version-made-by. The only variable
//     flag bit is 0x800 (names are UTF-8), set exactly when a name carries a
//     byte outside ASCII — a property of the name, so it is stable too.
//
// ZIP64: NO. Refused with std::runtime_error naming the entry whenever a size,
// an offset or the entry count would need a sentinel value — 4 GB of entry,
// 4 GB of archive, or 65535 entries. The reader refuses to read those, so
// writing them would only produce a file threepp itself could not open. A
// contained ZIP64 addition can happen the day a real scene reaches that size.

#ifndef THREEPP_ZIPWRITER_HPP
#define THREEPP_ZIPWRITER_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    class ZipWriter {

    public:
        // Adds one stored entry. The name is normalised the way ZipReader
        // normalises lookups (backslashes to forward slashes, leading "./"
        // stripped), so the name that goes in is the name that comes back out.
        //
        // Throws std::runtime_error naming the entry on a name that is empty,
        // ends in '/' (a directory entry carries no data and this writer emits
        // none), is longer than 65535 bytes, or repeats a name already added —
        // a duplicate is legal ZIP but the reader would only ever hand back the
        // last one, so silently keeping both would be a lie about the archive.
        void add(std::string name, std::vector<unsigned char> data);

        void add(std::string name, const std::string& text);

        // The whole archive, in memory. Deterministic: same entries in, same
        // bytes out, whatever order they were added in.
        [[nodiscard]] std::vector<unsigned char> build() const;

        // build() plus a crash-safe write: the bytes go to a sibling temp file
        // which is then renamed over the target, so an export that dies halfway
        // leaves the previous archive intact rather than a truncated one. The
        // temp file is <target>.tmp — two processes saving the same path at the
        // same time will fight over it, which is the same race they already
        // have over the target itself.
        void writeTo(const std::filesystem::path& path) const;

        [[nodiscard]] std::size_t size() const { return entries_.size(); }

    private:
        struct Entry {
            std::string name;
            std::vector<unsigned char> data;
            std::uint32_t crc{};
        };

        std::vector<Entry> entries_;
    };

}// namespace threepp

#endif//THREEPP_ZIPWRITER_HPP
