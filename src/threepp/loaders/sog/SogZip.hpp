// A read-only reader for the ZIP containers PlayCanvas SOG / SuperSplat "SSOG"
// Gaussian-splat scans are distributed in.
//
// Provenance: clean-room, written from the PKWARE .ZIP File Format Specification
// (APPNOTE, public since 1989). No third-party zip code was consulted and none
// is linked. That is affordable only because SOG archives use a tiny subset of
// the format: every entry is STORED (compression method 0), because the payload
// is already-compressed WebP and JSON, so "reading an entry" is nothing but
// finding the offset its bytes start at. Nothing here inflates anything; a
// method other than stored is refused by name rather than guessed at.
//
// THE TRAP, and the reason this file exists rather than a ten-line offset walk:
// THE LOCAL FILE HEADER LIES. 153 of the 154 entries in the reference archive
// set general-purpose bit 3 (0x08, "sizes follow the data in a descriptor"),
// and on those the local header's crc32, compressed size and uncompressed size
// are all written as ZERO — the real values trail the data instead. A reader
// that takes sizes from the local file header therefore gets 153 zero-length
// entries and no error whatsoever: every image decodes to nothing, and the
// failure surfaces far away as a blank splat cloud.
//
// THE CENTRAL DIRECTORY IS THE ONLY TRUTH. Compression method, uncompressed
// size and the local header's offset are all taken from there. The local header
// is consulted for exactly two fields — its own name length and extra length —
// because those may legitimately differ from the central directory's copies
// (writers routinely put different extra fields in the two places), and they
// are what fixes where the data actually begins:
//
//     dataOffset = localHeaderOffset + 30 + localNameLen + localExtraLen
//
// Everything this cannot honestly read is refused with std::runtime_error
// naming the entry and the offending value — a non-stored method, an encrypted
// entry, the ZIP64 sentinels (0xFFFFFFFF in a size or offset, 0xFFFF in the
// entry count), a spanned archive, and any record or payload that runs past the
// end of the file. It is pointed at arbitrary user files, so every offset is
// bounds-checked in 64-bit arithmetic against the real file size before it is
// used to index; a truncated or malformed archive throws, it never returns the
// wrong bytes.

#ifndef THREEPP_SOG_SOGZIP_HPP
#define THREEPP_SOG_SOGZIP_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::sog {

    class SogZip {

    public:
        // Is this file plausibly a ZIP archive?
        //
        // Sniffs the first local file header's magic ("PK\x03\x04"), which is
        // what a SOG bundle starts with. Reads four bytes and never throws: a
        // missing, unreadable or truncated file is simply "not an archive",
        // which leaves the caller's plain-file path to report the failure in its
        // own words. Note the two deliberate false negatives — an empty archive
        // (which begins with its end-of-central-directory record) and one with a
        // prepended self-extracting stub — neither of which a SOG scan is.
        [[nodiscard]] static bool looksLikeZip(const std::filesystem::path& path);

        // Reads the whole file into memory and parses the central directory.
        //
        // Throws std::runtime_error, with the offending entry and value in the
        // message, on anything it cannot represent. Validation is done here for
        // every entry rather than lazily at read() time, so has() never promises
        // bytes that a later read() would refuse to deliver.
        explicit SogZip(const std::filesystem::path& path);

        // Lookup names are normalised: backslashes become forward slashes and a
        // leading "./" is stripped, so "0_0/meta.json" finds the entry however
        // the writer happened to spell it.
        [[nodiscard]] bool has(const std::string& name) const;

        // The entry's uncompressed bytes, sized from the central directory.
        // Throws if there is no such entry — a caller that is unsure should ask
        // has() first.
        [[nodiscard]] std::vector<unsigned char> read(const std::string& name) const;

        // Every entry, normalised, in central directory order. Directory
        // entries (names ending in '/') are not listed; they carry no data.
        [[nodiscard]] std::vector<std::string> names() const;

    private:
        struct Entry {
            std::uint64_t offset{};
            std::uint64_t size{};
        };

        // Walks the central directory of the already-loaded bytes_. Separate
        // from the constructor only so the constructor can name the file in
        // whatever it throws.
        void parse();

        // The whole file. These archives are ~180 MB and a caller loads most of
        // every one of them anyway, so a memory-mapped or streaming design would
        // buy little and cost the portability of a per-platform mapping layer.
        std::vector<unsigned char> bytes_;

        std::unordered_map<std::string, Entry> entries_;
        std::vector<std::string> names_;
    };

}// namespace threepp::sog

#endif//THREEPP_SOG_SOGZIP_HPP
