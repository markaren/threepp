
#include "threepp/utils/ZipWriter.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

using namespace threepp;

namespace {

    constexpr std::uint32_t LOCAL_SIG = 0x04034b50;  // "PK\x03\x04"
    constexpr std::uint32_t CENTRAL_SIG = 0x02014b50;// "PK\x01\x02"
    constexpr std::uint32_t EOCD_SIG = 0x06054b50;   // "PK\x05\x06"

    constexpr std::uint64_t LOCAL_FIXED = 30;
    constexpr std::uint64_t CENTRAL_FIXED = 46;
    constexpr std::uint64_t EOCD_FIXED = 22;

    // 2.0 in both places: the floor for a reader that understands directories
    // and stored entries, which is all this writer produces. The high byte of
    // version-made-by is the host system, and 0 (MS-DOS/FAT) is the one value
    // that carries no attributes worth disagreeing about across platforms —
    // holding it fixed is what keeps a Windows export and a Linux export of the
    // same scene byte-identical.
    constexpr std::uint16_t VERSION_MADE_BY = 20;
    constexpr std::uint16_t VERSION_NEEDED = 20;

    // 1980-01-01 00:00, the epoch of the DOS timestamp: year 0 of the 7-bit
    // year field, month 1, day 1. Anything later would be a clock reading, and
    // a clock reading in the archive is a diff in every autosave.
    constexpr std::uint16_t DOS_TIME = 0;
    constexpr std::uint16_t DOS_DATE = (0 << 9) | (1 << 5) | 1;

    // Names are UTF-8. The bit is set only when it can matter, so a plain ASCII
    // archive is byte-identical to what a writer without the notion would emit.
    constexpr std::uint16_t FLAG_UTF8 = 0x0800;

    constexpr std::uint32_t MAX32 = 0xFFFFFFFFu;
    constexpr std::uint16_t MAX16 = 0xFFFFu;

    [[noreturn]] void fail(const std::string& msg) {

        throw std::runtime_error("ZipWriter: " + msg);
    }

    std::string named(const std::string& s) {

        return "\"" + s + "\"";
    }

    std::string normalizeName(std::string name) {

        std::replace(name.begin(), name.end(), '\\', '/');
        while (name.rfind("./", 0) == 0) name.erase(0, 2);
        return name;
    }

    // CRC-32/ISO-HDLC, the one ZIP uses: reflected input and output, so the
    // polynomial appears here bit-reversed (0x04C11DB7 -> 0xEDB88320) and the
    // register shifts right rather than left. The table is built once, on the
    // first archive written, rather than spelled out as 256 magic constants.
    const std::array<std::uint32_t, 256>& crcTable() {

        static const std::array<std::uint32_t, 256> table = [] {
            std::array<std::uint32_t, 256> t{};
            for (std::uint32_t i = 0; i < 256; ++i) {

                std::uint32_t c = i;
                for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                t[i] = c;
            }
            return t;
        }();

        return table;
    }

    std::uint32_t crc32(const std::vector<unsigned char>& data) {

        const auto& table = crcTable();
        std::uint32_t c = 0xFFFFFFFFu;
        for (const unsigned char byte : data) {

            c = table[(c ^ byte) & 0xFFu] ^ (c >> 8);
        }
        return c ^ 0xFFFFFFFFu;
    }

    void put16(std::vector<unsigned char>& out, std::uint16_t v) {

        out.push_back(static_cast<unsigned char>(v & 0xFFu));
        out.push_back(static_cast<unsigned char>((v >> 8) & 0xFFu));
    }

    void put32(std::vector<unsigned char>& out, std::uint32_t v) {

        out.push_back(static_cast<unsigned char>(v & 0xFFu));
        out.push_back(static_cast<unsigned char>((v >> 8) & 0xFFu));
        out.push_back(static_cast<unsigned char>((v >> 16) & 0xFFu));
        out.push_back(static_cast<unsigned char>((v >> 24) & 0xFFu));
    }

    void putName(std::vector<unsigned char>& out, const std::string& name) {

        out.insert(out.end(), name.begin(), name.end());
    }

    std::uint16_t flagsFor(const std::string& name) {

        const bool ascii = std::none_of(name.begin(), name.end(),
                                        [](char c) { return static_cast<unsigned char>(c) >= 0x80; });
        return ascii ? 0 : FLAG_UTF8;
    }

    // Fewer path components first, then lexicographic. A general rule with a
    // specific consequence the scene format leans on: scene.json lands ahead of
    // buffers/* and images/*, so the document a reader wants first is also the
    // first thing in the file.
    bool before(const std::string& a, const std::string& b) {

        const auto depth = [](const std::string& s) {
            return std::count(s.begin(), s.end(), '/');
        };
        const auto da = depth(a);
        const auto db = depth(b);
        if (da != db) return da < db;
        return a < b;
    }

}// namespace


void ZipWriter::add(std::string name, std::vector<unsigned char> data) {

    name = normalizeName(std::move(name));

    if (name.empty()) fail("an entry cannot have an empty name");
    if (name.back() == '/') {

        fail(named(name) + " names a directory; this writer stores files only");
    }
    if (name.size() > MAX16) {

        fail(named(name.substr(0, 64) + "...") + " is " + std::to_string(name.size()) +
             " bytes long; a ZIP entry name cannot exceed " + std::to_string(MAX16));
    }
    if (data.size() > MAX32) {

        fail(named(name) + " is " + std::to_string(data.size()) +
             " bytes; entries of 4 GB or more need ZIP64, which is not supported");
    }

    const auto duplicate = std::find_if(entries_.begin(), entries_.end(),
                                        [&](const Entry& e) { return e.name == name; });
    if (duplicate != entries_.end()) {

        fail(named(name) + " was added twice; entry names must be unique");
    }

    const auto crc = crc32(data);
    entries_.push_back(Entry{std::move(name), std::move(data), crc});
}

void ZipWriter::add(std::string name, const std::string& text) {

    add(std::move(name), std::vector<unsigned char>(text.begin(), text.end()));
}

std::vector<unsigned char> ZipWriter::build() const {

    if (entries_.size() >= MAX16) {

        fail("archive has " + std::to_string(entries_.size()) +
             " entries; 65535 or more need ZIP64, which is not supported");
    }

    // Sorted by name rather than in add() order, so the caller's traversal
    // order cannot leak into the bytes.
    std::vector<const Entry*> ordered;
    ordered.reserve(entries_.size());
    for (const auto& entry : entries_) ordered.push_back(&entry);
    std::sort(ordered.begin(), ordered.end(),
              [](const Entry* a, const Entry* b) { return before(a->name, b->name); });

    std::vector<unsigned char> out;

    std::uint64_t total = 0;
    for (const auto* entry : ordered) {

        total += LOCAL_FIXED + entry->name.size() + entry->data.size() +
                 CENTRAL_FIXED + entry->name.size();
    }
    out.reserve(static_cast<std::size_t>(total + EOCD_FIXED));

    std::vector<std::uint32_t> localOffsets;
    localOffsets.reserve(ordered.size());

    for (const auto* entry : ordered) {

        // Checked before the header is written rather than after: the offset
        // that overflows is the one that would silently alias entry 0's.
        if (out.size() > MAX32) {

            fail(named(entry->name) + " starts at offset " + std::to_string(out.size()) +
                 "; archives of 4 GB or more need ZIP64, which is not supported");
        }
        localOffsets.push_back(static_cast<std::uint32_t>(out.size()));

        const auto size = static_cast<std::uint32_t>(entry->data.size());

        put32(out, LOCAL_SIG);
        put16(out, VERSION_NEEDED);
        put16(out, flagsFor(entry->name));
        put16(out, 0);// stored
        put16(out, DOS_TIME);
        put16(out, DOS_DATE);
        put32(out, entry->crc);
        put32(out, size);// compressed == uncompressed, by definition of stored
        put32(out, size);
        put16(out, static_cast<std::uint16_t>(entry->name.size()));
        put16(out, 0);// no extra field
        putName(out, entry->name);

        out.insert(out.end(), entry->data.begin(), entry->data.end());
    }

    const auto cdStart = out.size();

    for (std::size_t i = 0; i < ordered.size(); ++i) {

        const auto* entry = ordered[i];
        const auto size = static_cast<std::uint32_t>(entry->data.size());

        put32(out, CENTRAL_SIG);
        put16(out, VERSION_MADE_BY);
        put16(out, VERSION_NEEDED);
        put16(out, flagsFor(entry->name));
        put16(out, 0);// stored
        put16(out, DOS_TIME);
        put16(out, DOS_DATE);
        put32(out, entry->crc);
        put32(out, size);
        put32(out, size);
        put16(out, static_cast<std::uint16_t>(entry->name.size()));
        put16(out, 0);// no extra field
        put16(out, 0);// no comment
        put16(out, 0);// disk number
        put16(out, 0);// internal attributes
        put32(out, 0);// external attributes: none, so the archive says nothing
                      // about a mode or a hidden bit it did not observe
        put32(out, localOffsets[i]);
        putName(out, entry->name);
    }

    const auto cdSize = out.size() - cdStart;

    if (cdStart > MAX32 || cdSize > MAX32) {

        fail("central directory is " + std::to_string(cdSize) + " bytes at offset " +
             std::to_string(cdStart) + "; archives of 4 GB or more need ZIP64, which is not supported");
    }

    put32(out, EOCD_SIG);
    put16(out, 0);// this disk
    put16(out, 0);// disk the central directory starts on
    put16(out, static_cast<std::uint16_t>(ordered.size()));
    put16(out, static_cast<std::uint16_t>(ordered.size()));
    put32(out, static_cast<std::uint32_t>(cdSize));
    put32(out, static_cast<std::uint32_t>(cdStart));
    put16(out, 0);// no archive comment

    return out;
}

void ZipWriter::writeTo(const std::filesystem::path& path) const {

    // Built first: a refusal (ZIP64, a duplicate name) must not have already
    // deleted the archive it was asked to replace.
    const auto bytes = build();

    auto temp = path;
    temp += ".tmp";

    {
        // From the path OBJECT, never path.string(): on Windows the narrow
        // overload goes through the ANSI code page and a directory with an
        // accented name silently fails to open.
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) fail("cannot open '" + temp.string() + "' for writing");

        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        out.flush();

        if (!out) {

            out.close();
            std::error_code ignored;
            std::filesystem::remove(temp, ignored);
            fail("short write to '" + temp.string() + "'");
        }
    }

    // Replaces the target if it exists on both platforms (MoveFileExW with
    // MOVEFILE_REPLACE_EXISTING on Windows, ::rename on POSIX), which is the
    // whole point: the swap is what the reader either sees or does not.
    std::error_code ec;
    std::filesystem::rename(temp, path, ec);
    if (ec) {

        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        fail("cannot rename '" + temp.string() + "' onto '" + path.string() + "': " + ec.message());
    }
}
