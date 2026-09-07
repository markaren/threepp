
#include "threepp/utils/ZipReader.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

using namespace threepp;

namespace {

    // Signatures, as the little-endian words they are on disk.
    constexpr std::uint32_t LOCAL_SIG = 0x04034b50;  // "PK\x03\x04"
    constexpr std::uint32_t CENTRAL_SIG = 0x02014b50;// "PK\x01\x02"
    constexpr std::uint32_t EOCD_SIG = 0x06054b50;   // "PK\x05\x06"

    // Fixed parts of each record, before any variable-length name/extra/comment.
    constexpr std::uint64_t LOCAL_FIXED = 30;
    constexpr std::uint64_t CENTRAL_FIXED = 46;
    constexpr std::uint64_t EOCD_FIXED = 22;

    // The archive comment's length field is 16-bit, so the end-of-central-directory
    // record cannot start further back than this from the end of the file.
    constexpr std::uint64_t MAX_COMMENT = 0xFFFF;

    // The values a ZIP64 archive parks in the 32/16-bit fields to say "the real
    // number is in the ZIP64 extra field". Reading one as a literal size or
    // count is how a naive reader turns a 5 GB archive into nonsense.
    constexpr std::uint32_t ZIP64_32 = 0xFFFFFFFFu;
    constexpr std::uint16_t ZIP64_16 = 0xFFFFu;

    // General purpose bit flags that change how a record must be read.
    constexpr std::uint16_t FLAG_ENCRYPTED = 0x0001;

    [[noreturn]] void fail(const std::string& msg) {

        throw std::runtime_error("ZipReader: " + msg);
    }

    std::string quoted(const std::string& s) {

        return "\"" + s + "\"";
    }

    std::string hex32(std::uint32_t v) {

        static const char* DIGITS = "0123456789abcdef";
        std::string s(8, '0');
        for (int i = 7; i >= 0; --i) {

            s[static_cast<size_t>(i)] = DIGITS[v & 0xFu];
            v >>= 4;
        }
        return "0x" + s;
    }

    std::uint16_t rd16(const unsigned char* p) {

        return static_cast<std::uint16_t>(static_cast<unsigned>(p[0]) |
                                          (static_cast<unsigned>(p[1]) << 8));
    }

    std::uint32_t rd32(const unsigned char* p) {

        return static_cast<std::uint32_t>(p[0]) |
               (static_cast<std::uint32_t>(p[1]) << 8) |
               (static_cast<std::uint32_t>(p[2]) << 16) |
               (static_cast<std::uint32_t>(p[3]) << 24);
    }

    // Named where the name is more use than the number; the number is always
    // printed, so an unknown method is still reported honestly.
    std::string methodName(std::uint16_t method) {

        switch (method) {
            case 1: return "shrunk";
            case 6: return "imploded";
            case 8: return "deflate";
            case 9: return "deflate64";
            case 12: return "bzip2";
            case 14: return "lzma";
            case 93: return "zstd";
            case 95: return "xz";
            case 98: return "ppmd";
            default: return "";
        }
    }

    // Entry names are stored with forward slashes by every writer that follows
    // the spec, but not by every writer, and a caller should not have to care.
    // Applied to both sides of the lookup, so has("0_0\\meta.json"),
    // has("./0_0/meta.json") and has("0_0/meta.json") are one question.
    std::string normalizeName(std::string name) {

        std::replace(name.begin(), name.end(), '\\', '/');
        while (name.rfind("./", 0) == 0) name.erase(0, 2);
        return name;
    }

    // The whole file, in one allocation. These archives are ~180 MB and a caller
    // reads most of every one of them anyway, so a memory-mapped or streaming
    // design would save little and cost a per-platform mapping layer.
    //
    // The stream is constructed from the path OBJECT, never from path.string():
    // on Windows the narrow overload goes through the ANSI code page, so a name
    // like "Sainte-Anne-de-Beaupre" with an accented 'e' silently fails to open.
    // (The path only appears in messages as a string, where a mangled character
    // is cosmetic.)
    std::vector<unsigned char> readWholeFile(const std::filesystem::path& path) {

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) fail("cannot open '" + path.string() + "'");

        const std::streamoff end = in.tellg();
        if (end < 0) fail("cannot determine the size of '" + path.string() + "'");

        const auto size = static_cast<std::uint64_t>(end);
        if (size > static_cast<std::uint64_t>((std::numeric_limits<size_t>::max)())) {

            fail("'" + path.string() + "' is " + std::to_string(size) +
                 " bytes, too large to read into memory on this platform");
        }

        std::vector<unsigned char> bytes(static_cast<size_t>(size));
        in.seekg(0, std::ios::beg);
        if (size > 0) {

            in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
            if (static_cast<std::uint64_t>(in.gcount()) != size) {

                fail("short read from '" + path.string() + "': got " + std::to_string(in.gcount()) +
                     " of " + std::to_string(size) + " bytes");
            }
        }
        return bytes;
    }

}// namespace


bool ZipReader::looksLikeZip(const std::filesystem::path& path) {

    try {

        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        unsigned char magic[4] = {};
        in.read(reinterpret_cast<char*>(magic), sizeof(magic));
        if (in.gcount() != static_cast<std::streamsize>(sizeof(magic))) return false;

        return rd32(magic) == LOCAL_SIG;

    } catch (...) {

        // Documented never to throw: a caller uses this to choose a loader, and
        // an exception here would pre-empt the error the real loader would give.
        return false;
    }
}

ZipReader::ZipReader(const std::filesystem::path& path)
    : bytes_(readWholeFile(path)) {

    try {

        parse();

    } catch (const std::runtime_error& e) {

        // Re-throw with the file named — a bare "compression method 8" is
        // useless when a scene pulls in several archives.
        throw std::runtime_error(std::string(e.what()) + " (in '" + path.string() + "')");
    }
}

void ZipReader::parse() {

    const unsigned char* const d = bytes_.data();
    const std::uint64_t size = bytes_.size();

    if (size < EOCD_FIXED) {

        fail("file is " + std::to_string(size) + " bytes, too small to hold even an empty ZIP archive");
    }

    // 1. The end-of-central-directory record, found by scanning BACKWARD for
    //    its signature. The first hit must not be trusted blindly: a stored
    //    entry's payload, or a decoy planted in the archive comment, can carry
    //    the same four bytes. A candidate is accepted only when its recorded
    //    comment length reaches EXACTLY end-of-file.
    std::uint64_t eocd = 0;
    bool foundEocd = false;
    const std::uint64_t maxBack = (std::min)(size, EOCD_FIXED + MAX_COMMENT);

    for (std::uint64_t back = EOCD_FIXED; back <= maxBack; ++back) {

        const std::uint64_t pos = size - back;
        if (rd32(d + pos) != EOCD_SIG) continue;
        if (pos + EOCD_FIXED + rd16(d + pos + 20) != size) continue;

        eocd = pos;
        foundEocd = true;
        break;
    }

    if (!foundEocd) {

        fail("no end-of-central-directory record in the last " + std::to_string(maxBack) +
             " bytes; not a ZIP archive, or truncated");
    }

    // 2. Its fields. Everything the walk needs comes from here or from the
    //    central directory; nothing is inferred from the file's layout.
    const unsigned char* const e = d + eocd;
    const std::uint16_t thisDisk = rd16(e + 4);
    const std::uint16_t cdDisk = rd16(e + 6);
    const std::uint16_t entriesHere = rd16(e + 8);
    const std::uint16_t entriesTotal = rd16(e + 10);
    const std::uint32_t cdSize = rd32(e + 12);
    const std::uint32_t cdOffset = rd32(e + 16);

    if (entriesTotal == ZIP64_16) {

        fail("entry count is the ZIP64 sentinel 0xFFFF; ZIP64 archives are not supported");
    }
    if (cdSize == ZIP64_32 || cdOffset == ZIP64_32) {

        fail("central directory size " + std::to_string(cdSize) + " / offset " + std::to_string(cdOffset) +
             " hits the ZIP64 sentinel 0xFFFFFFFF; ZIP64 archives are not supported");
    }
    if (thisDisk != 0 || cdDisk != 0 || entriesHere != entriesTotal) {

        fail("spanned (multi-disk) archives are not supported: disk " + std::to_string(thisDisk) +
             ", directory on disk " + std::to_string(cdDisk) + ", " + std::to_string(entriesHere) +
             " of " + std::to_string(entriesTotal) + " entries here");
    }

    // 3. Where the central directory actually starts. Normally that is exactly
    //    cdOffset. An archive with something prepended — a self-extracting stub,
    //    or a concatenation — has every recorded offset shifted by the size of
    //    that prefix, and the shift is recoverable because the directory always
    //    ends where the EOCD begins. The correction is applied only when the
    //    plain reading does not land on a central header and the shifted one
    //    does, and the same delta then applies to every local header offset.
    std::uint64_t cdStart = cdOffset;
    const auto looksLikeCd = [&](std::uint64_t at) {
        if (at > size || cdSize > size - at) return false;
        return entriesTotal == 0 || (cdSize >= 4 && rd32(d + at) == CENTRAL_SIG);
    };

    if (!looksLikeCd(cdStart) && cdSize <= eocd && looksLikeCd(eocd - cdSize)) {

        cdStart = eocd - cdSize;
    }

    if (cdStart > size || cdSize > size - cdStart) {

        fail("central directory (" + std::to_string(cdSize) + " bytes at offset " + std::to_string(cdOffset) +
             ") runs past the end of the " + std::to_string(size) + "-byte file");
    }

    const std::int64_t base = static_cast<std::int64_t>(cdStart) - static_cast<std::int64_t>(cdOffset);
    const std::uint64_t cdEnd = cdStart + cdSize;

    // 4. Walk the central directory. It is the authority: the local file header
    //    of an entry written with a data descriptor (general purpose bit 3) has
    //    zero for crc, compressed size AND uncompressed size, so a reader that
    //    believes it comes away with an empty file and no error to show for it.
    entries_.reserve(entriesTotal);
    names_.reserve(entriesTotal);

    std::uint64_t pos = cdStart;

    for (std::uint32_t i = 0; i < entriesTotal; ++i) {

        if (pos + CENTRAL_FIXED > cdEnd) {

            fail("central directory ends after " + std::to_string(i) + " of " +
                 std::to_string(entriesTotal) + " entries");
        }

        const unsigned char* const c = d + pos;
        if (rd32(c) != CENTRAL_SIG) {

            fail("central directory entry " + std::to_string(i) + " has signature " + hex32(rd32(c)) +
                 ", expected " + hex32(CENTRAL_SIG));
        }

        const std::uint16_t flags = rd16(c + 8);
        const std::uint16_t method = rd16(c + 10);
        const std::uint32_t compSize = rd32(c + 20);
        const std::uint32_t uncompSize = rd32(c + 24);
        const std::uint16_t nameLen = rd16(c + 28);
        const std::uint16_t extraLen = rd16(c + 30);
        const std::uint16_t commentLen = rd16(c + 32);
        const std::uint32_t localOffset = rd32(c + 42);

        // Widened before adding: three 16-bit lengths and a 64-bit position
        // cannot overflow a uint64, and pos <= cdEnd <= size throughout.
        const std::uint64_t recordEnd = pos + CENTRAL_FIXED + nameLen + extraLen + commentLen;
        if (recordEnd > cdEnd) {

            fail("central directory entry " + std::to_string(i) + " (name " + std::to_string(nameLen) +
                 " + extra " + std::to_string(extraLen) + " + comment " + std::to_string(commentLen) +
                 " bytes) runs past the end of the directory");
        }
        if (nameLen == 0) fail("central directory entry " + std::to_string(i) + " has an empty name");

        // The name is raw bytes. The UTF-8 flag (0x800) is set by some writers
        // and not by others — the reference archive leaves it clear on plain
        // ASCII names — and guessing a code page for the rest would only invent
        // a second spelling of a name the caller already has. Lookup is
        // byte-exact after normalisation.
        const std::string name = normalizeName(
                std::string(reinterpret_cast<const char*>(c + CENTRAL_FIXED), nameLen));

        pos = recordEnd;

        if (flags & FLAG_ENCRYPTED) {

            fail(quoted(name) + " is encrypted (general purpose bit 0 set); encrypted entries are not supported");
        }

        if (method != 0) {

            const std::string named = methodName(method);
            fail(quoted(name) + ": compression method " + std::to_string(method) +
                 (named.empty() ? "" : " (" + named + ")") + " is not supported, only stored entries");
        }

        if (uncompSize == ZIP64_32 || compSize == ZIP64_32 || localOffset == ZIP64_32) {

            fail(quoted(name) + ": size " + std::to_string(uncompSize) + " / local header offset " +
                 std::to_string(localOffset) + " hits the ZIP64 sentinel 0xFFFFFFFF; ZIP64 archives are not supported");
        }

        // Stored means the two sizes are the same number by definition. If they
        // disagree the record is corrupt and there is no honest way to pick one.
        if (compSize != uncompSize) {

            fail(quoted(name) + ": stored entry declares compressed size " + std::to_string(compSize) +
                 " but uncompressed size " + std::to_string(uncompSize));
        }

        // Directory entries hold no data. The reference archive has none, but
        // other writers emit them, and listing a zero-byte "0_0/" would only
        // invite a caller to read it. A name that normalises away entirely
        // (".", "./") is the same thing said differently.
        if (name.empty() || name.back() == '/') continue;

        // 5. The local file header, for its two length fields ONLY. They may
        //    legitimately differ from the central directory's copies — writers
        //    put different extra fields in the two places — and they are what
        //    decides where the payload begins.
        const std::int64_t localStart = static_cast<std::int64_t>(localOffset) + base;
        if (localStart < 0 || static_cast<std::uint64_t>(localStart) + LOCAL_FIXED > size) {

            fail(quoted(name) + ": local header offset " + std::to_string(localOffset) +
                 " is outside the " + std::to_string(size) + "-byte file");
        }

        const unsigned char* const l = d + localStart;
        if (rd32(l) != LOCAL_SIG) {

            fail(quoted(name) + ": no local file header at offset " + std::to_string(localStart) +
                 " (found signature " + hex32(rd32(l)) + ", expected " + hex32(LOCAL_SIG) + ")");
        }

        const std::uint64_t dataOffset =
                static_cast<std::uint64_t>(localStart) + LOCAL_FIXED + rd16(l + 26) + rd16(l + 28);

        // Subtraction rather than dataOffset + uncompSize, so a size near the
        // 32-bit ceiling cannot wrap the comparison.
        if (dataOffset > size || uncompSize > size - dataOffset) {

            fail(quoted(name) + ": data runs past the end of the file (" + std::to_string(uncompSize) +
                 " bytes at offset " + std::to_string(dataOffset) + ", file is " + std::to_string(size) + " bytes)");
        }

        const auto inserted = entries_.emplace(name, Entry{dataOffset, uncompSize});
        if (inserted.second) {

            names_.push_back(name);

        } else {

            // A repeated name is legal — appending to an archive can leave the
            // superseded record in place — and the later one is the live one.
            inserted.first->second = Entry{dataOffset, uncompSize};
        }
    }
}

bool ZipReader::has(const std::string& name) const {

    return entries_.find(normalizeName(name)) != entries_.end();
}

std::vector<unsigned char> ZipReader::read(const std::string& name) const {

    const auto it = entries_.find(normalizeName(name));
    if (it == entries_.end()) {

        fail("no entry named " + quoted(name) + " in the archive (" +
             std::to_string(names_.size()) + " entries)");
    }

    // Both bounds were checked against the real file size while parsing, so
    // this is the one place that does not re-derive them.
    const auto offset = static_cast<size_t>(it->second.offset);
    const auto length = static_cast<size_t>(it->second.size);
    const unsigned char* const first = bytes_.data() + offset;
    return std::vector<unsigned char>(first, first + length);
}

std::vector<std::string> ZipReader::names() const {

    return names_;
}
