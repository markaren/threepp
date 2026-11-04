// Rewrites the .def that `cmake -E __create_def` produced for a DLL, without the
// symbols defined inside namespace std. cmake/WindowsDllExports.cmake says why
// a shared threepp needs this on MSVC and runs it before every link of the DLL.
//
// Usage: threepp_dll_exports <in.def> <out.def>
//
// The decision is made on the undecorated name: dbghelp's UnDecorateSymbolName
// with UNDNAME_NAME_ONLY turns a member of a std template into
// "std::vector<threepp::Vector3,...>::push_back" and a threepp function that
// merely takes a std argument into "threepp::foo", so no hand-rolled parse of
// the MSVC mangling is involved. A name dbghelp cannot undecorate stays
// exported, as does every unmangled (C) symbol.

#include <windows.h>

#include <dbghelp.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

    bool definedInStd(const std::string& symbol, std::vector<char>& buffer) {
        if (symbol.empty() || symbol[0] != '?') return false;
        const DWORD n = UnDecorateSymbolName(symbol.c_str(), buffer.data(),
                                             static_cast<DWORD>(buffer.size()), UNDNAME_NAME_ONLY);
        if (n == 0) return false;
        return std::string_view(buffer.data(), n).starts_with("std::");
    }

}// namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <in.def> <out.def>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "%s: cannot read %s\n", argv[0], argv[1]);
        return 1;
    }
    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "%s: cannot write %s\n", argv[0], argv[2]);
        return 1;
    }

    std::vector<char> buffer(1 << 16);
    std::size_t kept = 0;
    std::size_t dropped = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // __create_def writes "EXPORTS", then one "\t<symbol>" or
        // "\t<symbol> \t DATA" per export.
        const auto begin = line.find_first_not_of("\t ");
        if (line.empty() || line[0] != '\t' || begin == std::string::npos) {
            out << line << '\n';
            continue;
        }
        const auto end = line.find_first_of(" \t", begin);
        const std::string symbol = line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (definedInStd(symbol, buffer)) {
            ++dropped;
            continue;
        }
        out << line << '\n';
        ++kept;
    }
    out.flush();
    if (!out) {
        std::fprintf(stderr, "%s: writing %s failed\n", argv[0], argv[2]);
        return 1;
    }
    std::printf("%s: %zu exports, %zu std:: instantiations left out\n", argv[2], kept, dropped);
    return 0;
}
