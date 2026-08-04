
#include "threepp/loaders/xacro/PackageResolver.hpp"

#include "pugixml.hpp"

#include <cstdlib>
#include <system_error>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    std::string trim(std::string s) {

        const auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
        std::size_t b = 0;
        std::size_t e = s.size();
        while (b < e && isSpace(s[b])) ++b;
        while (e > b && isSpace(s[e - 1])) --e;
        return s.substr(b, e - b);
    }

    bool isDirectory(const std::filesystem::path& p) {

        std::error_code ec;
        return std::filesystem::is_directory(p, ec);
    }

    bool isFile(const std::filesystem::path& p) {

        std::error_code ec;
        return std::filesystem::is_regular_file(p, ec);
    }

    std::vector<std::filesystem::path> ancestors(const std::filesystem::path& fromDoc) {

        std::error_code ec;
        std::filesystem::path dir = std::filesystem::weakly_canonical(fromDoc, ec);
        if (ec) dir = fromDoc;
        if (!isDirectory(dir)) dir = dir.parent_path();

        std::vector<std::filesystem::path> out;
        while (!dir.empty()) {
            out.push_back(dir);
            const auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
        return out;
    }

    std::vector<std::filesystem::path> envPaths(const char* variable) {

        std::vector<std::filesystem::path> out;
        const char* raw = std::getenv(variable);
        if (!raw) return out;

        const std::string value(raw);
        const char sep = static_cast<char>(std::filesystem::path::preferred_separator) == '\\' ? ';' : ':';

        std::size_t start = 0;
        while (start <= value.size()) {
            const std::size_t end = value.find(sep, start);
            const std::string entry = trim(value.substr(start, end == std::string::npos ? std::string::npos : end - start));
            if (!entry.empty()) out.emplace_back(entry);
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return out;
    }

    bool named(const std::filesystem::path& dir, const std::string& package) {

        const auto name = PackageResolver::manifestName(dir);
        if (name) return *name == package;
        return isFile(dir / "package.xml") && dir.filename().string() == package;
    }

}// namespace

void PackageResolver::addPackagePath(const std::string& package, const std::filesystem::path& dir) {

    registry_[package] = dir;
    cache_.erase(package);
}

void PackageResolver::clearCache() {

    cache_.clear();
}

std::optional<std::string> PackageResolver::manifestName(const std::filesystem::path& dir) {

    const auto manifest = dir / "package.xml";
    if (!isFile(manifest)) return std::nullopt;

    pugi::xml_document doc;
    if (!doc.load_file(manifest.string().c_str())) return std::nullopt;

    const auto name = doc.child("package").child("name");
    if (!name) return std::nullopt;

    const std::string text = trim(name.text().as_string());
    if (text.empty()) return std::nullopt;

    return text;
}

std::optional<std::filesystem::path> PackageResolver::resolve(const std::string& package,
                                                              const std::filesystem::path& fromDoc,
                                                              std::vector<std::string>* tried) {

    const auto note = [&](const std::filesystem::path& p) {
        if (tried) tried->push_back(p.string());
    };

    if (const auto it = registry_.find(package); it != registry_.end()) return it->second;
    if (const auto it = cache_.find(package); it != cache_.end()) return it->second;

    const auto accept = [&](const std::filesystem::path& p) {
        cache_[package] = p;
        return std::optional<std::filesystem::path>(p);
    };

    const auto chain = ancestors(fromDoc);

    for (const auto& dir : chain) {
        note(dir);
        if (named(dir, package)) return accept(dir);
    }

    for (const auto& dir : chain) {
        const auto candidate = dir / package;
        if (!isDirectory(candidate)) continue;
        note(candidate);
        if (named(candidate, package)) return accept(candidate);
    }

    for (const auto& entry : envPaths("ROS_PACKAGE_PATH")) {
        note(entry);
        if (isDirectory(entry) && (named(entry, package) || entry.filename().string() == package)) {
            return accept(entry);
        }
        const auto candidate = entry / package;
        note(candidate);
        if (isDirectory(candidate)) return accept(candidate);
    }

    for (const auto& entry : envPaths("AMENT_PREFIX_PATH")) {
        const auto candidate = entry / "share" / package;
        note(candidate);
        if (isDirectory(candidate)) return accept(candidate);
    }

    return std::nullopt;
}
