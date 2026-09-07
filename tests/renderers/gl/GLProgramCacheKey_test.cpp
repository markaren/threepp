// Structural tripwire for the GL program cache key.
//
// getProgramCacheKey() delegates entirely to the hand-maintained
// ProgramParameters::hash(). Adding a variant flag to the struct and to the
// shader-define list, but forgetting the one line in hash(), does not fail to
// compile and does not fail any rendering test that only ever draws one
// material — the cache simply serves the previously compiled program for the
// new flag. It shows up much later as "the second material ignores the
// setting", which is a miserable thing to debug.
//
// So this test reads both files and checks that every scalar member declared
// on ProgramParameters is actually mentioned inside hash(). It is a source
// check rather than a behavioural one on purpose: it fails the moment the
// field is added, in the same commit, naming the field.
//
// The exclusion list below is for members that are legitimately not part of
// the key — the shader text and defines are hashed by GLPrograms via a
// different route, and the rest are outputs rather than variant selectors.
// Adding to it should be a deliberate, argued act.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace {

    std::string readFile(const std::string& path) {
        std::ifstream in(path);
        REQUIRE(in.is_open());
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    // Members that are deliberately outside the cache key.
    const std::set<std::string> kNotPartOfTheKey{
            "shaderID",             // selects vertexShader/fragmentShader, hashed via those
            "shaderName",           // cosmetic: only feeds #define SHADER_NAME
            "defines",              // hashed separately by GLPrograms
            "vertexShader",         // hashed separately by GLPrograms
            "fragmentShader",       // hashed separately by GLPrograms
            "isRawShaderMaterial",  // implied by the shader text itself
            "precision",            // constant in practice; changing it needs a full reload
            "maxBones",             // derived from skinning + capabilities
            "index0AttributeName",  // an attribute-binding detail, not a variant
            "uniforms",             // a pointer to live uniform storage, not a variant
    };

}// namespace

TEST_CASE("Program cache key: every ProgramParameters member reaches hash()") {
    const std::string header = readFile(std::string(PROGRAM_PARAMETERS_DIR) + "/ProgramParameters.hpp");
    const std::string source = readFile(std::string(PROGRAM_PARAMETERS_DIR) + "/ProgramParameters.cpp");

    // Body of hash(), from its definition to the closing `return s.str();`.
    const auto hashStart = source.find("std::string ProgramParameters::hash() const");
    REQUIRE(hashStart != std::string::npos);
    const auto hashEnd = source.find("return s.str();", hashStart);
    REQUIRE(hashEnd != std::string::npos);
    const std::string hashBody = source.substr(hashStart, hashEnd - hashStart);

    // Member declarations: everything between `struct ProgramParameters {` and
    // the constructor. Matches `<type> name{};`, `<type> name;`, `<type> name = ...;`.
    const auto structStart = header.find("struct ProgramParameters {");
    REQUIRE(structStart != std::string::npos);
    const auto structEnd = header.find("ProgramParameters(", structStart);
    REQUIRE(structEnd != std::string::npos);
    const std::string members = header.substr(structStart, structEnd - structStart);

    const std::regex decl(R"(^\s{12}[A-Za-z_][\w:<>,\s\*]*?\b(\w+)\s*(\{\}|=[^;]*)?;\s*$)");

    std::set<std::string> missing;
    int seen = 0;

    std::istringstream lines(members);
    for (std::string line; std::getline(lines, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::smatch m;
        if (!std::regex_match(line, m, decl)) continue;

        const std::string name = m[1].str();
        if (kNotPartOfTheKey.count(name)) continue;
        ++seen;

        // A bare `name` token inside hash() — not a prefix of a longer
        // identifier (envMap must not be satisfied by envMapMode).
        const std::regex use("\\b" + name + "\\b");
        if (!std::regex_search(hashBody, use)) missing.insert(name);
    }

    // Guard the parse itself: if the regex ever stops matching declarations the
    // test would pass vacuously. There are ~70 hashed members today.
    INFO("parsed " << seen << " cache-key members from ProgramParameters.hpp");
    CHECK(seen > 60);

    std::string report;
    for (const auto& n : missing) report += n + " ";
    INFO("members declared but never read by hash(): " << report);
    CHECK(missing.empty());
}
