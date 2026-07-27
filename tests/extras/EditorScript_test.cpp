
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/objects/Group.hpp"

using namespace threepp;
using namespace threepp::editor;


TEST_CASE("ScriptConfig round-trips through userData", "[editor]") {

    auto group = Group::create();

    // Nothing attached: no entries, and read() says so rather than handing back
    // a default-constructed config nobody asked for.
    CHECK_FALSE(ScriptConfig::read(*group).has_value());

    ScriptConfig config;
    config.path = "C:/projects/scripts/spinner.py";
    config.setField("speed", "1.5");
    config.setField("clockwise", "1");
    config.setField("label", "spin");
    config.write(*group);

    // The path gets its own plain key: it contains ':' and '/' and, on a real
    // Windows path, could contain the flat format's delimiters too.
    CHECK(group->userData.contains(ScriptConfig::pathKey));
    CHECK(group->userData.contains(ScriptConfig::fieldsKey));

    const auto read = ScriptConfig::read(*group);
    REQUIRE(read.has_value());
    CHECK(*read == config);
    CHECK(read->path == config.path);
    REQUIRE(read->fields.size() == 3);
    // Insertion order survives, which is what keeps a saved document stable.
    CHECK(read->fields[0].name == "speed");
    CHECK(read->fields[1].name == "clockwise");
    CHECK(read->fields[2].name == "label");

    // Clearing the script leaves no trace in the file.
    ScriptConfig{}.write(*group);
    CHECK_FALSE(ScriptConfig::read(*group).has_value());
    CHECK_FALSE(group->userData.contains(ScriptConfig::pathKey));
    CHECK_FALSE(group->userData.contains(ScriptConfig::fieldsKey));
}

TEST_CASE("a script with no fields writes only the path", "[editor]") {

    auto group = Group::create();

    ScriptConfig config;
    config.path = "spinner.py";
    config.write(*group);

    CHECK(group->userData.contains(ScriptConfig::pathKey));
    CHECK_FALSE(group->userData.contains(ScriptConfig::fieldsKey));

    const auto read = ScriptConfig::read(*group);
    REQUIRE(read.has_value());
    CHECK(read->fields.empty());
}

TEST_CASE("delimiters are stripped from encoded fields", "[editor]") {

    ScriptConfig config;
    config.setField("na;me", "va=lue;here");

    const auto decoded = ScriptConfig::decodeFields(config.encodeFields());
    REQUIRE(decoded.size() == 1);
    CHECK(decoded[0].name == "name");
    CHECK(decoded[0].value == "valuehere");
}

TEST_CASE("field values encode and parse back", "[editor]") {

    // Locale-independent, trailing zeros trimmed: an unchanged value has to
    // encode byte-identically or documents stop diffing cleanly.
    CHECK(ScriptConfig::toText(1.5f) == "1.5");
    CHECK(ScriptConfig::toText(2.f) == "2");
    CHECK(ScriptConfig::toText(0.f) == "0");
    CHECK(ScriptConfig::toText(-0.25f) == "-0.25");
    CHECK(ScriptConfig::toText(7) == "7");
    CHECK(ScriptConfig::toText(true) == "1");
    CHECK(ScriptConfig::toText(false) == "0");

    CHECK(ScriptConfig::toFloat("1.5") == 1.5f);
    CHECK(ScriptConfig::toInt("7") == 7);
    CHECK(ScriptConfig::toBool("1"));
    CHECK_FALSE(ScriptConfig::toBool("0"));
    // Hand-edited files are not required to use the editor's spelling.
    CHECK(ScriptConfig::toBool("true"));
    CHECK_FALSE(ScriptConfig::toBool("False"));

    // Garbage falls back instead of throwing: this text arrives from a file the
    // editor did not necessarily write.
    CHECK(ScriptConfig::toFloat("not a number", 3.f) == 3.f);
    CHECK(ScriptConfig::toInt("", -1) == -1);
    CHECK(ScriptConfig::toBool("maybe", true));
}

TEST_CASE("setField replaces in place and retainFields prunes", "[editor]") {

    ScriptConfig config;
    config.setField("speed", "1");
    config.setField("radius", "2");
    config.setField("speed", "9");

    REQUIRE(config.fields.size() == 2);
    CHECK(config.fields[0].name == "speed");
    CHECK(config.fields[0].value == "9");
    REQUIRE(config.field("radius").has_value());
    CHECK(*config.field("radius") == "2");
    CHECK_FALSE(config.field("missing").has_value());

    // A class that lost an attribute should not keep its value in the document
    // forever.
    config.retainFields({"radius"});
    REQUIRE(config.fields.size() == 1);
    CHECK(config.fields[0].name == "radius");

    config.eraseField("radius");
    CHECK(config.fields.empty());
}

TEST_CASE("a malformed fields entry loads what it can", "[editor]") {

    auto group = Group::create();
    group->userData[ScriptConfig::pathKey] = std::string("spinner.py");
    // Trailing separator, a token with no '=', and an empty name.
    group->userData[ScriptConfig::fieldsKey] = std::string("speed=2;garbage;=orphan;ok=1;");

    const auto read = ScriptConfig::read(*group);
    REQUIRE(read.has_value());
    REQUIRE(read->fields.size() == 2);
    CHECK(read->fields[0].name == "speed");
    CHECK(read->fields[1].name == "ok");
}

TEST_CASE("fields without a path are not written", "[editor]") {

    auto group = Group::create();

    // Values with nothing to apply them to are meaningless; a cleared script
    // must not leave its parameters behind.
    ScriptConfig config;
    config.setField("speed", "1.5");
    config.write(*group);

    CHECK_FALSE(group->userData.contains(ScriptConfig::pathKey));
    CHECK_FALSE(group->userData.contains(ScriptConfig::fieldsKey));
}
