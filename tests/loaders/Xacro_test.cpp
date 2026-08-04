#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/loaders/xacro/Expr.hpp"
#include "threepp/loaders/xacro/PackageResolver.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Substitution.hpp"
#include "threepp/loaders/xacro/Value.hpp"
#include "threepp/loaders/xacro/YamlLite.hpp"
#include "threepp/objects/Robot.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    struct TempDir {

        std::filesystem::path path;

        explicit TempDir(const std::string& tag) {

            static int counter = 0;
            path = std::filesystem::temp_directory_path() /
                   ("threepp_xacro_" + tag + "_" + std::to_string(++counter));
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path);
        }

        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
    };

    void writeFile(const std::filesystem::path& file, const std::string& text) {

        std::filesystem::create_directories(file.parent_path());
        std::ofstream out(file, std::ios::binary);
        out << text;
    }

    void writeManifest(const std::filesystem::path& dir, const std::string& name) {

        writeFile(dir / "package.xml",
                  "<?xml version=\"1.0\"?>\n<package format=\"3\">\n  <name>" + name + "</name>\n</package>\n");
    }

    void setEnv(const char* name, const char* value) {

#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) ::setenv(name, value, 1);
        else ::unsetenv(name);
#endif
    }

    Value eval(const std::string& expression, const Scope* scope = nullptr,
               const std::filesystem::path& document = {}) {

        EvalContext ctx;
        ctx.scope = scope;
        ctx.document = document;
        return evaluate(expression, ctx);
    }

    std::string wrap(const std::string& body) {

        return "<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"test\">\n" + body + "\n</robot>\n";
    }

    ProcessResult process(const std::string& body,
                          const std::map<std::string, std::string>& args = {},
                          const std::filesystem::path& baseDir = std::filesystem::temp_directory_path()) {

        Processor processor;
        processor.setArgs(args);
        return processor.processString(wrap(body), baseDir);
    }

    bool contains(const std::string& haystack, const std::string& needle) {

        return haystack.find(needle) != std::string::npos;
    }

    std::string requireOk(const ProcessResult& result) {

        const std::string why = result.errors.empty() ? std::string{} : result.errors.front();
        INFO(why);
        REQUIRE(result.ok);
        return result.xml;
    }

}// namespace


TEST_CASE("Value classification and rendering") {

    REQUIRE(classify("5").isInt());
    REQUIRE(classify("5").asInt() == 5);
    REQUIRE(classify("  -12  ").isInt());
    REQUIRE(classify("1.5").isDouble());
    REQUIRE(classify("1e3").isDouble());
    REQUIRE(classify("true").isString());
    REQUIRE(classify("").isString());
    REQUIRE(classify("1.2.3").isString());
    REQUIRE(classify("0x10").isString());

    REQUIRE(Value(1.0).toString() == "1.0");
    REQUIRE(Value(0.5).toString() == "0.5");
    REQUIRE(Value(2LL).toString() == "2");
    REQUIRE(Value(true).toString() == "True");
    REQUIRE(Value().toString() == "None");
    REQUIRE(Value("plain").toString() == "plain");
    REQUIRE(Value(0.1 + 0.2).toString() == "0.30000000000000004");

    REQUIRE(Value(List{Value(1LL), Value("a")}).toString() == "[1, 'a']");
    REQUIRE(Value(Dict{{"a", Value(1LL)}, {"b", Value("x")}}).toString() == "{'a': 1, 'b': 'x'}");

    REQUIRE(Value(1LL) == Value(1.0));
    REQUIRE(Value(true) == Value(1LL));
    REQUIRE_FALSE(Value("1") == Value(1LL));

    REQUIRE(Value("0").truthy());
    REQUIRE_FALSE(Value("").truthy());
    REQUIRE_FALSE(Value(0.0).truthy());
    REQUIRE_FALSE(Value(List{}).truthy());
    REQUIRE_FALSE(Value().truthy());
}

TEST_CASE("Scope frames restore in reverse write order") {

    Scope scope;
    scope.set("a", Value(1LL));

    scope.pushFrame();
    scope.set("a", Value(2LL));
    scope.set("b", Value("inner"));
    REQUIRE(scope.get("a") == Value(2LL));
    scope.set("a", Value(3LL));
    scope.popFrame();

    REQUIRE(scope.get("a") == Value(1LL));
    REQUIRE_FALSE(scope.has("b"));

    SECTION("scope=parent survives the current frame") {

        scope.pushFrame();
        scope.pushFrame();
        scope.set("c", Value("local"));
        scope.setParent("c", Value("caller"));
        scope.popFrame();
        REQUIRE(scope.get("c") == Value("caller"));
        scope.popFrame();
        REQUIRE_FALSE(scope.has("c"));
    }

    SECTION("scope=global survives every frame") {

        scope.pushFrame();
        scope.pushFrame();
        scope.set("g", Value(1LL));
        scope.setGlobal("g", Value(2LL));
        scope.popFrame();
        scope.popFrame();
        REQUIRE(scope.get("g") == Value(2LL));
    }
}

TEST_CASE("YamlLite reads typed scalars") {

    const Value v = parseYaml("i: 3\nf: 1.5\nneg: -2\nb: true\nB: False\ns: hello\nq: 'true'\nn: ~\nempty:\n");
    const Dict& d = v.asDict();

    REQUIRE(d.at("i").isInt());
    REQUIRE(d.at("i").asInt() == 3);
    REQUIRE(d.at("f").asNumber() == Catch::Approx(1.5));
    REQUIRE(d.at("neg").asInt() == -2);
    REQUIRE(d.at("b").isBool());
    REQUIRE(d.at("b").truthy());
    REQUIRE(d.at("B").isBool());
    REQUIRE_FALSE(d.at("B").truthy());
    REQUIRE(d.at("s").asString() == "hello");
    REQUIRE(d.at("q").isString());
    REQUIRE(d.at("q").asString() == "true");
    REQUIRE(d.at("n").isNone());
    REQUIRE(d.at("empty").isNone());
}

TEST_CASE("YamlLite reads nested blocks and sequences") {

    const std::string text =
            "root:\n"
            "  child:\n"
            "    value: 3\n"
            "  list:\n"
            "    - 1\n"
            "    - 2\n"
            "items:\n"
            "- name: a\n"
            "  id: 1\n"
            "- name: b\n"
            "  id: 2\n";

    const Value v = parseYaml(text);
    REQUIRE(v.asDict().at("root").asDict().at("child").asDict().at("value").asInt() == 3);

    const List& list = v.asDict().at("root").asDict().at("list").asList();
    REQUIRE(list.size() == 2);
    REQUIRE(list[1].asInt() == 2);

    const List& items = v.asDict().at("items").asList();
    REQUIRE(items.size() == 2);
    REQUIRE(items[0].asDict().at("name").asString() == "a");
    REQUIRE(items[1].asDict().at("id").asInt() == 2);
}

TEST_CASE("YamlLite reads flow collections, comments and CRLF") {

    const Value v = parseYaml("# leading comment\r\n"
                              "a: [1, 2, three]  # trailing\r\n"
                              "b: {x: 1, y: 'two'}\r\n"
                              "\r\n"
                              "c: 4\r\n");

    REQUIRE(v.asDict().at("a").asList().size() == 3);
    REQUIRE(v.asDict().at("a").asList()[2].asString() == "three");
    REQUIRE(v.asDict().at("b").asDict().at("y").asString() == "two");
    REQUIRE(v.asDict().at("c").asInt() == 4);
    REQUIRE(parseYaml("[]").asList().empty());
}

TEST_CASE("YamlLite converts the !degrees and !radians tags") {

    // The two tags python xacro registers on its yaml loader; robot joint limit packs are
    // written with them (ur_description's config/*/joint_limits.yaml, for one).
    const Value v = parseYaml("full: !degrees  360.0\n"
                              "back: !degrees -360.0\n"
                              "half: !degrees 180\n"
                              "raw: !radians 1.5\n"
                              "list:\n"
                              "  - !degrees 90\n"
                              "  - !radians 2\n"
                              "flow: [!degrees 90, 3]\n"
                              "commented: !degrees 180  # half a turn\n");

    constexpr double pi = 3.14159265358979323846;

    REQUIRE(v.asDict().at("full").isDouble());
    REQUIRE(v.asDict().at("full").asNumber() == Catch::Approx(2.0 * pi).margin(1e-15));
    REQUIRE(v.asDict().at("back").asNumber() == Catch::Approx(-2.0 * pi).margin(1e-15));
    REQUIRE(v.asDict().at("half").asNumber() == Catch::Approx(pi).margin(1e-15));
    REQUIRE(v.asDict().at("raw").isDouble());
    REQUIRE(v.asDict().at("raw").asNumber() == Catch::Approx(1.5));
    REQUIRE(v.asDict().at("list").asList()[0].asNumber() == Catch::Approx(pi / 2.0).margin(1e-15));
    REQUIRE(v.asDict().at("list").asList()[1].asNumber() == Catch::Approx(2.0));
    REQUIRE(v.asDict().at("flow").asList()[0].asNumber() == Catch::Approx(pi / 2.0).margin(1e-15));
    REQUIRE(v.asDict().at("flow").asList()[1].asInt() == 3);
    REQUIRE(v.asDict().at("commented").asNumber() == Catch::Approx(pi).margin(1e-15));

    REQUIRE_THROWS_AS(parseYaml("a: !degrees\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: !degrees text\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: !degrees true\n"), XacroError);
}

TEST_CASE("YamlLite resolves anchors and aliases") {

    // How an inertia config says "this finger is the other one" — franka_description
    // writes its two fingers exactly this way.
    const auto v = parseYaml(
            "left: &finger\n"
            "  mass: 0.03\n"
            "  inertia:\n"
            "    xx: 0.5\n"
            "right: *finger\n"
            "scalar: &m 2.5\n"
            "again: *m\n");

    REQUIRE(v.asDict().at("right").asDict().at("mass").asNumber() == Catch::Approx(0.03));
    REQUIRE(v.asDict().at("right").asDict().at("inertia").asDict().at("xx").asNumber() == Catch::Approx(0.5));
    REQUIRE(v.asDict().at("again").asNumber() == Catch::Approx(2.5));

    SECTION("a merge key folds the aliased mapping in, and loses to what the document says") {

        const auto merged = parseYaml(
                "base: &base\n"
                "  a: 1\n"
                "  b: 2\n"
                "derived:\n"
                "  <<: *base\n"
                "  b: 3\n");

        const auto& derived = merged.asDict().at("derived").asDict();
        REQUIRE(derived.at("a").asInt() == 1);
        REQUIRE(derived.at("b").asInt() == 3);
        REQUIRE(derived.count("<<") == 0);
    }

    SECTION("a merge key written in block form merges the same way") {

        const auto merged = parseYaml(
                "derived:\n"
                "  <<:\n"
                "    a: 9\n"
                "    b: 2\n"
                "  b: 3\n");

        const auto& derived = merged.asDict().at("derived").asDict();
        REQUIRE(derived.at("a").asInt() == 9);
        REQUIRE(derived.at("b").asInt() == 3);
        REQUIRE(derived.count("<<") == 0);
    }

    SECTION("an alias with no anchor is an error, not an empty value") {

        REQUIRE_THROWS_AS(parseYaml("a: *nobody\n"), XacroError);
    }
}

TEST_CASE("YamlLite rejects what it does not support") {

    REQUIRE_THROWS_AS(parseYaml("a: !!str 1\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: !custom 1\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: !Degrees 1\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: |\n  text\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a:\n\tb: 1\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: 1\n---\nb: 2\n"), XacroError);
}

TEST_CASE("Expr arithmetic and precedence") {

    REQUIRE(eval("1 + 2 * 3") == Value(7LL));
    REQUIRE(eval("(1 + 2) * 3") == Value(9LL));
    REQUIRE(eval("2 ** 3 ** 2") == Value(512LL));
    REQUIRE(eval("-2 ** 2") == Value(-4LL));
    REQUIRE(eval("7 / 2") == Value(3.5));
    REQUIRE(eval("7 // 2") == Value(3LL));
    REQUIRE(eval("-7 // 2") == Value(-4LL));
    REQUIRE(eval("7 % 3") == Value(1LL));
    REQUIRE(eval("-7 % 3") == Value(2LL));
    REQUIRE(eval("1 + 2.0").isDouble());
    REQUIRE(eval("2 * 3").isInt());
}

TEST_CASE("Expr comparisons, booleans and membership") {

    REQUIRE(eval("1 < 2 < 3").truthy());
    REQUIRE_FALSE(eval("1 < 2 < 2").truthy());
    REQUIRE(eval("1 == 1.0").truthy());
    REQUIRE(eval("'a' != 'b'").truthy());
    REQUIRE(eval("not 0").truthy());
    REQUIRE(eval("1 and 2") == Value(2LL));
    REQUIRE(eval("0 or 'x'") == Value("x"));

    REQUIRE(eval("2 in [1, 2, 3]").truthy());
    REQUIRE(eval("4 not in [1, 2, 3]").truthy());
    REQUIRE(eval("'oo' in 'foo'").truthy());
    REQUIRE(eval("'a' in dict(a=1)").truthy());
    REQUIRE(eval("'b' not in dict(a=1)").truthy());
}

TEST_CASE("Expr strings, containers and builtins") {

    REQUIRE(eval("'a' + 'b'") == Value("ab"));
    REQUIRE(eval("len('abc')") == Value(3LL));
    REQUIRE(eval("str(1.5)") == Value("1.5"));
    REQUIRE(eval("int('7') + 1") == Value(8LL));
    REQUIRE(eval("float('1.5')") == Value(1.5));
    REQUIRE(eval("bool('')").truthy() == false);

    REQUIRE(eval("[10, 20, 30][1]") == Value(20LL));
    REQUIRE(eval("[10, 20, 30][-1]") == Value(30LL));
    REQUIRE(eval("dict(a=1, b='x')['b']") == Value("x"));
    REQUIRE(eval("dict(outer=dict(inner=5))['outer']['inner']") == Value(5LL));
    REQUIRE(eval("'lo' in 'hello' if True else False").truthy());
    REQUIRE(eval("1 if 0 else 2") == Value(2LL));

    REQUIRE(eval("radians(180)").asNumber() == Catch::Approx(3.14159265358979));
    REQUIRE(eval("degrees(pi)").asNumber() == Catch::Approx(180.0));
    REQUIRE(eval("math.pi").asNumber() == Catch::Approx(3.14159265358979));
    REQUIRE(eval("round(sqrt(16))") == Value(4LL));
    REQUIRE(eval("max(1, 5, 3)") == Value(5LL));
    REQUIRE(eval("min([4, 2, 9])") == Value(2LL));
    REQUIRE(eval("abs(-3)").isInt());
}

TEST_CASE("Expr resolves names through the scope") {

    Scope scope;
    scope.set("width", Value(0.5));
    scope.set("name", Value("robot"));
    scope.set("limits", Value(Dict{{"upper", Value(2LL)}}));

    REQUIRE(eval("width * 2", &scope) == Value(1.0));
    REQUIRE(eval("name + '_link'", &scope) == Value("robot_link"));
    REQUIRE(eval("limits['upper']", &scope) == Value(2LL));

    // A mapping answers to a dotted name as well as a subscript.
    REQUIRE(eval("limits.upper", &scope) == Value(2LL));
    REQUIRE_THROWS_AS(eval("limits.lower", &scope), XacroError);
    REQUIRE_THROWS_AS(eval("width.upper", &scope), XacroError);
}

TEST_CASE("Expr slices lists and strings") {

    Scope scope;
    scope.set("types", Value(List{Value("base"), Value("left"), Value("right")}));

    // "every arm but the first", which is how a two-armed description splits its list.
    REQUIRE(eval("types[1:]", &scope).asList().size() == 2);
    REQUIRE(eval("types[1:]", &scope).asList()[0] == Value("left"));
    REQUIRE(eval("types[:1]", &scope).asList()[0] == Value("base"));
    REQUIRE(eval("types[1:2]", &scope).asList().size() == 1);
    REQUIRE(eval("types[:]", &scope).asList().size() == 3);
    REQUIRE(eval("types[-1:]", &scope).asList()[0] == Value("right"));
    REQUIRE(eval("types[::2]", &scope).asList().size() == 2);
    REQUIRE(eval("types[::-1]", &scope).asList()[0] == Value("right"));

    // Out of range clamps, the way Python does, rather than throwing.
    REQUIRE(eval("types[5:]", &scope).asList().empty());
    REQUIRE(eval("types[:99]", &scope).asList().size() == 3);

    REQUIRE(eval("'prefix_link'[7:]") == Value("link"));
    REQUIRE(eval("len(types[1:])", &scope) == Value(2LL));

    REQUIRE_THROWS_AS(eval("types[::0]", &scope), XacroError);
    REQUIRE_THROWS_AS(eval("types['a':]", &scope), XacroError);
    REQUIRE_THROWS_AS(eval("5[1:]"), XacroError);
}

TEST_CASE("A substitution inside an expression is expanded before it is evaluated") {

    // `${xacro.load_yaml('$(find pkg)/config/x.yaml')}` is how a ROS description names a
    // file it wants to read - franka_description opens its inertias exactly this way, and
    // the path it builds is full of backslashes on Windows for the lexer to trip over.
    const TempDir dir("nested_subst");
    writeFile(dir.path / "config" / "limits.yaml", "shoulder:\n  upper: 6.28\n");

    Processor processor;
    processor.addPackagePath("arm_description", dir.path);

    const auto xml = requireOk(processor.processString(
            wrap(R"XML(<xacro:arg name="which" default="limits"/>
                       <xacro:property name="limits"
                                       value="${xacro.load_yaml('$(find arm_description)/config/$(arg which).yaml')}"/>
                       <limit upper="${limits.shoulder.upper}" home="$(dirname)"/>)XML"),
            dir.path));

    REQUIRE(contains(xml, R"XML(upper="6.28")XML"));
    REQUIRE(contains(xml, "home=\"" + dir.path.string() + "\""));

    SECTION("an opener with no closer is a dollar in a string, not a substitution") {

        const auto result = process(R"XML(<a v="${'cost is $(unknown'}"/>)XML");
        const std::string got = result.ok ? result.xml : result.errors.front();
        INFO(got);
        REQUIRE(result.ok);
        REQUIRE(contains(result.xml, "cost is $(unknown"));
    }
}

TEST_CASE("Expr errors throw instead of defaulting to zero") {

    REQUIRE_THROWS_AS(eval("1 / 0"), XacroError);
    REQUIRE_THROWS_AS(eval("1 % 0"), XacroError);
    REQUIRE_THROWS_AS(eval("nosuchname"), XacroError);
    REQUIRE_THROWS_AS(eval("1 +"), XacroError);
    REQUIRE_THROWS_AS(eval("'a' + 1"), XacroError);
    REQUIRE_THROWS_AS(eval("dict(a=1)['missing']"), XacroError);
    REQUIRE_THROWS_AS(eval("[1, 2][5]"), XacroError);
    REQUIRE_THROWS_AS(eval("sin('x')"), XacroError);
    REQUIRE_THROWS_AS(eval("math.nosuch(1)"), XacroError);
    REQUIRE_THROWS_AS(eval("1 @ 2"), XacroError);
}

TEST_CASE("Expr load_yaml resolves against the document directory") {

    const TempDir dir("yaml");
    writeFile(dir.path / "config" / "limits.yaml", "joint:\n  upper: 6.28\n  name: shoulder\n");

    const auto document = dir.path / "robot.urdf.xacro";

    REQUIRE(eval("load_yaml('config/limits.yaml')['joint']['upper']", nullptr, document).asNumber() ==
            Catch::Approx(6.28));
    REQUIRE(eval("xacro.load_yaml('config/limits.yaml')['joint']['name']", nullptr, document) ==
            Value("shoulder"));
    REQUIRE_THROWS_AS(eval("load_yaml('missing.yaml')", nullptr, document), XacroError);
}

TEST_CASE("Substitution evaluates expressions and keeps whole-span types") {

    Scope scope;
    scope.set("count", Value(3LL));

    SubstCtx ctx;
    ctx.scope = &scope;

    const auto whole = substitute("${count * 2}", ctx);
    REQUIRE(whole.text == "6");
    REQUIRE(whole.whole.has_value());
    REQUIRE(whole.whole->isInt());

    const auto partial = substitute("link_${count}", ctx);
    REQUIRE(partial.text == "link_3");
    REQUIRE_FALSE(partial.whole.has_value());

    REQUIRE(substitute("${'{' + '}'}", ctx).text == "{}");
    REQUIRE(substitute("$${count}", ctx).text == "${count}");
    REQUIRE(substitute("a$$b", ctx).text == "a$b");
    REQUIRE(substitute("plain text", ctx).text == "plain text");
    REQUIRE_THROWS_AS(substitute("${count", ctx), XacroError);
}

TEST_CASE("Substitution commands") {

    const TempDir dir("subst");
    const auto document = dir.path / "sub" / "robot.xacro";
    writeFile(document, "<robot/>");

    Scope scope;
    scope.set("pkg", Value("subst_pkg"));

    std::map<std::string, Value> args{{"prefix", Value("left_")}, {"n", Value(2LL)}};

    PackageResolver packages;
    packages.addPackagePath("subst_pkg", dir.path / "subst_pkg");

    SubstCtx ctx;
    ctx.scope = &scope;
    ctx.args = &args;
    ctx.packages = &packages;
    ctx.document = document;

    REQUIRE(substitute("$(arg prefix)joint", ctx).text == "left_joint");
    REQUIRE(substitute("$(arg n)", ctx).text == "2");
    REQUIRE_THROWS_AS(substitute("$(arg missing)", ctx), XacroError);
    REQUIRE_THROWS_AS(substitute("$(nosuchcmd x)", ctx), XacroError);

    REQUIRE(substitute("$(dirname)", ctx).text == (dir.path / "sub").string());

    setEnv("THREEPP_XACRO_TEST_VAR", "from_env");
    REQUIRE(substitute("$(env THREEPP_XACRO_TEST_VAR)", ctx).text == "from_env");
    REQUIRE(substitute("$(optenv THREEPP_XACRO_TEST_VAR fallback)", ctx).text == "from_env");
    setEnv("THREEPP_XACRO_TEST_VAR", nullptr);
    REQUIRE_THROWS_AS(substitute("$(env THREEPP_XACRO_TEST_VAR)", ctx), XacroError);
    REQUIRE(substitute("$(optenv THREEPP_XACRO_UNSET_VAR fallback)", ctx).text == "fallback");
    REQUIRE(substitute("$(optenv THREEPP_XACRO_UNSET_VAR)", ctx).text.empty());

    const auto eval = substitute("$(eval 1 + 2)", ctx);
    REQUIRE(eval.text == "3");
    REQUIRE(eval.whole.has_value());
    REQUIRE(eval.whole->isInt());
    REQUIRE_THROWS_AS(substitute("x$(eval 1 + 2)", ctx), XacroError);

    const std::string expected = (dir.path / "subst_pkg").string() + "/meshes/base.dae";
    REQUIRE(substitute("$(find ${pkg})/meshes/base.dae", ctx).text == expected);
    REQUIRE_THROWS_AS(substitute("$(find not_a_package_at_all)", ctx), XacroError);
}

TEST_CASE("PackageResolver honours the explicit registry") {

    const TempDir dir("registry");
    PackageResolver packages;
    packages.addPackagePath("registered_pkg", dir.path);

    const auto found = packages.resolve("registered_pkg", dir.path / "any.xacro");
    REQUIRE(found.has_value());
    REQUIRE(*found == dir.path);
}

TEST_CASE("PackageResolver answers with a directory that stands on its own") {

    // A registered path may be spelled relative to the working directory - on Windows also
    // drive-relative, as "D:pkg". The answer must not stay that way: callers join filenames
    // onto it, and a relative answer would be re-anchored at the including document instead.
    const TempDir dir("relative_registry");
    writeFile(dir.path / "urdf" / "part.xacro", wrap(R"XML(<link name="part"/>)XML"));

    std::error_code ec;
    const auto spelled = std::filesystem::relative(dir.path, std::filesystem::current_path(), ec);
    if (ec || spelled.empty()) SKIP("the temp directory has no relative spelling from here");

    PackageResolver packages;
    packages.addPackagePath("relative_pkg", spelled);

    const auto found = packages.resolve("relative_pkg", dir.path / "any.xacro");
    REQUIRE(found.has_value());
    REQUIRE(found->is_absolute());
    REQUIRE(std::filesystem::equivalent(*found, dir.path));

    const TempDir elsewhere("relative_registry_doc");
    const auto document = elsewhere.path / "robot.xacro";
    writeFile(document, wrap(R"XML(<xacro:include filename="package://relative_pkg/urdf/part.xacro"/>)XML"));

    Processor processor;
    processor.addPackagePath("relative_pkg", spelled);

    const auto xml = requireOk(processor.processFile(document));
    REQUIRE(contains(xml, R"XML(<link name="part")XML"));
}

TEST_CASE("PackageResolver walks up to a matching manifest") {

    const TempDir dir("walkup");
    const auto root = dir.path / "some_folder_name";
    writeManifest(root, "walkup_pkg");
    const auto document = root / "urdf" / "robot.xacro";
    writeFile(document, "<robot/>");

    PackageResolver packages;
    const auto found = packages.resolve("walkup_pkg", document);
    REQUIRE(found.has_value());
    REQUIRE(std::filesystem::equivalent(*found, root));

    PackageResolver other;
    REQUIRE_FALSE(other.resolve("some_folder_name", document).has_value());
}

TEST_CASE("PackageResolver finds colcon siblings") {

    const TempDir dir("siblings");
    writeManifest(dir.path / "src" / "sibling_pkg", "sibling_pkg");
    const auto document = dir.path / "src" / "other_pkg" / "urdf" / "robot.xacro";
    writeFile(document, "<robot/>");

    PackageResolver packages;
    const auto found = packages.resolve("sibling_pkg", document);
    REQUIRE(found.has_value());
    REQUIRE(std::filesystem::equivalent(*found, dir.path / "src" / "sibling_pkg"));
}

TEST_CASE("PackageResolver reads the environment") {

    const TempDir install("env_install");
    const TempDir workspace("env_workspace");

    writeManifest(install.path / "env_pkg", "env_pkg");
    writeManifest(install.path / "share" / "ament_pkg", "ament_pkg");

    const auto document = workspace.path / "robot.xacro";
    writeFile(document, "<robot/>");

    setEnv("ROS_PACKAGE_PATH", install.path.string().c_str());
    setEnv("AMENT_PREFIX_PATH", install.path.string().c_str());

    PackageResolver packages;
    const auto fromRos = packages.resolve("env_pkg", document);
    REQUIRE(fromRos.has_value());
    REQUIRE(std::filesystem::equivalent(*fromRos, install.path / "env_pkg"));

    const auto fromAment = packages.resolve("ament_pkg", document);
    REQUIRE(fromAment.has_value());
    REQUIRE(std::filesystem::equivalent(*fromAment, install.path / "share" / "ament_pkg"));

    setEnv("ROS_PACKAGE_PATH", nullptr);
    setEnv("AMENT_PREFIX_PATH", nullptr);
}

TEST_CASE("Expand evaluates properties in document order and keeps their type") {

    const auto xml = requireOk(process(
            R"XML(<xacro:property name="width" value="0.5"/>
                  <xacro:property name="count" value="3"/>
                  <xacro:property name="base" value="base"/>
                  <xacro:property name="empty"/>
                  <link name="${base}_link" w="${width * 2}" n="${count + 1}" raw="${width}" e="[${empty}]"/>
                  <xacro:property name="width" value="${width * 2}"/>
                  <link name="second" w="${width}"/>)XML"));

    REQUIRE(contains(xml, R"XML(<link name="base_link" w="1.0" n="4" raw="0.5" e="[]")XML"));
    REQUIRE(contains(xml, R"XML(<link name="second" w="1.0")XML"));
}

TEST_CASE("Expand honours property scopes") {

    SECTION("a macro's own properties do not escape it") {

        const auto xml = requireOk(process(
                R"XML(<xacro:property name="l" value="outer"/>
                      <xacro:macro name="shadow">
                        <xacro:property name="l" value="inner"/>
                        <a l="${l}"/>
                      </xacro:macro>
                      <xacro:shadow/>
                      <b l="${l}"/>)XML"));

        REQUIRE(contains(xml, R"XML(<a l="inner")XML"));
        REQUIRE(contains(xml, R"XML(<b l="outer")XML"));
    }

    SECTION("parent and global write through") {

        const auto xml = requireOk(process(
                R"XML(<xacro:macro name="inner">
                        <xacro:property name="p" value="from_inner" scope="parent"/>
                        <xacro:property name="g" value="from_inner" scope="global"/>
                      </xacro:macro>
                      <xacro:macro name="outer">
                        <xacro:inner/>
                        <a p="${p}"/>
                      </xacro:macro>
                      <xacro:outer/>
                      <b g="${g}"/>)XML"));

        REQUIRE(contains(xml, R"XML(<a p="from_inner")XML"));
        REQUIRE(contains(xml, R"XML(<b g="from_inner")XML"));
    }

    SECTION("a parent write does not leak past the caller") {

        const auto result = process(
                R"XML(<xacro:macro name="inner">
                        <xacro:property name="p" value="from_inner" scope="parent"/>
                      </xacro:macro>
                      <xacro:macro name="outer"><xacro:inner/></xacro:macro>
                      <xacro:outer/>
                      <b p="${p}"/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "undefined name 'p'"));
    }

    SECTION("an unknown scope is an error") {

        const auto result = process(R"XML(<xacro:property name="a" value="1" scope="somewhere"/>)XML");
        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "unknown scope"));
    }

    SECTION("a block body is rejected rather than half supported") {

        const auto result = process(R"XML(<xacro:property name="a"><child/></xacro:property>)XML");
        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "block body"));
    }
}

TEST_CASE("Expand resolves arguments, defaults and overrides") {

    const std::string body =
            R"XML(<xacro:arg name="prefix" default="left_"/>
                  <xacro:arg name="count" default="2"/>
                  <link name="$(arg prefix)base" n="$(arg count)"/>)XML";

    SECTION("declared defaults apply") {

        const auto xml = requireOk(process(body));
        REQUIRE(contains(xml, R"XML(<link name="left_base" n="2")XML"));
    }

    SECTION("a supplied argument wins over the default") {

        const auto xml = requireOk(process(body, {{"prefix", "right_"}}));
        REQUIRE(contains(xml, R"XML(<link name="right_base" n="2")XML"));
    }

    SECTION("an undeclared argument is an error, not an empty string") {

        const auto result = process(R"XML(<link name="$(arg nope)"/>)XML");
        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "undefined arg 'nope'"));
    }

    SECTION("$(arg) yields a string even for numeric text") {

        const auto xml = requireOk(process(
                R"XML(<xacro:arg name="n" default="2"/>
                      <link n="$(arg n)0"/>)XML"));
        REQUIRE(contains(xml, R"XML(<link n="20")XML"));
    }
}

TEST_CASE("Expand instantiates macros") {

    SECTION("parameters split on any whitespace run and defaults fill in") {

        const auto xml = requireOk(process(
                "<xacro:macro name=\"wheel\" params=\"name\n"
                "        radius:=0.1\n"
                "        parent\">\n"
                "  <link name=\"${name}\" r=\"${radius}\" p=\"${parent}\"/>\n"
                "</xacro:macro>\n"
                "<xacro:wheel name=\"w1\" parent=\"base\"/>\n"
                "<xacro:wheel name=\"w2\" parent=\"base\" radius=\"0.2\"/>"));

        REQUIRE(contains(xml, R"XML(<link name="w1" r="0.1" p="base")XML"));
        REQUIRE(contains(xml, R"XML(<link name="w2" r="0.2" p="base")XML"));
    }

    SECTION("defaults are bound in declaration order") {

        const auto xml = requireOk(process(
                R"XML(<xacro:macro name="pair" params="a b:=${a + 1}">
                        <link a="${a}" b="${b}"/>
                      </xacro:macro>
                      <xacro:pair a="4"/>)XML"));

        REQUIRE(contains(xml, R"XML(<link a="4" b="5")XML"));
    }

    SECTION("^ inherits the enclosing binding") {

        const auto xml = requireOk(process(
                R"XML(<xacro:property name="prefix" value="A_"/>
                      <xacro:macro name="inh" params="prefix:=^ suffix:=^|tip">
                        <link name="${prefix}${suffix}"/>
                      </xacro:macro>
                      <xacro:inh/>
                      <xacro:inh suffix="base"/>)XML"));

        REQUIRE(contains(xml, R"XML(<link name="A_tip")XML"));
        REQUIRE(contains(xml, R"XML(<link name="A_base")XML"));
    }

    SECTION("a bare ^ with nothing to inherit is an error") {

        const auto result = process(
                R"XML(<xacro:macro name="inh" params="missing:=^"><link/></xacro:macro>
                      <xacro:inh/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "nothing to inherit"));
    }

    SECTION("a missing required parameter is an error") {

        const auto result = process(
                R"XML(<xacro:macro name="need" params="a"><link a="${a}"/></xacro:macro>
                      <xacro:need/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "missing parameter 'a'"));
    }

    SECTION("macros nest and see the caller's properties") {

        const auto xml = requireOk(process(
                R"XML(<xacro:macro name="leaf" params="n"><link name="leaf_${n}"/></xacro:macro>
                      <xacro:macro name="branch" params="n">
                        <xacro:leaf n="${n}"/>
                        <xacro:leaf n="${n + 1}"/>
                      </xacro:macro>
                      <xacro:branch n="1"/>)XML"));

        REQUIRE(contains(xml, R"XML(<link name="leaf_1")XML"));
        REQUIRE(contains(xml, R"XML(<link name="leaf_2")XML"));
    }

    SECTION("a default may be written with '=' as well as ':='") {

        // franka_description mixes the two forms inside one params attribute.
        const auto xml = requireOk(process(
                R"XML(<xacro:macro name="link" params="name prefix=front_ rpy:='0 0 0'">
                        <link name="${prefix}${name}" rpy="${rpy}"/>
                      </xacro:macro>
                      <xacro:link name="wheel"/>
                      <xacro:link name="hand" prefix="left_"/>)XML"));

        REQUIRE(contains(xml, R"XML(<link name="front_wheel" rpy="0 0 0")XML"));
        REQUIRE(contains(xml, R"XML(<link name="left_hand")XML"));
    }

    SECTION("unbounded recursion hits the budget instead of the stack") {

        const auto result = process(
                R"XML(<xacro:macro name="deep" params="n"><xacro:deep n="${n + 1}"/></xacro:macro>
                      <xacro:deep n="0"/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "nested more than"));
    }

    SECTION("the same file included twice redefines nothing") {

        // xacro has no include guards, so a shared utils file lands once per branch of the
        // robot. franka's does, and every macro in it was drawing a warning.
        const TempDir dir("reinclude");
        writeFile(dir.path / "utils.xacro",
                  wrap(R"XML(<xacro:macro name="util"><link name="util"/></xacro:macro>)XML"));
        writeFile(dir.path / "robot.xacro",
                  wrap(R"XML(<xacro:include filename="utils.xacro"/>
                             <xacro:include filename="utils.xacro"/>
                             <xacro:util/>)XML"));

        Processor processor;
        const auto result = processor.processFile(dir.path / "robot.xacro");

        REQUIRE(result.ok);
        REQUIRE(result.warnings.empty());
    }

    SECTION("two files that disagree take turns; each definition says so once") {

        const TempDir dir("pingpong");
        writeFile(dir.path / "a.xacro", wrap(R"XML(<xacro:macro name="m"><a/></xacro:macro>)XML"));
        writeFile(dir.path / "b.xacro", wrap(R"XML(<xacro:macro name="m"><b/></xacro:macro>)XML"));
        writeFile(dir.path / "top.xacro",
                  wrap(R"XML(<xacro:include filename="a.xacro"/>
                             <xacro:include filename="b.xacro"/>
                             <xacro:include filename="a.xacro"/>
                             <xacro:include filename="b.xacro"/>
                             <xacro:m/>)XML"));

        Processor processor;
        const auto result = processor.processFile(dir.path / "top.xacro");
        REQUIRE(result.ok);

        std::size_t redefinitions = 0;
        for (const auto& w : result.warnings) {
            if (contains(w, "redefined")) ++redefinitions;
        }
        REQUIRE(redefinitions == 2);
    }

    SECTION("a later definition replaces an earlier one, with a warning") {

        const auto result = process(
                R"XML(<xacro:macro name="dup"><a/></xacro:macro>
                      <xacro:macro name="dup"><b/></xacro:macro>
                      <xacro:dup/>)XML");

        REQUIRE(result.ok);
        REQUIRE(contains(result.xml, "<b"));
        REQUIRE_FALSE(contains(result.xml, "<a"));
        REQUIRE_FALSE(result.warnings.empty());
        REQUIRE(contains(result.warnings.front(), "redefined"));
    }

    SECTION("an unknown xacro element names the macros that do exist") {

        const auto result = process(
                R"XML(<xacro:macro name="known"><a/></xacro:macro>
                      <xacro:nosuch/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "unknown element 'xacro:nosuch'"));
        REQUIRE(contains(result.errors.front(), "known"));
    }
}

TEST_CASE("Expand binds and inserts blocks") {

    const auto xml = requireOk(process(
            R"XML(<xacro:macro name="mount" params="name *origin **extras">
                    <xacro:property name="off" value="0.25"/>
                    <joint name="${name}">
                      <xacro:insert_block name="origin"/>
                      <extra><xacro:insert_block name="extras"/></extra>
                    </joint>
                  </xacro:macro>
                  <xacro:mount name="j1">
                    <origin xyz="0 0 ${off}"/>
                    <bag><p1 v="1"/><p2 v="2"/></bag>
                  </xacro:mount>)XML"));

    REQUIRE(contains(xml, R"XML(<joint name="j1">)XML"));
    REQUIRE(contains(xml, R"XML(<origin xyz="0 0 0.25" />)XML"));
    REQUIRE(contains(xml, "<extra>"));
    REQUIRE(contains(xml, R"XML(<p1 v="1" />)XML"));
    REQUIRE(contains(xml, R"XML(<p2 v="2" />)XML"));
    REQUIRE_FALSE(contains(xml, "<bag"));
}

TEST_CASE("Expand reports block arguments that are not there") {

    SECTION("a missing block argument is an error") {

        const auto result = process(
                R"XML(<xacro:macro name="mount" params="*origin"><joint><xacro:insert_block name="origin"/></joint></xacro:macro>
                      <xacro:mount/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "missing block parameter 'origin'"));
    }

    SECTION("insert_block for an unbound name is an error") {

        const auto result = process(R"XML(<xacro:insert_block name="nope"/>)XML");
        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "no such block"));
    }
}

TEST_CASE("Expand takes conditionals literally, then as expressions") {

    const auto xml = requireOk(process(
            R"XML(<xacro:arg name="flag" default="true"/>
                  <xacro:property name="n" value="3"/>
                  <xacro:if value="$(arg flag)"><a/></xacro:if>
                  <xacro:unless value="$(arg flag)"><b/></xacro:unless>
                  <xacro:if value="FALSE"><c/></xacro:if>
                  <xacro:if value="1"><d/></xacro:if>
                  <xacro:if value="${n > 2}"><e/></xacro:if>
                  <xacro:if value="n == 3"><f/></xacro:if>
                  <xacro:unless value="${n > 5}"><g/></xacro:unless>)XML"));

    REQUIRE(contains(xml, "<a"));
    REQUIRE_FALSE(contains(xml, "<b"));
    REQUIRE_FALSE(contains(xml, "<c"));
    REQUIRE(contains(xml, "<d"));
    REQUIRE(contains(xml, "<e"));
    REQUIRE(contains(xml, "<f"));
    REQUIRE(contains(xml, "<g"));

    SECTION("a value that is not a boolean is an error") {

        const auto result = process(R"XML(<xacro:if value="${2.5}"><a/></xacro:if>)XML");
        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "not a boolean"));
    }
}

TEST_CASE("Expand includes files relative to the document that names them") {

    const TempDir dir("include");

    writeFile(dir.path / "sub" / "leaf.xacro",
              wrap(R"XML(<xacro:property name="leaf_dir" value="$(dirname)"/>
                         <xacro:macro name="leaf"><link name="leaf"/></xacro:macro>)XML"));

    writeFile(dir.path / "sub" / "branch.xacro",
              wrap(R"XML(<xacro:include filename="leaf.xacro"/>
                         <xacro:property name="branch_dir" value="$(dirname)"/>)XML"));

    writeFile(dir.path / "top.xacro",
              wrap(R"XML(<xacro:include filename="sub/branch.xacro"/>
                         <xacro:leaf/>
                         <dirs branch="${branch_dir}" leaf="${leaf_dir}" top="$(dirname)"/>)XML"));

    Processor processor;
    const auto xml = requireOk(processor.processFile(dir.path / "top.xacro"));

    REQUIRE(contains(xml, R"XML(<link name="leaf")XML"));
    REQUIRE(contains(xml, "branch=\"" + (dir.path / "sub").string() + "\""));
    REQUIRE(contains(xml, "leaf=\"" + (dir.path / "sub").string() + "\""));
    REQUIRE(contains(xml, "top=\"" + dir.path.string() + "\""));
}

TEST_CASE("Expand rejects include cycles, missing files and namespaced includes") {

    const TempDir dir("cycle");

    writeFile(dir.path / "a.xacro", wrap(R"XML(<xacro:include filename="b.xacro"/>)XML"));
    writeFile(dir.path / "b.xacro", wrap(R"XML(<xacro:include filename="a.xacro"/>)XML"));

    Processor processor;
    const auto cycle = processor.processFile(dir.path / "a.xacro");
    REQUIRE_FALSE(cycle.ok);
    REQUIRE(contains(cycle.errors.front(), "include cycle"));

    const auto missing = process(R"XML(<xacro:include filename="not_here.xacro"/>)XML", {}, dir.path);
    REQUIRE_FALSE(missing.ok);
    REQUIRE(contains(missing.errors.front(), "no such file"));

    const auto namespaced = process(R"XML(<xacro:include filename="b.xacro" ns="other"/>)XML", {}, dir.path);
    REQUIRE_FALSE(namespaced.ok);
    REQUIRE(contains(namespaced.errors.front(), "'ns' attribute is not supported"));
}

TEST_CASE("Processor round-trips a document and drops the xacro namespace") {

    const TempDir dir("roundtrip");
    const auto file = dir.path / "robot.urdf.xacro";

    writeFile(file,
              wrap(R"XML(<!-- dropped -->
                         <xacro:property name="side" value="0.4"/>
                         <link name="base"><size>${side}</size></link>)XML"));

    Processor processor;
    const auto xml = requireOk(processor.processFile(file));

    REQUIRE(contains(xml, R"XML(<robot name="test">)XML"));
    REQUIRE_FALSE(contains(xml, "xmlns:xacro"));
    REQUIRE_FALSE(contains(xml, "dropped"));
    REQUIRE(contains(xml, "<size>0.4</size>"));
}

TEST_CASE("Processor resolves $(find) through a registered package") {

    const TempDir dir("find");
    writeFile(dir.path / "shapes" / "meshes" / "base.dae", "");

    Processor processor;
    processor.addPackagePath("shapes", dir.path / "shapes");

    const auto xml = requireOk(processor.processString(
            wrap(R"XML(<mesh filename="$(find shapes)/meshes/base.dae"/>)XML"), dir.path));

    REQUIRE(contains(xml, (dir.path / "shapes").string() + "/meshes/base.dae"));
}

TEST_CASE("URDFLoader loads a xacro robot built from a macro, a property and an include") {

    const TempDir dir("urdf");

    writeFile(dir.path / "macros.xacro",
              "<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\">\n"
              R"XML(  <xacro:property name="side" value="0.2"/>
                      <xacro:macro name="box_link" params="name *origin">
                        <link name="${name}">
                          <visual>
                            <xacro:insert_block name="origin"/>
                            <geometry><box size="${side} ${side} ${side}"/></geometry>
                          </visual>
                        </link>
                      </xacro:macro>)XML"
              "\n</robot>\n");

    const auto file = dir.path / "robot.urdf.xacro";
    writeFile(file,
              "<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"$(arg robot_name)\">\n"
              R"XML(  <xacro:arg name="robot_name" default="smoke"/>
                      <xacro:include filename="macros.xacro"/>
                      <xacro:property name="reach" value="0.5"/>
                      <xacro:box_link name="base_link"><origin xyz="0 0 0"/></xacro:box_link>
                      <xacro:box_link name="arm_link"><origin xyz="0 0 ${reach / 2}"/></xacro:box_link>
                      <joint name="base_to_arm" type="revolute">
                        <parent link="base_link"/>
                        <child link="arm_link"/>
                        <origin xyz="0 0 ${reach}"/>
                        <axis xyz="0 0 1"/>
                        <limit lower="${-pi}" upper="${pi}" effort="10" velocity="1"/>
                      </joint>)XML"
              "\n</robot>\n");

    URDFLoader loader;
    const auto robot = loader.load(file);

    REQUIRE(robot);
    CHECK(robot->name == "smoke");
    REQUIRE(robot->getObjectByName("base_link"));
    REQUIRE(robot->getObjectByName("arm_link"));
    REQUIRE(robot->numDOF() == 1);

    const auto joints = robot->getArticulatedJointInfo();
    CHECK(joints.front().name == "base_to_arm");
    REQUIRE(joints.front().range.has_value());
    CHECK(joints.front().range->max == Catch::Approx(3.14159265).epsilon(1e-6));

    URDFLoader named;
    named.setArgs({{"robot_name", "override"}});
    const auto renamed = named.load(file);
    REQUIRE(renamed);
    CHECK(renamed->name == "override");
}

TEST_CASE("URDFLoader keeps exposing its arguments as properties") {

    const TempDir dir("urdfargs");
    const auto file = dir.path / "robot.urdf.xacro";

    writeFile(file,
              "<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"args\">\n"
              R"XML(  <xacro:arg name="prefix" default="a_"/>
                      <xacro:arg name="count" default="1"/>
                      <link name="${prefix}link_${count + 1}"/>)XML"
              "\n</robot>\n");

    URDFLoader loader;
    loader.setArgs({{"prefix", "b_"}});

    const auto robot = loader.load(file);
    REQUIRE(robot);
    REQUIRE(robot->getObjectByName("b_link_2"));
}

TEST_CASE("scanArgs reports what a document asks to be told") {

    SECTION("declarations in order, with and without defaults") {

        const auto args = scanArgsFromString(wrap(
                R"XML(  <xacro:arg name="ur_type" default="ur5x"/>
                        <xacro:arg name="tf_prefix"/>
                        <xacro:arg name="joint_limits" default="$(find ur_description)/config/$(arg ur_type)/joint_limits.yaml"/>)XML"));

        REQUIRE(args.size() == 3);
        CHECK(args[0].name == "ur_type");
        CHECK(args[0].hasDefault);
        CHECK(args[0].defaultValue == "ur5x");
        CHECK(args[1].name == "tf_prefix");
        CHECK_FALSE(args[1].hasDefault);
        CHECK(args[1].defaultValue.empty());
        CHECK(args[2].name == "joint_limits");
        // Still a recipe, not a path: the dialog shows what the file will do if
        // left alone, which is the thing an override would replace.
        CHECK(args[2].defaultValue == "$(find ur_description)/config/$(arg ur_type)/joint_limits.yaml");
    }

    SECTION("a repeated name keeps the first declaration, as expansion does") {

        const auto args = scanArgsFromString(wrap(
                R"XML(  <xacro:arg name="dof" default="6"/>
                        <xacro:arg name="dof" default="7"/>)XML"));

        REQUIRE(args.size() == 1);
        CHECK(args.front().defaultValue == "6");
    }

    SECTION("the document's own prefix, not the conventional one") {

        const auto args = scanArgsFromString(
                "<robot xmlns:x=\"http://ros.org/wiki/xacro\" name=\"test\">\n"
                "  <x:arg name=\"dof\" default=\"2\"/>\n"
                "  <xacro:arg name=\"not_bound\" default=\"ignored\"/>\n"
                "</robot>\n");

        REQUIRE(args.size() == 1);
        CHECK(args.front().name == "dof");
    }

    SECTION("shallow on purpose: a nested declaration is not a document argument") {

        const auto args = scanArgsFromString(wrap(
                R"XML(  <xacro:arg name="top" default="1"/>
                        <xacro:macro name="thing" params="n">
                          <xacro:arg name="inside" default="2"/>
                        </xacro:macro>)XML"));

        REQUIRE(args.size() == 1);
        CHECK(args.front().name == "top");
    }

    SECTION("a plain URDF declares nothing") {

        CHECK(scanArgsFromString("<robot name=\"arm\"><link name=\"base_link\"/></robot>").empty());
    }

    SECTION("malformed and missing files answer empty rather than throw") {

        CHECK(scanArgsFromString("<robot name=\"unclosed\">").empty());
        CHECK(scanArgsFromString("").empty());

        const TempDir dir("scanargs");
        CHECK(scanArgs(dir.path / "nothing_here.xacro").empty());
    }

    SECTION("from a file, which is how the editor asks") {

        const TempDir dir("scanargsfile");
        const auto file = dir.path / "robot.urdf.xacro";
        writeFile(file, wrap(R"XML(<xacro:arg name="dof" default="1"/>)XML"));

        const auto args = scanArgs(file);
        REQUIRE(args.size() == 1);
        CHECK(args.front().name == "dof");
        CHECK(args.front().defaultValue == "1");
    }
}

TEST_CASE("Errors say which file and which line") {

    const TempDir dir("lines");

    SECTION("the line is the element the author wrote, not the document's first") {

        const auto file = dir.path / "robot.urdf.xacro";
        writeFile(file, wrap(R"XML(<link name="fine"/>
<link name="${nope}"/>)XML"));

        Processor processor;
        const auto result = processor.processFile(file);

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), file.string() + ":3: "));
    }

    SECTION("an error under an include belongs to the included file") {

        const auto leaf = dir.path / "leaf.xacro";
        writeFile(leaf, wrap(R"XML(<link name="ok"/>
<link name="${nope}"/>)XML"));

        const auto top = dir.path / "top.xacro";
        writeFile(top, wrap(R"XML(<xacro:include filename="leaf.xacro"/>)XML"));

        Processor processor;
        const auto result = processor.processFile(top);

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), leaf.string() + ":3: "));
    }

    SECTION("a macro is reported where its body is, not where it was called") {

        const auto file = dir.path / "macro.urdf.xacro";
        writeFile(file, wrap(R"XML(<xacro:macro name="arm">
  <link name="${nope}"/>
</xacro:macro>
<xacro:arm/>)XML"));

        Processor processor;
        const auto result = processor.processFile(file);

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), file.string() + ":3: "));
    }

    SECTION("a document given as a string is counted the same way") {

        const auto result = process(R"XML(<link name="ok"/>
<link name="${nope}"/>)XML");

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "(string):3: "));
    }

    SECTION("a warning is placed the same way an error is") {

        const auto result = process(R"XML(<xacro:macro name="dup"><a/></xacro:macro>
<xacro:macro name="dup"><b/></xacro:macro>
<xacro:dup/>)XML");

        REQUIRE(result.ok);
        REQUIRE_FALSE(result.warnings.empty());
        REQUIRE(contains(result.warnings.front(), "(string):3: "));
    }

    SECTION("an error with no element behind it still names the file") {

        const auto result = process(R"XML(<xacro:include filename="not_here.xacro"/>)XML", {}, dir.path);

        REQUIRE_FALSE(result.ok);
        REQUIRE(contains(result.errors.front(), "(string):2: "));
        REQUIRE(contains(result.errors.front(), "no such file"));
    }
}

TEST_CASE("URDFLoader says why a load failed") {

    const TempDir dir("lasterror");

    SECTION("a xacro that cannot be expanded names its cause") {

        const auto file = dir.path / "robot.urdf.xacro";
        writeFile(file, wrap(R"XML(<link name="${no_such_property}"/>)XML"));

        URDFLoader loader;
        CHECK_FALSE(loader.load(file));

        const auto error = loader.lastError();
        CHECK_FALSE(error.empty());
        CHECK(contains(error, "no_such_property"));
        CHECK_FALSE(loader.diagnostics().empty());
    }

    SECTION("a file that is not there, and one that is not XML") {

        URDFLoader loader;
        CHECK_FALSE(loader.load(dir.path / "absent.urdf"));
        CHECK(contains(loader.lastError(), "absent.urdf"));

        const auto junk = dir.path / "junk.urdf";
        writeFile(junk, "this is not a document");
        CHECK_FALSE(loader.load(junk));
        CHECK_FALSE(loader.lastError().empty());
    }

    SECTION("XML without a <robot> root is a failure with a reason too") {

        const auto file = dir.path / "notarobot.urdf";
        writeFile(file, "<model name=\"arm\"/>");

        URDFLoader loader;
        CHECK_FALSE(loader.load(file));
        CHECK(contains(loader.lastError(), "<robot>"));
    }

    SECTION("a successful load clears what the failed one left behind") {

        const auto broken = dir.path / "broken.urdf.xacro";
        writeFile(broken, wrap(R"XML(<link name="${nope}"/>)XML"));

        const auto good = dir.path / "good.urdf";
        writeFile(good, "<robot name=\"arm\"><link name=\"base_link\"/></robot>");

        URDFLoader loader;
        CHECK_FALSE(loader.load(broken));
        REQUIRE_FALSE(loader.diagnostics().empty());

        REQUIRE(loader.load(good));
        CHECK(loader.lastError().empty());
        CHECK(loader.diagnostics().empty());
    }
}
