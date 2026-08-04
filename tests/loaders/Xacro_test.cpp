#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/xacro/Expr.hpp"
#include "threepp/loaders/xacro/PackageResolver.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Substitution.hpp"
#include "threepp/loaders/xacro/Value.hpp"
#include "threepp/loaders/xacro/YamlLite.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

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

TEST_CASE("YamlLite rejects what it does not support") {

    REQUIRE_THROWS_AS(parseYaml("a: &anchor 1\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: *alias\n"), XacroError);
    REQUIRE_THROWS_AS(parseYaml("a: !!str 1\n"), XacroError);
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
