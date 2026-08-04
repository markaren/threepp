// Dynamic value model for xacro properties, arguments and expression results.
// Mirrors the python types xacro exposes: None, bool, int, float, str, list, dict.

#ifndef THREEPP_XACRO_VALUE_HPP
#define THREEPP_XACRO_VALUE_HPP

#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace threepp::xacro {

    class Value;

    using List = std::vector<Value>;
    using Dict = std::map<std::string, Value>;

    class Value {

    public:
        using Variant = std::variant<std::monostate, bool, long long, double, std::string, List, Dict>;

        Value() = default;
        Value(bool v): v_(v) {}
        Value(int v): v_(static_cast<long long>(v)) {}
        Value(long long v): v_(v) {}
        Value(double v): v_(v) {}
        Value(const char* v): v_(std::string(v)) {}
        Value(std::string v): v_(std::move(v)) {}
        Value(List v): v_(std::move(v)) {}
        Value(Dict v): v_(std::move(v)) {}

        [[nodiscard]] bool isNone() const { return v_.index() == 0; }
        [[nodiscard]] bool isBool() const { return v_.index() == 1; }
        [[nodiscard]] bool isInt() const { return v_.index() == 2; }
        [[nodiscard]] bool isDouble() const { return v_.index() == 3; }
        [[nodiscard]] bool isString() const { return v_.index() == 4; }
        [[nodiscard]] bool isList() const { return v_.index() == 5; }
        [[nodiscard]] bool isDict() const { return v_.index() == 6; }

        // bools count as numbers, as in python
        [[nodiscard]] bool isNumber() const { return isBool() || isInt() || isDouble(); }

        [[nodiscard]] const Variant& raw() const { return v_; }

        [[nodiscard]] std::string typeName() const;

        [[nodiscard]] bool truthy() const;

        // python str(): what gets written back into the document
        [[nodiscard]] std::string toString() const;

        // python repr(): as str(), but strings are quoted. Used for list/dict members.
        [[nodiscard]] std::string repr() const;

        [[nodiscard]] double asNumber() const;

        [[nodiscard]] long long asInt() const;

        [[nodiscard]] const std::string& asString() const;

        [[nodiscard]] const List& asList() const;

        [[nodiscard]] const Dict& asDict() const;

        [[nodiscard]] List& asList();

        [[nodiscard]] Dict& asDict();

        // -1/0/1 ordering for numbers, strings and lists. Throws for anything else.
        static int compare(const Value& a, const Value& b);

    private:
        Variant v_;
    };

    bool operator==(const Value& a, const Value& b);

    // Text -> typed value using xacro's rules: a full int parse wins, then a full float
    // parse, otherwise the (untrimmed) text stays a string. "true"/"false" stay strings.
    [[nodiscard]] Value classify(std::string_view text);

    [[nodiscard]] std::string formatDouble(double d);

}// namespace threepp::xacro

#endif
