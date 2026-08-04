
#include "threepp/loaders/xacro/Value.hpp"

#include "threepp/loaders/xacro/Diagnostics.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <system_error>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    std::string_view trimmed(std::string_view s) {

        const auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
        while (!s.empty() && isSpace(s.front())) s.remove_prefix(1);
        while (!s.empty() && isSpace(s.back())) s.remove_suffix(1);
        return s;
    }

    bool parseInt(std::string_view s, long long& out) {

        if (s.empty()) return false;
        if (s.front() == '+') s.remove_prefix(1);
        if (s.empty()) return false;
        const char* begin = s.data();
        const char* end = begin + s.size();
        auto [ptr, ec] = std::from_chars(begin, end, out);
        return ec == std::errc{} && ptr == end;
    }

    bool parseDouble(std::string_view s, double& out) {

        if (s.empty()) return false;
        if (s.front() == '+') s.remove_prefix(1);
        if (s.empty()) return false;
#if defined(__cpp_lib_to_chars)
        const char* begin = s.data();
        const char* end = begin + s.size();
        auto [ptr, ec] = std::from_chars(begin, end, out);
        return ec == std::errc{} && ptr == end;
#else
        // libc++ without floating-point from_chars (see ColladaLoader.cpp): strtod
        // needs a terminator, and the view is a substring, so copy first.
        const std::string copy(s);
        char* next = nullptr;
        out = std::strtod(copy.c_str(), &next);
        return next == copy.c_str() + copy.size();
#endif
    }

    int compareNumbers(double a, double b) {

        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    }

}// namespace

std::string threepp::xacro::formatDouble(double d) {

    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";

    char buf[64];
#if defined(__cpp_lib_to_chars)
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), d);
    std::string s(buf, ptr);
#else
    // Shortest round-trip by hand: widen the precision until strtod gives the
    // value back. %.17g alone would render 0.1 as 0.10000000000000001.
    std::string s;
    for (int precision = 1; precision <= 17; ++precision) {
        std::snprintf(buf, sizeof(buf), "%.*g", precision, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    s = buf;
#endif

    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string Value::typeName() const {

    switch (v_.index()) {
        case 0: return "NoneType";
        case 1: return "bool";
        case 2: return "int";
        case 3: return "float";
        case 4: return "str";
        case 5: return "list";
        default: return "dict";
    }
}

bool Value::truthy() const {

    switch (v_.index()) {
        case 0: return false;
        case 1: return std::get<bool>(v_);
        case 2: return std::get<long long>(v_) != 0;
        case 3: return std::get<double>(v_) != 0.0;
        case 4: return !std::get<std::string>(v_).empty();
        case 5: return !std::get<List>(v_).empty();
        default: return !std::get<Dict>(v_).empty();
    }
}

std::string Value::toString() const {

    switch (v_.index()) {
        case 0:
            return "None";
        case 1:
            return std::get<bool>(v_) ? "True" : "False";
        case 2:
            return std::to_string(std::get<long long>(v_));
        case 3:
            return formatDouble(std::get<double>(v_));
        case 4:
            return std::get<std::string>(v_);
        case 5: {
            std::string s = "[";
            const auto& l = std::get<List>(v_);
            for (std::size_t i = 0; i < l.size(); ++i) {
                if (i) s += ", ";
                s += l[i].repr();
            }
            return s + "]";
        }
        default: {
            std::string s = "{";
            const auto& d = std::get<Dict>(v_);
            bool first = true;
            for (const auto& [k, v] : d) {
                if (!first) s += ", ";
                first = false;
                s += Value(k).repr() + ": " + v.repr();
            }
            return s + "}";
        }
    }
}

std::string Value::repr() const {

    if (!isString()) return toString();

    std::string s = "'";
    for (char c : std::get<std::string>(v_)) {
        if (c == '\\' || c == '\'') s += '\\';
        s += c;
    }
    return s + "'";
}

double Value::asNumber() const {

    switch (v_.index()) {
        case 1: return std::get<bool>(v_) ? 1.0 : 0.0;
        case 2: return static_cast<double>(std::get<long long>(v_));
        case 3: return std::get<double>(v_);
        default: throw XacroError("expected a number, got " + typeName() + " (" + toString() + ")");
    }
}

long long Value::asInt() const {

    switch (v_.index()) {
        case 1:
            return std::get<bool>(v_) ? 1 : 0;
        case 2:
            return std::get<long long>(v_);
        case 3: {
            const double d = std::get<double>(v_);
            if (std::floor(d) != d || std::isinf(d) || std::isnan(d)) {
                throw XacroError("expected an integer, got float " + toString());
            }
            return static_cast<long long>(d);
        }
        default:
            throw XacroError("expected an integer, got " + typeName() + " (" + toString() + ")");
    }
}

const std::string& Value::asString() const {

    if (!isString()) throw XacroError("expected a string, got " + typeName() + " (" + toString() + ")");
    return std::get<std::string>(v_);
}

const List& Value::asList() const {

    if (!isList()) throw XacroError("expected a list, got " + typeName() + " (" + toString() + ")");
    return std::get<List>(v_);
}

const Dict& Value::asDict() const {

    if (!isDict()) throw XacroError("expected a dict, got " + typeName() + " (" + toString() + ")");
    return std::get<Dict>(v_);
}

List& Value::asList() {

    if (!isList()) throw XacroError("expected a list, got " + typeName() + " (" + toString() + ")");
    return std::get<List>(v_);
}

Dict& Value::asDict() {

    if (!isDict()) throw XacroError("expected a dict, got " + typeName() + " (" + toString() + ")");
    return std::get<Dict>(v_);
}

bool threepp::xacro::operator==(const Value& a, const Value& b) {

    // python treats bool as an int, so True == 1 and 1.0 == 1
    if (a.isNumber() && b.isNumber()) return a.asNumber() == b.asNumber();

    if (a.raw().index() != b.raw().index()) return false;

    switch (a.raw().index()) {
        case 0:
            return true;
        case 4:
            return a.asString() == b.asString();
        case 5: {
            const auto& l = a.asList();
            const auto& r = b.asList();
            if (l.size() != r.size()) return false;
            for (std::size_t i = 0; i < l.size(); ++i) {
                if (!(l[i] == r[i])) return false;
            }
            return true;
        }
        default: {
            const auto& l = a.asDict();
            const auto& r = b.asDict();
            if (l.size() != r.size()) return false;
            auto it = l.begin();
            auto jt = r.begin();
            for (; it != l.end(); ++it, ++jt) {
                if (it->first != jt->first || !(it->second == jt->second)) return false;
            }
            return true;
        }
    }
}

int Value::compare(const Value& a, const Value& b) {

    if (a.isNumber() && b.isNumber()) return compareNumbers(a.asNumber(), b.asNumber());

    if (a.isString() && b.isString()) {
        const auto& l = a.asString();
        const auto& r = b.asString();
        return l < r ? -1 : (l == r ? 0 : 1);
    }

    if (a.isList() && b.isList()) {
        const auto& l = a.asList();
        const auto& r = b.asList();
        for (std::size_t i = 0; i < l.size() && i < r.size(); ++i) {
            const int c = compare(l[i], r[i]);
            if (c != 0) return c;
        }
        if (l.size() == r.size()) return 0;
        return l.size() < r.size() ? -1 : 1;
    }

    throw XacroError("cannot order " + a.typeName() + " and " + b.typeName());
}

Value threepp::xacro::classify(std::string_view text) {

    const auto t = trimmed(text);
    if (t.empty()) return Value(std::string(text));

    long long i{};
    if (parseInt(t, i)) return Value(i);

    double d{};
    if (parseDouble(t, d)) return Value(d);

    return Value(std::string(text));
}

std::optional<bool> threepp::xacro::booleanLiteral(std::string_view text) {

    // The spellings get_boolean_value() answers to, and only those: it compares against
    // 'true'/'True'/'false'/'False' outright, so "TRUE" raises there and is not a boolean
    // here either. Surrounding space is threepp's own leniency - upstream leaves "  false  "
    // a string - and it is the same leniency classify() already extends to a number and
    // xacro:if to its condition.
    const auto t = trimmed(text);
    if (t == "true" || t == "True") return true;
    if (t == "false" || t == "False") return false;
    return std::nullopt;
}
