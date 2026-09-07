// A structural read of an expanded URDF: the links it declares, and the links its joints
// name. The xacro tests share it because "the output contains this string" is blind to the
// failure that actually matters. A description that expands one subtree under a prefix and
// another without it still emits every element, every mesh and every inertia the text
// assertions look for - the only trace is a <parent link> naming a link that is not there,
// and the subtree hanging off it is detached from the robot.

#ifndef THREEPP_TESTS_URDF_STRUCTURE_HPP
#define THREEPP_TESTS_URDF_STRUCTURE_HPP

#include <set>
#include <string>
#include <vector>

namespace urdf_structure {

    inline bool isSpace(char c) {

        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }

    // Every `attribute="..."` carried by an <element ...> tag, in document order. A plain
    // scan rather than a parse: the tests only ever hand this what threepp's own writer
    // produced, and a second XML dependency in a test is not worth the coverage.
    inline std::vector<std::string> attributesOf(const std::string& xml, const std::string& element,
                                                 const std::string& attribute) {

        std::vector<std::string> out;

        const std::string open = "<" + element;
        const std::string key = attribute + "=\"";

        for (auto at = xml.find(open); at != std::string::npos; at = xml.find(open, at + open.size())) {

            // <link>, <link ...> and <link/>, but not <linkage>
            const auto after = at + open.size();
            if (after < xml.size() && !isSpace(xml[after]) && xml[after] != '>' && xml[after] != '/') continue;

            const auto tagEnd = xml.find('>', at);
            if (tagEnd == std::string::npos) break;

            // ...="..." has to start on a boundary, or `name=` would match `filename=`
            auto found = xml.find(key, after);
            while (found != std::string::npos && found < tagEnd && !isSpace(xml[found - 1])) {
                found = xml.find(key, found + key.size());
            }
            if (found == std::string::npos || found > tagEnd) continue;

            const auto from = found + key.size();
            const auto to = xml.find('"', from);
            if (to == std::string::npos || to > tagEnd) continue;

            out.emplace_back(xml, from, to - from);
        }

        return out;
    }

    inline std::set<std::string> linkNames(const std::string& xml) {

        const auto declared = attributesOf(xml, "link", "name");
        return {declared.begin(), declared.end()};
    }

    // Every joint endpoint - <parent link="..."/> and <child link="..."/> - that names a
    // link the document never declared. Empty is the only healthy answer; what comes back
    // otherwise is written for an assertion message.
    inline std::vector<std::string> danglingJointEndpoints(const std::string& xml) {

        const auto links = linkNames(xml);

        std::vector<std::string> dangling;
        for (const std::string end : {"parent", "child"}) {
            for (const auto& named : attributesOf(xml, end, "link")) {
                if (!links.count(named)) dangling.push_back("<" + end + " link=\"" + named + "\"/>");
            }
        }

        return dangling;
    }

    inline std::string joined(const std::vector<std::string>& items) {

        std::string s;
        for (const auto& item : items) {
            if (!s.empty()) s += ", ";
            s += item;
        }
        return s;
    }

}// namespace urdf_structure

#endif
