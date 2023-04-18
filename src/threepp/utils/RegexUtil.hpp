// https://stackoverflow.com/questions/57193450/c-regex-replace-one-by-one

#ifndef THREEPP_REGEX_CALLBACK_HPP
#define THREEPP_REGEX_CALLBACK_HPP

#include <cstdlib>
#include <regex>
#include <string>

namespace {

    template<class BidirIt, class Traits, class CharT, class UnaryFunction>
    std::basic_string<CharT> regex_replace(BidirIt first, BidirIt last,
                                           const std::basic_regex<CharT, Traits>& re, UnaryFunction f) {
        std::basic_string<CharT> s;

        typename std::match_results<BidirIt>::difference_type
                positionOfLastMatch = 0;
        auto endOfLastMatch = first;

        auto callback = [&](const std::match_results<BidirIt>& match) {
            auto positionOfThisMatch = match.position(0);
            auto diff = positionOfThisMatch - positionOfLastMatch;

            auto startOfThisMatch = endOfLastMatch;
            std::advance(startOfThisMatch, diff);

            s.append(endOfLastMatch, startOfThisMatch);
            s.append(f(match));

            auto lengthOfMatch = match.length(0);

            positionOfLastMatch = positionOfThisMatch + lengthOfMatch;

            endOfLastMatch = startOfThisMatch;
            std::advance(endOfLastMatch, lengthOfMatch);
        };

        std::sregex_iterator begin(first, last, re), end;
        std::for_each(begin, end, callback);

        s.append(endOfLastMatch, last);

        return s;
    }

    template<class Traits, class CharT, class UnaryFunction>
    std::string regex_replace(const std::string& s, const std::basic_regex<CharT, Traits>& re, UnaryFunction f) {
        return regex_replace(s.cbegin(), s.cend(), re, f);
    }

}// namespace

#endif//THREEPP_REGEX_CALLBACK_HPP
