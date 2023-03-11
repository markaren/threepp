
#ifndef THREEPP_URLFETCHER_HPP
#define THREEPP_URLFETCHER_HPP

#include <memory>
#include <string>
#include <vector>

namespace threepp::utils {

    struct UrlFetcher {

        UrlFetcher();

        bool fetch(const std::string& url, std::vector<unsigned char>& data);

        ~UrlFetcher();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp::utils

#endif//THREEPP_URLFETCHER_HPP
