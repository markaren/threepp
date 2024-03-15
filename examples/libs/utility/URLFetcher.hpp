#ifndef THREEPP_URLFETCHER_HPP
#define THREEPP_URLFETCHER_HPP

#if __has_include(<curl/curl.h>)
#define THREEPP_WITH_CURL
#include <curl/curl.h>
#endif

#include <string>
#include <vector>

namespace threepp::utils {

#ifdef THREEPP_WITH_CURL

    inline static size_t write_data(char* ptr, size_t size, size_t nmemb, void* userdata) {
        auto stream = (std::vector<unsigned char>*) userdata;
        size_t count = size * nmemb;
        stream->insert(stream->end(), ptr, ptr + count);
        return count;
    }

#endif

    struct UrlFetcher {

#ifndef THREEPP_WITH_CURL
        bool fetch(const std::string&, std::vector<unsigned char>&) {

            return false;
        }
#else
        UrlFetcher(): curl(curl_easy_init()) {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &write_data);
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "threepp/1.0 (Cross-Platform; C++)");
        }

        bool fetch(const std::string& url, std::vector<unsigned char>& data) {

            if (cache_.count(url)) {
                data = cache_[url];
                return true;
            }

            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            CURLcode res = curl_easy_perform(curl);

            if (res == CURLE_OK) {
                cache_[url] = data;
            }

            return res == CURLE_OK;
        }

        ~UrlFetcher() {
            curl_easy_cleanup(curl);
        }

    private:
        CURL* curl;
        std::unordered_map<std::string, std::vector<unsigned char>> cache_{};
#endif
    };

}// namespace threepp::utils

#endif//THREEPP_URLFETCHER_HPP