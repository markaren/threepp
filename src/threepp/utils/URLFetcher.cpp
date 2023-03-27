
#include "threepp/utils/URLFetcher.hpp"

#include <curl/curl.h>

using namespace threepp::utils;

namespace {

    size_t write_data(char* ptr, size_t size, size_t nmemb, void* userdata) {
        auto stream = (std::vector<unsigned char>*) userdata;
        size_t count = size * nmemb;
        stream->insert(stream->end(), ptr, ptr + count);
        return count;
    }

}// namespace

struct UrlFetcher::Impl {

    Impl() {

        curl = curl_easy_init();
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);// pass the writefunction
    }

    bool fetch(const std::string& url, std::vector<unsigned char>& data) {

        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);// pass the stream ptr to the writefunction
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());//the img url
        CURLcode res = curl_easy_perform(curl);

        return res == 0;
    }

    ~Impl() {
        curl_easy_cleanup(curl);
    }

private:
    CURL* curl;
};

threepp::utils::UrlFetcher::UrlFetcher()
    : pimpl_(std::make_unique<Impl>()) {}

bool threepp::utils::UrlFetcher::fetch(const std::string& url, std::vector<unsigned char>& data) {

    return pimpl_->fetch(url, data);
}

threepp::utils::UrlFetcher::~UrlFetcher() = default;
