
#ifndef THREEPP_GROUP_HPP
#define THREEPP_GROUP_HPP

namespace threepp {

    class Group {

    public:
        const unsigned int start;
        const unsigned int count;
        const unsigned int materialIndex;

        Group(unsigned int start, unsigned int count, unsigned int materialIndex) : start(start), count(count), materialIndex(materialIndex) {}
    };

}// namespace threepp

#endif//THREEPP_GROUP_HPP
