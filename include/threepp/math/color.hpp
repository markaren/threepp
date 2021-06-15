
#ifndef THREEPP_COLOR_HPP
#define THREEPP_COLOR_HPP

namespace threepp {

    class color {

    public:
        float r = 1.0;
        float g = 1.0;
        float b = 1.0;

        color() = default;

        color(float r, float g, float b);

        color &setRGB(float r, float g, float b);

        template<class ArrayLike>
        color &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->r = array[offset];
            this->g = array[offset + 1];
            this->b = array[offset + 2];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) {

            array[offset] = this.r;
            array[offset + 1] = this.g;
            array[offset + 2] = this.b;
        }
    };

}// namespace threepp

#endif//THREEPP_COLOR_HPP
