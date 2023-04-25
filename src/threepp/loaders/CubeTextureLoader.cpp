#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/URLFetcher.hpp"

#include <iostream>
#include <regex>
#include <vector>

using namespace threepp;
 
 
std::shared_ptr<CubeTexture> CubeTextureLoader::load(const list<filesystem::path>& filePaths, bool flipY) 
{ 
    TextureLoader loader;
    std::shared_ptr<CubeTexture> cubeTexture = CubeTexture::create();
    for (auto path : filePaths)
    {
        std::shared_ptr<Texture> texture = loader.load(path);
        cubeTexture->images.push_back(texture);
    } 
    cubeTexture->needsUpdate();
    return cubeTexture;
}
  
