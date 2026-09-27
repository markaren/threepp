// GLTFParser: images, textures and materials. See GLTFParser.hpp.

#include "GLTFParser.hpp"

namespace threepp::gltf {

    // `embedded`, when given, receives the encoded bytes for an image
    // the .glb carries INSIDE itself (a bufferView, or a data: uri) —
    // the ones with no file anywhere for an exporter to point at. An
    // image referenced by path keeps having a file, so it gets nothing
    // and the caller stays on the path it always had.
    std::optional<Image> GLTFParser::loadImageData(int imageIdx, std::vector<uint8_t>* embedded) {
        const auto& imgDef = gltf["images"][imageIdx];
        std::vector<uint8_t> raw;
        bool isEmbedded = true;

        if (imgDef.contains("bufferView")) {
            int bvIdx = imgDef["bufferView"].get<int>();
            const auto& bv = gltf["bufferViews"][bvIdx];
            int bufIdx = bv["buffer"].get<int>();
            size_t off = bv.value("byteOffset", 0);
            size_t len = bv["byteLength"].get<size_t>();
            const auto& buf = resolveBuffer(bufIdx);
            raw.assign(buf.data() + off, buf.data() + off + len);
        } else if (imgDef.contains("uri")) {
            std::string uri = imgDef["uri"].get<std::string>();
            if (uri.rfind("data:", 0) == 0) {
                auto comma = uri.find(',');
                raw = base64Decode(uri.substr(comma + 1));
            } else {
                fs::path p = basePath / percentDecode(uri);
                std::ifstream f(p, std::ios::binary);
                if (!f) throw std::runtime_error("Cannot open image: " + p.string());
                raw = readAllBytes(f, p);
                isEmbedded = false;
            }
        } else {
            throw std::runtime_error("Image " + std::to_string(imageIdx) + " has no source");
        }

        ImageLoader loader;
        // flipY false: glTF's UV origin is the top-left corner, so the
        // rows are used in the order the file stores them. Whatever
        // reads the retained bytes back has to agree.
        auto image = loader.load(raw, 4, false);
        if (image && embedded && isEmbedded) *embedded = std::move(raw);

        return image;
    }

    std::shared_ptr<Texture> GLTFParser::loadTexture(int texIdx, ColorSpace cs) {
        const std::pair<int, int> key{texIdx, static_cast<int>(cs)};
        if (auto it = textureCache.find(key); it != textureCache.end())
            return it->second;

        const auto& texDef = gltf["textures"][texIdx];
        int imageIdx = texDef.value("source", -1);
        // EXT_texture_webp keeps the WebP image's index INSIDE the
        // extension. The top-level `source` beside it is the PNG/JPEG
        // fallback for readers that cannot decode WebP, and an asset that
        // lists the extension as REQUIRED leaves it out altogether
        // (Khronos sample SheenWoodLeatherSofa: all 13 textures). Read
        // only from the top level, such a texture had no image at all, and
        // the model came up untextured on every backend without a word.
        // ImageLoader has decoded WebP since the splat work (it sniffs the
        // RIFF/WEBP magic, so the bytes need no other help from here), so
        // the extension's image is the one to prefer, as the spec asks of
        // a reader that supports it.
        if (const auto ext = texDef.find("extensions"); ext != texDef.end()) {
            if (const auto webp = ext->find("EXT_texture_webp");
                webp != ext->end() && webp->contains("source")) {
                imageIdx = (*webp)["source"].get<int>();
            }
        }
        if (imageIdx < 0) return nullptr;

        std::vector<uint8_t> encoded;
        auto image = loadImageData(imageIdx, &encoded);
        if (!image) return nullptr;

        // Move the decoded pixels straight into the texture — no copy and
        // no lingering decode buffer. The (texIdx, colorSpace) texture
        // cache above means each variant is built exactly once, so the
        // rare second colour-space role of one texture re-decodes rather
        // than retaining every decoded image for the whole load (which
        // would inflate peak memory on texture-heavy scenes).
        auto tex = Texture::create(std::vector<Image>{std::move(*image)});
        tex->colorSpace = cs;
        // The bytes this came in as, for a texture the .glb keeps inside
        // itself and that therefore has no sourceFile. Carries the same
        // flipY the decode above used (see Texture::encodedSource).
        tex->encodedSource = Texture::EncodedImage::from(std::move(encoded), false);
        tex->needsUpdate();

        // glTF 2.0 §3.8.4: when sampler is undefined, repeat wrapping
        // and auto filtering must be used. Texture's C++ default is
        // ClampToEdge, so always default to Repeat for glTF assets.
        tex->wrapS = TextureWrapping::Repeat;
        tex->wrapT = TextureWrapping::Repeat;

        // Apply sampler settings if present
        if (texDef.contains("sampler") && gltf.contains("samplers")) {
            const auto& samp = gltf["samplers"][texDef["sampler"].get<int>()];
            int wrapS = samp.value("wrapS", WRAP_REPEAT);
            int wrapT = samp.value("wrapT", WRAP_REPEAT);
            auto toWrap = [](int w) -> TextureWrapping {
                if (w == WRAP_CLAMP_TO_EDGE) return TextureWrapping::ClampToEdge;
                if (w == WRAP_MIRRORED_REPEAT) return TextureWrapping::MirroredRepeat;
                return TextureWrapping::Repeat;
            };
            tex->wrapS = toWrap(wrapS);
            tex->wrapT = toWrap(wrapT);

            // Filters: only override threepp's defaults (magFilter=Linear,
            // minFilter=LinearMipmapLinear, generateMipmaps=true) when the
            // sampler explicitly specifies one — an unspecified filter is
            // glTF "auto", which those defaults already match.
            if (samp.contains("magFilter")) {
                tex->magFilter = samp["magFilter"].get<int>() == FILTER_NEAREST
                                         ? Filter::Nearest
                                         : Filter::Linear;
            }
            if (samp.contains("minFilter")) {
                switch (samp["minFilter"].get<int>()) {
                    case FILTER_NEAREST:                tex->minFilter = Filter::Nearest; break;
                    case FILTER_LINEAR:                 tex->minFilter = Filter::Linear; break;
                    case FILTER_NEAREST_MIPMAP_NEAREST: tex->minFilter = Filter::NearestMipmapNearest; break;
                    case FILTER_LINEAR_MIPMAP_NEAREST:  tex->minFilter = Filter::LinearMipmapNearest; break;
                    case FILTER_NEAREST_MIPMAP_LINEAR:  tex->minFilter = Filter::NearestMipmapLinear; break;
                    case FILTER_LINEAR_MIPMAP_LINEAR:   tex->minFilter = Filter::LinearMipmapLinear; break;
                    default: break;
                }
                // A non-mipmap min filter samples only the base level, so
                // mipmap generation is pointless (and the GL renderer keys
                // its glGenerateMipmap on this flag).
                const int mf = samp["minFilter"].get<int>();
                tex->generateMipmaps =
                        (mf != FILTER_NEAREST && mf != FILTER_LINEAR);
            }
        }

        textureCache[key] = tex;
        return tex;
    }

    // Load a texture and apply KHR_texture_transform (+ non-zero
    // texCoord) from a textureInfo JSON node. When no transform is
    // present and texCoord == 0 the shared base texture is returned
    // unchanged (no clone). Otherwise a transformed clone is produced
    // and cached by (texIdx, colorSpace, offset/scale/rotation/texCoord),
    // so identical transforms clone at most once instead of per slot.
    std::shared_ptr<Texture> GLTFParser::applyTextureTransform(
            const json& texInfo, int texIdx, ColorSpace cs) {
        auto tex = loadTexture(texIdx, cs);
        if (!tex) return tex;
        int texCoordVal = texInfo.value("texCoord", 0);
        bool hasTransform = false;
        float offX = 0, offY = 0, scX = 1, scY = 1, rot = 0;
        if (texInfo.contains("extensions") &&
            texInfo["extensions"].contains("KHR_texture_transform")) {
            hasTransform = true;
            const auto& tt = texInfo["extensions"]["KHR_texture_transform"];
            if (tt.contains("offset")) {
                offX = tt["offset"][0].get<float>();
                offY = tt["offset"][1].get<float>();
            }
            if (tt.contains("scale")) {
                scX = tt["scale"][0].get<float>();
                scY = tt["scale"][1].get<float>();
            }
            rot = tt.value("rotation", 0.0f);
            texCoordVal = tt.value("texCoord", texCoordVal);
        }
        if (!hasTransform && texCoordVal == 0) return tex;

        const std::tuple<int, int, float, float, float, float, float, int> key{
                texIdx, static_cast<int>(cs), offX, offY, scX, scY, rot, texCoordVal};
        if (auto it = textureTransformCache.find(key); it != textureTransformCache.end())
            return it->second;

        // Clone to avoid sharing transforms between channels
        auto clone = tex->clone();
        clone->offset = {offX, offY};
        clone->repeat = {scX, scY};
        clone->rotation = rot;
        clone->center = {0, 0};
        clone->channel = texCoordVal;
        clone->updateMatrix();
        textureTransformCache[key] = clone;
        return clone;
    }

    std::shared_ptr<Material> GLTFParser::loadMaterial(int matIdx) {
        auto it = materialCache.find(matIdx);
        if (it != materialCache.end()) return it->second;

        const auto& matDef = gltf["materials"][matIdx];

        // KHR_materials_unlit → MeshBasicMaterial
        if (matDef.contains("extensions") &&
            matDef["extensions"].contains("KHR_materials_unlit")) {
            auto basicMat = MeshBasicMaterial::create();
            basicMat->name = matDef.value("name", "");
            if (matDef.contains("pbrMetallicRoughness")) {
                const auto& pbr = matDef["pbrMetallicRoughness"];
                if (pbr.contains("baseColorFactor")) {
                    auto f = pbr["baseColorFactor"].get<std::vector<float>>();
                    basicMat->color = Color(f[0], f[1], f[2]);
                    if (f.size() > 3) basicMat->opacity = f[3];
                }
                if (pbr.contains("baseColorTexture")) {
                    int ti = pbr["baseColorTexture"]["index"].get<int>();
                    basicMat->map = applyTextureTransform(pbr["baseColorTexture"], ti);
                }
            }
            std::string alphaMode = matDef.value("alphaMode", "OPAQUE");
            if (alphaMode == "BLEND") {
                basicMat->transparent = true;
            } else if (alphaMode == "MASK") {
                basicMat->alphaTest = matDef.value("alphaCutoff", 0.5f);
            }
            if (matDef.value("doubleSided", false)) {
                basicMat->side = Side::Double;
            }
            materialCache[matIdx] = basicMat;
            return basicMat;
        }

        // Check if we need MeshPhysicalMaterial (for transmission, clearcoat, etc.)
        bool needsPhysical = false;
        if (matDef.contains("extensions")) {
            const auto& ext = matDef["extensions"];
            if (ext.contains("KHR_materials_transmission") ||
                ext.contains("KHR_materials_clearcoat") ||
                ext.contains("KHR_materials_ior") ||
                ext.contains("KHR_materials_dispersion") ||
                ext.contains("KHR_materials_specular") ||
                ext.contains("KHR_materials_sheen") ||
                ext.contains("KHR_materials_volume") ||
                ext.contains("KHR_materials_iridescence")) {
                needsPhysical = true;
            }
        }

        std::shared_ptr<MeshStandardMaterial> mat;
        std::shared_ptr<MeshPhysicalMaterial> physMat;
        if (needsPhysical) {
            physMat = MeshPhysicalMaterial::create();
            mat = physMat;
        } else {
            mat = MeshStandardMaterial::create();
        }
        mat->name = matDef.value("name", "");

        // PBR Metallic-Roughness
        if (matDef.contains("pbrMetallicRoughness")) {
            const auto& pbr = matDef["pbrMetallicRoughness"];

            // Base color factor
            if (pbr.contains("baseColorFactor")) {
                auto f = pbr["baseColorFactor"].get<std::vector<float>>();
                mat->color = Color(f[0], f[1], f[2]);
                if (f.size() > 3) mat->opacity = f[3];
            }

            // Base color texture
            if (pbr.contains("baseColorTexture")) {
                int ti = pbr["baseColorTexture"]["index"].get<int>();
                mat->map = applyTextureTransform(pbr["baseColorTexture"], ti);
            }

            // Metalness / roughness
            mat->metalness = pbr.value("metallicFactor", 1.0f);
            mat->roughness = pbr.value("roughnessFactor", 1.0f);

            // Metallic-roughness texture (G=roughness, B=metalness per spec)
            if (pbr.contains("metallicRoughnessTexture")) {
                int ti = pbr["metallicRoughnessTexture"]["index"].get<int>();
                auto tex = applyTextureTransform(pbr["metallicRoughnessTexture"], ti, ColorSpace::Linear);
                mat->metalnessMap = tex;
                mat->roughnessMap = tex;
            }
        }

        // Normal map
        if (matDef.contains("normalTexture")) {
            int ti = matDef["normalTexture"]["index"].get<int>();
            mat->normalMap = applyTextureTransform(matDef["normalTexture"], ti, ColorSpace::Linear);
            float scale = matDef["normalTexture"].value("scale", 1.0f);
            mat->normalScale = Vector2{scale, scale};
        }

        // Occlusion map
        if (matDef.contains("occlusionTexture")) {
            int ti = matDef["occlusionTexture"]["index"].get<int>();
            mat->aoMap = applyTextureTransform(matDef["occlusionTexture"], ti, ColorSpace::Linear);
            mat->aoMapIntensity = matDef["occlusionTexture"].value("strength", 1.0f);
        }

        // Emissive
        if (matDef.contains("emissiveFactor")) {
            auto e = matDef["emissiveFactor"].get<std::vector<float>>();
            mat->emissive = Color(e[0], e[1], e[2]);
        }
        if (matDef.contains("emissiveTexture")) {
            int ti = matDef["emissiveTexture"]["index"].get<int>();
            mat->emissiveMap = applyTextureTransform(matDef["emissiveTexture"], ti);
        }

        // Alpha mode
        std::string alphaMode = matDef.value("alphaMode", "OPAQUE");
        if (alphaMode == "BLEND") {
            mat->transparent = true;
        } else if (alphaMode == "MASK") {
            mat->alphaTest = matDef.value("alphaCutoff", 0.5f);
        }

        // Double-sided
        if (matDef.value("doubleSided", false)) {
            mat->side = Side::Double;
        }

        // Extensions (MeshPhysicalMaterial properties)
        if (physMat && matDef.contains("extensions")) {
            const auto& ext = matDef["extensions"];

            // KHR_materials_transmission
            if (ext.contains("KHR_materials_transmission")) {
                const auto& tr = ext["KHR_materials_transmission"];
                physMat->transmission = tr.value("transmissionFactor", 0.0f);
                if (tr.contains("transmissionTexture")) {
                    int ti = tr["transmissionTexture"]["index"].get<int>();
                    physMat->transmissionMap = loadTexture(ti, ColorSpace::Linear);
                }
                // glTF: transmission WITHOUT KHR_materials_volume = a THIN-WALLED
                // surface (infinitely thin, e.g. a watch crystal / car window).
                // The volume block below resets this to false when volume is
                // present (a closed solid). Without this the renderer treated
                // every transmissive surface as solid → 2-interface refraction
                // that warps/destroys whatever is just behind a thin pane.
                physMat->thinWalled = true;
            }

            // KHR_materials_ior
            if (ext.contains("KHR_materials_ior")) {
                const auto& iorExt = ext["KHR_materials_ior"];
                physMat->setIor(iorExt.value("ior", 1.5f));
            }

            // KHR_materials_dispersion
            if (ext.contains("KHR_materials_dispersion")) {
                const auto& dispExt = ext["KHR_materials_dispersion"];
                physMat->dispersion = dispExt.value("dispersion", 0.0f);
            }

            // KHR_materials_emissive_strength
            if (ext.contains("KHR_materials_emissive_strength")) {
                float strength = ext["KHR_materials_emissive_strength"].value("emissiveStrength", 1.0f);
                mat->emissiveIntensity = strength;
            }

            // KHR_materials_sheen
            if (ext.contains("KHR_materials_sheen")) {
                const auto& sh = ext["KHR_materials_sheen"];
                if (sh.contains("sheenColorFactor")) {
                    auto c = sh["sheenColorFactor"];
                    physMat->sheenColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                }
                physMat->sheenRoughness = sh.value("sheenRoughnessFactor", 0.0f);
            }

            // KHR_materials_specular
            if (ext.contains("KHR_materials_specular")) {
                const auto& sp = ext["KHR_materials_specular"];
                physMat->specularIntensity = sp.value("specularFactor", 1.0f);
                if (sp.contains("specularColorFactor")) {
                    auto c = sp["specularColorFactor"];
                    physMat->specularColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                }
            }

            // KHR_materials_volume
            if (ext.contains("KHR_materials_volume")) {
                const auto& vol = ext["KHR_materials_volume"];
                physMat->attenuationDistance = vol.value("attenuationDistance", 0.0f);
                if (vol.contains("attenuationColor")) {
                    auto c = vol["attenuationColor"];
                    physMat->attenuationColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                }
                physMat->thickness = vol.value("thicknessFactor", 0.0f);
                // A volume = a closed solid → 2-interface (closed-mesh) refraction,
                // not the thin-shell path the transmission block assumed above.
                physMat->thinWalled = (physMat->thickness <= 0.0f);
            }

            // KHR_materials_clearcoat
            if (ext.contains("KHR_materials_clearcoat")) {
                const auto& cc = ext["KHR_materials_clearcoat"];
                physMat->clearcoat = cc.value("clearcoatFactor", 0.0f);
                if (cc.contains("clearcoatTexture")) {
                    int ti = cc["clearcoatTexture"]["index"].get<int>();
                    physMat->clearcoatMap = loadTexture(ti, ColorSpace::Linear);
                }
                physMat->clearcoatRoughness = cc.value("clearcoatRoughnessFactor", 0.0f);
                if (cc.contains("clearcoatRoughnessTexture")) {
                    int ti = cc["clearcoatRoughnessTexture"]["index"].get<int>();
                    physMat->clearcoatRoughnessMap = loadTexture(ti, ColorSpace::Linear);
                }
                if (cc.contains("clearcoatNormalTexture")) {
                    int ti = cc["clearcoatNormalTexture"]["index"].get<int>();
                    physMat->clearcoatNormalMap = loadTexture(ti, ColorSpace::Linear);
                    float scale = cc["clearcoatNormalTexture"].value("scale", 1.0f);
                    physMat->clearcoatNormalScale = Vector2{scale, scale};
                }
            }

            // KHR_materials_iridescence
            if (ext.contains("KHR_materials_iridescence")) {
                const auto& ir = ext["KHR_materials_iridescence"];
                physMat->iridescence = ir.value("iridescenceFactor", 0.0f);
                physMat->iridescenceIOR = ir.value("iridescenceIor", 1.3f);
                // Spec: thicknessMaximum is used as the constant thickness
                // when no thickness texture is present (which we don't sample yet).
                physMat->iridescenceThicknessNm = ir.value("iridescenceThicknessMaximum", 400.0f);
            }
        }

        materialCache[matIdx] = mat;
        return mat;
    }

}// namespace threepp::gltf
