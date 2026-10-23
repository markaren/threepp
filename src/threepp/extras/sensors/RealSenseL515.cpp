// Media Foundation (streams) and WinUSB (calibration) half of the L515
// capture. Windows headers stay in this translation unit; the public header
// pulls in no platform headers.

#include "threepp/extras/sensors/RealSenseL515.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <cfgmgr32.h>
#include <winusb.h>

#include <algorithm>
#include <cwctype>
#endif

using namespace threepp;

RealSenseL515::Intrinsics RealSenseL515::Intrinsics::fromFov(int width, int height, float hfovDeg, float vfovDeg) {
    constexpr float degToRad = 3.14159265358979f / 180.f;
    Intrinsics k;
    k.width = width;
    k.height = height;
    k.fx = 0.5f * static_cast<float>(width) / std::tan(0.5f * hfovDeg * degToRad);
    k.fy = 0.5f * static_cast<float>(height) / std::tan(0.5f * vfovDeg * degToRad);
    k.ppx = 0.5f * static_cast<float>(width);
    k.ppy = 0.5f * static_cast<float>(height);
    return k;
}

namespace {

    // Datasheet fields of view, the fallback when no calibration was read.
    constexpr float depthHfov = 70.f, depthVfov = 55.f;
    constexpr float colorHfov = 70.f, colorVfov = 43.f;

    struct Calibration {
        bool valid = false;
        std::vector<RealSenseL515::Intrinsics> depth;// landscape models, one per resolution
        std::vector<RealSenseL515::Intrinsics> color;
        RealSenseL515::Extrinsics depthToColor;

        static RealSenseL515::Intrinsics find(const std::vector<RealSenseL515::Intrinsics>& list,
                                              int width, int height, float hfov, float vfov) {
            for (const auto& k : list) {
                if (k.width == width && k.height == height) return k;
            }
            return RealSenseL515::Intrinsics::fromFov(width, height, hfov, vfov);
        }
    };

    template<class Frame>
    struct Stream {
        std::thread thread;
        std::atomic<bool> open{false};
        std::mutex mutex;
        Frame latest;
        bool fresh = false;

        bool poll(Frame& out) {
            std::lock_guard lock(mutex);
            if (!fresh) return false;
            out.width = latest.width;
            out.height = latest.height;
            out.intrinsics = latest.intrinsics;
            out.frameNumber = latest.frameNumber;
            out.timestamp = latest.timestamp;
            if constexpr (requires { out.data; }) {
                out.depthUnit = latest.depthUnit;
                out.data.assign(latest.data.begin(), latest.data.end());
            } else {
                out.rgba.assign(latest.rgba.begin(), latest.rgba.end());
            }
            fresh = false;
            return true;
        }
    };

}// namespace

#ifdef _WIN32

namespace {

    template<class T>
    struct ComPtr {
        T* p = nullptr;
        ComPtr() = default;
        ComPtr(const ComPtr&) = delete;
        ComPtr& operator=(const ComPtr&) = delete;
        ~ComPtr() { reset(); }
        void reset() {
            if (p) p->Release();
            p = nullptr;
        }
        T** put() {
            reset();
            return &p;
        }
        T* operator->() const { return p; }
        explicit operator bool() const { return p != nullptr; }
    };

    // ---- Hardware monitor (MI_06, WinUSB bulk) ----
    //
    // Request: uint16 length (bytes after this field and the magic), uint16
    // magic 0xCDAB, uint32 opcode, uint32 param[4]. Reply: int32 status (the
    // opcode on success, negative on error) followed by the payload. Only the
    // three read commands below are ever sent; librealsense issues the same
    // ones (over the depth interface's extension unit) to report calibration.
    constexpr std::uint32_t opRgbIntrinsics = 0x81;
    constexpr std::uint32_t opRgbExtrinsics = 0x82;
    constexpr std::uint32_t opDepthIntrinsics = 0x7F;

    class HwMonitor {
    public:
        // `instanceId` is the PnP id of the unit's MI_06 interface.
        explicit HwMonitor(const std::wstring& instanceId) {
            const auto path = interfacePath(instanceId);
            if (path.empty()) return;
            file_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
            if (file_ == INVALID_HANDLE_VALUE || !WinUsb_Initialize(file_, &usb_)) return;

            USB_INTERFACE_DESCRIPTOR desc{};
            if (!WinUsb_QueryInterfaceSettings(usb_, 0, &desc)) return;
            for (UCHAR i = 0; i < desc.bNumEndpoints; ++i) {
                WINUSB_PIPE_INFORMATION pipe{};
                if (!WinUsb_QueryPipe(usb_, 0, i, &pipe) || pipe.PipeType != UsbdPipeTypeBulk) continue;
                (USB_ENDPOINT_DIRECTION_IN(pipe.PipeId) ? in_ : out_) = pipe.PipeId;
            }
            ULONG timeoutMs = 1000;
            WinUsb_SetPipePolicy(usb_, in_, PIPE_TRANSFER_TIMEOUT, sizeof(timeoutMs), &timeoutMs);
            WinUsb_SetPipePolicy(usb_, out_, PIPE_TRANSFER_TIMEOUT, sizeof(timeoutMs), &timeoutMs);
        }

        ~HwMonitor() {
            if (usb_) WinUsb_Free(usb_);
            if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        }

        HwMonitor(const HwMonitor&) = delete;
        HwMonitor& operator=(const HwMonitor&) = delete;

        [[nodiscard]] bool valid() const { return usb_ && in_ && out_; }

        // The reply payload, or empty on any failure.
        std::vector<std::uint8_t> command(std::uint32_t opcode) {
            std::uint8_t request[24] = {};
            const std::uint16_t length = 20, magic = 0xCDAB;
            std::memcpy(request + 0, &length, 2);
            std::memcpy(request + 2, &magic, 2);
            std::memcpy(request + 4, &opcode, 4);
            ULONG n = 0;
            if (!WinUsb_WritePipe(usb_, out_, request, sizeof(request), &n, nullptr)) return {};

            std::vector<std::uint8_t> reply(1024);
            if (!WinUsb_ReadPipe(usb_, in_, reply.data(), static_cast<ULONG>(reply.size()), &n, nullptr) || n < 4) return {};
            std::uint32_t status = 0;
            std::memcpy(&status, reply.data(), 4);
            if (status != opcode) return {};
            return {reply.begin() + 4, reply.begin() + n};
        }

    private:
        HANDLE file_ = INVALID_HANDLE_VALUE;
        WINUSB_INTERFACE_HANDLE usb_ = nullptr;
        UCHAR in_ = 0, out_ = 0;

        // The interface GUID comes from the device's MS OS descriptor, stored
        // under its hardware key as DeviceInterfaceGUID.
        static std::wstring interfacePath(const std::wstring& instanceId) {
            DEVINST inst{};
            auto* id = const_cast<wchar_t*>(instanceId.c_str());
            if (CM_Locate_DevNodeW(&inst, id, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS) return {};
            HKEY key{};
            if (CM_Open_DevNode_Key(inst, KEY_READ, 0, RegDisposition_OpenExisting, &key, CM_REGISTRY_HARDWARE) != CR_SUCCESS) return {};
            wchar_t guidText[64] = {};
            DWORD size = sizeof(guidText) - sizeof(wchar_t);
            const LSTATUS st = RegQueryValueExW(key, L"DeviceInterfaceGUID", nullptr, nullptr,
                                                reinterpret_cast<BYTE*>(guidText), &size);
            RegCloseKey(key);
            GUID guid{};
            if (st != ERROR_SUCCESS || CLSIDFromString(guidText, &guid) != NOERROR) return {};

            ULONG len = 0;
            if (CM_Get_Device_Interface_List_SizeW(&len, &guid, id, CM_GET_DEVICE_INTERFACE_LIST_PRESENT) != CR_SUCCESS || len <= 1) return {};
            std::vector<wchar_t> list(len);
            if (CM_Get_Device_Interface_ListW(&guid, id, list.data(), len, CM_GET_DEVICE_INTERFACE_LIST_PRESENT) != CR_SUCCESS) return {};
            return list.data();
        }
    };

    std::uint32_t readU32(const std::vector<std::uint8_t>& b, size_t at) {
        std::uint32_t v = 0;
        std::memcpy(&v, b.data() + at, 4);
        return v;
    }

    float readF32(const std::vector<std::uint8_t>& b, size_t at) {
        float v = 0;
        std::memcpy(&v, b.data() + at, 4);
        return v;
    }

    // uint32 width, uint32 height, float fx, fy, ppx, ppy, float coeffs[5].
    RealSenseL515::Intrinsics readPinhole(const std::vector<std::uint8_t>& b, size_t at) {
        RealSenseL515::Intrinsics k;
        k.width = static_cast<int>(readU32(b, at));
        k.height = static_cast<int>(readU32(b, at + 4));
        k.fx = readF32(b, at + 8);
        k.fy = readF32(b, at + 12);
        k.ppx = readF32(b, at + 16);
        k.ppy = readF32(b, at + 20);
        for (size_t i = 0; i < 5; ++i) k.coeffs[i] = readF32(b, at + 24 + 4 * i);
        return k;
    }

    // Depth table (0x7F): 16-byte header, then one 112-byte entry per
    // resolution: the portrait sensor model (56 bytes) and the landscape model
    // (56 bytes: the pinhole above plus three floats not used here). The
    // landscape model is the one matching the frames this class delivers.
    // RGB table (0x81): 12-byte header, then 44-byte pinhole entries.
    // RGB extrinsics (0x82): row-major rotation, then translation in mm.
    // Unused trailing entries have width 0.
    Calibration readCalibration(const std::wstring& parent) {
        Calibration calib;
        if (parent.empty()) return calib;
        std::wstring upper = parent;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towupper(c)); });
        HwMonitor hwm(L"USB\\VID_8086&PID_0B64&MI_06\\" + upper + L"&0006");
        if (!hwm.valid()) return calib;

        const auto depth = hwm.command(opDepthIntrinsics);
        for (size_t at = 16; at + 112 <= depth.size(); at += 112) {
            const auto k = readPinhole(depth, at + 56);
            if (k.width <= 0 || k.height <= 0) break;
            calib.depth.push_back(k);
        }

        const auto color = hwm.command(opRgbIntrinsics);
        for (size_t at = 12; at + 44 <= color.size(); at += 44) {
            const auto k = readPinhole(color, at);
            if (k.width <= 0 || k.height <= 0) break;
            calib.color.push_back(k);
        }

        const auto ext = hwm.command(opRgbExtrinsics);
        if (ext.size() >= 48) {
            for (size_t i = 0; i < 9; ++i) calib.depthToColor.rotation[i] = readF32(ext, 4 * i);
            for (size_t i = 0; i < 3; ++i) calib.depthToColor.translation[i] = readF32(ext, 36 + 4 * i) * 0.001f;
        }

        calib.valid = !calib.depth.empty() && !calib.color.empty() && ext.size() >= 48;
        if (!calib.valid) calib = {};
        return calib;
    }

    std::string hrText(const char* what, HRESULT hr) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s failed (HRESULT 0x%08lx)", what, static_cast<unsigned long>(hr));
        return buf;
    }

    // Keeps COM (MTA) and Media Foundation initialised for one thread.
    struct MfThreadScope {
        bool com = false, mf = false;
        MfThreadScope() {
            com = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
            mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE));
        }
        ~MfThreadScope() {
            if (mf) MFShutdown();
            if (com) CoUninitialize();
        }
    };

    struct DeviceEntry {
        std::wstring link;  ///< lower-cased symbolic link
        std::wstring parent;///< composite-device instance prefix shared by all interfaces
    };

    // \\?\usb#vid_8086&pid_0b64&mi_00#6&1c496b7e&2&0000#{...}\global
    //        interface tag ----^^^^^  ^^^^^^^^^^^^^^ parent  ^^^^ interface number
    std::wstring parentOf(const std::wstring& link) {
        const auto first = link.find(L'#');
        if (first == std::wstring::npos) return {};
        const auto second = link.find(L'#', first + 1);
        if (second == std::wstring::npos) return {};
        auto inst = link.substr(second + 1);
        inst = inst.substr(0, inst.find(L'#'));
        const auto amp = inst.rfind(L'&');
        return amp == std::wstring::npos ? inst : inst.substr(0, amp);
    }

    // Capture devices of the given L515 interface, in enumeration order.
    std::vector<DeviceEntry> enumerate(const wchar_t* interfaceTag) {
        std::vector<DeviceEntry> out;
        ComPtr<IMFAttributes> attr;
        if (FAILED(MFCreateAttributes(attr.put(), 1))) return out;
        attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

        IMFActivate** devs = nullptr;
        UINT32 n = 0;
        if (FAILED(MFEnumDeviceSources(attr.p, &devs, &n))) return out;
        for (UINT32 i = 0; i < n; ++i) {
            WCHAR* link = nullptr;
            UINT32 len = 0;
            if (SUCCEEDED(devs[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &link, &len))) {
                std::wstring s(link, len);
                std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
                if (s.find(L"vid_8086&pid_0b64&") != std::wstring::npos && s.find(interfaceTag) != std::wstring::npos) {
                    out.push_back({s, parentOf(s)});
                }
                CoTaskMemFree(link);
            }
            devs[i]->Release();
        }
        CoTaskMemFree(devs);
        return out;
    }

    HRESULT openReader(const std::wstring& link, ComPtr<IMFMediaSource>& source, ComPtr<IMFSourceReader>& reader) {
        ComPtr<IMFAttributes> attr;
        HRESULT hr = MFCreateAttributes(attr.put(), 2);
        if (FAILED(hr)) return hr;
        attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        attr->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, link.c_str());
        hr = MFCreateDeviceSource(attr.p, source.put());
        if (FAILED(hr)) return hr;

        ComPtr<IMFAttributes> readerAttr;
        hr = MFCreateAttributes(readerAttr.put(), 1);
        if (FAILED(hr)) return hr;
        // Z16 has no decoder; keep every stream in its native format.
        readerAttr->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
        return MFCreateSourceReaderFromMediaSource(source.p, readerAttr.p, reader.put());
    }

    DWORD fourcc(const char (&s)[5]) {
        return static_cast<DWORD>(s[0]) | static_cast<DWORD>(s[1]) << 8 |
               static_cast<DWORD>(s[2]) << 16 | static_cast<DWORD>(s[3]) << 24;
    }

    // Finds the native type matching subtype/size/fps across all streams;
    // returns the stream index or -1.
    int selectNativeType(IMFSourceReader* reader, DWORD subtypeFourcc, UINT32 width, UINT32 height, UINT32 fps) {
        for (DWORD s = 0;; ++s) {
            ComPtr<IMFMediaType> mt;
            if (FAILED(reader->GetNativeMediaType(s, 0, mt.put()))) return -1;
            for (DWORD i = 0;; ++i) {
                if (FAILED(reader->GetNativeMediaType(s, i, mt.put()))) break;
                GUID sub{};
                UINT32 w = 0, h = 0, fn = 0, fd = 1;
                mt->GetGUID(MF_MT_SUBTYPE, &sub);
                MFGetAttributeSize(mt.p, MF_MT_FRAME_SIZE, &w, &h);
                MFGetAttributeRatio(mt.p, MF_MT_FRAME_RATE, &fn, &fd);
                const UINT32 rate = fd ? static_cast<UINT32>(std::lround(static_cast<double>(fn) / fd)) : 0;
                if (sub.Data1 == subtypeFourcc && w == width && h == height && rate == fps) {
                    if (FAILED(reader->SetCurrentMediaType(s, nullptr, mt.p))) return -1;
                    reader->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS), FALSE);
                    reader->SetStreamSelection(s, TRUE);
                    return static_cast<int>(s);
                }
            }
        }
    }

    // Locks a sample's pixels, reporting the row pitch in bytes.
    struct LockedSample {
        ComPtr<IMFMediaBuffer> buffer;
        ComPtr<IMF2DBuffer> buffer2d;
        BYTE* data = nullptr;
        LONG pitch = 0;
        DWORD length = 0;

        bool lock(IMFSample* sample, LONG defaultPitch) {
            if (FAILED(sample->GetBufferByIndex(0, buffer.put()))) return false;
            if (SUCCEEDED(buffer->QueryInterface(IID_PPV_ARGS(buffer2d.put()))) &&
                SUCCEEDED(buffer2d->Lock2D(&data, &pitch))) {
                return true;
            }
            buffer2d.reset();
            pitch = defaultPitch;
            return SUCCEEDED(buffer->Lock(&data, nullptr, &length));
        }

        ~LockedSample() {
            if (!data) return;
            if (buffer2d) buffer2d->Unlock2D();
            else buffer->Unlock();
        }
    };

    std::uint8_t clamp8(int v) {
        return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    // BT.601 limited-range YUY2 -> RGBA8.
    void yuy2ToRgba(const BYTE* src, LONG pitch, int width, int height, std::uint8_t* dst) {
        for (int y = 0; y < height; ++y) {
            const BYTE* row = src + static_cast<ptrdiff_t>(pitch) * y;
            std::uint8_t* out = dst + static_cast<size_t>(y) * width * 4;
            for (int x = 0; x < width; x += 2) {
                const int y0 = row[0] - 16, u = row[1] - 128, y1 = row[2] - 16, v = row[3] - 128;
                row += 4;
                const int rv = 409 * v, guv = -100 * u - 208 * v, bu = 516 * u;
                const int c0 = 298 * y0 + 128, c1 = 298 * y1 + 128;
                out[0] = clamp8((c0 + rv) >> 8);
                out[1] = clamp8((c0 + guv) >> 8);
                out[2] = clamp8((c0 + bu) >> 8);
                out[3] = 255;
                out[4] = clamp8((c1 + rv) >> 8);
                out[5] = clamp8((c1 + guv) >> 8);
                out[6] = clamp8((c1 + bu) >> 8);
                out[7] = 255;
                out += 8;
            }
        }
    }

}// namespace

struct RealSenseL515::Impl {
    Config config;
    std::atomic<bool> stop{false};

    std::mutex errorMutex;
    std::string error;

    Stream<DepthFrame> depth;
    Stream<ColorFrame> color;

    Calibration calib;

    explicit Impl(const Config& cfg): config(cfg) {
        {
            MfThreadScope mf;
            if (mf.mf) calib = readCalibration(unitParent());
        }
        if (config.depth) start(depth, [this](std::promise<void>& ready) { runDepth(ready); });
        if (config.color) start(color, [this](std::promise<void>& ready) { runColor(ready); });
    }

    ~Impl() {
        stop = true;
        if (depth.thread.joinable()) depth.thread.join();
        if (color.thread.joinable()) color.thread.join();
    }

    void setError(std::string msg) {
        std::lock_guard lock(errorMutex);
        error = std::move(msg);
    }

    // Starts the stream's thread and waits until it has opened (or failed),
    // so depthOpen()/colorOpen() are meaningful as soon as the constructor returns.
    template<class Frame, class Fn>
    void start(Stream<Frame>& stream, Fn fn) {
        std::promise<void> ready;
        auto opened = ready.get_future();
        stream.thread = std::thread([fn, r = std::move(ready)]() mutable { fn(r); });
        opened.wait();
    }

    // The composite-device prefix of the configured unit, from its depth interface.
    std::wstring unitParent() {
        const auto units = enumerate(L"&mi_00#");
        if (config.deviceIndex < 0 || config.deviceIndex >= static_cast<int>(units.size())) return {};
        return units[config.deviceIndex].parent;
    }

    bool openStream(const wchar_t* interfaceTag, const char* what, DWORD subtype,
                    UINT32 w, UINT32 h, UINT32 fps,
                    ComPtr<IMFMediaSource>& source, ComPtr<IMFSourceReader>& reader, int& streamIndex) {
        const auto parent = unitParent();
        if (parent.empty()) {
            setError(std::string("no RealSense L515 found at device index ") + std::to_string(config.deviceIndex));
            return false;
        }
        const DeviceEntry* entry = nullptr;
        const auto devices = enumerate(interfaceTag);
        for (const auto& d : devices) {
            if (d.parent == parent) entry = &d;
        }
        if (!entry) {
            setError(std::string("L515 ") + what + " interface not found");
            return false;
        }
        if (const HRESULT hr = openReader(entry->link, source, reader); FAILED(hr)) {
            setError(hrText((std::string("opening L515 ") + what).c_str(), hr));
            return false;
        }
        streamIndex = selectNativeType(reader.p, subtype, w, h, fps);
        if (streamIndex < 0) {
            setError(std::string("L515 ") + what + " mode " + std::to_string(w) + "x" + std::to_string(h) +
                     "@" + std::to_string(fps) + " not offered");
            return false;
        }
        return true;
    }

    void runDepth(std::promise<void>& ready) {
        MfThreadScope mf;
        ComPtr<IMFMediaSource> source;
        ComPtr<IMFSourceReader> reader;
        const int outW = config.depthWidth, outH = config.depthHeight;
        // The UVC mode is portrait: width and height swapped.
        const int srcW = outH, srcH = outW;
        int streamIndex = -1;
        const bool ok = mf.mf && openStream(L"&mi_00#", "depth", fourcc("Z16 "), srcW, srcH, 30, source, reader, streamIndex);
        depth.open = ok;
        ready.set_value();
        if (!ok) return;

        const auto intrinsics = Calibration::find(calib.depth, outW, outH, depthHfov, depthVfov);
        std::vector<std::uint16_t> rotated(static_cast<size_t>(outW) * outH);
        std::uint64_t count = 0;

        while (!stop) {
            DWORD actual = 0, flags = 0;
            LONGLONG ts = 0;
            ComPtr<IMFSample> sample;
            const HRESULT hr = reader->ReadSample(streamIndex, 0, &actual, &flags, &ts, sample.put());
            if (FAILED(hr) || (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM))) {
                setError(hrText("L515 depth ReadSample", hr));
                break;
            }
            if (!sample) continue;

            LockedSample px;
            if (!px.lock(sample.p, srcW * 2)) continue;
            if (!px.buffer2d && px.length < static_cast<DWORD>(srcW * srcH * 2)) continue;
            // Portrait -> landscape is a reflection across the anti-diagonal
            // (90 degrees clockwise, then flipped vertically), checked against
            // the RGB frame: out(x, y) = src(col = srcW - 1 - y, row = srcH - 1 - x).
            for (int y = 0; y < outH; ++y) {
                std::uint16_t* out = rotated.data() + static_cast<size_t>(y) * outW;
                for (int x = 0; x < outW; ++x) {
                    const BYTE* row = px.data + static_cast<ptrdiff_t>(px.pitch) * (srcH - 1 - x);
                    out[x] = reinterpret_cast<const std::uint16_t*>(row)[srcW - 1 - y];
                }
            }

            std::lock_guard lock(depth.mutex);
            depth.latest.width = outW;
            depth.latest.height = outH;
            depth.latest.intrinsics = intrinsics;
            depth.latest.data.swap(rotated);
            depth.latest.frameNumber = ++count;
            depth.latest.timestamp = static_cast<double>(ts) * 1e-7;
            depth.fresh = true;
            rotated.resize(static_cast<size_t>(outW) * outH);
        }
        depth.open = false;
        source->Shutdown();
    }

    void runColor(std::promise<void>& ready) {
        MfThreadScope mf;
        ComPtr<IMFMediaSource> source;
        ComPtr<IMFSourceReader> reader;
        const int w = config.colorWidth, h = config.colorHeight;
        int streamIndex = -1;
        const bool ok = mf.mf && openStream(L"&mi_04#", "RGB", fourcc("YUY2"), w, h, config.colorFps, source, reader, streamIndex);
        color.open = ok;
        ready.set_value();
        if (!ok) return;

        const auto intrinsics = Calibration::find(calib.color, w, h, colorHfov, colorVfov);
        std::vector<std::uint8_t> rgba(static_cast<size_t>(w) * h * 4);
        std::uint64_t count = 0;

        while (!stop) {
            DWORD actual = 0, flags = 0;
            LONGLONG ts = 0;
            ComPtr<IMFSample> sample;
            const HRESULT hr = reader->ReadSample(streamIndex, 0, &actual, &flags, &ts, sample.put());
            if (FAILED(hr) || (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM))) {
                setError(hrText("L515 RGB ReadSample", hr));
                break;
            }
            if (!sample) continue;

            LockedSample px;
            if (!px.lock(sample.p, w * 2)) continue;
            if (!px.buffer2d && px.length < static_cast<DWORD>(w * h * 2)) continue;
            yuy2ToRgba(px.data, px.pitch, w, h, rgba.data());

            std::lock_guard lock(color.mutex);
            color.latest.width = w;
            color.latest.height = h;
            color.latest.intrinsics = intrinsics;
            color.latest.rgba.swap(rgba);
            color.latest.frameNumber = ++count;
            color.latest.timestamp = static_cast<double>(ts) * 1e-7;
            color.fresh = true;
            rgba.resize(static_cast<size_t>(w) * h * 4);
        }
        color.open = false;
        source->Shutdown();
    }
};

int RealSenseL515::deviceCount() {
    MfThreadScope mf;
    return mf.mf ? static_cast<int>(enumerate(L"&mi_00#").size()) : 0;
}

#else// !_WIN32

struct RealSenseL515::Impl {
    Config config;
    std::string error = "RealSenseL515 is only implemented on Windows (Media Foundation)";
    std::mutex errorMutex;
    Stream<DepthFrame> depth;
    Stream<ColorFrame> color;
    Calibration calib;

    explicit Impl(const Config& cfg): config(cfg) {}
};

int RealSenseL515::deviceCount() {
    return 0;
}

#endif

RealSenseL515::RealSenseL515(): RealSenseL515(Config{}) {}

RealSenseL515::RealSenseL515(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

RealSenseL515::~RealSenseL515() = default;

bool RealSenseL515::depthOpen() const {
    return pimpl_->depth.open;
}

bool RealSenseL515::colorOpen() const {
    return pimpl_->color.open;
}

bool RealSenseL515::calibrated() const {
    return pimpl_->calib.valid;
}

RealSenseL515::Extrinsics RealSenseL515::depthToColor() const {
    return pimpl_->calib.depthToColor;
}

std::string RealSenseL515::lastError() const {
    std::lock_guard lock(pimpl_->errorMutex);
    return pimpl_->error;
}

bool RealSenseL515::pollDepth(DepthFrame& out) {
    return pimpl_->depth.poll(out);
}

bool RealSenseL515::pollColor(ColorFrame& out) {
    return pimpl_->color.poll(out);
}
