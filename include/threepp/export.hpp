
#ifndef THREEPP_EXPORT_HPP
#define THREEPP_EXPORT_HPP

// THREEPP_API marks the few symbols a shared threepp has to export by hand.
//
// The DLL build exports every function through CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS,
// but that mechanism cannot export static data members: a consumer that reads
// ColorManagement::workingColorSpace gets "unresolved external symbol" unless
// the declaration itself carries dllimport. THREEPP_SHARED is defined (PUBLIC,
// so for consumers too) when threepp is built as a shared library, and
// threepp_EXPORTS is CMake's DEFINE_SYMBOL for the library's own objects.
// A static threepp, and every non-Windows build, see an empty macro.
#if defined(_WIN32) && defined(THREEPP_SHARED)
#if defined(threepp_EXPORTS)
#define THREEPP_API __declspec(dllexport)
#else
#define THREEPP_API __declspec(dllimport)
#endif
#else
#define THREEPP_API
#endif

#endif//THREEPP_EXPORT_HPP
