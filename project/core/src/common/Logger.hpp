#ifndef CODECUDA_LOGGER_HPP
#define CODECUDA_LOGGER_HPP

#include <cstdio>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <source_location>
#include <sstream>
#include <string_view>
#include <utility>

namespace CodeCommon
{
    // Sample host usage:
    //   CODECUDA_LOG_INFO("Initialized codeCudaLib");
    //   CODECUDA_LOG_WARNING("Allocation fallback for device ", device_index);
    //   CODECUDA_PRINT("matmul completed\n");
    //   CODECUDA_PRINTLN("matmul completed");
    //
    // Sample device usage:
    //   __global__ void kernel()
    //   {
    //       CODECUDA_DEVICE_LOG_DEBUG("row=%d col=%d", row, col);
    //       CODECUDA_DEVICE_PRINT("row=%d\n", row);
    //   }
    enum class LogLevel
    {
        debug = 0,
        info = 1,
        warning = 2,
        error = 3,
    };

    class Logger
    {
    public:
        static void SetMinimumLevel(LogLevel level)
        {
            MinimumLevelStorage() = level;
        }

        static LogLevel GetMinimumLevel()
        {
            return MinimumLevelStorage();
        }

        template <typename... Args>
        static void Log(LogLevel level, const std::source_location& location, Args&&... args)
        {
            if (!ShouldLog(level))
            {
                return;
            }

            std::lock_guard<std::mutex> lock(MutexStorage());
            std::ostream& stream = StreamFor(level);
            stream << "[" << TimestampNow() << "]"
                   << " [" << ToString(level) << "]"
                   << " [" << location.file_name() << ":" << location.line() << "] ";
            (stream << ... << std::forward<Args>(args));
            stream << '\n';
        }

        template <typename... Args>
        static void Print(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            (std::cout << ... << std::forward<Args>(args));
        }

        template <typename... Args>
        static void Println(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            (std::cout << ... << std::forward<Args>(args));
            std::cout << '\n';
        }
        
        template <typename... Args>
        static void PrintError(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            (std::cerr << ... << std::forward<Args>(args));
        }

    private:
        static bool ShouldLog(LogLevel level)
        {
            return static_cast<int>(level) >= static_cast<int>(GetMinimumLevel());
        }

        static LogLevel& MinimumLevelStorage()
        {
            static LogLevel level = LogLevel::info;
            return level;
        }

        static std::mutex& MutexStorage()
        {
            static std::mutex mutex;
            return mutex;
        }

        static std::ostream& StreamFor(LogLevel level)
        {
            if (level == LogLevel::error)
            {
                return std::cerr;
            }

            return std::clog;
        }

        static std::string_view ToString(LogLevel level)
        {
            switch (level)
            {
                case LogLevel::debug:
                    return "DEBUG";
                case LogLevel::info:
                    return "INFO";
                case LogLevel::warning:
                    return "WARN";
                case LogLevel::error:
                    return "ERROR";
            }

            return "UNKNOWN";
        }

        static std::string TimestampNow()
        {
            const auto now = std::chrono::system_clock::now();
            const auto time = std::chrono::system_clock::to_time_t(now);
            std::tm local_time{};
#ifdef _WIN32
            localtime_s(&local_time, &time);
#else
            localtime_r(&time, &local_time);
#endif

            std::ostringstream output;
            output << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
            return output.str();
        }
    };
}

#define CODECUDA_LOG_DEBUG(...) ::CodeCommon::Logger::Log(::CodeCommon::LogLevel::debug, std::source_location::current(), __VA_ARGS__)
#define CODECUDA_LOG_INFO(...) ::CodeCommon::Logger::Log(::CodeCommon::LogLevel::info, std::source_location::current(), __VA_ARGS__)
#define CODECUDA_LOG_WARNING(...) ::CodeCommon::Logger::Log(::CodeCommon::LogLevel::warning, std::source_location::current(), __VA_ARGS__)
#define CODECUDA_LOG_ERROR(...) ::CodeCommon::Logger::Log(::CodeCommon::LogLevel::error, std::source_location::current(), __VA_ARGS__)
#define CODECUDA_PRINT(...) ::CodeCommon::Logger::Print(__VA_ARGS__)
#define CODECUDA_PRINTLN(...) ::CodeCommon::Logger::Println(__VA_ARGS__)
#define CODECUDA_PRINT_ERROR(...) ::CodeCommon::Logger::PrintError(__VA_ARGS__)

#define CODECUDA_ENABLE_DEVICE_LOGS 1

#if defined(CODECUDA_ENABLE_DEVICE_LOGS)
    #define CODECUDA_DEVICE_LOG_ENABLED 1
#else
    #define CODECUDA_DEVICE_LOG_ENABLED 0
#endif

#ifdef __CUDA_ARCH__
    #if CODECUDA_DEVICE_LOG_ENABLED
        #define CODECUDA_DEVICE_LOG_IMPL(level, fmt, ...) printf("[%s] " fmt "\n", level, ##__VA_ARGS__)
        #define CODECUDA_DEVICE_PRINT_IMPL(fmt, ...) printf(fmt, ##__VA_ARGS__)
    #else
        #define CODECUDA_DEVICE_LOG_IMPL(level, fmt, ...)
        #define CODECUDA_DEVICE_PRINT_IMPL(fmt, ...)
    #endif
#else
    #define CODECUDA_DEVICE_LOG_IMPL(level, fmt, ...)
    #define CODECUDA_DEVICE_PRINT_IMPL(fmt, ...)
#endif

#define CODECUDA_DEVICE_LOG_DEBUG(fmt, ...) CODECUDA_DEVICE_LOG_IMPL("DEBUG", fmt, ##__VA_ARGS__)
#define CODECUDA_DEVICE_LOG_INFO(fmt, ...) CODECUDA_DEVICE_LOG_IMPL("INFO", fmt, ##__VA_ARGS__)
#define CODECUDA_DEVICE_LOG_WARNING(fmt, ...) CODECUDA_DEVICE_LOG_IMPL("WARN", fmt, ##__VA_ARGS__)
#define CODECUDA_DEVICE_LOG_ERROR(fmt, ...) CODECUDA_DEVICE_LOG_IMPL("ERROR", fmt, ##__VA_ARGS__)
#define CODECUDA_DEVICE_PRINT(fmt, ...) CODECUDA_DEVICE_PRINT_IMPL(fmt, ##__VA_ARGS__)
#define CODECUDA_DEVICE_PRINT_ERROR(fmt, ...) CODECUDA_DEVICE_PRINT_IMPL(fmt, ##__VA_ARGS__)


#endif // CODECUDA_LOGGER_HPP
