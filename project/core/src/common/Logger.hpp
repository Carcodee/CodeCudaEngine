#ifndef CODECUDA_LOGGER_HPP
#define CODECUDA_LOGGER_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <source_location>
#include <sstream>
#include <string>
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

        static void SetOutputFile(const std::string& path)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            SetOutputFileUnlocked(path);
        }

        static void DisableOutputFile()
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            OutputFileStorage().close();
            OutputFileEnvCheckedStorage() = true;
        }

        template <typename... Args>
        static void Log(LogLevel level, const std::source_location& location, Args&&... args)
        {
            if (!ShouldLog(level))
            {
                return;
            }

            std::lock_guard<std::mutex> lock(MutexStorage());
            EnsureOutputFileConfigured();
            std::ostream& stream = StreamFor(level);
            const std::string prefix = "[" + TimestampNow() + "] [" + std::string(ToString(level)) + "] [" +
                                       location.file_name() + ":" + std::to_string(location.line()) + "] ";
            std::ostringstream message;
            (message << ... << std::forward<Args>(args));

            stream << prefix << message.str() << '\n';
            WriteToOutputFile(prefix, message.str(), "\n");
        }

        template <typename... Args>
        static void Print(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            EnsureOutputFileConfigured();
            std::ostringstream message;
            (message << ... << std::forward<Args>(args));
            std::cout << message.str();
            WriteToOutputFile("", message.str(), "");
        }

        template <typename... Args>
        static void Println(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            EnsureOutputFileConfigured();
            std::ostringstream message;
            (message << ... << std::forward<Args>(args));
            std::cout << message.str();
            std::cout << '\n';
            WriteToOutputFile("", message.str(), "\n");
        }
        
        template <typename... Args>
        static void PrintError(Args&&... args)
        {
            std::lock_guard<std::mutex> lock(MutexStorage());
            EnsureOutputFileConfigured();
            std::ostringstream message;
            (message << ... << std::forward<Args>(args));
            std::cerr << message.str();
            WriteToOutputFile("", message.str(), "");
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

        static std::ofstream& OutputFileStorage()
        {
            static std::ofstream file;
            return file;
        }

        static bool& OutputFileEnvCheckedStorage()
        {
            static bool checked = false;
            return checked;
        }

        static void SetOutputFileUnlocked(const std::string& path)
        {
            std::ofstream& file = OutputFileStorage();
            file.close();
            file.open(path, std::ios::out | std::ios::app);
        }

        static void EnsureOutputFileConfigured()
        {
            bool& checked = OutputFileEnvCheckedStorage();
            if (checked)
            {
                return;
            }

            checked = true;
            const char* path = std::getenv("CODECUDA_LOG_FILE");
            if (path != nullptr && path[0] != '\0')
            {
                SetOutputFileUnlocked(path);
            }
        }

        static void WriteToOutputFile(const std::string& prefix, const std::string& message, const char* suffix)
        {
            std::ofstream& file = OutputFileStorage();
            if (!file.is_open())
            {
                return;
            }

            file << prefix << message << suffix;
            file.flush();
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
