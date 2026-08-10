//
// Created by carlo on 2026-07-29.
//

#ifndef CODECUDAPROJECT_CODEMATH_HPP
#define CODECUDAPROJECT_CODEMATH_HPP

#include <cmath>

namespace code_math
{
    // ============================================================
    // vec2
    // ============================================================

    struct vec2
    {
        float x = 0.0f;
        float y = 0.0f;

        __host__ __device__ constexpr vec2() {}

        __host__ __device__
        explicit constexpr vec2(float value)
            : x(value), y(value)
        {
        }

        __host__ __device__
        constexpr vec2(float x, float y)
            : x(x), y(y)
        {
        }

        __host__ __device__
        constexpr float& operator[](int index)
        {
            return index == 0 ? x : y;
        }

        __host__ __device__
        constexpr const float& operator[](int index) const
        {
            return index == 0 ? x : y;
        }

        __host__ __device__
        constexpr vec2 operator+() const
        {
            return *this;
        }

        __host__ __device__
        constexpr vec2 operator-() const
        {
            return {-x, -y};
        }

        __host__ __device__
        constexpr vec2& operator+=(const vec2& other)
        {
            x += other.x;
            y += other.y;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator-=(const vec2& other)
        {
            x -= other.x;
            y -= other.y;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator*=(const vec2& other)
        {
            x *= other.x;
            y *= other.y;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator/=(const vec2& other)
        {
            x /= other.x;
            y /= other.y;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator+=(float scalar)
        {
            x += scalar;
            y += scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator-=(float scalar)
        {
            x -= scalar;
            y -= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator*=(float scalar)
        {
            x *= scalar;
            y *= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec2& operator/=(float scalar)
        {
            x /= scalar;
            y /= scalar;
            return *this;
        }

        __host__ __device__
        constexpr bool operator==(const vec2& other) const
        {
            return x == other.x &&
                   y == other.y;
        }

        __host__ __device__
        constexpr bool operator!=(const vec2& other) const
        {
            return !(*this == other);
        }

        __host__ __device__
        constexpr float dot(const vec2& other) const
        {
            return x * other.x +
                   y * other.y;
        }

        __host__ __device__
        constexpr float length_squared() const
        {
            return dot(*this);
        }

        __host__ __device__
        float length() const
        {
            return sqrtf(length_squared());
        }

        __host__ __device__
        vec2 normalized(float epsilon = 1.0e-6f) const
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
                return {};

            const float inverse_length = 1.0f / sqrtf(squared);

            return {
                x * inverse_length,
                y * inverse_length
            };
        }

        __host__ __device__
        vec2& normalize(float epsilon = 1.0e-6f)
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
            {
                x = 0.0f;
                y = 0.0f;
                return *this;
            }

            return *this *= 1.0f / sqrtf(squared);
        }

        __host__ __device__
        constexpr float distance_squared(const vec2& other) const
        {
            const float dx = x - other.x;
            const float dy = y - other.y;

            return dx * dx + dy * dy;
        }

        __host__ __device__
        float distance(const vec2& other) const
        {
            return sqrtf(distance_squared(other));
        }
    };


    // ============================================================
    // vec3
    // ============================================================

    struct vec3
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;

        __host__ __device__ constexpr vec3() {}

        __host__ __device__
        explicit constexpr vec3(float value)
            : x(value), y(value), z(value)
        {
        }

        __host__ __device__
        constexpr vec3(float x, float y, float z)
            : x(x), y(y), z(z)
        {
        }

        __host__ __device__
        constexpr float& operator[](int index)
        {
            return index == 0 ? x :
                   index == 1 ? y : z;
        }

        __host__ __device__
        constexpr const float& operator[](int index) const
        {
            return index == 0 ? x :
                   index == 1 ? y : z;
        }

        __host__ __device__
        constexpr vec3 operator+() const
        {
            return *this;
        }

        __host__ __device__
        constexpr vec3 operator-() const
        {
            return {-x, -y, -z};
        }

        __host__ __device__
        constexpr vec3& operator+=(const vec3& other)
        {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator-=(const vec3& other)
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator*=(const vec3& other)
        {
            x *= other.x;
            y *= other.y;
            z *= other.z;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator/=(const vec3& other)
        {
            x /= other.x;
            y /= other.y;
            z /= other.z;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator+=(float scalar)
        {
            x += scalar;
            y += scalar;
            z += scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator-=(float scalar)
        {
            x -= scalar;
            y -= scalar;
            z -= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator*=(float scalar)
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec3& operator/=(float scalar)
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            return *this;
        }

        __host__ __device__
        constexpr bool operator==(const vec3& other) const
        {
            return x == other.x &&
                   y == other.y &&
                   z == other.z;
        }

        __host__ __device__
        constexpr bool operator!=(const vec3& other) const
        {
            return !(*this == other);
        }

        __host__ __device__
        constexpr float dot(const vec3& other) const
        {
            return x * other.x +
                   y * other.y +
                   z * other.z;
        }

        __host__ __device__
        constexpr vec3 cross(const vec3& other) const
        {
            return {
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x
            };
        }

        __host__ __device__
        constexpr float length_squared() const
        {
            return dot(*this);
        }

        __host__ __device__
        float length() const
        {
            return sqrtf(length_squared());
        }

        __host__ __device__
        vec3 normalized(float epsilon = 1.0e-6f) const
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
                return {};

            const float inverse_length = 1.0f / sqrtf(squared);

            return {
                x * inverse_length,
                y * inverse_length,
                z * inverse_length
            };
        }

        __host__ __device__
        vec3& normalize(float epsilon = 1.0e-6f)
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
            {
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                return *this;
            }

            return *this *= 1.0f / sqrtf(squared);
        }

        __host__ __device__
        constexpr float distance_squared(const vec3& other) const
        {
            const float dx = x - other.x;
            const float dy = y - other.y;
            const float dz = z - other.z;

            return dx * dx +
                   dy * dy +
                   dz * dz;
        }

        __host__ __device__
        float distance(const vec3& other) const
        {
            return sqrtf(distance_squared(other));
        }
    };


    // ============================================================
    // vec4
    // ============================================================

    struct vec4
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;

        __host__ __device__ constexpr vec4() {}

        __host__ __device__
        explicit constexpr vec4(float value)
            : x(value), y(value), z(value), w(value)
        {
        }

        __host__ __device__
        constexpr vec4(float x, float y, float z, float w)
            : x(x), y(y), z(z), w(w)
        {
        }

        __host__ __device__
        constexpr float& operator[](int index)
        {
            return index == 0 ? x :
                   index == 1 ? y :
                   index == 2 ? z : w;
        }

        __host__ __device__
        constexpr const float& operator[](int index) const
        {
            return index == 0 ? x :
                   index == 1 ? y :
                   index == 2 ? z : w;
        }

        __host__ __device__
        constexpr vec4 operator+() const
        {
            return *this;
        }

        __host__ __device__
        constexpr vec4 operator-() const
        {
            return {-x, -y, -z, -w};
        }

        __host__ __device__
        constexpr vec4& operator+=(const vec4& other)
        {
            x += other.x;
            y += other.y;
            z += other.z;
            w += other.w;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator-=(const vec4& other)
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            w -= other.w;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator*=(const vec4& other)
        {
            x *= other.x;
            y *= other.y;
            z *= other.z;
            w *= other.w;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator/=(const vec4& other)
        {
            x /= other.x;
            y /= other.y;
            z /= other.z;
            w /= other.w;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator+=(float scalar)
        {
            x += scalar;
            y += scalar;
            z += scalar;
            w += scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator-=(float scalar)
        {
            x -= scalar;
            y -= scalar;
            z -= scalar;
            w -= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator*=(float scalar)
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            w *= scalar;
            return *this;
        }

        __host__ __device__
        constexpr vec4& operator/=(float scalar)
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            w /= scalar;
            return *this;
        }

        __host__ __device__
        constexpr bool operator==(const vec4& other) const
        {
            return x == other.x &&
                   y == other.y &&
                   z == other.z &&
                   w == other.w;
        }

        __host__ __device__
        constexpr bool operator!=(const vec4& other) const
        {
            return !(*this == other);
        }

        __host__ __device__
        constexpr float dot(const vec4& other) const
        {
            return x * other.x +
                   y * other.y +
                   z * other.z +
                   w * other.w;
        }

        __host__ __device__
        constexpr float length_squared() const
        {
            return dot(*this);
        }

        __host__ __device__
        float length() const
        {
            return sqrtf(length_squared());
        }

        __host__ __device__
        vec4 normalized(float epsilon = 1.0e-6f) const
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
                return {};

            const float inverse_length = 1.0f / sqrtf(squared);

            return {
                x * inverse_length,
                y * inverse_length,
                z * inverse_length,
                w * inverse_length
            };
        }

        __host__ __device__
        vec4& normalize(float epsilon = 1.0e-6f)
        {
            const float squared = length_squared();

            if (squared <= epsilon * epsilon)
            {
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                w = 0.0f;
                return *this;
            }

            return *this *= 1.0f / sqrtf(squared);
        }

        __host__ __device__
        constexpr float distance_squared(const vec4& other) const
        {
            const float dx = x - other.x;
            const float dy = y - other.y;
            const float dz = z - other.z;
            const float dw = w - other.w;

            return dx * dx +
                   dy * dy +
                   dz * dz +
                   dw * dw;
        }

        __host__ __device__
        float distance(const vec4& other) const
        {
            return sqrtf(distance_squared(other));
        }
    };


    // ============================================================
    // vec2 operators
    // ============================================================

    __host__ __device__
    constexpr vec2 operator+(vec2 lhs, const vec2& rhs)
    {
        return lhs += rhs;
    }

    __host__ __device__
    constexpr vec2 operator-(vec2 lhs, const vec2& rhs)
    {
        return lhs -= rhs;
    }

    __host__ __device__
    constexpr vec2 operator*(vec2 lhs, const vec2& rhs)
    {
        return lhs *= rhs;
    }

    __host__ __device__
    constexpr vec2 operator/(vec2 lhs, const vec2& rhs)
    {
        return lhs /= rhs;
    }

    __host__ __device__
    constexpr vec2 operator+(vec2 value, float scalar)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec2 operator-(vec2 value, float scalar)
    {
        return value -= scalar;
    }

    __host__ __device__
    constexpr vec2 operator*(vec2 value, float scalar)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec2 operator/(vec2 value, float scalar)
    {
        return value /= scalar;
    }

    __host__ __device__
    constexpr vec2 operator+(float scalar, vec2 value)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec2 operator-(float scalar, const vec2& value)
    {
        return {
            scalar - value.x,
            scalar - value.y
        };
    }

    __host__ __device__
    constexpr vec2 operator*(float scalar, vec2 value)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec2 operator/(float scalar, const vec2& value)
    {
        return {
            scalar / value.x,
            scalar / value.y
        };
    }


    // ============================================================
    // vec3 operators
    // ============================================================

    __host__ __device__
    constexpr vec3 operator+(vec3 lhs, const vec3& rhs)
    {
        return lhs += rhs;
    }

    __host__ __device__
    constexpr vec3 operator-(vec3 lhs, const vec3& rhs)
    {
        return lhs -= rhs;
    }

    __host__ __device__
    constexpr vec3 operator*(vec3 lhs, const vec3& rhs)
    {
        return lhs *= rhs;
    }

    __host__ __device__
    constexpr vec3 operator/(vec3 lhs, const vec3& rhs)
    {
        return lhs /= rhs;
    }

    __host__ __device__
    constexpr vec3 operator+(vec3 value, float scalar)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec3 operator-(vec3 value, float scalar)
    {
        return value -= scalar;
    }

    __host__ __device__
    constexpr vec3 operator*(vec3 value, float scalar)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec3 operator/(vec3 value, float scalar)
    {
        return value /= scalar;
    }

    __host__ __device__
    constexpr vec3 operator+(float scalar, vec3 value)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec3 operator-(float scalar, const vec3& value)
    {
        return {
            scalar - value.x,
            scalar - value.y,
            scalar - value.z
        };
    }

    __host__ __device__
    constexpr vec3 operator*(float scalar, vec3 value)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec3 operator/(float scalar, const vec3& value)
    {
        return {
            scalar / value.x,
            scalar / value.y,
            scalar / value.z
        };
    }


    // ============================================================
    // vec4 operators
    // ============================================================

    __host__ __device__
    constexpr vec4 operator+(vec4 lhs, const vec4& rhs)
    {
        return lhs += rhs;
    }

    __host__ __device__
    constexpr vec4 operator-(vec4 lhs, const vec4& rhs)
    {
        return lhs -= rhs;
    }

    __host__ __device__
    constexpr vec4 operator*(vec4 lhs, const vec4& rhs)
    {
        return lhs *= rhs;
    }

    __host__ __device__
    constexpr vec4 operator/(vec4 lhs, const vec4& rhs)
    {
        return lhs /= rhs;
    }

    __host__ __device__
    constexpr vec4 operator+(vec4 value, float scalar)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec4 operator-(vec4 value, float scalar)
    {
        return value -= scalar;
    }

    __host__ __device__
    constexpr vec4 operator*(vec4 value, float scalar)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec4 operator/(vec4 value, float scalar)
    {
        return value /= scalar;
    }

    __host__ __device__
    constexpr vec4 operator+(float scalar, vec4 value)
    {
        return value += scalar;
    }

    __host__ __device__
    constexpr vec4 operator-(float scalar, const vec4& value)
    {
        return {
            scalar - value.x,
            scalar - value.y,
            scalar - value.z,
            scalar - value.w
        };
    }

    __host__ __device__
    constexpr vec4 operator*(float scalar, vec4 value)
    {
        return value *= scalar;
    }

    __host__ __device__
    constexpr vec4 operator/(float scalar, const vec4& value)
    {
        return {
            scalar / value.x,
            scalar / value.y,
            scalar / value.z,
            scalar / value.w
        };
    }


    // ============================================================
    // vec2 functions
    // ============================================================

    __host__ __device__
    constexpr float dot(const vec2& lhs, const vec2& rhs)
    {
        return lhs.dot(rhs);
    }

    __host__ __device__
    constexpr float length_squared(const vec2& value)
    {
        return value.length_squared();
    }

    __host__ __device__
    inline float length(const vec2& value)
    {
        return value.length();
    }

    __host__ __device__
    constexpr float distance_squared(const vec2& lhs, const vec2& rhs)
    {
        return lhs.distance_squared(rhs);
    }

    __host__ __device__
    inline float distance(const vec2& lhs, const vec2& rhs)
    {
        return lhs.distance(rhs);
    }

    __host__ __device__
    inline vec2 normalized(
        const vec2& value,
        float epsilon = 1.0e-6f)
    {
        return value.normalized(epsilon);
    }

    __host__ __device__
    constexpr vec2 lerp(
        const vec2& from,
        const vec2& to,
        float amount)
    {
        return from + (to - from) * amount;
    }

    __host__ __device__
    constexpr vec2 component_min(
        const vec2& lhs,
        const vec2& rhs)
    {
        return {
            lhs.x < rhs.x ? lhs.x : rhs.x,
            lhs.y < rhs.y ? lhs.y : rhs.y
        };
    }

    __host__ __device__
    constexpr vec2 component_max(
        const vec2& lhs,
        const vec2& rhs)
    {
        return {
            lhs.x > rhs.x ? lhs.x : rhs.x,
            lhs.y > rhs.y ? lhs.y : rhs.y
        };
    }

    __host__ __device__
    constexpr vec2 clamp(
        const vec2& value,
        const vec2& lower,
        const vec2& upper)
    {
        return component_min(
            component_max(value, lower),
            upper);
    }

    __host__ __device__
    constexpr vec2 clamp(
        const vec2& value,
        float lower,
        float upper)
    {
        return clamp(
            value,
            vec2(lower),
            vec2(upper));
    }

    __host__ __device__
    inline vec2 abs(const vec2& value)
    {
        return {
            fabsf(value.x),
            fabsf(value.y)
        };
    }

    __host__ __device__
    inline vec2 floor(const vec2& value)
    {
        return {
            floorf(value.x),
            floorf(value.y)
        };
    }

    __host__ __device__
    inline vec2 ceil(const vec2& value)
    {
        return {
            ceilf(value.x),
            ceilf(value.y)
        };
    }

    __host__ __device__
    inline vec2 round(const vec2& value)
    {
        return {
            roundf(value.x),
            roundf(value.y)
        };
    }

    __host__ __device__
    constexpr float component_sum(const vec2& value)
    {
        return value.x + value.y;
    }

    __host__ __device__
    constexpr float min_component(const vec2& value)
    {
        return value.x < value.y
            ? value.x
            : value.y;
    }

    __host__ __device__
    constexpr float max_component(const vec2& value)
    {
        return value.x > value.y
            ? value.x
            : value.y;
    }

    __host__ __device__
    inline bool nearly_equal(
        const vec2& lhs,
        const vec2& rhs,
        float epsilon = 1.0e-6f)
    {
        return
            fabsf(lhs.x - rhs.x) <= epsilon &&
            fabsf(lhs.y - rhs.y) <= epsilon;
    }

    __host__ __device__
    constexpr vec2 reflect(
        const vec2& incident,
        const vec2& normal)
    {
        return incident -
               2.0f * dot(incident, normal) * normal;
    }

    __host__ __device__
    inline vec2 project(
        const vec2& value,
        const vec2& onto,
        float epsilon = 1.0e-6f)
    {
        const float denominator = length_squared(onto);

        if (denominator <= epsilon * epsilon)
            return {};

        return onto *
               (dot(value, onto) / denominator);
    }

    __host__ __device__
    inline vec2 reject(
        const vec2& value,
        const vec2& from,
        float epsilon = 1.0e-6f)
    {
        return value - project(value, from, epsilon);
    }


    // ============================================================
    // vec3 functions
    // ============================================================

    __host__ __device__
    constexpr float dot(const vec3& lhs, const vec3& rhs)
    {
        return lhs.dot(rhs);
    }

    __host__ __device__
    constexpr vec3 cross(const vec3& lhs, const vec3& rhs)
    {
        return lhs.cross(rhs);
    }

    __host__ __device__
    constexpr float length_squared(const vec3& value)
    {
        return value.length_squared();
    }

    __host__ __device__
    inline float length(const vec3& value)
    {
        return value.length();
    }

    __host__ __device__
    constexpr float distance_squared(const vec3& lhs, const vec3& rhs)
    {
        return lhs.distance_squared(rhs);
    }

    __host__ __device__
    inline float distance(const vec3& lhs, const vec3& rhs)
    {
        return lhs.distance(rhs);
    }

    __host__ __device__
    inline vec3 normalized(
        const vec3& value,
        float epsilon = 1.0e-6f)
    {
        return value.normalized(epsilon);
    }

    __host__ __device__
    constexpr vec3 lerp(
        const vec3& from,
        const vec3& to,
        float amount)
    {
        return from + (to - from) * amount;
    }

    __host__ __device__
    constexpr vec3 component_min(
        const vec3& lhs,
        const vec3& rhs)
    {
        return {
            lhs.x < rhs.x ? lhs.x : rhs.x,
            lhs.y < rhs.y ? lhs.y : rhs.y,
            lhs.z < rhs.z ? lhs.z : rhs.z
        };
    }

    __host__ __device__
    constexpr vec3 component_max(
        const vec3& lhs,
        const vec3& rhs)
    {
        return {
            lhs.x > rhs.x ? lhs.x : rhs.x,
            lhs.y > rhs.y ? lhs.y : rhs.y,
            lhs.z > rhs.z ? lhs.z : rhs.z
        };
    }

    __host__ __device__
    constexpr vec3 clamp(
        const vec3& value,
        const vec3& lower,
        const vec3& upper)
    {
        return component_min(
            component_max(value, lower),
            upper);
    }

    __host__ __device__
    constexpr vec3 clamp(
        const vec3& value,
        float lower,
        float upper)
    {
        return clamp(
            value,
            vec3(lower),
            vec3(upper));
    }

    __host__ __device__
    inline vec3 abs(const vec3& value)
    {
        return {
            fabsf(value.x),
            fabsf(value.y),
            fabsf(value.z)
        };
    }

    __host__ __device__
    inline vec3 floor(const vec3& value)
    {
        return {
            floorf(value.x),
            floorf(value.y),
            floorf(value.z)
        };
    }

    __host__ __device__
    inline vec3 ceil(const vec3& value)
    {
        return {
            ceilf(value.x),
            ceilf(value.y),
            ceilf(value.z)
        };
    }

    __host__ __device__
    inline vec3 round(const vec3& value)
    {
        return {
            roundf(value.x),
            roundf(value.y),
            roundf(value.z)
        };
    }

    __host__ __device__
    constexpr float component_sum(const vec3& value)
    {
        return value.x +
               value.y +
               value.z;
    }

    __host__ __device__
    constexpr float min_component(const vec3& value)
    {
        return value.x < value.y
            ? (value.x < value.z ? value.x : value.z)
            : (value.y < value.z ? value.y : value.z);
    }

    __host__ __device__
    constexpr float max_component(const vec3& value)
    {
        return value.x > value.y
            ? (value.x > value.z ? value.x : value.z)
            : (value.y > value.z ? value.y : value.z);
    }

    __host__ __device__
    inline bool nearly_equal(
        const vec3& lhs,
        const vec3& rhs,
        float epsilon = 1.0e-6f)
    {
        return
            fabsf(lhs.x - rhs.x) <= epsilon &&
            fabsf(lhs.y - rhs.y) <= epsilon &&
            fabsf(lhs.z - rhs.z) <= epsilon;
    }

    __host__ __device__
    constexpr vec3 reflect(
        const vec3& incident,
        const vec3& normal)
    {
        return incident -
               2.0f * dot(incident, normal) * normal;
    }

    __host__ __device__
    inline vec3 project(
        const vec3& value,
        const vec3& onto,
        float epsilon = 1.0e-6f)
    {
        const float denominator = length_squared(onto);

        if (denominator <= epsilon * epsilon)
            return {};

        return onto *
               (dot(value, onto) / denominator);
    }

    __host__ __device__
    inline vec3 reject(
        const vec3& value,
        const vec3& from,
        float epsilon = 1.0e-6f)
    {
        return value - project(value, from, epsilon);
    }


    // ============================================================
    // vec4 functions
    // ============================================================

    __host__ __device__
    constexpr float dot(const vec4& lhs, const vec4& rhs)
    {
        return lhs.dot(rhs);
    }

    __host__ __device__
    constexpr float length_squared(const vec4& value)
    {
        return value.length_squared();
    }

    __host__ __device__
    inline float length(const vec4& value)
    {
        return value.length();
    }

    __host__ __device__
    constexpr float distance_squared(const vec4& lhs, const vec4& rhs)
    {
        return lhs.distance_squared(rhs);
    }

    __host__ __device__
    inline float distance(const vec4& lhs, const vec4& rhs)
    {
        return lhs.distance(rhs);
    }

    __host__ __device__
    inline vec4 normalized(
        const vec4& value,
        float epsilon = 1.0e-6f)
    {
        return value.normalized(epsilon);
    }

    __host__ __device__
    constexpr vec4 lerp(
        const vec4& from,
        const vec4& to,
        float amount)
    {
        return from + (to - from) * amount;
    }

    __host__ __device__
    constexpr vec4 component_min(
        const vec4& lhs,
        const vec4& rhs)
    {
        return {
            lhs.x < rhs.x ? lhs.x : rhs.x,
            lhs.y < rhs.y ? lhs.y : rhs.y,
            lhs.z < rhs.z ? lhs.z : rhs.z,
            lhs.w < rhs.w ? lhs.w : rhs.w
        };
    }

    __host__ __device__
    constexpr vec4 component_max(
        const vec4& lhs,
        const vec4& rhs)
    {
        return {
            lhs.x > rhs.x ? lhs.x : rhs.x,
            lhs.y > rhs.y ? lhs.y : rhs.y,
            lhs.z > rhs.z ? lhs.z : rhs.z,
            lhs.w > rhs.w ? lhs.w : rhs.w
        };
    }

    __host__ __device__
    constexpr vec4 clamp(
        const vec4& value,
        const vec4& lower,
        const vec4& upper)
    {
        return component_min(
            component_max(value, lower),
            upper);
    }

    __host__ __device__
    constexpr vec4 clamp(
        const vec4& value,
        float lower,
        float upper)
    {
        return clamp(
            value,
            vec4(lower),
            vec4(upper));
    }

    __host__ __device__
    inline vec4 abs(const vec4& value)
    {
        return {
            fabsf(value.x),
            fabsf(value.y),
            fabsf(value.z),
            fabsf(value.w)
        };
    }

    __host__ __device__
    inline vec4 floor(const vec4& value)
    {
        return {
            floorf(value.x),
            floorf(value.y),
            floorf(value.z),
            floorf(value.w)
        };
    }

    __host__ __device__
    inline vec4 ceil(const vec4& value)
    {
        return {
            ceilf(value.x),
            ceilf(value.y),
            ceilf(value.z),
            ceilf(value.w)
        };
    }

    __host__ __device__
    inline vec4 round(const vec4& value)
    {
        return {
            roundf(value.x),
            roundf(value.y),
            roundf(value.z),
            roundf(value.w)
        };
    }

    __host__ __device__
    constexpr float component_sum(const vec4& value)
    {
        return value.x +
               value.y +
               value.z +
               value.w;
    }

    __host__ __device__
    constexpr float min_component(const vec4& value)
    {
        float result = value.x;

        result = value.y < result ? value.y : result;
        result = value.z < result ? value.z : result;
        result = value.w < result ? value.w : result;

        return result;
    }

    __host__ __device__
    constexpr float max_component(const vec4& value)
    {
        float result = value.x;

        result = value.y > result ? value.y : result;
        result = value.z > result ? value.z : result;
        result = value.w > result ? value.w : result;

        return result;
    }

    __host__ __device__
    inline bool nearly_equal(
        const vec4& lhs,
        const vec4& rhs,
        float epsilon = 1.0e-6f)
    {
        return
            fabsf(lhs.x - rhs.x) <= epsilon &&
            fabsf(lhs.y - rhs.y) <= epsilon &&
            fabsf(lhs.z - rhs.z) <= epsilon &&
            fabsf(lhs.w - rhs.w) <= epsilon;
    }

    __host__ __device__
    constexpr vec4 reflect(
        const vec4& incident,
        const vec4& normal)
    {
        return incident -
               2.0f * dot(incident, normal) * normal;
    }

    __host__ __device__
    inline vec4 project(
        const vec4& value,
        const vec4& onto,
        float epsilon = 1.0e-6f)
    {
        const float denominator = length_squared(onto);

        if (denominator <= epsilon * epsilon)
            return {};

        return onto *
               (dot(value, onto) / denominator);
    }

    __host__ __device__
    inline vec4 reject(
        const vec4& value,
        const vec4& from,
        float epsilon = 1.0e-6f)
    {
        return value - project(value, from, epsilon);
    }

} // namespace code_math#endif // CODECUDAPROJECT_CODEMATH_HPP
#endif

