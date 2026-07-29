//
// Created by carlo on 2026-07-29.
//

#ifndef CODECUDAPROJECT_CODEMATH_HPP
#define CODECUDAPROJECT_CODEMATH_HPP


namespace code_math
{
    struct vec2
    {
        float x = 0.0f;
        float y = 0.0f;

        __host__ __device__ constexpr vec2() {}
        __host__ __device__ constexpr vec2(float x, float y) : x(x), y(y) {}
    };

    struct vec3
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;

        __host__ __device__ constexpr vec3() {}
        __host__ __device__ explicit constexpr vec3(float value) : x(value), y(value), z(value) {}
        __host__ __device__ constexpr vec3(float x, float y, float z) : x(x), y(y), z(z) {}

        __host__ __device__ constexpr float &operator[](int index)
        {
            return index == 0 ? x : (index == 1 ? y : z);
        }

        __host__ __device__ constexpr const float &operator[](int index) const
        {
            return index == 0 ? x : (index == 1 ? y : z);
        }

        __host__ __device__ constexpr vec3 operator+() const { return *this; }
        __host__ __device__ constexpr vec3 operator-() const { return {-x, -y, -z}; }

        __host__ __device__ constexpr vec3 &operator+=(const vec3 &other)
        {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator-=(const vec3 &other)
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator*=(const vec3 &other)
        {
            x *= other.x;
            y *= other.y;
            z *= other.z;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator/=(const vec3 &other)
        {
            x /= other.x;
            y /= other.y;
            z /= other.z;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator+=(float scalar)
        {
            x += scalar;
            y += scalar;
            z += scalar;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator-=(float scalar)
        {
            x -= scalar;
            y -= scalar;
            z -= scalar;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator*=(float scalar)
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }

        __host__ __device__ constexpr vec3 &operator/=(float scalar)
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            return *this;
        }

        __host__ __device__ constexpr bool operator==(const vec3 &other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }

        __host__ __device__ constexpr bool operator!=(const vec3 &other) const { return !(*this == other); }

        __host__ __device__ constexpr float dot(const vec3 &other) const
        {
            return x * other.x + y * other.y + z * other.z;
        }

        __host__ __device__ constexpr vec3 cross(const vec3 &other) const
        {
            return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
        }

        __host__ __device__ constexpr float length_squared() const { return dot(*this); }
        __host__ __device__ float length() const { return sqrtf(length_squared()); }

        __host__ __device__ vec3 normalized(float epsilon = 1.0e-6f) const
        {
            const float squared = length_squared();
            if (squared <= epsilon * epsilon)
            {
                return {};
            }

            const float inverse_length = 1.0f / sqrtf(squared);
            return {x * inverse_length, y * inverse_length, z * inverse_length};
        }

        __host__ __device__ vec3 &normalize(float epsilon = 1.0e-6f)
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

        __host__ __device__ constexpr float distance_squared(const vec3 &other) const
        {
            const float dx = x - other.x;
            const float dy = y - other.y;
            const float dz = z - other.z;
            return dx * dx + dy * dy + dz * dz;
        }

        __host__ __device__ float distance(const vec3 &other) const { return sqrtf(distance_squared(other)); }
    };

    __host__ __device__ constexpr vec3 operator+(vec3 lhs, const vec3 &rhs) { return lhs += rhs; }
    __host__ __device__ constexpr vec3 operator-(vec3 lhs, const vec3 &rhs) { return lhs -= rhs; }
    __host__ __device__ constexpr vec3 operator*(vec3 lhs, const vec3 &rhs) { return lhs *= rhs; }
    __host__ __device__ constexpr vec3 operator/(vec3 lhs, const vec3 &rhs) { return lhs /= rhs; }

    __host__ __device__ constexpr vec3 operator+(vec3 value, float scalar) { return value += scalar; }
    __host__ __device__ constexpr vec3 operator-(vec3 value, float scalar) { return value -= scalar; }
    __host__ __device__ constexpr vec3 operator*(vec3 value, float scalar) { return value *= scalar; }
    __host__ __device__ constexpr vec3 operator/(vec3 value, float scalar) { return value /= scalar; }

    __host__ __device__ constexpr vec3 operator+(float scalar, vec3 value) { return value += scalar; }
    __host__ __device__ constexpr vec3 operator-(float scalar, const vec3 &value)
    {
        return {scalar - value.x, scalar - value.y, scalar - value.z};
    }
    __host__ __device__ constexpr vec3 operator*(float scalar, vec3 value) { return value *= scalar; }
    __host__ __device__ constexpr vec3 operator/(float scalar, const vec3 &value)
    {
        return {scalar / value.x, scalar / value.y, scalar / value.z};
    }

    __host__ __device__ constexpr float dot(const vec3 &lhs, const vec3 &rhs) { return lhs.dot(rhs); }
    __host__ __device__ constexpr vec3 cross(const vec3 &lhs, const vec3 &rhs) { return lhs.cross(rhs); }
    __host__ __device__ constexpr float length_squared(const vec3 &value) { return value.length_squared(); }
    __host__ __device__ inline float length(const vec3 &value) { return value.length(); }
    __host__ __device__ constexpr float distance_squared(const vec3 &lhs, const vec3 &rhs)
    {
        return lhs.distance_squared(rhs);
    }
    __host__ __device__ inline float distance(const vec3 &lhs, const vec3 &rhs) { return lhs.distance(rhs); }
    __host__ __device__ inline vec3 normalized(const vec3 &value, float epsilon = 1.0e-6f)
    {
        return value.normalized(epsilon);
    }

    __host__ __device__ constexpr vec3 lerp(const vec3 &from, const vec3 &to, float amount)
    {
        return from + (to - from) * amount;
    }

    __host__ __device__ constexpr vec3 component_min(const vec3 &lhs, const vec3 &rhs)
    {
        return {lhs.x < rhs.x ? lhs.x : rhs.x, lhs.y < rhs.y ? lhs.y : rhs.y, lhs.z < rhs.z ? lhs.z : rhs.z};
    }

    __host__ __device__ constexpr vec3 component_max(const vec3 &lhs, const vec3 &rhs)
    {
        return {lhs.x > rhs.x ? lhs.x : rhs.x, lhs.y > rhs.y ? lhs.y : rhs.y, lhs.z > rhs.z ? lhs.z : rhs.z};
    }

    __host__ __device__ constexpr vec3 clamp(const vec3 &value, const vec3 &lower, const vec3 &upper)
    {
        return component_min(component_max(value, lower), upper);
    }

    __host__ __device__ constexpr vec3 clamp(const vec3 &value, float lower, float upper)
    {
        return clamp(value, vec3(lower), vec3(upper));
    }

    __host__ __device__ inline vec3 abs(const vec3 &value)
    {
        return {fabsf(value.x), fabsf(value.y), fabsf(value.z)};
    }

    __host__ __device__ inline vec3 floor(const vec3 &value)
    {
        return {floorf(value.x), floorf(value.y), floorf(value.z)};
    }

    __host__ __device__ inline vec3 ceil(const vec3 &value)
    {
        return {ceilf(value.x), ceilf(value.y), ceilf(value.z)};
    }

    __host__ __device__ inline vec3 round(const vec3 &value)
    {
        return {roundf(value.x), roundf(value.y), roundf(value.z)};
    }

    __host__ __device__ constexpr float component_sum(const vec3 &value) { return value.x + value.y + value.z; }

    __host__ __device__ constexpr float min_component(const vec3 &value)
    {
        return value.x < value.y ? (value.x < value.z ? value.x : value.z)
                                 : (value.y < value.z ? value.y : value.z);
    }

    __host__ __device__ constexpr float max_component(const vec3 &value)
    {
        return value.x > value.y ? (value.x > value.z ? value.x : value.z)
                                 : (value.y > value.z ? value.y : value.z);
    }

    __host__ __device__ inline bool nearly_equal(const vec3 &lhs, const vec3 &rhs, float epsilon = 1.0e-6f)
    {
        return fabsf(lhs.x - rhs.x) <= epsilon && fabsf(lhs.y - rhs.y) <= epsilon &&
               fabsf(lhs.z - rhs.z) <= epsilon;
    }

    __host__ __device__ constexpr vec3 reflect(const vec3 &incident, const vec3 &normal)
    {
        return incident - 2.0f * dot(incident, normal) * normal;
    }

    __host__ __device__ inline vec3 project(const vec3 &value, const vec3 &onto, float epsilon = 1.0e-6f)
    {
        const float denominator = length_squared(onto);
        if (denominator <= epsilon * epsilon)
        {
            return {};
        }

        return onto * (dot(value, onto) / denominator);
    }

    __host__ __device__ inline vec3 reject(const vec3 &value, const vec3 &from, float epsilon = 1.0e-6f)
    {
        return value - project(value, from, epsilon);
    }

    struct vec4
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;

        __host__ __device__ constexpr vec4() {}
        __host__ __device__ constexpr vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    };
} // namespace code_math
#endif // CODECUDAPROJECT_CODEMATH_HPP
