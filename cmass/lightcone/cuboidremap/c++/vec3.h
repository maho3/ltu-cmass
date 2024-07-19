#ifndef VEC3_H
#define VEC3_H

namespace cuboid {

template<typename T>
struct vec3 {
    T x, y, z;

    vec3() {
        x = y = z = 0;
    }

    vec3(T x_, T y_, T z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    vec3(const T* v) {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    template<typename U>
    vec3(const vec3<U>& v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    template<typename U>
    vec3<T>& operator=(const vec3<U>& v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    vec3<T>& operator+=(const vec3<T>& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    vec3<T>& operator-=(const vec3<T>& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    vec3<T>& operator*=(T s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    vec3<T>& operator/=(T s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    T operator[](int i) const {
        return (&x)[i];
    }

    T& operator[](int i) {
        return (&x)[i];
    }

};

typedef vec3<int> vec3i;
typedef vec3<float> vec3f;
typedef vec3<double> vec3d;


#include <cmath>

template<typename T>
inline vec3<T> operator+(const vec3<T>& v) {
    return v;
}

template<typename T>
inline vec3<T> operator-(const vec3<T>& v) {
    return vec3<T>(-v.x, -v.y, -v.z);
}

template<typename T>
inline vec3<T> operator+(const vec3<T>& a, const vec3<T>& b) {
    return vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename T>
inline vec3<T> operator-(const vec3<T>& a, const vec3<T>& b) {
    return vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename T>
inline vec3<T> operator*(T s, const vec3<T>& v) {
    return vec3<T>(s*v.x, s*v.y, s*v.z);
}

template<typename T>
inline vec3<T> operator*(const vec3<T>& v, T s) {
    return vec3<T>(v.x*s, v.y*s, v.z*s);
}

template<typename T>
inline vec3<T> operator/(const vec3<T>& v, T s) {
    return vec3<T>(v.x/s, v.y/s, v.z/s);
}

template<typename T>
inline T dot(const vec3<T>& a, const vec3<T>& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<typename T>
inline vec3<T> cross(const vec3<T>& a, const vec3<T>& b) {
    return vec3<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

template<typename T>
inline T sqr(const vec3<T>& v) {
    return dot(v, v);
}

template<typename T>
inline T len(const vec3<T>& v) {
    return sqrt(sqr(v));
}

//template<typename T>
//void normalize(vec3<T>& v) {
//    T s = len(v);
//    v /= s;
//}

template<typename T>
inline vec3<T> normal(const vec3<T>& v) {
    return v/len(v);
}

/* Return the triple scalar product u . (v ^ w) */
template<typename T>
inline T tsp(const vec3<T>& u, const vec3<T>& v, const vec3<T>& w) {
    return u.x*(v.y*w.z - v.z*w.y) + u.y*(v.z*w.x - v.x*w.z) + u.z*(v.x*w.y - v.y*w.x);
}

} // namespace cuboid

#endif // VEC3_H
