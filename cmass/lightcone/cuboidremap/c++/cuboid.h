#ifndef CUBOID_H
#define CUBOID_H

#include <vector>

#include "vec3.h"

namespace cuboid {

/* Plane defined by the relation $a*x + b*y + c*z + d = 0$. */
struct Plane {
    double a, b, c, d;

    /* Default to the x-y plane */
    Plane() {
        a = 0;
        b = 0;
        c = 1;
        d = 0;
    }

    Plane(double a_, double b_, double c_, double d_) {
        a = a_;
        b = b_;
        c = c_;
        d = d_;
    }

    /* Construct plane from a point and a normal vector */
    Plane(const vec3d& p, const vec3d& n) {
        a = n.x;
        b = n.y;
        c = n.z;
        d = -dot(p,n);
    }

    vec3d normal() const {
        return vec3d(a, b, c);
    }

    double test(double x, double y, double z) const {
        return a*x + b*y + c*z + d;
    }

    double test(const vec3d& r) const {
        return a*r.x + b*r.y + c*r.z + d;
    }
};

/* Cuboid remapping */
class Cuboid {
public:
    /* Default to the identity remapping */
    Cuboid();

    /* Initialize the remapping by passing an invertible integer matrix */
    Cuboid(const vec3i& u1, const vec3i& u2, const vec3i& u3);
    Cuboid(int u11, int u12, int u13, int u21, int u22, int u23, int u31, int u32, int u33);
    Cuboid(int u[]);

    /* For backwards compatibility... */
    Cuboid(double m, double n);


    /* Transform the point x in the unit cube [0,1]^3 to local coordinates in
     * the fiducial cuboid [0,L1]x[0,L2]x[0,L3] */
    vec3d Transform(const vec3d& x) const;
    void Transform(double x1, double x2, double x3, double& r1, double& r2, double& r3) const;

    // LFT added
    void VelocityTransform(double v1, double v2, double v3, double& r1, double& r2, double& r3) const;

    /* Transform the point r in the cuboid [0,L1]x[0,L2]x[0,L3] back to the
     * unit cube */
    vec3d InverseTransform(const vec3d& r) const;
    void InverseTransform(double r1, double r2, double r3, double& x1, double& x2, double& x3) const;

    /* Note: these maps should satisfy
     *   x == InverseTransform(Transform(x)) && r == Transform(InverseTransform(r))
     * to machine precision. */


    vec3d e1, e2, e3;   // vectors along the 3 primary directions
    vec3d n1, n2, n3;   // normal vectors along these directions
    double L1, L2, L3;  // dimensions of the cuboid
    vec3d v[8];         // 8 vertices [$v_{abc}$ <--> v[k] with k = (a << 2 ) + (b << 1) + c]

    /* A 'cell' represents the intersection of the oriented cuboid with a
     * particular replication of the unit cube.  The integers (ix,iy,iz) define
     * which replication [e.g. (0,0,0) represents the canonical unit cube], and
     * the list of planes defines the region (these are just the faces of the
     * oriented cuboid translated by (-ix,-iy,-iz)). */
    struct Cell {
        int ix, iy, iz;
        Plane face[6];
        int nfaces;

        bool contains(const vec3d& p) const;
        bool contains(double x, double y, double z) const;
    };

    std::vector<Cell> cells;

protected:
    /* Return +1, -1, or 0 if the unit cube is above, below, or intersecting the plane */
    static int UnitCubeTest(const Plane& P);

    /* Initialize cuboid once e1, e2, e3 are set */
    void Initialize();

    /* Initialize from lattice vectors */
    void Initialize(const vec3i& u1, const vec3i& u2, const vec3i& u3);
};

} // namespace cuboid

#endif // CUBOID_H
