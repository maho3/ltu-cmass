#include <cassert>
#include <cmath>
#include <cstdio>

#include "cuboid.h"

using namespace std;
using namespace cuboid;


void Cuboid::Initialize() {
    /* Assume that e1, e2, e3 are orthogonal (weird results if they aren't) */

    L1 = len(e1);
    L2 = len(e2);
    L3 = len(e3);
    n1 = e1/L1;
    n2 = e2/L2;
    n3 = e3/L3;
//    printf("e1 = (%g, %g, %g)\n", e1.x, e1.y, e1.z);
//    printf("e2 = (%g, %g, %g)\n", e2.x, e2.y, e2.z);
//    printf("e3 = (%g, %g, %g)\n", e3.x, e3.y, e3.z);

    v[0] = vec3d(0, 0, 0);
    v[1] = v[0] + e3;
    v[2] = v[0] + e2;
    v[3] = v[0] + e2 + e3;
    v[4] = v[0] + e1;
    v[5] = v[0] + e1 + e3;
    v[6] = v[0] + e1 + e2;
    v[7] = v[0] + e1 + e2 + e3;

    const double inf = 1e100;
    vec3d min = vec3d(+inf, +inf, +inf);
    vec3d max = vec3d(-inf, -inf, -inf);
    for(int k = 0; k < 8; k++) {
        if(v[k].x < min.x)
            min.x = v[k].x;
        if(v[k].x > max.x)
            max.x = v[k].x;
        if(v[k].y < min.y)
            min.y = v[k].y;
        if(v[k].y > max.y)
            max.y = v[k].y;
        if(v[k].z < min.z)
            min.z = v[k].z;
        if(v[k].z > max.z)
            max.z = v[k].z;
    }

    int ixmin = (int)floor(min.x);
    int ixmax = (int)ceil(max.x);
    int iymin = (int)floor(min.y);
    int iymax = (int)ceil(max.y);
    int izmin = (int)floor(min.z);
    int izmax = (int)ceil(max.z);

//    printf("ixmin, ixmax = %d, %d\n", ixmin, ixmax);
//    printf("iymin, iymax = %d, %d\n", iymin, iymax);
//    printf("izmin, izmax = %d, %d\n", izmin, izmax);

    /* Determine which cells (and which faces within those cells) are non-trivial */
    Cell c;
    vec3d shift;
    for(int ix = ixmin; ix < ixmax; ix++) {
        c.ix = ix;
        shift.x = -ix;
        for(int iy = iymin; iy < iymax; iy++) {
            c.iy = iy;
            shift.y = -iy;
            for(int iz = izmin; iz < izmax; iz++) {
                c.iz = iz;
                shift.z = -iz;

                /* Compute faces of cuboid shifted by (-ix,-iy,-iz) */
                c.face[0] = Plane(v[0] + shift, +n1);
                c.face[1] = Plane(v[4] + shift, -n1);
                c.face[2] = Plane(v[0] + shift, +n2);
                c.face[3] = Plane(v[2] + shift, -n2);
                c.face[4] = Plane(v[0] + shift, +n3);
                c.face[5] = Plane(v[1] + shift, -n3);

                /* Determine which faces actually define non-trivial regions
                 * within the fundamental cell */
                c.nfaces = 0;
                bool skipcell = false;
                for(int i = 0; i < 6; i++) {
                    switch(UnitCubeTest(c.face[i])) {
                    case +1:    // unit cube is trivially inside face; ignore face
                        break;
                    case 0:     // face intersects unit cube; keep face
                        c.face[c.nfaces++] = c.face[i];
                        break;
                    case -1:    // unit cube is outside face; ignore entire cell
                        skipcell = true;
                        break;
                    }
                }

                if(!skipcell) {
                    cells.push_back(c);
//                    printf("Adding cell at (%d,%d,%d)\n", ix, iy, iz);
                }
//                else
//                    printf("Skipping cell at (%d,%d,%d)\n", ix, iy, iz);
            }
        }
    }

    /* For debugging purposes, print the list of cells */
//    printf("%d cells\n", cells.size());
//    for(vector<Cell>::const_iterator iter = cells.begin(); iter != cells.end(); iter++) {
//        const Cell& c = *iter;
//        printf("Cell at (%d,%d,%d) has %d non-trivial planes\n", c.ix, c.iy, c.iz, c.nfaces);
//    }
}

void Cuboid::Initialize(const vec3i& u1, const vec3i& u2, const vec3i& u3) {
    if(tsp(u1, u2, u3) == 1) {
        double s1 = sqr(u1);
        double s2 = sqr(u2);
        double d12 = dot(u1, u2);
        double d23 = dot(u2, u3);
        double d13 = dot(u1, u3);
        double alpha = -d12/s1;
        double gamma = -(alpha*d13 + d23)/(alpha*d12 + s2);
        double beta = -(d13 + gamma*d12)/s1;
        e1 = vec3d(u1);
        e2 = vec3d(u2) + alpha*vec3d(u1);
        e3 = vec3d(u3) + beta*vec3d(u1) + gamma*vec3d(u2);
    }
    else {
        fprintf(stderr, "!! Invalid lattice vectors: u1 = (%d,%d,%d), u2 = (%d,%d,%d), u3 = (%d,%d,%d)\n", u1.x, u1.y, u1.z, u2.x, u2.y, u2.z, u3.x, u3.y, u3.z);
        e1 = vec3d(1,0,0);
        e2 = vec3d(0,1,0);
        e3 = vec3d(0,0,1);
    }
    Initialize();
}

Cuboid::Cuboid() {
    Initialize(vec3i(1,0,0), vec3i(0,1,0), vec3i(0,0,1));
}

Cuboid::Cuboid(int u11, int u12, int u13, int u21, int u22, int u23, int u31, int u32, int u33) {
    vec3i u1(u11, u12, u13);
    vec3i u2(u21, u22, u23);
    vec3i u3(u31, u32, u33);
    Initialize(u1, u2, u3);
}

Cuboid::Cuboid(int u[]) {
    vec3i u1(u[0], u[1], u[2]);
    vec3i u2(u[3], u[4], u[5]);
    vec3i u3(u[6], u[7], u[8]);
    Initialize(u1, u2, u3);
}

Cuboid::Cuboid(double m, double n) {
    e1 = vec3d((1 + n*n)/(1. + m*m + n*n), -m*n/(1. + m*m + n*n), -m/(1. + m*m + n*n));
    e2 = vec3d(0, 1/(1. + n*n), -n/(1. + n*n));
    e3 = vec3d(m, n, 1);
    Initialize();
}

vec3d Cuboid::Transform(const vec3d& x) const {
    vec3d r;
    Transform(x[0], x[1], x[2], r[0], r[1], r[2]);
    return r;
}

void Cuboid::Transform(double x1, double x2, double x3, double& r1, double& r2, double& r3) const {
    for(vector<Cell>::const_iterator iter = cells.begin(); iter != cells.end(); iter++) {
        const Cell& c = *iter;
        if(c.contains(x1, x2, x3)) {
            x1 += c.ix;
            x2 += c.iy;
            x3 += c.iz;
            vec3d p = vec3d(x1, x2, x3);
            r1 = dot(p, n1);
            r2 = dot(p, n2);
            r3 = dot(p, n3);
            return;
        }
    }

    fprintf(stderr, "!! point (%g,%g,%g) not contained by any cell\n", x1, x2, x3);
}

// LFT added, code adopted from Chang's
void Cuboid::VelocityTransform(double v1, double v2, double v3, double& r1, double& r2, double& r3) const {
    auto v = vec3(v1, v2, v3);
    r1 = dot(v, n1);
    r2 = dot(v, n2);
    r3 = dot(v, n3);
}

vec3d Cuboid::InverseTransform(const vec3d& r) const {
    vec3d x;
    InverseTransform(r[0], r[1], r[2], x[0], x[1], x[2]);
    return x;
}

void Cuboid::InverseTransform(double r1, double r2, double r3, double& x1, double& x2, double& x3) const {
    vec3d p = r1*n1 + r2*n2 + r3*n3;
    x1 = fmod(p[0], 1) + (p[0] < 0);
    x2 = fmod(p[1], 1) + (p[1] < 0);
    x3 = fmod(p[2], 1) + (p[2] < 0);
}

int Cuboid::UnitCubeTest(const Plane& P) {
    int above = 0;      // set to 1 if any of the unit cube's vertices lie above the plane
    int below = 0;      // set to 1 if any of the unit cube's vertices lie below the plane
    for(int a = 0; a < 2; a++) {
        for(int b = 0; b < 2; b++) {
            for(int c = 0; c < 2; c++) {
                double s = P.test(a, b, c);
                if(s > 0)
                    above = 1;
                else if(s < 0)
                    below = 1;
            }
        }
    }
    return above - below;
}


bool Cuboid::Cell::contains(const vec3d& p) const {
    return this->contains(p.x, p.y, p.z);
}

bool Cuboid::Cell::contains(double x, double y, double z) const {
//    for(int i = 0; i < nfaces; i++)
//        if(face[i].test(x, y, z) < 0)
//            return false;
//    return true;

    bool b = true;
    for(int i = 0; i < nfaces; i++)
        b = b && (face[i].test(x, y, z) >= 0);
    return b;
}
