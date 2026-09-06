#include "st_camera.h"
#include <math.h>

#define ST_UNDISTORT_ITERS 5

st_status_t st_pixel_to_vector(const st_camera_t *cam,
                               float u, float v,
                               st_vec3_t *out)
{
    float xd, yd, xn, yn, r2, denom, norm;
    int i;

    if (!cam || !out) return ST_ERR_NULL;
    if (cam->fx == 0.0f || cam->fy == 0.0f) return ST_ERR_BAD_PARAM;

    /* Observed, distorted normalized coordinates. */
    xd = (u - cam->cx) / cam->fx;
    yd = (v - cam->cy) / cam->fy;

    /* Invert the radial distortion by fixed-point iteration. Starting from
     * the distorted point and repeatedly dividing by the distortion factor
     * evaluated at the current estimate converges quickly for the small
     * distortions typical of a star tracker lens. */
    xn = xd;
    yn = yd;
    for (i = 0; i < ST_UNDISTORT_ITERS; ++i) {
        r2 = xn * xn + yn * yn;
        denom = 1.0f + cam->k1 * r2 + cam->k2 * r2 * r2;
        if (denom < 1e-6f) denom = 1e-6f;   /* guard a pathological lens fit */
        xn = xd / denom;
        yn = yd / denom;
    }

    /* A normalized coordinate (xn, yn) corresponds to the ray (xn, yn, 1).
     * Normalizing to unit length is what makes the later dot products equal
     * the cosine of the angle between two stars. */
    norm = sqrtf(xn * xn + yn * yn + 1.0f);
    out->x = xn / norm;
    out->y = yn / norm;
    out->z = 1.0f / norm;
    return ST_OK;
}

st_status_t st_pixels_to_vectors(const st_camera_t *cam,
                                 const st_star_t *stars, int n,
                                 st_vec3_t *out)
{
    int i;
    st_status_t rc;
    if (!cam || !stars || !out) return ST_ERR_NULL;
    for (i = 0; i < n; ++i) {
        rc = st_pixel_to_vector(cam, stars[i].x, stars[i].y, &out[i]);
        if (rc != ST_OK) return rc;
    }
    return ST_OK;
}
