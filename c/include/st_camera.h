/* st_camera.h — pixel coordinates to unit direction vectors. */
#ifndef ST_CAMERA_H
#define ST_CAMERA_H

#include "st_types.h"

/* Convert one star's pixel position to a unit direction vector in the camera
 * frame, undoing lens distortion on the way.
 *
 * The forward (physical) model is:
 *   xn = (u - cx) / fx                 <- ideal normalized coordinate
 *   r2 = xn^2 + yn^2
 *   xd = xn * (1 + k1*r2 + k2*r2^2)    <- what actually lands on the sensor
 * We observe xd and need xn, which has no closed form, so we iterate:
 *   xn <- xd / (1 + k1*r2 + k2*r2^2), with r2 recomputed from the current xn.
 * Five iterations converges to well below a pixel for any sane lens.
 *
 * Returns ST_OK, or ST_ERR_BAD_PARAM if fx or fy is zero.
 */
st_status_t st_pixel_to_vector(const st_camera_t *cam,
                               float u, float v,
                               st_vec3_t *out);

/* Convert a whole array of stars. out must have room for n vectors. */
st_status_t st_pixels_to_vectors(const st_camera_t *cam,
                                 const st_star_t *stars, int n,
                                 st_vec3_t *out);

#endif /* ST_CAMERA_H */
