/* st_types.h — shared plain-data types.
 *
 * Convention used everywhere in this codebase:
 *   - All floating point is `float` (32-bit). The Raspberry Pi 5 has hardware
 *     single-precision; doubles are slower and buy nothing here.
 *   - Functions return an st_status_t. Zero means success.
 *   - No function allocates. Callers own all buffers.
 */
#ifndef ST_TYPES_H
#define ST_TYPES_H

#include <stdint.h>
#include "st_config.h"

/* Return codes. Any non-zero value means the caller should not trust the
 * outputs of that call. */
typedef enum {
    ST_OK              = 0,
    ST_ERR_NULL        = 1,  /* a required pointer argument was NULL        */
    ST_ERR_TOO_FEW     = 2,  /* not enough stars detected to form a feature */
    ST_ERR_OVERFLOW    = 3,  /* an internal fixed buffer filled up          */
    ST_ERR_BAD_PARAM   = 4   /* a parameter was out of its valid range      */
} st_status_t;

/* A detected star in image coordinates.
 *   x, y      : sub-pixel centroid, in pixels, origin at the top-left corner
 *               of pixel (0,0), measured to the CENTER of that pixel.
 *   flux      : summed above-background intensity. Used only for ranking.
 *   n_pixels  : how many pixels were in the blob. Used for sanity filtering.
 */
typedef struct {
    float    x;
    float    y;
    float    flux;
    uint16_t n_pixels;
} st_star_t;

/* A unit-length direction vector in the camera body frame.
 * +z points out along the boresight, +x along image +u, +y along image +v. */
typedef struct {
    float x;
    float y;
    float z;
} st_vec3_t;

/* Pinhole camera model with two radial distortion terms.
 *   fx, fy  : focal length in pixels along each image axis
 *   cx, cy  : principal point (where the boresight pierces the sensor), pixels
 *   k1, k2  : radial distortion coefficients (Brown-Conrady, radial only)
 * A star at normalized coordinate r from the center is observed at
 *   r_distorted = r * (1 + k1*r^2 + k2*r^4)
 * so recovering the true direction requires undistorting. */
typedef struct {
    float fx, fy;
    float cx, cy;
    float k1, k2;
} st_camera_t;

/* The output of the classifier stage: a shortlist of candidate sky cells. */
typedef struct {
    uint16_t cell_id[ST_MAX_CANDIDATES]; /* cell indices, best first        */
    float    confidence[ST_MAX_CANDIDATES];
    uint8_t  n;                          /* how many entries are valid      */
    uint8_t  rejected;                   /* 1 if nothing cleared threshold  */
} st_shortlist_t;

#endif /* ST_TYPES_H */
