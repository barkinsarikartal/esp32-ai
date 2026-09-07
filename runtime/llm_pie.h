// ESP32-P4 vector kernel for the staged int8 matvec, using the PIE / Xespv
// extension (TRM chapter 4). Same arithmetic as matvec_i8_range in llm.h:
// per row, per group, an exact integer dot product scaled by the group's
// float scale, then by the activation scale. Results are bit-identical to the
// scalar kernel, which the firmware checks once at boot.
//
// ESP.VMULAS.S8.XACC multiplies 16 int8 pairs and adds all 16 products into
// the 40-bit XACC accumulator in one instruction; the .LD.IP form also loads
// the next 16 weights. A group of 128 is therefore 8 such steps.
//
// Requirements, all arranged by the caller:
//   - llm.h included with LLM_STAGE_ALIGN 16, so t->stride8 is a multiple of
//     16 and padding bytes are zero;
//   - t->w8 16-byte aligned (heap_caps_aligned_alloc), so every row is;
//   - xq 16-byte aligned and zero from cols up to stride8;
//   - t->group a multiple of 16.
// The PIE unit runs in force-alignment mode by default: a misaligned address
// silently reads the aligned data below it, so these are not optional.
#ifndef LLM_PIE_H
#define LLM_PIE_H
#include <stdint.h>
#include "llm.h"

#include "llm_pie_dot.h"

#if LLM_HAVE_PIE
#define llm_pie_dot llm_pie_dot_s8v

// Drop-in for matvec_i8_range on a staged tensor that meets the requirements.
static void matvec_pie_range(const QT *t, const int8_t *xq, float x_scale,
                             float *y, int row_begin, int row_end) {
  int g = t->group, ng = t->n_groups, stride = t->stride8;
  int gvec = g / 16;                       // vectors per full group
  for (int r = row_begin; r < row_end; r++) {
    const int8_t *w = t->w8 + (size_t)r * stride;
    const float *sc = t->scale8 + (size_t)r * ng;
    float acc = 0.f;
    for (int gi = 0; gi < ng; gi++) {
      int begin = gi * g;
      int span = stride - begin;           // padded bytes left in the row
      int nvec = span < g ? span / 16 : gvec;
      acc += (float)llm_pie_dot(w + begin, xq + begin, nvec) * sc[gi];
    }
    y[r] = acc * x_scale;
  }
}

// Whether a staged tensor can go through the vector kernel.
static inline int llm_pie_ok(const QT *t) {
  return t->w8 != NULL && (t->stride8 % 16) == 0 && (t->group % 16) == 0 &&
         (((uintptr_t)t->w8) & 15) == 0;
}
#endif

#endif
