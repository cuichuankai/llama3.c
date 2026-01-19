/* Inference for Llama-3 Transformer model in pure C, int8 quantized forward pass. */

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 4096 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  int8_t *q; // quantized values
  float *s;  // scaling factors
} QuantizedTensor;

typedef struct {
  // token embedding table
  QuantizedTensor *q_tokens;    // (vocab_size, dim)
  float *token_embedding_table; // same, but dequantized

  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
  QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  QuantizedTensor *w1; // (layer, hidden_dim, dim)
  QuantizedTensor *w2; // (layer, dim, hidden_dim)
  QuantizedTensor *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;           // activation at current time stamp (dim,)
  float *xb;          // same, but inside a residual branch (dim,)
  float *xb2;         // an additional buffer just for convenience (dim,)
  float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
  QuantizedTensor xq; // quantized x (dim,)
  QuantizedTensor hq; // quantized hb (hidden_dim,)
  float *q;           // query (dim,)
  float *k;           // key (dim,)
  float *v;           // value (dim,)
  float *att;         // buffer for scores/attention values (n_heads, seq_len)
  float *logits;      // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->xq = (QuantizedTensor){.q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float))};
  s->hq = (QuantizedTensor){.q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float))};
  s->q = calloc(p->dim, sizeof(float));
  s->k = calloc(kv_dim, sizeof(float));
  s->v = calloc(kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->xq.q);
  free(s->xq.s);
  free(s->hq.q);
  free(s->hq.s);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
  int num_groups = n / GS;
  for (int group = 0; group < num_groups; group++) {
    float scale = qx->s[group];
    int i = 0;
#if defined(__aarch64__)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 16 <= GS; i += 16) {
      int8x16_t b = vld1q_s8(qx->q + group * GS + i);
      
      int16x8_t b_low = vmovl_s8(vget_low_s8(b));
      int16x8_t b_high = vmovl_s8(vget_high_s8(b));
      
      int32x4_t b_low_low = vmovl_s16(vget_low_s16(b_low));
      int32x4_t b_low_high = vmovl_s16(vget_high_s16(b_low));
      int32x4_t b_high_low = vmovl_s16(vget_low_s16(b_high));
      int32x4_t b_high_high = vmovl_s16(vget_high_s16(b_high));
      
      float32x4_t f0 = vcvtq_f32_s32(b_low_low);
      float32x4_t f1 = vcvtq_f32_s32(b_low_high);
      float32x4_t f2 = vcvtq_f32_s32(b_high_low);
      float32x4_t f3 = vcvtq_f32_s32(b_high_high);
      
      vst1q_f32(x + group * GS + i, vmulq_f32(f0, vscale));
      vst1q_f32(x + group * GS + i + 4, vmulq_f32(f1, vscale));
      vst1q_f32(x + group * GS + i + 8, vmulq_f32(f2, vscale));
      vst1q_f32(x + group * GS + i + 12, vmulq_f32(f3, vscale));
    }
#endif
    for (; i < GS; i++) {
      x[group * GS + i] = qx->q[group * GS + i] * scale;
    }
  }
}

void quantize(QuantizedTensor *qx, float *x, int n) {
  int num_groups = n / GS;
  float Q_MAX = 127.0f;

  for (int group = 0; group < num_groups; group++) {

    // find the max absolute value in the current group
    float wmax = 0.0f;
    int i = 0;
#if defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (; i + 16 <= GS; i += 16) {
      float32x4_t v0 = vld1q_f32(x + group * GS + i);
      float32x4_t v1 = vld1q_f32(x + group * GS + i + 4);
      float32x4_t v2 = vld1q_f32(x + group * GS + i + 8);
      float32x4_t v3 = vld1q_f32(x + group * GS + i + 12);
      vmax = vmaxq_f32(vmax, vabsq_f32(v0));
      vmax = vmaxq_f32(vmax, vabsq_f32(v1));
      vmax = vmaxq_f32(vmax, vabsq_f32(v2));
      vmax = vmaxq_f32(vmax, vabsq_f32(v3));
    }
    float max_val = vgetq_lane_f32(vmax, 0);
    max_val = fmaxf(max_val, vgetq_lane_f32(vmax, 1));
    max_val = fmaxf(max_val, vgetq_lane_f32(vmax, 2));
    max_val = fmaxf(max_val, vgetq_lane_f32(vmax, 3));
    if (max_val > wmax) wmax = max_val;
#endif
    for (; i < GS; i++) {
      float val = fabs(x[group * GS + i]);
      if (val > wmax) {
        wmax = val;
      }
    }

    // calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    qx->s[group] = scale;

    // calculate and write the quantized values
    if (scale == 0.0f) {
      memset(qx->q + group * GS, 0, GS * sizeof(int8_t));
    } else {
      float inv_scale = 1.0f / scale;
      i = 0;
#if defined(__aarch64__)
      float32x4_t vinv = vdupq_n_f32(inv_scale);
      for (; i + 16 <= GS; i += 16) {
        float32x4_t v0 = vld1q_f32(x + group * GS + i);
        float32x4_t v1 = vld1q_f32(x + group * GS + i + 4);
        float32x4_t v2 = vld1q_f32(x + group * GS + i + 8);
        float32x4_t v3 = vld1q_f32(x + group * GS + i + 12);

        v0 = vmulq_f32(v0, vinv);
        v1 = vmulq_f32(v1, vinv);
        v2 = vmulq_f32(v2, vinv);
        v3 = vmulq_f32(v3, vinv);

        int32x4_t vi0 = vcvtaq_s32_f32(v0);
        int32x4_t vi1 = vcvtaq_s32_f32(v1);
        int32x4_t vi2 = vcvtaq_s32_f32(v2);
        int32x4_t vi3 = vcvtaq_s32_f32(v3);

        int16x4_t vn0 = vqmovn_s32(vi0); // narrow to 16-bit
        int16x4_t vn1 = vqmovn_s32(vi1); // narrow to 16-bit
        int16x4_t vn2 = vqmovn_s32(vi2); // narrow to 16-bit
        int16x4_t vn3 = vqmovn_s32(vi3); // narrow to 16-bit

        int16x8_t vq0 = vcombine_s16(vn0, vn1);
        int16x8_t vq1 = vcombine_s16(vn2, vn3);
        
        int8x8_t vb0 = vqmovn_s16(vq0);
        int8x8_t vb1 = vqmovn_s16(vq1);
        
        int8x16_t vfinal = vcombine_s8(vb0, vb1);
        vst1q_s8(qx->q + group * GS + i, vfinal);
      }
#endif
      for (; i < GS; i++) {
        float quant_value = x[group * GS + i] * inv_scale;
        int8_t quantized = (int8_t)roundf(quant_value);
        qx->q[group * GS + i] = quantized;
      }
    }
  }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
  void *p = *ptr;
  QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
  for (int i = 0; i < n; i++) {
    /* map quantized int8 values*/
    res[i].q = (int8_t *)p;
    p = (int8_t *)p + size_each;
    /* map scale factors */
    res[i].s = (float *)p;
    p = (float *)p + size_each / GS;
  }
  *ptr = p; // advance ptr to current position
  return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr, uint8_t shared_classifier) {
  int head_size = p->dim / p->n_heads;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr; // cast our pointer to float*
  w->rms_att_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_ffn_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_final_weight = fptr;
  fptr += p->dim;

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
  // dequantize token embedding table
  w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
  dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

  w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
  w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
  w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
  w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

  w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
  w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
  w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

  w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 2
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // read in flags
  uint8_t shared_classifier; // a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  int group_size; // the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  GS = group_size; // set as global, as it will be used in many places
                   // figure out the file size
#if defined _WIN32
  _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = _ftelli64(file); // get the file size, in bytes
#else
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
#endif
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  void *weights_ptr = ((char *)*data) + header_size; // skip header bytes. char is 1 byte
  memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  // free QuantizedTensors
  free(t->weights.q_tokens);
  free(t->weights.token_embedding_table);
  free(t->weights.wq);
  free(t->weights.wk);
  free(t->weights.wv);
  free(t->weights.wo);
  free(t->weights.w1);
  free(t->weights.w2);
  free(t->weights.w3);
  if (t->weights.wcls != t->weights.q_tokens) {
    free(t->weights.wcls);
  }
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
  
#if defined(__aarch64__)
  int i = 0;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  for (; i + 16 <= size; i += 16) {
    float32x4_t x0 = vld1q_f32(x + i);
    float32x4_t x1 = vld1q_f32(x + i + 4);
    float32x4_t x2 = vld1q_f32(x + i + 8);
    float32x4_t x3 = vld1q_f32(x + i + 12);
    acc0 = vmlaq_f32(acc0, x0, x0);
    acc1 = vmlaq_f32(acc1, x1, x1);
    acc2 = vmlaq_f32(acc2, x2, x2);
    acc3 = vmlaq_f32(acc3, x3, x3);
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + 4 <= size; i += 4) {
    float32x4_t x0 = vld1q_f32(x + i);
    acc = vmlaq_f32(acc, x0, x0);
  }
  float ss = vaddvq_f32(acc);
  for (; i < size; i++) {
    ss += x[i] * x[i];
  }
  ss /= size;
  ss += 1e-5f;
  float inv = 1.0f / sqrtf(ss);
  i = 0;
  float32x4_t vinv = vdupq_n_f32(inv);
  for (; i + 16 <= size; i += 16) {
    float32x4_t x0 = vld1q_f32(x + i);
    float32x4_t w0 = vld1q_f32(weight + i);
    float32x4_t x1 = vld1q_f32(x + i + 4);
    float32x4_t w1 = vld1q_f32(weight + i + 4);
    float32x4_t x2 = vld1q_f32(x + i + 8);
    float32x4_t w2 = vld1q_f32(weight + i + 8);
    float32x4_t x3 = vld1q_f32(x + i + 12);
    float32x4_t w3 = vld1q_f32(weight + i + 12);
    vst1q_f32(o + i, vmulq_f32(w0, vmulq_f32(vinv, x0)));
    vst1q_f32(o + i + 4, vmulq_f32(w1, vmulq_f32(vinv, x1)));
    vst1q_f32(o + i + 8, vmulq_f32(w2, vmulq_f32(vinv, x2)));
    vst1q_f32(o + i + 12, vmulq_f32(w3, vmulq_f32(vinv, x3)));
  }
  for (; i + 4 <= size; i += 4) {
    float32x4_t xv = vld1q_f32(x + i);
    float32x4_t wv = vld1q_f32(weight + i);
    vst1q_f32(o + i, vmulq_f32(wv, vmulq_f32(vinv, xv)));
  }
  for (; i < size; i++) {
    o[i] = weight[i] * (inv * x[i]);
  }
#else
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
#endif
}

static inline float fast_exp(float x) {
  if (x < -80.0f) return 0.0f;
  if (x > 80.0f) x = 80.0f;
  const float INV_LN2 = 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;
  float y = x * INV_LN2;
  float n = floorf(y + 0.5f);
  float f = y - n;
  float z = f * LN2;
  float p = 1.0f + z * (1.0f + z * (0.5f + z * (0.1666666716f + z * (0.0416666679f + z * 0.0083333338f))));
  return ldexpf(p, (int)n);
}

static inline float fast_sigmoid(float x) {
  return 1.0f / (1.0f + fast_exp(-x));
}

void softmax(float *x, int size) {
  float max_val;
#if defined(__aarch64__)
  int i = 0;
  float32x4_t vmaxv = vdupq_n_f32(-FLT_MAX);
  for (; i + 16 <= size; i += 16) {
    float32x4_t xv0 = vld1q_f32(x + i);
    float32x4_t xv1 = vld1q_f32(x + i + 4);
    float32x4_t xv2 = vld1q_f32(x + i + 8);
    float32x4_t xv3 = vld1q_f32(x + i + 12);
    vmaxv = vmaxq_f32(vmaxv, xv0);
    vmaxv = vmaxq_f32(vmaxv, xv1);
    vmaxv = vmaxq_f32(vmaxv, xv2);
    vmaxv = vmaxq_f32(vmaxv, xv3);
  }
  for (; i + 4 <= size; i += 4) {
    float32x4_t xv = vld1q_f32(x + i);
    vmaxv = vmaxq_f32(vmaxv, xv);
  }
  max_val = vgetq_lane_f32(vmaxv, 0);
  max_val = fmaxf(max_val, vgetq_lane_f32(vmaxv, 1));
  max_val = fmaxf(max_val, vgetq_lane_f32(vmaxv, 2));
  max_val = fmaxf(max_val, vgetq_lane_f32(vmaxv, 3));
  for (; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }
  float sum = 0.0f;
  for (i = 0; i < size; i++) {
    x[i] = fast_exp(x[i] - max_val);
    sum += x[i];
  }
  float32x4_t vsum = vdupq_n_f32(sum);
  i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t xv = vld1q_f32(x + i);
    xv = vdivq_f32(xv, vsum);
    vst1q_f32(x + i, xv);
  }
  for (; i < size; i++) {
    x[i] /= sum;
  }
#else
  max_val = x[0];
  for (int j = 1; j < size; j++) {
    if (x[j] > max_val) max_val = x[j];
  }
  float sum = 0.0f;
  for (int j = 0; j < size; j++) {
    x[j] = fast_exp(x[j] - max_val);
    sum += x[j];
  }
  for (int j = 0; j < size; j++) {
    x[j] /= sum;
  }
#endif
}

#if defined(__aarch64__)
static inline int32_t neon_dot_s8_block16(const int8_t *a, const int8_t *b) {
  int32x4_t acc = vdupq_n_s32(0);
  int8x16_t va = vld1q_s8(a);
  int8x16_t vb = vld1q_s8(b);
  acc = vdotq_s32(acc, va, vb);
  return vaddvq_s32(acc);
}
#endif

void matmul(float *__restrict xout, QuantizedTensor *__restrict x, QuantizedTensor *__restrict w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  int i;
#pragma omp parallel for schedule(static) private(i)
  for (i = 0; i < d; i++) {

    float val = 0.0f;
    int32_t ival = 0;
    int in = i * n;

    // do the matmul in groups of GS
    int j;
    for (j = 0; j <= n - GS; j += GS) {
      float scale = w->s[(in + j) / GS] * x->s[j / GS];
      __builtin_prefetch(&w->q[in + j + GS], 0, 0); // Prefetch next block of weights

#if defined(__aarch64__)
      int k = 0;
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc3 = vdupq_n_s32(0);
      
      for (; k + 64 <= GS; k += 64) {
        int8x16_t va0 = vld1q_s8(x->q + j + k);
        int8x16_t vb0 = vld1q_s8(w->q + in + j + k);
        vacc0 = vdotq_s32(vacc0, va0, vb0);
        
        int8x16_t va1 = vld1q_s8(x->q + j + k + 16);
        int8x16_t vb1 = vld1q_s8(w->q + in + j + k + 16);
        vacc1 = vdotq_s32(vacc1, va1, vb1);
        
        int8x16_t va2 = vld1q_s8(x->q + j + k + 32);
        int8x16_t vb2 = vld1q_s8(w->q + in + j + k + 32);
        vacc2 = vdotq_s32(vacc2, va2, vb2);
        
        int8x16_t va3 = vld1q_s8(x->q + j + k + 48);
        int8x16_t vb3 = vld1q_s8(w->q + in + j + k + 48);
        vacc3 = vdotq_s32(vacc3, va3, vb3);
      }
      
      for (; k + 16 <= GS; k += 16) {
        int8x16_t va = vld1q_s8(x->q + j + k);
        int8x16_t vb = vld1q_s8(w->q + in + j + k);
        vacc0 = vdotq_s32(vacc0, va, vb);
      }
      
      vacc0 = vaddq_s32(vacc0, vacc1);
      vacc2 = vaddq_s32(vacc2, vacc3);
      ival = vaddvq_s32(vaddq_s32(vacc0, vacc2));
      
      for (; k < GS; k++) {
        ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
      }
#else
      for (int k = 0; k < GS; k++) {
        ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
      }
#endif
      val += ((float)ival) * scale;
      ival = 0;
    }

    xout[i] = val;
  }
}

float *forward(Transformer *transformer, int token, int pos) {

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {

    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // qkv matmuls for this position
    quantize(&s->xq, s->xb, dim);
    matmul(s->q, &s->xq, w->wq + l, dim, dim);
    matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
    matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // Precompute cos/sin for this position
    float fcr_cache[head_size];
    float fci_cache[head_size];
    for (int j = 0; j < head_size; j += 2) {
      float freq = 1.0f / powf(500000.0f, (float)j / (float)head_size);
      float val = pos * freq;
      float c = cosf(val);
      float s_val = sinf(val);
      fcr_cache[j] = c;
      fcr_cache[j + 1] = c;
      fci_cache[j] = s_val;
      fci_cache[j + 1] = s_val;
    }

    for (int i = 0; i < p->n_heads; i++) {
      int j = 0;
#if defined(__aarch64__)
      for (; j + 8 <= head_size; j += 8) {
         // Process 4 pairs (8 elements) at a time
         // Load q
         float32x4x2_t q_vec = vld2q_f32(s->q + i * head_size + j);
         float32x4_t q_real = q_vec.val[0];
         float32x4_t q_imag = q_vec.val[1];
         
         // Load cos/sin
         // fcr_cache has duplicates: c0, c0, c2, c2...
         // We need c0, c2, c4, c6 for the vector op?
         // vld2q_f32 on cache will separate evens and odds.
         // Since evens == odds, both val[0] and val[1] will be c0, c2, c4, c6.
         // Perfect.
         float32x4x2_t fcr_vec = vld2q_f32(fcr_cache + j);
         float32x4x2_t fci_vec = vld2q_f32(fci_cache + j);
         float32x4_t fcr = fcr_vec.val[0];
         float32x4_t fci = fci_vec.val[0];
         
         // Rotate q
         // q_real_new = q_real * fcr - q_imag * fci
         // q_imag_new = q_real * fci + q_imag * fcr
         float32x4_t q_real_new = vsubq_f32(vmulq_f32(q_real, fcr), vmulq_f32(q_imag, fci));
         float32x4_t q_imag_new = vaddq_f32(vmulq_f32(q_real, fci), vmulq_f32(q_imag, fcr));
         
         q_vec.val[0] = q_real_new;
         q_vec.val[1] = q_imag_new;
         vst2q_f32(s->q + i * head_size + j, q_vec);
         
         if (i < p->n_kv_heads) {
             float32x4x2_t k_vec = vld2q_f32(s->k + i * head_size + j);
             float32x4_t k_real = k_vec.val[0];
             float32x4_t k_imag = k_vec.val[1];
             
             float32x4_t k_real_new = vsubq_f32(vmulq_f32(k_real, fcr), vmulq_f32(k_imag, fci));
             float32x4_t k_imag_new = vaddq_f32(vmulq_f32(k_real, fci), vmulq_f32(k_imag, fcr));
             
             k_vec.val[0] = k_real_new;
             k_vec.val[1] = k_imag_new;
             vst2q_f32(s->k + i * head_size + j, k_vec);
         }
      }
#endif
      for (; j < head_size; j += 2) {
        float fcr = fcr_cache[j];
        float fci = fci_cache[j];
        float q0 = s->q[i * head_size + j];
        float q1 = s->q[i * head_size + j + 1];
        s->q[i * head_size + j] = q0 * fcr - q1 * fci;
        s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
        if (i < p->n_kv_heads) {
          float k0 = s->k[i * head_size + j];
          float k1 = s->k[i * head_size + j + 1];
          s->k[i * head_size + j] = k0 * fcr - k1 * fci;
          s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
        }
      }
    }

    // save key,value at this time step (pos) to our kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    float *key_cache_row = s->key_cache + loff + pos * kv_dim;
    float *value_cache_row = s->value_cache + loff + pos * kv_dim;
    memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      int t = 0;
#if defined(__aarch64__)
      for (; t <= pos - 3; t += 4) {
          float *k0 = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
          float *k1 = s->key_cache + loff + (t+1) * kv_dim + (h / kv_mul) * head_size;
          float *k2 = s->key_cache + loff + (t+2) * kv_dim + (h / kv_mul) * head_size;
          float *k3 = s->key_cache + loff + (t+3) * kv_dim + (h / kv_mul) * head_size;
          
          float32x4_t vacc0 = vdupq_n_f32(0.0f);
          float32x4_t vacc1 = vdupq_n_f32(0.0f);
          float32x4_t vacc2 = vdupq_n_f32(0.0f);
          float32x4_t vacc3 = vdupq_n_f32(0.0f);
          
          int vi = 0;
          for (; vi + 4 <= head_size; vi += 4) {
              float32x4_t qv = vld1q_f32(q + vi);
              
              float32x4_t kv0 = vld1q_f32(k0 + vi);
              float32x4_t kv1 = vld1q_f32(k1 + vi);
              float32x4_t kv2 = vld1q_f32(k2 + vi);
              float32x4_t kv3 = vld1q_f32(k3 + vi);
              
              vacc0 = vmlaq_f32(vacc0, qv, kv0);
              vacc1 = vmlaq_f32(vacc1, qv, kv1);
              vacc2 = vmlaq_f32(vacc2, qv, kv2);
              vacc3 = vmlaq_f32(vacc3, qv, kv3);
          }
          
          float score0 = vaddvq_f32(vacc0);
          float score1 = vaddvq_f32(vacc1);
          float score2 = vaddvq_f32(vacc2);
          float score3 = vaddvq_f32(vacc3);
          
          for (; vi < head_size; vi++) {
              float qval = q[vi];
              score0 += qval * k0[vi];
              score1 += qval * k1[vi];
              score2 += qval * k2[vi];
              score3 += qval * k3[vi];
          }
          
          float scale = 1.0f / sqrtf(head_size);
          att[t] = score0 * scale;
          att[t+1] = score1 * scale;
          att[t+2] = score2 * scale;
          att[t+3] = score3 * scale;
      }
#endif
      for (; t <= pos; t++) {
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
#if defined(__aarch64__)
        int vi = 0;
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; vi + 4 <= head_size; vi += 4) {
          float32x4_t qv = vld1q_f32(q + vi);
          float32x4_t kv = vld1q_f32(k + vi);
          vacc = vmlaq_f32(vacc, qv, kv);
        }
        score = vaddvq_f32(vacc);
        for (; vi < head_size; vi++) {
          score += q[vi] * k[vi];
        }
#else
        int vi = 0;
        for (; vi < head_size; vi++) {
          score += q[vi] * k[vi];
        }
#endif
        score /= sqrtf(head_size);
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      t = 0;
#if defined(__aarch64__)
      // loop unrolling for weighted sum
      for (; t <= pos - 3; t += 4) {
        float *v0 = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float *v1 = s->value_cache + loff + (t + 1) * kv_dim + (h / kv_mul) * head_size;
        float *v2 = s->value_cache + loff + (t + 2) * kv_dim + (h / kv_mul) * head_size;
        float *v3 = s->value_cache + loff + (t + 3) * kv_dim + (h / kv_mul) * head_size;
        float a0 = att[t];
        float a1 = att[t + 1];
        float a2 = att[t + 2];
        float a3 = att[t + 3];
        float32x4_t va0 = vdupq_n_f32(a0);
        float32x4_t va1 = vdupq_n_f32(a1);
        float32x4_t va2 = vdupq_n_f32(a2);
        float32x4_t va3 = vdupq_n_f32(a3);
        int vi = 0;
        for (; vi + 4 <= head_size; vi += 4) {
          float32x4_t xbv = vld1q_f32(xb + vi);
          float32x4_t vb0 = vld1q_f32(v0 + vi);
          float32x4_t vb1 = vld1q_f32(v1 + vi);
          float32x4_t vb2 = vld1q_f32(v2 + vi);
          float32x4_t vb3 = vld1q_f32(v3 + vi);
          xbv = vmlaq_f32(xbv, vb0, va0);
          xbv = vmlaq_f32(xbv, vb1, va1);
          xbv = vmlaq_f32(xbv, vb2, va2);
          xbv = vmlaq_f32(xbv, vb3, va3);
          vst1q_f32(xb + vi, xbv);
        }
        for (; vi < head_size; vi++) {
          xb[vi] += a0 * v0[vi] + a1 * v1[vi] + a2 * v2[vi] + a3 * v3[vi];
        }
      }
#endif
      for (; t <= pos; t++) {
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t];
#if defined(__aarch64__)
        int vi = 0;
        float32x4_t va = vdupq_n_f32(a);
        for (; vi + 4 <= head_size; vi += 4) {
          float32x4_t vb = vld1q_f32(v + vi);
          float32x4_t xbv = vld1q_f32(xb + vi);
          xbv = vmlaq_f32(xbv, vb, va);
          vst1q_f32(xb + vi, xbv);
        }
        for (; vi < head_size; vi++) {
          xb[vi] += a * v[vi];
        }
#else
        for (int vi = 0; vi < head_size; vi++) {
          xb[vi] += a * v[vi];
        }
#endif
      }
    }

    // final matmul to get the output of the attention
    quantize(&s->xq, s->xb, dim);
    matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(&s->xq, s->xb, dim);
    matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
    matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

    // SwiGLU non-linearity
    int gi = 0;
#if defined(__aarch64__)
    for (; gi + 16 <= hidden_dim; gi += 16) {
      float32x4_t vhb0 = vld1q_f32(s->hb + gi);
      float32x4_t vhb1 = vld1q_f32(s->hb + gi + 4);
      float32x4_t vhb2 = vld1q_f32(s->hb + gi + 8);
      float32x4_t vhb3 = vld1q_f32(s->hb + gi + 12);
      
      float32x4_t vhb2_0 = vld1q_f32(s->hb2 + gi);
      float32x4_t vhb2_1 = vld1q_f32(s->hb2 + gi + 4);
      float32x4_t vhb2_2 = vld1q_f32(s->hb2 + gi + 8);
      float32x4_t vhb2_3 = vld1q_f32(s->hb2 + gi + 12);

      // sigmoid approximation for 4 vectors
      float32x4_t vsig0, vsig1, vsig2, vsig3;
      
      // Vector 0
      {
          float sig0 = fast_sigmoid(vgetq_lane_f32(vhb0, 0));
          float sig1 = fast_sigmoid(vgetq_lane_f32(vhb0, 1));
          float sig2 = fast_sigmoid(vgetq_lane_f32(vhb0, 2));
          float sig3 = fast_sigmoid(vgetq_lane_f32(vhb0, 3));
          vsig0 = (float32x4_t){sig0, sig1, sig2, sig3};
      }
      // Vector 1
      {
          float sig0 = fast_sigmoid(vgetq_lane_f32(vhb1, 0));
          float sig1 = fast_sigmoid(vgetq_lane_f32(vhb1, 1));
          float sig2 = fast_sigmoid(vgetq_lane_f32(vhb1, 2));
          float sig3 = fast_sigmoid(vgetq_lane_f32(vhb1, 3));
          vsig1 = (float32x4_t){sig0, sig1, sig2, sig3};
      }
      // Vector 2
      {
          float sig0 = fast_sigmoid(vgetq_lane_f32(vhb2, 0));
          float sig1 = fast_sigmoid(vgetq_lane_f32(vhb2, 1));
          float sig2 = fast_sigmoid(vgetq_lane_f32(vhb2, 2));
          float sig3 = fast_sigmoid(vgetq_lane_f32(vhb2, 3));
          vsig2 = (float32x4_t){sig0, sig1, sig2, sig3};
      }
      // Vector 3
      {
          float sig0 = fast_sigmoid(vgetq_lane_f32(vhb3, 0));
          float sig1 = fast_sigmoid(vgetq_lane_f32(vhb3, 1));
          float sig2 = fast_sigmoid(vgetq_lane_f32(vhb3, 2));
          float sig3 = fast_sigmoid(vgetq_lane_f32(vhb3, 3));
          vsig3 = (float32x4_t){sig0, sig1, sig2, sig3};
      }

      vst1q_f32(s->hb + gi, vmulq_f32(vmulq_f32(vhb0, vsig0), vhb2_0));
      vst1q_f32(s->hb + gi + 4, vmulq_f32(vmulq_f32(vhb1, vsig1), vhb2_1));
      vst1q_f32(s->hb + gi + 8, vmulq_f32(vmulq_f32(vhb2, vsig2), vhb2_2));
      vst1q_f32(s->hb + gi + 12, vmulq_f32(vmulq_f32(vhb3, vsig3), vhb2_3));
    }
    for (; gi + 4 <= hidden_dim; gi += 4) {
      float32x4_t vhb = vld1q_f32(s->hb + gi);
      float sig0 = fast_sigmoid(vgetq_lane_f32(vhb, 0));
      float sig1 = fast_sigmoid(vgetq_lane_f32(vhb, 1));
      float sig2 = fast_sigmoid(vgetq_lane_f32(vhb, 2));
      float sig3 = fast_sigmoid(vgetq_lane_f32(vhb, 3));
      float32x4_t vsig = (float32x4_t){sig0, sig1, sig2, sig3};
      float32x4_t vhb2 = vld1q_f32(s->hb2 + gi);
      float32x4_t vout = vmulq_f32(vmulq_f32(vhb, vsig), vhb2);
      vst1q_f32(s->hb + gi, vout);
    }
#endif
    for (; gi < hidden_dim; gi++) {
      float val = s->hb[gi];
      val *= fast_sigmoid(val);
      val *= s->hb2[gi];
      s->hb[gi] = val;
    }

    // final matmul to get the output of the ffn
    quantize(&s->hq, s->hb, hidden_dim);
    matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  quantize(&s->xq, x, dim);
  matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=128000) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 128000;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ? UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    int best_len = 2; // length of the best merge sequence (2 for pair, 3 for triple)

    // first, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // if no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // check if we can merge the triple (tokens[i], tokens[i+1], tokens[i+2])
        sprintf(str_buffer, "%s%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]], t->vocab[tokens[i + 2]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs or triples to merge, so we're done
    }

    // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2]) into new token best_id
    tokens[best_idx] = best_id;
    // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    (*n_tokens) -= (best_len - 1); // token length decreased by the number of merged tokens minus one
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 128001;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
      break;
    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are somewhat haphazardly and unsafely set atm
  char *system_prompt = (char *)malloc(32768 * sizeof(char));
  char *user_prompt = (char *)malloc(32768 * sizeof(char));
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *system_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *user_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int user_idx = 0;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;             // will store the next token in the sequence
  int token;            // stores the current token to feed into the transformer

  int pos = 0; // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
        prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 9125;   // "system"
        prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, 32768);
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
        if (system_prompt != NULL) {
          int num_system_prompt_tokens = 0;
          encode(tokenizer, system_prompt, 0, 0, system_prompt_tokens, &num_system_prompt_tokens);
          for (int i = 0; i < num_system_prompt_tokens; i++) {
            prompt_tokens[num_prompt_tokens++] = system_prompt_tokens[i];
          }
        }
        prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      } else {
        num_prompt_tokens = 0;
      }
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 882;    // "user"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User (or exit): ", user_prompt, 32768);
        if (strcmp(user_prompt, "exit") == 0)
          break;
      }
      int num_user_prompt_tokens = 0;
      // encode the user prompt into tokens
      encode(tokenizer, user_prompt, 0, 0, user_prompt_tokens, &num_user_prompt_tokens);
      for (int i = 0; i < num_user_prompt_tokens; i++) {
        prompt_tokens[num_prompt_tokens++] = user_prompt_tokens[i];
      }
      prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=128009) token ends the Assistant turn
    if (user_idx >= num_prompt_tokens && (token == 128009 || token == 128001)) {
      user_turn = 1;
    }

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
      // the Assistant is responding, so print its output
      char *piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (user_idx >= num_prompt_tokens && next == 128009 || next == 128001) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
  free(system_prompt_tokens);
  free(user_prompt_tokens);
  free(system_prompt);
  free(user_prompt);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4096;                // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";         // generate|chat
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
#ifndef TESTING
#if defined(_OPENMP)
  omp_set_num_threads(4);
#endif
#endif
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
