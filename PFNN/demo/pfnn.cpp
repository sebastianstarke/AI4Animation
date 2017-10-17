#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <eigen3/Eigen/Dense>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <stdarg.h>
#include <time.h>

using namespace Eigen;

/* Options */

#ifdef HIGH_QUALITY
enum { WINDOW_WIDTH  = 1280, WINDOW_HEIGHT = 720 };
#else
enum { WINDOW_WIDTH  = 720, WINDOW_HEIGHT = 480 };
#endif

enum {
  GAMEPAD_TRIGGER_L  = 4,
  GAMEPAD_TRIGGER_R  = 5,
  GAMEPAD_SHOULDER_L = 8,
  GAMEPAD_SHOULDER_R = 9,
};

struct Options {
  
  bool invert_y;
  
  bool enable_ik;
  
  bool display_debug;
  bool display_debug_heights;
  bool display_debug_joints;
  bool display_debug_pfnn;
  bool display_hud_options;
  bool display_hud_stick;
  bool display_hud_speed;
  
  bool display_areas_jump;
  bool display_areas_walls;
  
  float display_scale;
  
  float extra_direction_smooth;
  float extra_velocity_smooth;
  float extra_strafe_smooth;
  float extra_crouched_smooth;
  float extra_gait_smooth;
  float extra_joint_smooth;
  
  Options()
    : invert_y(false)
    , enable_ik(true)
    , display_debug(true)
    , display_debug_heights(true)
    , display_debug_joints(false)
    , display_debug_pfnn(false)
    , display_hud_options(true)
    , display_hud_stick(true)
    , display_hud_speed(true)
    , display_areas_jump(false)
    , display_areas_walls(false)
#ifdef HIGH_QUALITY
    , display_scale(3.0)
#else
    , display_scale(2.0)
#endif
    , extra_direction_smooth(0.9)
    , extra_velocity_smooth(0.9)
    , extra_strafe_smooth(0.9)
    , extra_crouched_smooth(0.9)
    , extra_gait_smooth(0.1)
    , extra_joint_smooth(0.5)
    {}
};

static Options* options = NULL;

static int X = 0;
static int Y = 0;
static bool W, A, S, D;

/* Helper Functions */

static glm::vec3 mix_vectors(glm::vec3 a, glm::vec3 b, float c) {
  return c * a + (1.f-c) * b;
}

static glm::vec2 mix_vectors(glm::vec2 a, glm::vec2 b, float c) {
  return c * a + (1.f-c) * b;
}

static glm::vec3 mix_directions(glm::vec3 x, glm::vec3 y, float a) {
  glm::quat x_q = glm::angleAxis(atan2f(x.x, x.z), glm::vec3(0,1,0));
  glm::quat y_q = glm::angleAxis(atan2f(y.x, y.z), glm::vec3(0,1,0));
  glm::quat z_q = glm::slerp(x_q, y_q, a);
  return z_q * glm::vec3(0,0,1);
}

static glm::quat quat_exp(glm::vec3 l) {
  float w = glm::length(l);
  glm::quat q = w < 0.01 ? glm::quat(1,0,0,0) : glm::quat(
    cosf(w),
    l.x * (sinf(w) / w),
    l.y * (sinf(w) / w),
    l.z * (sinf(w) / w));
  return q / sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); 
}

static glm::vec2 segment_nearest(glm::vec2 v, glm::vec2 w, glm::vec2 p) {
  float l2 = glm::dot(v - w, v - w);
  if (l2 == 0.0) return v;
  float t = glm::clamp(glm::dot(p - v, w - v) / l2, 0.0f, 1.0f);
  return v + t * (w - v);
}

/* Phase-Functioned Neural Network */

struct PFNN {
  
  enum { XDIM = 342, YDIM = 311, HDIM = 512 };
  enum { MODE_CONSTANT, MODE_LINEAR, MODE_CUBIC };

  int mode;
  
  ArrayXf Xmean, Xstd;
  ArrayXf Ymean, Ystd;
  
  std::vector<ArrayXXf> W0, W1, W2;
  std::vector<ArrayXf>  b0, b1, b2;
  
  ArrayXf  Xp, Yp;
  ArrayXf  H0,  H1;
  ArrayXXf W0p, W1p, W2p;
  ArrayXf  b0p, b1p, b2p;
   
  PFNN(int pfnnmode)
    : mode(pfnnmode) { 
    
    Xp = ArrayXf((int)XDIM);
    Yp = ArrayXf((int)YDIM);
    
    H0 = ArrayXf((int)HDIM);
    H1 = ArrayXf((int)HDIM);
    
    W0p = ArrayXXf((int)HDIM, (int)XDIM);
    W1p = ArrayXXf((int)HDIM, (int)HDIM);
    W2p = ArrayXXf((int)YDIM, (int)HDIM);
    
    b0p = ArrayXf((int)HDIM);
    b1p = ArrayXf((int)HDIM);
    b2p = ArrayXf((int)YDIM);
  }
  
  static void load_weights(ArrayXXf &A, int rows, int cols, const char* fmt, ...) {
    va_list valist;
    va_start(valist, fmt);
    char filename[512];
    vsprintf(filename, fmt, valist);
    va_end(valist);

    FILE *f = fopen(filename, "rb");
    if (f == NULL) { fprintf(stderr, "Couldn't load file %s\n", filename); exit(1); }

    A = ArrayXXf(rows, cols);
    int elements = 0;
    for (int x = 0; x < rows; x++)
    for (int y = 0; y < cols; y++) {
      float item = 0.0;
      elements += fread(&item, sizeof(float), 1, f);
      A(x, y) = item;
    }
    printf("Read %u weight elements.\n", elements);
    fclose(f); 
  }

  static void load_weights(ArrayXf &V, int items, const char* fmt, ...) {
    va_list valist;
    va_start(valist, fmt);
    char filename[512];
    vsprintf(filename, fmt, valist);
    va_end(valist);
    
    FILE *f = fopen(filename, "rb"); 
    if (f == NULL) { fprintf(stderr, "Couldn't load file %s\n", filename); exit(1); }
    
    V = ArrayXf(items);
    int elements = 0;
    for (int i = 0; i < items; i++) {
      float item = 0.0;
      elements += fread(&item, sizeof(float), 1, f);
      V(i) = item;
    }
    printf("Read %u weight elements.\n", elements);
    fclose(f); 
  }  
  
  void load() {
    
    load_weights(Xmean, XDIM, "./network/pfnn/Xmean.bin");
    load_weights(Xstd,  XDIM, "./network/pfnn/Xstd.bin");
    load_weights(Ymean, YDIM, "./network/pfnn/Ymean.bin");
    load_weights(Ystd,  YDIM, "./network/pfnn/Ystd.bin");
    
    switch (mode) {
      
      case MODE_CONSTANT:
        
        W0.resize(50); W1.resize(50); W2.resize(50);
        b0.resize(50); b1.resize(50); b2.resize(50);
      
        for (int i = 0; i < 50; i++) {            
          load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", i);
          load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", i);
          load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", i);
          load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", i);
          load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", i);
          load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", i);            
        }
        
      break;
      
      case MODE_LINEAR:
      
        W0.resize(10); W1.resize(10); W2.resize(10);
        b0.resize(10); b1.resize(10); b2.resize(10);
      
        for (int i = 0; i < 10; i++) {
          load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", i * 5);
          load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", i * 5);
          load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", i * 5);
          load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", i * 5);
          load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", i * 5);
          load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", i * 5);  
        }
      
      break;
      
      case MODE_CUBIC:
      
        W0.resize(4); W1.resize(4); W2.resize(4);
        b0.resize(4); b1.resize(4); b2.resize(4);
      
        for (int i = 0; i < 4; i++) {
          load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", (int)(i * 12.5));
          load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", (int)(i * 12.5));
          load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", (int)(i * 12.5));
          load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", (int)(i * 12.5));
          load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", (int)(i * 12.5));
          load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", (int)(i * 12.5));  
        }
        
      break;
    }
    
  }
  
  static void ELU(ArrayXf &x) {
    x = x.max(0) + x.min(0).exp() - 1;
  }

  static void linear(ArrayXf  &o, const ArrayXf  &y0, const ArrayXf  &y1, float mu) {
    o = (1.0f-mu) * y0 + (mu) * y1;
  }

  static void linear(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, float mu) {
    o = (1.0f-mu) * y0 + (mu) * y1;
  }
  
  static void cubic(ArrayXf  &o, const ArrayXf &y0, const ArrayXf &y1, const ArrayXf &y2, const ArrayXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
  }
  
  static void cubic(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, const ArrayXXf &y2, const ArrayXXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
  }

  void predict(float P) {
    
    float pamount;
    int pindex_0, pindex_1, pindex_2, pindex_3;
    
    Xp = (Xp - Xmean) / Xstd;
    
    switch (mode) {
      
      case MODE_CONSTANT:
        pindex_1 = (int)((P / (2*M_PI)) * 50);
        H0 = (W0[pindex_1].matrix() * Xp.matrix()).array() + b0[pindex_1]; ELU(H0);
        H1 = (W1[pindex_1].matrix() * H0.matrix()).array() + b1[pindex_1]; ELU(H1);
        Yp = (W2[pindex_1].matrix() * H1.matrix()).array() + b2[pindex_1];
      break;
      
      case MODE_LINEAR:
        pamount = fmod((P / (2*M_PI)) * 10, 1.0);
        pindex_1 = (int)((P / (2*M_PI)) * 10);
        pindex_2 = ((pindex_1+1) % 10);
        linear(W0p, W0[pindex_1], W0[pindex_2], pamount);
        linear(W1p, W1[pindex_1], W1[pindex_2], pamount);
        linear(W2p, W2[pindex_1], W2[pindex_2], pamount);
        linear(b0p, b0[pindex_1], b0[pindex_2], pamount);
        linear(b1p, b1[pindex_1], b1[pindex_2], pamount);
        linear(b2p, b2[pindex_1], b2[pindex_2], pamount);
        H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
        H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
        Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
      break;
      
      case MODE_CUBIC:
        pamount = fmod((P / (2*M_PI)) * 4, 1.0);
        pindex_1 = (int)((P / (2*M_PI)) * 4);
        pindex_0 = ((pindex_1+3) % 4);
        pindex_2 = ((pindex_1+1) % 4);
        pindex_3 = ((pindex_1+2) % 4);
        cubic(W0p, W0[pindex_0], W0[pindex_1], W0[pindex_2], W0[pindex_3], pamount);
        cubic(W1p, W1[pindex_0], W1[pindex_1], W1[pindex_2], W1[pindex_3], pamount);
        cubic(W2p, W2[pindex_0], W2[pindex_1], W2[pindex_2], W2[pindex_3], pamount);
        cubic(b0p, b0[pindex_0], b0[pindex_1], b0[pindex_2], b0[pindex_3], pamount);
        cubic(b1p, b1[pindex_0], b1[pindex_1], b1[pindex_2], b1[pindex_3], pamount);
        cubic(b2p, b2[pindex_0], b2[pindex_1], b2[pindex_2], b2[pindex_3], pamount);
        H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
        H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
        Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
      break;
      
      default:
      break;
    }
    
    Yp = (Yp * Ystd) + Ymean;

  }
  
  
  
};

static PFNN* pfnn = NULL;

/* Joystick */

static SDL_Joystick* stick = NULL;

/* Camera */

struct CameraOrbit {
  
  glm::vec3 target;
  float pitch, yaw;
  float distance;
  
  CameraOrbit()
    : target(glm::vec3(0))
    , pitch(M_PI/6)
    , yaw(0)
    , distance(300) {}
   
  glm::vec3 position() {
    glm::vec3 posn = glm::mat3(glm::rotate(yaw, glm::vec3(0,1,0))) * glm::vec3(distance, 0, 0);
    glm::vec3 axis = glm::normalize(glm::cross(posn, glm::vec3(0,1,0)));
    return glm::mat3(glm::rotate(pitch, axis)) * posn + target;
  }
 
  glm::vec3 direction() {
    return glm::normalize(target - position());
  }
  
  glm::mat4 view_matrix() {
    return glm::lookAt(position(), target, glm::vec3(0,1,0));
  }
  
  glm::mat4 proj_matrix() {
    return glm::perspective(45.0f, (float)WINDOW_WIDTH/(float)WINDOW_HEIGHT, 10.0f, 10000.0f);
  }
    
};

static CameraOrbit* camera = NULL;

/* Rendering */

struct LightDirectional {
  
  glm::vec3 target;
  glm::vec3 position;
  
  GLuint fbo;
  GLuint buf;
  GLuint tex;
  
  LightDirectional()
    : target(glm::vec3(0))
    , position(glm::vec3(3000, 3700, 1500))
    , fbo(0)
    , buf(0)
    , tex(0) {
      
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    
    glGenRenderbuffers(1, &buf);
    glBindRenderbuffer(GL_RENDERBUFFER, buf);
  #ifdef HIGH_QUALITY
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 2048, 2048);
  #else
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
  #endif
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buf);  
    
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
  #ifdef HIGH_QUALITY
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 2048, 2048, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  #else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  #endif
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
      
  }
  
  ~LightDirectional() {
    glDeleteBuffers(1, &fbo);
    glDeleteBuffers(1, &buf);
    glDeleteTextures(1, &tex);
  }
  
};

static LightDirectional* light = NULL;

/* Heightmap */

struct Heightmap {
  
  float hscale;
  float vscale;
  float offset;
  std::vector<std::vector<float>> data;
  GLuint vbo;
  GLuint tbo;
  
  Heightmap()
    : hscale(3.937007874)
    //, vscale(3.937007874)
    , vscale(3.0)
    , offset(0.0)
    , vbo(0)
    , tbo(0) {}
  
  ~Heightmap() {
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; } 
  }

  void load(const char* filename, float multiplier) {
    
    vscale = multiplier * vscale;
    
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }
    
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &tbo);
    
    data.clear();
    
    std::ifstream file(filename);
    
    std::string line;
    while (std::getline(file, line)) {
      std::vector<float> row;
      std::istringstream iss(line);
      while (iss) {
        float f;
        iss >> f;
        row.push_back(f);
      }
      data.push_back(row);
    }
    
    int w = data.size();
    int h = data[0].size();
    
    offset = 0.0;
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      offset += data[x][y];
    }
    offset /= w * h;
    
    printf("Loaded Heightmap '%s' (%i %i)\n", filename, (int)w, (int)h);
    
    glm::vec3* posns = (glm::vec3*)malloc(sizeof(glm::vec3) * w * h);
    glm::vec3* norms = (glm::vec3*)malloc(sizeof(glm::vec3) * w * h);
    float* aos   = (float*)malloc(sizeof(float) * w * h);
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      float cx = hscale * x, cy = hscale * y, cw = hscale * w, ch = hscale * h;
      posns[x+y*w] = glm::vec3(cx - cw/2, sample(glm::vec2(cx-cw/2, cy-ch/2)), cy - ch/2);
    }
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      norms[x+y*w] = (x > 0 && x < w-1 && y > 0 && y < h-1) ?
        glm::normalize(
          mix_vectors(
            glm::cross(
              posns[(x+0)+(y+1)*w] - posns[x+y*w],
              posns[(x+1)+(y+0)*w] - posns[x+y*w]),
            glm::cross(
              posns[(x+0)+(y-1)*w] - posns[x+y*w],
              posns[(x-1)+(y+0)*w] - posns[x+y*w]), 0.5f)
          ) : glm::vec3(0,1,0);
    }

    char ao_filename[512];
    memcpy(ao_filename, filename, strlen(filename)-4);
    ao_filename[strlen(filename)-4] = '\0';
    strcat(ao_filename, "_ao.txt");
    
    srand(0);

    FILE* ao_file = fopen(ao_filename, "r");
    bool ao_generate = false;
    if (ao_file == NULL || ao_generate) {
      ao_file = fopen(ao_filename, "w");
      //ao_generate = true;
    }
   
   int elements = 0;

    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      
      if (ao_generate) {
      
        float ao_amount = 0.0;
        float ao_radius = 50.0;
        int ao_samples = 1024;
        int ao_steps = 5;
        for (int i = 0; i < ao_samples; i++) {
          glm::vec3 off = glm::normalize(glm::vec3(rand() % 10000 - 5000, rand() % 10000 - 5000, rand() % 10000 - 5000));
          if (glm::dot(off, norms[x+y*w]) < 0.0f) { off = -off; }
          for (int j = 1; j <= ao_steps; j++) {
            glm::vec3 next = posns[x+y*w] + (((float)j) / ao_steps) * ao_radius * off;
            if (sample(glm::vec2(next.x, next.z)) > next.y) { ao_amount += 1.0; break; }
          }
        }
        
        aos[x+y*w] = 1.0 - (ao_amount / ao_samples);
        fprintf(ao_file, y == h-1 ? "%f\n" : "%f ", aos[x+y*w]);
      } else {
        elements += fscanf(ao_file, y == h-1 ? "%f\n" : "%f ", &aos[x+y*w]);
      }
      
    }

    printf("Read %u heightmap elements.\n", elements);
    
    fclose(ao_file);

    float *vbo_data = (float*)malloc(sizeof(float) * 7 * w * h);
  #ifdef HIGH_QUALITY
    uint32_t *tbo_data = (uint32_t*)malloc(sizeof(uint32_t) * 3 * 2 * (w-1) * (h-1));
  #else
    uint32_t *tbo_data = (uint32_t*)malloc(sizeof(uint32_t) * 3 * 2 * ((w-1)/2) * ((h-1)/2));
  #endif
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      vbo_data[x*7+y*7*w+0] = posns[x+y*w].x; 
      vbo_data[x*7+y*7*w+1] = posns[x+y*w].y;
      vbo_data[x*7+y*7*w+2] = posns[x+y*w].z;
      vbo_data[x*7+y*7*w+3] = norms[x+y*w].x;
      vbo_data[x*7+y*7*w+4] = norms[x+y*w].y;
      vbo_data[x*7+y*7*w+5] = norms[x+y*w].z; 
      vbo_data[x*7+y*7*w+6] = aos[x+y*w]; 
    }
    
    free(posns);
    free(norms);
    free(aos);
    
  #ifdef HIGH_QUALITY
    for (int x = 0; x < (w-1); x++)
    for (int y = 0; y < (h-1); y++) {
      tbo_data[x*3*2+y*3*2*(w-1)+0] = (x+0)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+1] = (x+0)+(y+1)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+2] = (x+1)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+3] = (x+1)+(y+1)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+4] = (x+1)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+5] = (x+0)+(y+1)*w;
    }
  #else
    for (int x = 0; x < (w-1)/2; x++)
    for (int y = 0; y < (h-1)/2; y++) {
      tbo_data[x*3*2+y*3*2*((w-1)/2)+0] = (x*2+0)+(y*2+0)*w;
      tbo_data[x*3*2+y*3*2*((w-1)/2)+1] = (x*2+0)+(y*2+2)*w;
      tbo_data[x*3*2+y*3*2*((w-1)/2)+2] = (x*2+2)+(y*2+0)*w;
      tbo_data[x*3*2+y*3*2*((w-1)/2)+3] = (x*2+2)+(y*2+2)*w;
      tbo_data[x*3*2+y*3*2*((w-1)/2)+4] = (x*2+2)+(y*2+0)*w;
      tbo_data[x*3*2+y*3*2*((w-1)/2)+5] = (x*2+0)+(y*2+2)*w;
    }
  #endif
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * w * h, vbo_data, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tbo);

  #ifdef HIGH_QUALITY
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * 3 * 2 * (w-1) * (h-1), tbo_data, GL_STATIC_DRAW);
  #else
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * 3 * 2 * ((w-1)/2) * ((h-1)/2), tbo_data, GL_STATIC_DRAW);  
  #endif
    
    free(vbo_data);
    free(tbo_data);
    
  }
  
  float sample(glm::vec2 pos) {
  
    int w = data.size();
    int h = data[0].size();
    
    pos.x = (pos.x/hscale) + w/2;
    pos.y = (pos.y/hscale) + h/2;
    
    float a0 = fmod(pos.x, 1.0);
    float a1 = fmod(pos.y, 1.0);
    
    int x0 = (int)std::floor(pos.x), x1 = (int)std::ceil(pos.x);
    int y0 = (int)std::floor(pos.y), y1 = (int)std::ceil(pos.y);
    
    x0 = x0 < 0 ? 0 : x0; x0 = x0 >= w ? w-1 : x0;
    x1 = x1 < 0 ? 0 : x1; x1 = x1 >= w ? w-1 : x1;
    y0 = y0 < 0 ? 0 : y0; y0 = y0 >= h ? h-1 : y0;
    y1 = y1 < 0 ? 0 : y1; y1 = y1 >= h ? h-1 : y1;
    
    float s0 = vscale * (data[x0][y0] - offset);
    float s1 = vscale * (data[x1][y0] - offset);
    float s2 = vscale * (data[x0][y1] - offset);
    float s3 = vscale * (data[x1][y1] - offset);
    
    return (s0 * (1-a0) + s1 * a0) * (1-a1) + (s2 * (1-a0) + s3 * a0) * a1;
  
  }
  
};

static Heightmap* heightmap = NULL;

/* Shader */

struct Shader {
  
  GLuint program;
  GLuint vs, fs;
  
  Shader()
    : program(0)
    , vs(0)
    , fs(0) { }
  
  ~Shader() {
    if (vs != 0) { glDeleteShader(vs); vs = 0; }
    if (fs != 0) { glDeleteShader(fs); fs = 0; }
    if (program != 0) { glDeleteShader(program); program = 0; }
  }
  
  void load_shader(const char* filename, GLenum type, GLuint *shader) {

    SDL_RWops* file = SDL_RWFromFile(filename, "r");
    if(file == NULL) {
      fprintf(stderr, "Cannot load file %s\n", filename);
      exit(1);
    }
    
    long size = SDL_RWseek(file,0,SEEK_END);
    char* contents = (char*)malloc(size+1);
    contents[size] = '\0';
    
    SDL_RWseek(file, 0, SEEK_SET);
    SDL_RWread(file, contents, size, 1);
    SDL_RWclose(file);
    
    *shader = glCreateShader(type);
    
    glShaderSource(*shader, 1, (const char**)&contents, NULL);
    glCompileShader(*shader);
    
    free(contents);
    
    char log[2048];
    int i;
    glGetShaderInfoLog(*shader, 2048, &i, log);
    log[i] = '\0';
    if (strcmp(log, "") != 0) { printf("%s\n", log); }
    
    int compile_error = 0;
    glGetShaderiv(*shader, GL_COMPILE_STATUS, &compile_error);
    if (compile_error == GL_FALSE) {
      fprintf(stderr, "Compiler Error on Shader %s.\n", filename);
      exit(1);
    }
  }

  
  void load(const char* vertex, const char* fragment) {
    
    if (vs != 0) { glDeleteShader(vs); vs = 0; }
    if (fs != 0) { glDeleteShader(fs); fs = 0; }
    if (program != 0) { glDeleteShader(program); program = 0; }
    
    program = glCreateProgram();
    load_shader(vertex, GL_VERTEX_SHADER, &vs);
    load_shader(fragment, GL_FRAGMENT_SHADER, &fs);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    char log[2048];
    int i;
    glGetProgramInfoLog(program, 2048, &i, log);
    log[i] = '\0';
    if (strcmp(log, "") != 0) { printf("%s\n", log); }    
  }
  
};

static Shader* shader_terrain = NULL;
static Shader* shader_terrain_shadow = NULL;
static Shader* shader_character = NULL;
static Shader* shader_character_shadow = NULL;

/* Character */

struct Character {
  
  enum { JOINT_NUM = 31 };
  
  GLuint vbo, tbo;
  int ntri, nvtx;
  float phase;
  float strafe_amount;
  float strafe_target;
  float crouched_amount;
  float crouched_target;
  float responsive;
  
  glm::vec3 joint_positions[JOINT_NUM];
  glm::vec3 joint_velocities[JOINT_NUM];
  glm::mat3 joint_rotations[JOINT_NUM];
  
  glm::mat4 joint_anim_xform[JOINT_NUM];
  glm::mat4 joint_rest_xform[JOINT_NUM];
  glm::mat4 joint_mesh_xform[JOINT_NUM];
  glm::mat4 joint_global_rest_xform[JOINT_NUM];
  glm::mat4 joint_global_anim_xform[JOINT_NUM];

  int joint_parents[JOINT_NUM];
  
  enum {
    JOINT_ROOT_L = 1,
    JOINT_HIP_L  = 2,
    JOINT_KNEE_L = 3,  
    JOINT_HEEL_L = 4,
    JOINT_TOE_L  = 5,  
      
    JOINT_ROOT_R = 6,  
    JOINT_HIP_R  = 7,  
    JOINT_KNEE_R = 8,  
    JOINT_HEEL_R = 9,
    JOINT_TOE_R  = 10  
  };
  
  Character()
    : vbo(0)
    , tbo(0)
    , ntri(66918)
    , nvtx(11200)
    , phase(0)
    , strafe_amount(0)
    , strafe_target(0)
    , crouched_amount(0) 
    , crouched_target(0) 
    , responsive(0) {}
    
  ~Character() {
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }
  }
    
  void load(const char* filename_v, const char* filename_t, const char* filename_p, const char* filename_r) {
    printf("Read Character '%s %s'\n", filename_v, filename_t);
    
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }
    
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &tbo);
    
    FILE *f;
    
    int elements;

    f = fopen(filename_v, "rb");
    float *vbo_data = (float*)malloc(sizeof(float) * 15 * nvtx);
    elements = fread(vbo_data, sizeof(float) * 15 * nvtx, 1, f);
    printf("Read %u VBO elements.\n", elements);
    fclose(f);
    
    f = fopen(filename_t, "rb");
    uint32_t *tbo_data = (uint32_t*)malloc(sizeof(uint32_t) * ntri);  
    elements = fread(tbo_data, sizeof(uint32_t) * ntri, 1, f);
    printf("Read %u TBO elements.\n", elements);
    fclose(f);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 15 * nvtx, vbo_data, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * ntri, tbo_data, GL_STATIC_DRAW);
    
    free(vbo_data);
    free(tbo_data);
    
    f = fopen(filename_p, "rb");
    float fparents[JOINT_NUM];
    elements = fread(fparents, sizeof(float) * JOINT_NUM, 1, f);
    printf("Read %u joint elements.\n", elements);
    for (int i = 0; i < JOINT_NUM; i++) { joint_parents[i] = (int)fparents[i]; }
    fclose(f);
    
    f = fopen(filename_r, "rb");
    elements = fread(glm::value_ptr(joint_rest_xform[0]), sizeof(float) * JOINT_NUM * 4 * 4, 1, f);
    printf("Read %u xform elements.\n", elements);
    for (int i = 0; i < JOINT_NUM; i++) { joint_rest_xform[i] = glm::transpose(joint_rest_xform[i]); }
    fclose(f);
  }

  bool loadOBJ(
      const char * path,
      std::vector < glm::vec3 > & out_vertices,
      std::vector < glm::vec2 > & out_uvs,
      std::vector < glm::vec3 > & out_normals
  ) {
    std::vector< unsigned int > vertexIndices, uvIndices, normalIndices;
    std::vector< glm::vec3 > temp_vertices;
    std::vector< glm::vec2 > temp_uvs;
    std::vector< glm::vec3 > temp_normals;

    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return false;
    }

    while( 1 ){

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader

        if ( strcmp( lineHeader, "v" ) == 0 ){
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            temp_vertices.push_back(vertex);

        }else if ( strcmp( lineHeader, "vt" ) == 0 ){
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y );
            temp_uvs.push_back(uv);

        }else if ( strcmp( lineHeader, "vn" ) == 0 ){
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
            temp_normals.push_back(normal);

        }else if ( strcmp( lineHeader, "f" ) == 0 ){
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            if (matches != 9){
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices    .push_back(uvIndex[0]);
            uvIndices    .push_back(uvIndex[1]);
            uvIndices    .push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
        }
    }

      // For each vertex of each triangle
      for( unsigned int i=0; i<vertexIndices.size(); i++ ){
        unsigned int vertexIndex = vertexIndices[i];
        glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
        out_vertices.push_back(vertex);
      }
  
    }
  
  void forward_kinematics() {

    for (int i = 0; i < JOINT_NUM; i++) {
      joint_global_anim_xform[i] = joint_anim_xform[i];
      joint_global_rest_xform[i] = joint_rest_xform[i];
      int j = joint_parents[i];
      while (j != -1) {
        joint_global_anim_xform[i] = joint_anim_xform[j] * joint_global_anim_xform[i];
        joint_global_rest_xform[i] = joint_rest_xform[j] * joint_global_rest_xform[i];
        j = joint_parents[j];
      }
      joint_mesh_xform[i] = joint_global_anim_xform[i] * glm::inverse(joint_global_rest_xform[i]);
    }
    
  }
  
};

static Character* character = NULL;

/* Trajectory */

struct Trajectory {
  
  enum { LENGTH = 120 };
  
  float width;

  glm::vec3 positions[LENGTH];
  glm::vec3 directions[LENGTH];
  glm::mat3 rotations[LENGTH];
  float heights[LENGTH];
  
  float gait_stand[LENGTH];
  float gait_walk[LENGTH];
  float gait_jog[LENGTH];
  float gait_crouch[LENGTH];
  float gait_jump[LENGTH];
  float gait_bump[LENGTH];
  
  glm::vec3 target_dir, target_vel;
  
  Trajectory()
    : width(25)
    , target_dir(glm::vec3(0,0,1))
    , target_vel(glm::vec3(0)) {}
  
};

static Trajectory* trajectory = NULL;

/* IK */

struct IK {
  
  enum { HL = 0, HR = 1, TL = 2, TR = 3 };
  
  float lock[4];
  glm::vec3 position[4]; 
  float height[4];
  float fade;
  float threshold;
  float smoothness;
  float heel_height;
  float toe_height;
  
  IK()
    : fade(0.075)
    , threshold(0.8)
    , smoothness(0.5)
    , heel_height(5.0)
    , toe_height(4.0) {
    memset(lock, 4, sizeof(float));
    memset(position, 4, sizeof(glm::vec3));
    memset(height, 4, sizeof(float));
  }
  
  void two_joint(
    glm::vec3 a, glm::vec3 b, 
    glm::vec3 c, glm::vec3 t, float eps, 
    glm::mat4& a_pR, glm::mat4& b_pR,
    glm::mat4& a_gR, glm::mat4& b_gR,
    glm::mat4& a_lR, glm::mat4& b_lR) {
    
    float lc = glm::length(b - a);
    float la = glm::length(b - c);
    float lt = glm::clamp(glm::length(t - a), eps, lc + la - eps);
    
    if (glm::length(c - t) < eps) { return; }

    float ac_ab_0 = acosf(glm::clamp(glm::dot(glm::normalize(c - a), glm::normalize(b - a)), -1.0f, 1.0f));
    float ba_bc_0 = acosf(glm::clamp(glm::dot(glm::normalize(a - b), glm::normalize(c - b)), -1.0f, 1.0f));
    float ac_at_0 = acosf(glm::clamp(glm::dot(glm::normalize(c - a), glm::normalize(t - a)), -1.0f, 1.0f));
    
    float ac_ab_1 = acosf(glm::clamp((la*la - lc*lc - lt*lt) / (-2*lc*lt), -1.0f, 1.0f));
    float ba_bc_1 = acosf(glm::clamp((lt*lt - lc*lc - la*la) / (-2*lc*la), -1.0f, 1.0f));
    
    glm::vec3 a0 = glm::normalize(glm::cross(b - a, c - a));
    glm::vec3 a1 = glm::normalize(glm::cross(t - a, c - a));
    
    glm::mat3 r0 = glm::mat3(glm::rotate(ac_ab_1 - ac_ab_0, -a0));
    glm::mat3 r1 = glm::mat3(glm::rotate(ba_bc_1 - ba_bc_0, -a0));
    glm::mat3 r2 = glm::mat3(glm::rotate(ac_at_0,           -a1));
    
    glm::mat3 a_lRR = glm::inverse(glm::mat3(a_pR)) * (r2 * r0 * glm::mat3(a_gR)); 
    glm::mat3 b_lRR = glm::inverse(glm::mat3(b_pR)) * (r1 * glm::mat3(b_gR)); 
    
    for (int x = 0; x < 3; x++)
    for (int y = 0; y < 3; y++) {
      a_lR[x][y] = a_lRR[x][y];
      b_lR[x][y] = b_lRR[x][y];
    }
    
  }
  
};

static IK* ik = NULL;

/* Areas */

struct Areas {
  
  std::vector<glm::vec3> crouch_pos;
  std::vector<glm::vec2> crouch_size;
  static constexpr float CROUCH_WAVE = 50;
  
  std::vector<glm::vec3> jump_pos;
  std::vector<float> jump_size;
  std::vector<float> jump_falloff;
  
  std::vector<glm::vec2> wall_start;
  std::vector<glm::vec2> wall_stop;
  std::vector<float> wall_width;
  
  void clear() {
    crouch_pos.clear();
    crouch_size.clear();
    jump_pos.clear();
    jump_size.clear();
    jump_falloff.clear();
    wall_start.clear();
    wall_stop.clear();
    wall_width.clear();
  }
  
  void add_wall(glm::vec2 start, glm::vec2 stop, float width) {
    wall_start.push_back(start);
    wall_stop.push_back(stop);
    wall_width.push_back(width);
  }
  
  void add_crouch(glm::vec3 pos, glm::vec2 size) {
    crouch_pos.push_back(pos);
    crouch_size.push_back(size);
  }
  
  void add_jump(glm::vec3 pos, float size, float falloff) {
    jump_pos.push_back(pos);
    jump_size.push_back(size);
    jump_falloff.push_back(falloff);
  }
  
  int num_walls() { return wall_start.size(); }
  int num_crouches() { return crouch_pos.size(); }
  int num_jumps() { return jump_pos.size(); }
  
};

static Areas* areas = NULL;

/* Reset */

static void reset(glm::vec2 position) {

  ArrayXf Yp = pfnn->Ymean;

  glm::vec3 root_position = glm::vec3(position.x, heightmap->sample(position), position.y);
  glm::mat3 root_rotation = glm::mat3();
  
  for (int i = 0; i < Trajectory::LENGTH; i++) {
    trajectory->positions[i] = root_position;
    trajectory->rotations[i] = root_rotation;
    trajectory->directions[i] = glm::vec3(0,0,1);
    trajectory->heights[i] = root_position.y;
    trajectory->gait_stand[i] = 0.0;
    trajectory->gait_walk[i] = 0.0;
    trajectory->gait_jog[i] = 0.0;
    trajectory->gait_crouch[i] = 0.0;
    trajectory->gait_jump[i] = 0.0;
    trajectory->gait_bump[i] = 0.0;
  }
  
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    
    int opos = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*0);
    int ovel = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*1);
    int orot = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*2);
    
    glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos+i*3+0), Yp(opos+i*3+1), Yp(opos+i*3+2))) + root_position;
    glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel+i*3+0), Yp(ovel+i*3+1), Yp(ovel+i*3+2)));
    glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot+i*3+0), Yp(orot+i*3+1), Yp(orot+i*3+2)))));
    
    character->joint_positions[i]  = pos;
    character->joint_velocities[i] = vel;
    character->joint_rotations[i]  = rot;
  }
  
  character->phase = 0.0;
  
  ik->position[IK::HL] = glm::vec3(0,0,0); ik->lock[IK::HL] = 0; ik->height[IK::HL] = root_position.y;
  ik->position[IK::HR] = glm::vec3(0,0,0); ik->lock[IK::HR] = 0; ik->height[IK::HR] = root_position.y;
  ik->position[IK::TL] = glm::vec3(0,0,0); ik->lock[IK::TL] = 0; ik->height[IK::TL] = root_position.y;
  ik->position[IK::TR] = glm::vec3(0,0,0); ik->lock[IK::TR] = 0; ik->height[IK::TR] = root_position.y;
  
}

/* Load Worlds */

static void load_world0(void) {
  
  printf("Loading World 0\n");
  
  heightmap->load("./heightmaps/hmap_000_smooth.txt", 1.0);
  
  areas->clear();
  areas->add_wall(glm::vec2( 975, -975), glm::vec2( 975, 975), 20);
  areas->add_wall(glm::vec2( 975,  975), glm::vec2(-975, 975), 20);
  areas->add_wall(glm::vec2(-975,  975), glm::vec2(-975,-975), 20);
  areas->add_wall(glm::vec2(-975, -975), glm::vec2( 975,-975), 20);
  
  reset(glm::vec2(0, 0));
  
}

static void load_world1(void) {
  
  printf("Loading World 1\n");
  
  heightmap->load("./heightmaps/hmap_000_smooth.txt", 1.0);

  areas->clear();
  areas->add_crouch(glm::vec3(0,5,0), glm::vec2(1000.0f, 250.0f));
  areas->add_wall(glm::vec2( 975, -975), glm::vec2( 975, 975), 20);
  areas->add_wall(glm::vec2( 975,  975), glm::vec2(-975, 975), 20);
  areas->add_wall(glm::vec2(-975,  975), glm::vec2(-975,-975), 20);
  areas->add_wall(glm::vec2(-975, -975), glm::vec2( 975,-975), 20);
  
  reset(glm::vec2(0, 0));

}

static void load_world2(void) {
  
  printf("Loading World 2\n");
  
  heightmap->load("./heightmaps/hmap_004_smooth.txt", 1.0);
  
  areas->clear();
  areas->add_wall(glm::vec2(1013.78, -1023.47), glm::vec2( 1013.78,  1037.65), 20);
  areas->add_wall(glm::vec2(1013.78,  1037.65), glm::vec2(-1005.93,  1032.48), 20);
  areas->add_wall(glm::vec2(-1005.93, 1032.48), glm::vec2( -1012.46, -985.26), 20);
  areas->add_wall(glm::vec2(-1012.46, -985.26), glm::vec2( -680.57, -1001.82), 20);
  areas->add_wall(glm::vec2(-680.57, -1001.82), glm::vec2( -571.86, -1008.58), 20);
  areas->add_wall(glm::vec2(-571.86, -1008.58), glm::vec2( -441.50, -1025.14), 20);
  areas->add_wall(glm::vec2(-441.50, -1025.14), glm::vec2( -205.33, -1023.47), 20);
  areas->add_wall(glm::vec2(-205.33, -1023.47), glm::vec2( 1018.95, -1023.47), 20);
  
  reset(glm::vec2(0, 0));

}

static void load_world3(void) {
  
  printf("Loading World 3\n");
  
  heightmap->load("./heightmaps/hmap_007_smooth.txt", 1.0);

  areas->clear();
  areas->add_wall(glm::vec2(1137.99,  -2583.42), glm::vec2(1154.53,   2604.02), 20);
  areas->add_wall(glm::vec2(1154.53,   2604.02), glm::vec2(644.10,    2602.73), 20);
  areas->add_wall(glm::vec2(644.10,    2602.73), glm::vec2(504.73,    2501.38), 20);
  areas->add_wall(glm::vec2(504.73,    2501.38), glm::vec2(12.73,     2522.49), 20);
  areas->add_wall(glm::vec2(12.73,     2522.49), glm::vec2(-84.41,    2497.15), 20);
  areas->add_wall(glm::vec2(-84.41,    2497.15), glm::vec2(-342.03,   2481.34), 20);
  areas->add_wall(glm::vec2(-342.03,   2481.34), glm::vec2(-436.74,   2453.81), 20);
  areas->add_wall(glm::vec2(-436.74,   2453.81), glm::vec2(-555.85,   2480.54), 20);
  areas->add_wall(glm::vec2(-555.85,   2480.54), glm::vec2(-776.98,   2500.82), 20);
  areas->add_wall(glm::vec2(-776.98,   2500.82), glm::vec2(-877.50,   2466.82), 20);
  areas->add_wall(glm::vec2(-877.50,   2466.82), glm::vec2(-975.67,   2488.11), 20);
  areas->add_wall(glm::vec2(-975.67,   2488.11), glm::vec2(-995.97,   2607.62), 20);
  areas->add_wall(glm::vec2(-995.97,   2607.62), glm::vec2(-1142.54,  2612.13), 20);
  areas->add_wall(glm::vec2(-1142.54,  2612.13), glm::vec2(-1151.56,  2003.29), 20);
  areas->add_wall(glm::vec2(-1151.56,  2003.29), glm::vec2(-1133.52,  1953.68), 20);
  areas->add_wall(glm::vec2(-1133.52,  1953.68), glm::vec2(-1153.82,  1888.29), 20);
  areas->add_wall(glm::vec2(-1153.82,  1888.29), glm::vec2(-1151.56, -2608.12), 20);
  areas->add_wall(glm::vec2(-1151.56, -2608.12), glm::vec2(-1126.76, -2608.12), 20);
  areas->add_wall(glm::vec2(-1126.76, -2608.12), glm::vec2(-1133.52,  -427.57), 20);
  areas->add_wall(glm::vec2(-1133.52,  -427.57), glm::vec2(-1074.89,  -184.03), 20);
  areas->add_wall(glm::vec2(-1074.89,  -184.03), glm::vec2(-973.42,     48.23), 20);
  areas->add_wall(glm::vec2(-973.42,     48.23), glm::vec2(-928.32,    217.35), 20);
  areas->add_wall(glm::vec2(-928.32,    217.35), glm::vec2(-732.14,    535.30), 20);
  areas->add_wall(glm::vec2(-732.14,    535.30), glm::vec2(-734.39,    436.09), 20);
  areas->add_wall(glm::vec2(-734.39,    436.09), glm::vec2(-838.12,    167.75), 20);
  areas->add_wall(glm::vec2(-838.12,    167.75), glm::vec2(-937.34,   -427.57), 20);
  areas->add_wall(glm::vec2(-937.34,   -427.57), glm::vec2(-930.57,  -1164.94), 20);
  areas->add_wall(glm::vec2(-930.57,  -1164.94), glm::vec2(-844.88,  -1478.38), 20);
  areas->add_wall(glm::vec2(-844.88,  -1478.38), glm::vec2(-691.55,  -2166.15), 20);
  areas->add_wall(glm::vec2(-691.55,  -2166.15), glm::vec2(-648.70,  -2610.37), 20);
  areas->add_wall(glm::vec2(-648.70,  -2610.37), glm::vec2(1139.49,  -2581.06), 20);
  areas->add_wall(glm::vec2(-314.97,  -2472.82), glm::vec2(-258.59,  -2508.90), 20);
  areas->add_wall(glm::vec2(-258.59,  -2508.90), glm::vec2(-195.45,  -2504.39), 20);
  areas->add_wall(glm::vec2(-195.45,  -2504.39), glm::vec2(-199.96,  -2477.33), 20);
  areas->add_wall(glm::vec2(-199.96,  -2477.33), glm::vec2(-238.30,  -2450.27), 20);
  areas->add_wall(glm::vec2(-238.30,  -2450.27), glm::vec2(-281.14,  -2441.25), 20);
  areas->add_wall(glm::vec2(-281.14,  -2441.25), glm::vec2(-310.46,  -2466.06), 20);
  
  reset(glm::vec2(0, 0));

}

static void load_world4(void) {
  
  printf("Loading World 4\n");
  
  heightmap->load("./heightmaps/hmap_013_smooth.txt", 1.0);
  
  areas->clear();
  areas->add_wall(glm::vec2( 1225, -1000), glm::vec2( 1225, 1000), 20);
  areas->add_wall(glm::vec2( 1225,  1000), glm::vec2(-1225, 1000), 20);
  areas->add_wall(glm::vec2(-1225,  1000), glm::vec2(-1225,-1000), 20);
  areas->add_wall(glm::vec2(-1225, -1000), glm::vec2( 1225,-1000), 20);
  
  areas->add_jump(glm::vec3( 237.64, 5,  452.98), 75, 100);
  areas->add_jump(glm::vec3( 378.40, 5,  679.64), 75, 100);
  areas->add_jump(glm::vec3( 227.17, 5,  866.28), 75, 100);
  areas->add_jump(glm::vec3( -43.93, 5,  609.78), 75, 100);
  areas->add_jump(glm::vec3( 810.12, 5,  897.37), 75, 100);
  areas->add_jump(glm::vec3( 945.85, 5,  493.90), 75, 100);
  areas->add_jump(glm::vec3( 618.69, 5,  220.01), 75, 100);
  areas->add_jump(glm::vec3( 950.29, 5,  246.37), 75, 100);
  areas->add_jump(glm::vec3( 703.68, 5, -262.97), 75, 100);
  areas->add_jump(glm::vec3( 798.17, 5, -579.91), 75, 100);
  areas->add_jump(glm::vec3(1137.51, 5, -636.69), 75, 100);
  areas->add_jump(glm::vec3( 212.80, 5, -638.25), 75, 100);
  areas->add_jump(glm::vec3(  79.65, 5, -909.37), 75, 100);
  areas->add_jump(glm::vec3(-286.95, 5, -771.64), 75, 100);
  areas->add_jump(glm::vec3(-994.98, 5, -547.12), 75, 100);
  areas->add_jump(glm::vec3(-384.53, 5,  245.73), 75, 100);
  areas->add_jump(glm::vec3(-559.39, 5,  672.81), 75, 100);
  areas->add_jump(glm::vec3(-701.95, 5,  902.13), 75, 100);

  reset(glm::vec2(0, 0));
  
}

static void load_world5(void) {
  
  printf("Loading World 5\n");
  
  heightmap->load("./heightmaps/hmap_urban_001_smooth.txt", 1.0);

  areas->clear(); 
  areas->add_wall(glm::vec2(477.54,     762.20), glm::vec2(261.43,     980.61), 20);
  areas->add_wall(glm::vec2(450.08,     735.91), glm::vec2(236.42,     950.49), 20);
  areas->add_wall(glm::vec2(770.18,     811.10), glm::vec2(770.18,    1137.49), 20);
  areas->add_wall(glm::vec2(808.33,     811.10), glm::vec2(810.45,    1137.49), 20);
  areas->add_wall(glm::vec2(810.45,     804.10), glm::vec2(936.78,     799.44), 20);
  areas->add_wall(glm::vec2(936.78,     799.44), glm::vec2(939.11,     634.34), 20);
  areas->add_wall(glm::vec2(939.11,     634.34), glm::vec2(1286.05,    634.55), 20);
  areas->add_wall(glm::vec2(1286.05,    634.55), glm::vec2(1280.97,   1137.49), 20);
  areas->add_wall(glm::vec2(1280.97,   1137.49), glm::vec2(1149.56,   1146.81), 20);
  areas->add_wall(glm::vec2(1149.56,   1146.81), glm::vec2(1151.68,   1315.53), 20);
  areas->add_wall(glm::vec2(1151.68,   1315.53), glm::vec2(914.31,    1317.65), 20);
  areas->add_wall(glm::vec2(914.31,    1317.65), glm::vec2(911.97,    1142.15), 20);
  areas->add_wall(glm::vec2(911.97,    1142.15), glm::vec2(818.93,    1137.91), 20);
  areas->add_wall(glm::vec2(272.09,    1003.74), glm::vec2(401.21,    1137.59), 20);
  areas->add_wall(glm::vec2(401.21,    1137.59), glm::vec2(755.16,    1135.40), 20);
  areas->add_wall(glm::vec2(492.50,     772.82), glm::vec2(528.66,     798.22), 20);
  areas->add_wall(glm::vec2(528.66,     798.22), glm::vec2(765.94,     797.79), 20);
  areas->add_wall(glm::vec2(431.19,     713.73), glm::vec2(204.21,     482.63), 20);
  areas->add_wall(glm::vec2(204.21,     482.63), glm::vec2(350.53,     325.74), 20);
  areas->add_wall(glm::vec2(350.53,     325.74), glm::vec2(335.69,     310.90), 20);
  areas->add_wall(glm::vec2(335.69,     310.90), glm::vec2(-189.93,    840.77), 20);
  areas->add_wall(glm::vec2(-189.93,    840.77), glm::vec2(-172.98,    861.96), 20);
  areas->add_wall(glm::vec2(-172.98,    861.96), glm::vec2(-24.54,     709.28), 20);
  areas->add_wall(glm::vec2(-24.54,     709.28), glm::vec2(210.85,     940.50), 20);
  areas->add_wall(glm::vec2(-1084.78,   917.48), glm::vec2(-1082.57,  1027.92), 20);
  areas->add_wall(glm::vec2(-1082.57,  1027.92), glm::vec2(-1204.06,  1030.13), 20);
  areas->add_wall(glm::vec2(-1204.06,  1030.13), glm::vec2(-1201.85,   917.48), 20);
  areas->add_wall(glm::vec2(-1201.85,   917.48), glm::vec2(-1082.57,   917.48), 20);
  areas->add_wall(glm::vec2(-1089.20,   270.27), glm::vec2(-1086.99,   382.93), 20);
  areas->add_wall(glm::vec2(-1086.99,   382.93), glm::vec2(-1204.06,   387.34), 20);
  areas->add_wall(glm::vec2(-1204.06,   387.34), glm::vec2(-1204.06,   270.27), 20);
  areas->add_wall(glm::vec2(-1204.06,   270.27), glm::vec2(-1084.78,   272.48), 20);
  areas->add_wall(glm::vec2(1491.61,    265.82), glm::vec2(1494.18,    386.74), 20);
  areas->add_wall(glm::vec2(1494.18,    386.74), glm::vec2(1378.41,    386.74), 20);
  areas->add_wall(glm::vec2(1378.41,    386.74), glm::vec2(1378.41,    265.82), 20);
  areas->add_wall(glm::vec2(1378.41,    265.82), glm::vec2(1499.33,    270.96), 20);
  areas->add_wall(glm::vec2(1494.18,   -367.10), glm::vec2(1494.18,   -246.18), 20);
  areas->add_wall(glm::vec2(1494.18,   -246.18), glm::vec2(1375.83,   -246.18), 20);
  areas->add_wall(glm::vec2(1375.83,   -246.18), glm::vec2(1373.26,   -364.53), 20);
  areas->add_wall(glm::vec2(1373.26,   -364.53), glm::vec2(1499.33,   -364.53), 20);
  areas->add_wall(glm::vec2(-1088.90, -1665.86), glm::vec2(-1084.93, -1552.75), 20);
  areas->add_wall(glm::vec2(-1084.93, -1552.75), glm::vec2(-1202.01, -1552.75), 20);
  areas->add_wall(glm::vec2(-1202.01, -1552.75), glm::vec2(-1203.99, -1663.88), 20);
  areas->add_wall(glm::vec2(-1203.99, -1663.88), glm::vec2(-1086.91, -1663.88), 20);
  areas->add_wall(glm::vec2(-78.35,   -1210.59), glm::vec2(-94.14,   -1023.36), 20);
  areas->add_wall(glm::vec2(-94.14,   -1023.36), glm::vec2(-155.05,  -1032.38), 20);
  areas->add_wall(glm::vec2(-155.05,  -1032.38), glm::vec2(-179.86,   -820.34), 20);
  areas->add_wall(glm::vec2(-179.86,   -820.34), glm::vec2(-116.70,   -811.32), 20);
  areas->add_wall(glm::vec2(-116.70,   -811.32), glm::vec2(-89.63,   -1000.80), 20);
  areas->add_wall(glm::vec2(-89.63,   -1000.80), glm::vec2(-33.24,    -994.04), 20);
  areas->add_wall(glm::vec2(-33.24,    -994.04), glm::vec2(-12.93,   -1208.33), 20);
  areas->add_wall(glm::vec2(-12.93,   -1208.33), glm::vec2(-76.10,   -1212.84), 20);
  areas->add_wall(glm::vec2(-118.56,   -142.29), glm::vec2(-176.02,   -108.24), 20);
  areas->add_wall(glm::vec2(-176.02,   -108.24), glm::vec2(-278.18,   -291.27), 20);
  areas->add_wall(glm::vec2(-278.18,   -291.27), glm::vec2(-227.10,   -323.20), 20);
  areas->add_wall(glm::vec2(-227.10,   -323.20), glm::vec2(-114.30,   -140.16), 20);
  areas->add_wall(glm::vec2(-33.43,     -57.16), glm::vec2(153.87,      51.39), 20);
  areas->add_wall(glm::vec2(153.87,      51.39), glm::vec2(119.81,     104.60), 20);
  areas->add_wall(glm::vec2(119.81,     104.60), glm::vec2(-63.22,       0.31), 20);
  areas->add_wall(glm::vec2(-63.22,       0.31), glm::vec2(-29.17,     -52.90), 20);
  areas->add_wall(glm::vec2(-449.57,  -1219.96), glm::vec2(-443.12,  -1196.31), 20);
  areas->add_wall(glm::vec2(-443.12,  -1196.31), glm::vec2(-1201.49, -1192.57), 20);
  areas->add_wall(glm::vec2(-1201.49, -1192.57), glm::vec2(-1201.49, -1218.79), 20);
  areas->add_wall(glm::vec2(-1201.49, -1218.79), glm::vec2(-442.73,  -1218.74), 20);
  areas->add_wall(glm::vec2(996.02,     633.86), glm::vec2(1052.59,    633.86), 20);
  areas->add_wall(glm::vec2(1052.59,    633.86), glm::vec2(1067.35,    382.99), 20);
  areas->add_wall(glm::vec2(1067.35,    382.99), glm::vec2(1010.78,    382.99), 20);
  areas->add_wall(glm::vec2(1010.78,    382.99), glm::vec2(998.48,     638.78), 20);
  areas->add_wall(glm::vec2(1208.98,    633.86), glm::vec2(1213.90,    392.83), 20);
  areas->add_wall(glm::vec2(1213.90,    392.83), glm::vec2(1154.87,    390.37), 20);
  areas->add_wall(glm::vec2(1154.87,    390.37), glm::vec2(1157.33,    633.86), 20);
  areas->add_wall(glm::vec2(1157.33,    633.86), glm::vec2(1204.07,    633.86), 20);
  areas->add_wall(glm::vec2(-1217.82,   590.42), glm::vec2(-1173.38,   622.52), 20);
  areas->add_wall(glm::vec2(-1173.38,   622.52), glm::vec2(-1200.54,   671.91), 20);
  areas->add_wall(glm::vec2(-1200.54,   671.91), glm::vec2(-1267.21,   711.42), 20);
  areas->add_wall(glm::vec2(-1267.21,   711.42), glm::vec2(-1286.96,   664.50), 20);
  areas->add_wall(glm::vec2(-1286.96,   664.50), glm::vec2(-1274.62,   592.89), 20);
  areas->add_wall(glm::vec2(-1274.62,   592.89), glm::vec2(-1217.82,   592.89), 20);
  areas->add_wall(glm::vec2(861.99,     264.16), glm::vec2(852.86,     172.92), 20);
  areas->add_wall(glm::vec2(852.86,     172.92), glm::vec2(904.56,     142.51), 20);
  areas->add_wall(glm::vec2(904.56,     142.51), glm::vec2(971.47,     191.17), 20);
  areas->add_wall(glm::vec2(971.47,     191.17), glm::vec2(941.06,     273.28), 20);
  areas->add_wall(glm::vec2(941.06,     273.28), glm::vec2(865.03,     267.20), 20);
  areas->add_wall(glm::vec2(2.04,       273.54), glm::vec2(76.94,      344.69), 20);
  areas->add_wall(glm::vec2(76.94,      344.69), glm::vec2(222.99,     191.15), 20);
  areas->add_wall(glm::vec2(222.99,     191.15), glm::vec2(252.95,     217.36), 20);
  areas->add_wall(glm::vec2(252.95,     217.36), glm::vec2(-290.07,    756.64), 20);
  areas->add_wall(glm::vec2(-290.07,    756.64), glm::vec2(-308.80,    730.43), 20);
  areas->add_wall(glm::vec2(-308.80,    730.43), glm::vec2(-159.00,    580.63), 20);
  areas->add_wall(glm::vec2(-159.00,    580.63), glm::vec2(-226.41,    520.71), 20);
  areas->add_wall(glm::vec2(-226.41,    520.71), glm::vec2(9.53,       281.03), 20);
  areas->add_wall(glm::vec2(1494.76,   -496.66), glm::vec2(1496.73,   1442.38), 20);
  areas->add_wall(glm::vec2(1496.73,   1442.38), glm::vec2(1013.95,   1438.43), 20);
  areas->add_wall(glm::vec2(1013.95,   1438.43), glm::vec2(1011.97,   1468.10), 20);
  areas->add_wall(glm::vec2(1011.97,   1468.10), glm::vec2(1534.33,   1476.02), 20);
  areas->add_wall(glm::vec2(1534.33,   1476.02), glm::vec2(1534.33,   -465.01), 20);
  areas->add_wall(glm::vec2(1534.33,   -465.01), glm::vec2(1970.84,   -465.01), 20);
  areas->add_wall(glm::vec2(1970.84,   -465.01), glm::vec2(1971.08,   -340.88), 20);
  areas->add_wall(glm::vec2(1971.08,   -340.88), glm::vec2(1658.98,   -332.44), 20);
  areas->add_wall(glm::vec2(1658.98,   -332.44), glm::vec2(1662.94,   1602.65), 20);
  areas->add_wall(glm::vec2(1662.94,   1602.65), glm::vec2(1073.05,   1599.36), 20);
  areas->add_wall(glm::vec2(1073.05,   1599.36), glm::vec2(1066.82,   1585.85), 20);
  areas->add_wall(glm::vec2(1066.82,   1585.85), glm::vec2(1054.49,   1564.70), 20);
  areas->add_wall(glm::vec2(1054.49,   1564.70), glm::vec2(915.16,    1554.76), 20);
  areas->add_wall(glm::vec2(915.16,    1554.76), glm::vec2(911.60,    1608.03), 20);
  areas->add_wall(glm::vec2(911.60,    1608.03), glm::vec2(825.93,    1602.97), 20);
  areas->add_wall(glm::vec2(825.93,    1602.97), glm::vec2(-46.59,    1598.69), 20);
  areas->add_wall(glm::vec2(-46.59,    1598.69), glm::vec2(-48.57,    1248.48), 20);
  areas->add_wall(glm::vec2(-48.57,    1248.48), glm::vec2(-442.32,   1211.85), 20);
  areas->add_wall(glm::vec2(-442.32,   1211.85), glm::vec2(-434.40,   1610.57), 20);
  areas->add_wall(glm::vec2(-434.40,   1610.57), glm::vec2(-359.27,   1629.55), 20);
  areas->add_wall(glm::vec2(-359.27,   1629.55), glm::vec2(-374.44,   1759.60), 20);
  areas->add_wall(glm::vec2(-374.44,   1759.60), glm::vec2(-439.46,   1757.43), 20);
  areas->add_wall(glm::vec2(-439.46,   1757.43), glm::vec2(-483.22,   1764.33), 20);
  areas->add_wall(glm::vec2(-483.22,   1764.33), glm::vec2(-1118.99,  1750.84), 20);
  areas->add_wall(glm::vec2(-1118.99,  1750.84), glm::vec2(-1124.37,  1601.37), 20);
  areas->add_wall(glm::vec2(-1124.37,  1601.37), glm::vec2(-1774.60,  1599.21), 20);
  areas->add_wall(glm::vec2(-1774.60,  1599.21), glm::vec2(-1770.27, -1903.69), 20);
  areas->add_wall(glm::vec2(-1770.27, -1903.69), glm::vec2(-1646.72, -1901.88), 20);
  areas->add_wall(glm::vec2(-1646.72, -1901.88), glm::vec2(-1655.39,  1471.33), 20);
  areas->add_wall(glm::vec2(-1655.39,  1471.33), glm::vec2(-1122.20,  1471.33), 20);
  areas->add_wall(glm::vec2(-1122.20,  1471.33), glm::vec2(-1126.54,  1434.48), 20);
  areas->add_wall(glm::vec2(-1126.54,  1434.48), glm::vec2(-1612.04,  1438.82), 20);
  areas->add_wall(glm::vec2(-1612.04,  1438.82), glm::vec2(-1557.86,  1408.47), 20);
  areas->add_wall(glm::vec2(-1557.86,  1408.47), glm::vec2(-1534.02,  1339.11), 20);
  areas->add_wall(glm::vec2(-1534.02,  1339.11), glm::vec2(-1547.02,  1274.09), 20);
  areas->add_wall(glm::vec2(-1547.02,  1274.09), glm::vec2(-1605.54,  1241.58), 20);
  areas->add_wall(glm::vec2(-1605.54,  1241.58), glm::vec2(-1612.04,    29.98), 20);
  areas->add_wall(glm::vec2(-1612.04,    29.98), glm::vec2(-1577.37,    29.98), 20);
  areas->add_wall(glm::vec2(-1577.37,    29.98), glm::vec2(-1573.03,    68.99), 20);
  areas->add_wall(glm::vec2(-1573.03,    68.99), glm::vec2(-1521.01,    82.00), 20);
  areas->add_wall(glm::vec2(-1521.01,    82.00), glm::vec2(-1525.35,  -134.74), 20);
  areas->add_wall(glm::vec2(-1525.35,  -134.74), glm::vec2(-891.62,   -140.12), 20);
  areas->add_wall(glm::vec2(-891.62,   -140.12), glm::vec2(-665.51,     83.84), 20);
  areas->add_wall(glm::vec2(-665.51,     83.84), glm::vec2(-430.78,   -161.66), 20);
  areas->add_wall(glm::vec2(-430.78,   -161.66), glm::vec2(-646.13,   -381.31), 20);
  areas->add_wall(glm::vec2(-646.13,   -381.31), glm::vec2(-650.43,  -1029.50), 20);
  areas->add_wall(glm::vec2(-650.43,  -1029.50), glm::vec2(-448.01,  -1025.19), 20);
  areas->add_wall(glm::vec2(-448.01,  -1025.19), glm::vec2(-443.70,  -1059.64), 20);
  areas->add_wall(glm::vec2(-443.70,  -1059.64), glm::vec2(-1203.87, -1063.95), 20);
  areas->add_wall(glm::vec2(-1203.87, -1063.95), glm::vec2(-1203.87, -1025.19), 20);
  areas->add_wall(glm::vec2(-1203.87, -1025.19), glm::vec2(-1001.45, -1025.19), 20);
  areas->add_wall(glm::vec2(-1001.45, -1025.19), glm::vec2(-1001.45,  -824.92), 20);
  areas->add_wall(glm::vec2(-1001.45,  -824.92), glm::vec2(-1324.46,  -495.44), 20);
  areas->add_wall(glm::vec2(-1324.46,  -495.44), glm::vec2(-1524.73,  -486.83), 20);
  areas->add_wall(glm::vec2(-1524.73,  -486.83), glm::vec2(-1524.73,  -710.78), 20);
  areas->add_wall(glm::vec2(-1524.73,  -710.78), glm::vec2(-1574.26,  -708.63), 20);
  areas->add_wall(glm::vec2(-1574.26,  -708.63), glm::vec2(-1578.57,  -669.87), 20);
  areas->add_wall(glm::vec2(-1578.57,  -669.87), glm::vec2(-1604.41,  -667.72), 20);
  areas->add_wall(glm::vec2(-1604.41,  -667.72), glm::vec2(-1602.56, -1714.29), 20);
  areas->add_wall(glm::vec2(-1602.56, -1714.29), glm::vec2(-1544.11, -1750.90), 20);
  areas->add_wall(glm::vec2(-1544.11, -1750.90), glm::vec2(-1526.89, -1826.27), 20);
  areas->add_wall(glm::vec2(-1526.89, -1826.27), glm::vec2(-1524.73, -1908.10), 20);
  areas->add_wall(glm::vec2(-1524.73, -1908.10), glm::vec2(-1505.35, -1847.81), 20);
  areas->add_wall(glm::vec2(-1505.35, -1847.81), glm::vec2(-1429.98, -1856.42), 20);
  areas->add_wall(glm::vec2(-1429.98, -1856.42), glm::vec2(-1410.60, -1912.41), 20);
  areas->add_wall(glm::vec2(-1410.60, -1912.41), glm::vec2(-878.70,  -1918.87), 20);
  areas->add_wall(glm::vec2(-878.70,  -1918.87), glm::vec2(-863.63,  -1854.27), 20);
  areas->add_wall(glm::vec2(-863.63,  -1854.27), glm::vec2(-777.49,  -1854.27), 20);
  areas->add_wall(glm::vec2(-777.49,  -1854.27), glm::vec2(-762.41,  -1918.87), 20);
  areas->add_wall(glm::vec2(-762.41,  -1918.87), glm::vec2(-233.87,  -1915.62), 20);
  areas->add_wall(glm::vec2(-233.87,  -1915.62), glm::vec2(-227.54,  -1873.38), 20);
  areas->add_wall(glm::vec2(-227.54,  -1873.38), glm::vec2(-455.62,  -1871.27), 20);
  areas->add_wall(glm::vec2(-455.62,  -1871.27), glm::vec2(-451.39,  -1497.48), 20);
  areas->add_wall(glm::vec2(-451.39,  -1497.48), glm::vec2(-278.26,  -1502.82), 20);
  areas->add_wall(glm::vec2(-278.26,  -1502.82), glm::vec2(-276.28,  -1447.04), 20);
  areas->add_wall(glm::vec2(-276.28,  -1447.04), glm::vec2(-73.87,   -1453.75), 20);
  areas->add_wall(glm::vec2(-73.87,   -1453.75), glm::vec2(-71.27,   -1503.82), 20);
  areas->add_wall(glm::vec2(-71.27,   -1503.82), glm::vec2(-45.79,   -1879.72), 20);
  areas->add_wall(glm::vec2(-45.79,   -1879.72), glm::vec2(-121.95,  -1877.61), 20);
  areas->add_wall(glm::vec2(-121.95,  -1877.61), glm::vec2(-113.50,  -1924.07), 20);
  areas->add_wall(glm::vec2(-113.50,  -1924.07), glm::vec2(-115.61,  -2006.43), 20);
  areas->add_wall(glm::vec2(-115.61,  -2006.43), glm::vec2(629.86,   -2016.99), 20);
  areas->add_wall(glm::vec2(629.86,   -2016.99), glm::vec2(661.53,   -1909.28), 20);
  areas->add_wall(glm::vec2(661.53,   -1909.28), glm::vec2(765.01,   -1900.84), 20);
  areas->add_wall(glm::vec2(765.01,   -1900.84), glm::vec2(798.68,   -1926.36), 20);
  areas->add_wall(glm::vec2(798.68,   -1926.36), glm::vec2(1574.44,  -1920.78), 20);
  areas->add_wall(glm::vec2(1574.44,  -1920.78), glm::vec2(1616.07,  -1917.73), 20);
  areas->add_wall(glm::vec2(1616.07,  -1917.73), glm::vec2(1647.75,  -1913.51), 20);
  areas->add_wall(glm::vec2(1647.75,  -1913.51), glm::vec2(1654.08,  -1746.67), 20);
  areas->add_wall(glm::vec2(1654.08,  -1746.67), glm::vec2(1816.69,  -1748.79), 20);
  areas->add_wall(glm::vec2(1816.69,  -1748.79), glm::vec2(1812.47,  -1584.06), 20);
  areas->add_wall(glm::vec2(1812.47,  -1584.06), glm::vec2(2017.32,  -1590.40), 20);
  areas->add_wall(glm::vec2(2013.48,  -1595.35), glm::vec2(2005.79,  -1038.14), 20);
  areas->add_wall(glm::vec2(2005.79,  -1038.14), glm::vec2(1967.37,  -1030.45), 20);
  areas->add_wall(glm::vec2(1967.37,  -1030.45), glm::vec2(1975.05,   -895.95), 20);
  areas->add_wall(glm::vec2(1975.05,   -895.95), glm::vec2(1579.23,   -892.11), 20);
  areas->add_wall(glm::vec2(1579.23,   -892.11), glm::vec2(1579.23,   -849.84), 20);
  areas->add_wall(glm::vec2(1579.23,   -849.84), glm::vec2(1967.37,   -853.68), 20);
  areas->add_wall(glm::vec2(1967.37,   -853.68), glm::vec2(1955.84,   -500.13), 20);
  areas->add_wall(glm::vec2(1955.84,   -500.13), glm::vec2(1579.23,   -515.50), 20);
  areas->add_wall(glm::vec2(1579.23,   -515.50), glm::vec2(1575.39,   -496.29), 20);
  areas->add_wall(glm::vec2(1575.39,   -496.29), glm::vec2(1490.62,   -500.36), 20);

  reset(glm::vec2(200, 0));

}

static void pre_render() {
        
  /* Update Camera */

  int x_move = -X;
  int y_move = 0;
  
  if (abs(x_move) + abs(y_move) < 10000) { x_move = 0; y_move = 0; };
  
  if (options->invert_y) { y_move = -y_move; }
  
  camera->pitch = glm::clamp(camera->pitch + (y_move / 32768.0) * 0.03, M_PI/16, 2*M_PI/5);
  camera->yaw = camera->yaw + (x_move / 32768.0) * 0.03;
  
  float zoom_i = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_L) * 20.0;
  float zoom_o = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_R) * 20.0;
  
  if (zoom_i > 1e-5) { camera->distance = glm::clamp(camera->distance + zoom_i, 10.0f, 10000.0f); }
  if (zoom_o > 1e-5) { camera->distance = glm::clamp(camera->distance - zoom_o, 10.0f, 10000.0f); }
        
  /* Update Target Direction / Velocity */
    
  int x_vel = -X;
  int y_vel = -Y;
  glm::vec3 trajectory_target_direction_new = glm::normalize(glm::vec3(camera->direction().x, 0.0, camera->direction().z));
  glm::mat3 trajectory_target_rotation = glm::mat3(glm::rotate(atan2f(
    trajectory_target_direction_new.x,
    trajectory_target_direction_new.z), glm::vec3(0,1,0)));

  glm::vec3 trajectory_target_velocity_new = (trajectory_target_rotation * 2.0f * glm::vec3(x_vel, 0, y_vel));
  trajectory->target_vel = glm::mix(trajectory->target_vel, trajectory_target_velocity_new, options->extra_velocity_smooth);
    
  character->strafe_target = ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_L) / 32768.0) + 1.0) / 2.0;
  character->strafe_amount = glm::mix(character->strafe_amount, character->strafe_target, options->extra_strafe_smooth);
  
  glm::vec3 trajectory_target_velocity_dir = glm::length(trajectory->target_vel) < 1e-05 ? trajectory->target_dir : glm::normalize(trajectory->target_vel);
  trajectory_target_direction_new = mix_directions(trajectory_target_velocity_dir, trajectory_target_direction_new, character->strafe_amount);  
  trajectory->target_dir = mix_directions(trajectory->target_dir, trajectory_target_direction_new, options->extra_direction_smooth);
  
  character->crouched_amount = glm::mix(character->crouched_amount, character->crouched_target, options->extra_crouched_smooth);

  /* Update Gait */
  
  if (glm::length(trajectory->target_vel) < 0.1)  {
    float stand_amount = 1.0f - glm::clamp(glm::length(trajectory->target_vel) / 0.1f, 0.0f, 1.0f);
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  stand_amount, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    0.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
  } else if (character->crouched_amount > 0.1) {
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  0.0f, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    0.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], character->crouched_amount, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
  } else if ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_R) / 32768.0) + 1.0) {
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  0.0f, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    1.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);    
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);    
  } else {
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  0.0f, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   1.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    0.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);  
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);  
  }

  /* Predict Future Trajectory */
  glm::vec3 trajectory_positions_blend[Trajectory::LENGTH];
  trajectory_positions_blend[Trajectory::LENGTH/2] = trajectory->positions[Trajectory::LENGTH/2];

  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    
    float bias_pos = character->responsive ? glm::mix(2.0f, 2.0f, character->strafe_amount) : glm::mix(0.5f, 1.0f, character->strafe_amount);
    float bias_dir = character->responsive ? glm::mix(5.0f, 3.0f, character->strafe_amount) : glm::mix(2.0f, 0.5f, character->strafe_amount);
    
    float scale_pos = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH/2) / (Trajectory::LENGTH/2)), bias_pos));
    float scale_dir = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH/2) / (Trajectory::LENGTH/2)), bias_dir));

    trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + glm::mix(
        trajectory->positions[i] - trajectory->positions[i-1], 
        trajectory->target_vel,
        scale_pos);
        
    /* Collide with walls */
    for (int j = 0; j < areas->num_walls(); j++) {
      glm::vec2 trjpoint = glm::vec2(trajectory_positions_blend[i].x, trajectory_positions_blend[i].z);
      if (glm::length(trjpoint - ((areas->wall_start[j] + areas->wall_stop[j]) / 2.0f)) > 
          glm::length(areas->wall_start[j] - areas->wall_stop[j])) { continue; }
      glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
      float segdist = glm::length(segpoint - trjpoint);
      if (segdist < areas->wall_width[j] + 100.0) {
        glm::vec2 prjpoint0 = (areas->wall_width[j] +   0.0f) * glm::normalize(trjpoint - segpoint) + segpoint; 
        glm::vec2 prjpoint1 = (areas->wall_width[j] + 100.0f) * glm::normalize(trjpoint - segpoint) + segpoint; 
        glm::vec2 prjpoint = glm::mix(prjpoint0, prjpoint1, glm::clamp((segdist - areas->wall_width[j]) / 100.0f, 0.0f, 1.0f));
        trajectory_positions_blend[i].x = prjpoint.x;
        trajectory_positions_blend[i].z = prjpoint.y;
      }
    }

    trajectory->directions[i] = mix_directions(trajectory->directions[i], trajectory->target_dir, scale_dir);
    
    trajectory->heights[i] = trajectory->heights[Trajectory::LENGTH/2]; 
    
    trajectory->gait_stand[i]  = trajectory->gait_stand[Trajectory::LENGTH/2]; 
    trajectory->gait_walk[i]   = trajectory->gait_walk[Trajectory::LENGTH/2];  
    trajectory->gait_jog[i]    = trajectory->gait_jog[Trajectory::LENGTH/2];   
    trajectory->gait_crouch[i] = trajectory->gait_crouch[Trajectory::LENGTH/2];
    trajectory->gait_jump[i]   = trajectory->gait_jump[Trajectory::LENGTH/2];  
    trajectory->gait_bump[i]   = trajectory->gait_bump[Trajectory::LENGTH/2];  
  }
  
  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    trajectory->positions[i] = trajectory_positions_blend[i];
  }
  
  /* Jumps */
  for (int i = Trajectory::LENGTH/2; i < Trajectory::LENGTH; i++) {
    trajectory->gait_jump[i] = 0.0;
    for (int j = 0; j < areas->num_jumps(); j++) {
      float dist = glm::length(trajectory->positions[i] - areas->jump_pos[j]);
      trajectory->gait_jump[i] = std::max(trajectory->gait_jump[i], 
        1.0f-glm::clamp((dist - areas->jump_size[j]) / areas->jump_falloff[j], 0.0f, 1.0f));
    }
  }
  
  /* Crouch Area */
  for (int i = Trajectory::LENGTH/2; i < Trajectory::LENGTH; i++) {
    for (int j = 0; j < areas->num_crouches(); j++) {
      float dist_x = abs(trajectory->positions[i].x - areas->crouch_pos[j].x);
      float dist_z = abs(trajectory->positions[i].z - areas->crouch_pos[j].z);
      float height = (sinf(trajectory->positions[i].x/Areas::CROUCH_WAVE)+1.0)/2.0;
      trajectory->gait_crouch[i] = glm::mix(1.0f-height, trajectory->gait_crouch[i], 
          glm::clamp(
            ((dist_x - (areas->crouch_size[j].x/2)) + 
             (dist_z - (areas->crouch_size[j].y/2))) / 100.0f, 0.0f, 1.0f));
    }
  }
    
  /* Walls */
  for (int i = 0; i < Trajectory::LENGTH; i++) {
    trajectory->gait_bump[i] = 0.0;
    for (int j = 0; j < areas->num_walls(); j++) {
      glm::vec2 trjpoint = glm::vec2(trajectory->positions[i].x, trajectory->positions[i].z);
      glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
      float segdist = glm::length(segpoint - trjpoint);
      trajectory->gait_bump[i] = glm::max(trajectory->gait_bump[i], 1.0f-glm::clamp((segdist - areas->wall_width[j]) / 10.0f, 0.0f, 1.0f));
    } 
  }
    
  /* Trajectory Rotation */
  for (int i = 0; i < Trajectory::LENGTH; i++) {
    trajectory->rotations[i] = glm::mat3(glm::rotate(atan2f(
      trajectory->directions[i].x,
      trajectory->directions[i].z), glm::vec3(0,1,0)));
  }
    
  /* Trajectory Heights */
  for (int i = Trajectory::LENGTH/2; i < Trajectory::LENGTH; i++) {
    trajectory->positions[i].y = heightmap->sample(glm::vec2(trajectory->positions[i].x, trajectory->positions[i].z));
  }
    
  trajectory->heights[Trajectory::LENGTH/2] = 0.0;
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    trajectory->heights[Trajectory::LENGTH/2] += (trajectory->positions[i].y / ((Trajectory::LENGTH)/10));
  }
          
  glm::vec3 root_position = glm::vec3(
    trajectory->positions[Trajectory::LENGTH/2].x, 
    trajectory->heights[Trajectory::LENGTH/2],
    trajectory->positions[Trajectory::LENGTH/2].z);
          
  glm::mat3 root_rotation = trajectory->rotations[Trajectory::LENGTH/2];
      
  /* Input Trajectory Positions / Directions */
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    glm::vec3 pos = glm::inverse(root_rotation) * (trajectory->positions[i] - root_position);
    glm::vec3 dir = glm::inverse(root_rotation) * trajectory->directions[i];  
    pfnn->Xp((w*0)+i/10) = pos.x;
    pfnn->Xp((w*1)+i/10) = pos.z;
    pfnn->Xp((w*2)+i/10) = dir.x; 
    pfnn->Xp((w*3)+i/10) = dir.z;
  }

  /* Input Trajectory Positions / Directions */
  /*
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    pfnn->Xp((w*0)+i/10) = 0.0;
    pfnn->Xp((w*1)+i/10) = 0.0;
    pfnn->Xp((w*2)+i/10) = 0.0; 
    pfnn->Xp((w*3)+i/10) = 0.0;
  }
  */
    
  /* Input Trajectory Gaits */
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    pfnn->Xp((w*4)+i/10) = trajectory->gait_stand[i];
    pfnn->Xp((w*5)+i/10) = trajectory->gait_walk[i];
    pfnn->Xp((w*6)+i/10) = trajectory->gait_jog[i];
    pfnn->Xp((w*7)+i/10) = trajectory->gait_crouch[i];
    pfnn->Xp((w*8)+i/10) = trajectory->gait_jump[i];
    pfnn->Xp((w*9)+i/10) = 0.0; // Unused.
  }

  /* MANIPULATE Trajectory Gaits */
  /*
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    pfnn->Xp((w*4)+i/10) = 0.0;
    pfnn->Xp((w*5)+i/10) = 0.0;
    pfnn->Xp((w*6)+i/10) = 0.0;
    pfnn->Xp((w*7)+i/10) = 0.0;
    pfnn->Xp((w*8)+i/10) = 0.0;
    pfnn->Xp((w*9)+i/10) = 0.0; // Unused.
  }
  */

  /* Input Joint Previous Positions / Velocities / Rotations */
  glm::vec3 prev_root_position = glm::vec3(
    trajectory->positions[Trajectory::LENGTH/2-1].x, 
    trajectory->heights[Trajectory::LENGTH/2-1],
    trajectory->positions[Trajectory::LENGTH/2-1].z);
  glm::mat3 prev_root_rotation = trajectory->rotations[Trajectory::LENGTH/2-1];
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    int o = (((Trajectory::LENGTH)/10)*10);  
    glm::vec3 pos = glm::inverse(prev_root_rotation) * (character->joint_positions[i] - prev_root_position);
    glm::vec3 prv = glm::inverse(prev_root_rotation) *  character->joint_velocities[i];
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+0) = pos.x;
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+1) = pos.y;
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+2) = pos.z;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+0) = prv.x;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+1) = prv.y;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+2) = prv.z;
  }
    
  /* Input Trajectory Heights */
  for (int i = 0; i < Trajectory::LENGTH; i += 10) {
    int o = (((Trajectory::LENGTH)/10)*10)+Character::JOINT_NUM*3*2;
    int w = (Trajectory::LENGTH)/10;
    glm::vec3 position_r = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3( trajectory->width, 0, 0));
    glm::vec3 position_l = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(-trajectory->width, 0, 0));
    pfnn->Xp(o+(w*0)+(i/10)) = heightmap->sample(glm::vec2(position_r.x, position_r.z)) - root_position.y;
    pfnn->Xp(o+(w*1)+(i/10)) = trajectory->positions[i].y - root_position.y;
    pfnn->Xp(o+(w*2)+(i/10)) = heightmap->sample(glm::vec2(position_l.x, position_l.z)) - root_position.y;
  }

  /* MANIPULATE Trajectory Heights */
  /*
  for (int i = 0; i < Trajectory::LENGTH; i += 10) {
    int o = (((Trajectory::LENGTH)/10)*10)+Character::JOINT_NUM*3*2;
    int w = (Trajectory::LENGTH)/10;
    pfnn->Xp(o+(w*0)+(i/10)) = 0.0;
    pfnn->Xp(o+(w*1)+(i/10)) = 0.0;
    pfnn->Xp(o+(w*2)+(i/10)) = 0.0;
  }
  */

  //TEST
  /*
  character->phase = 0.0;
  for(int i=0; i<342; i++) {
    pfnn->Xp(i) = (float)(i+1)/342.0;
  }
  std::cout << "INPUT" << std::endl;
  for(int i=0; i<342; i++) {
    std::cout << i << ": " << pfnn->Xp(i) << " ";
  }
  std::cout << std::endl;
  
  pfnn->predict(character->phase);

  std::cout << "OUTPUT" << std::endl;
  for(int i=0; i<311; i++) {
    std::cout << i << ": " << pfnn->Yp(i) << " ";
  }
  std::cout << std::endl;
  */
  //
    
  /* Perform Regression */
  
  clock_t time_start = clock();
    
  pfnn->predict(character->phase);

  clock_t time_end = clock();

  /* Timing */
  
  enum { TIME_MSAMPLES = 500 };
  static int time_nsamples = 0;
  static float time_samples[TIME_MSAMPLES];
  
  time_samples[time_nsamples] = (float)(time_end - time_start) / CLOCKS_PER_SEC;
  time_nsamples++;
  if (time_nsamples == TIME_MSAMPLES) {
    float time_avg = 0.0;
    for (int i = 0; i < TIME_MSAMPLES; i++) {
      time_avg += (time_samples[i] / TIME_MSAMPLES);
    }
    printf("PFNN: %0.5f ms\n", time_avg * 1000);
    time_nsamples = 0;
  }
    
  /* Build Local Transforms */
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    int opos = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*0);
    int ovel = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*1);
    int orot = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*2);
    
    glm::vec3 pos = (root_rotation * glm::vec3(pfnn->Yp(opos+i*3+0), pfnn->Yp(opos+i*3+1), pfnn->Yp(opos+i*3+2))) + root_position;
    glm::vec3 vel = (root_rotation * glm::vec3(pfnn->Yp(ovel+i*3+0), pfnn->Yp(ovel+i*3+1), pfnn->Yp(ovel+i*3+2)));
    glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(pfnn->Yp(orot+i*3+0), pfnn->Yp(orot+i*3+1), pfnn->Yp(orot+i*3+2)))));

    /*
    ** Blending Between the predicted positions and
    ** the previous positions plus the velocities 
    ** smooths out the motion a bit in the case 
    ** where the two disagree with each other.
    */
    
    character->joint_positions[i]  = glm::mix(character->joint_positions[i] + vel, pos, options->extra_joint_smooth);
    character->joint_velocities[i] = vel;
    character->joint_rotations[i]  = rot;
    
    character->joint_global_anim_xform[i] = glm::transpose(glm::mat4(
      rot[0][0], rot[1][0], rot[2][0], pos[0],
      rot[0][1], rot[1][1], rot[2][1], pos[1],
      rot[0][2], rot[1][2], rot[2][2], pos[2],
              0,         0,         0,      1));
  }
  
  /* Convert to local space ... yes I know this is inefficient. */
  
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    if (i == 0) {
      character->joint_anim_xform[i] = character->joint_global_anim_xform[i];
    } else {
      character->joint_anim_xform[i] = glm::inverse(character->joint_global_anim_xform[character->joint_parents[i]]) * character->joint_global_anim_xform[i];
    }
  }
  
  character->forward_kinematics();
}

void post_render() {
            
  /* Update Past Trajectory */
  
  for (int i = 0; i < Trajectory::LENGTH/2; i++) {
    trajectory->positions[i]  = trajectory->positions[i+1];
    trajectory->directions[i] = trajectory->directions[i+1];
    trajectory->rotations[i] = trajectory->rotations[i+1];
    trajectory->heights[i] = trajectory->heights[i+1];
    trajectory->gait_stand[i] = trajectory->gait_stand[i+1];
    trajectory->gait_walk[i] = trajectory->gait_walk[i+1];
    trajectory->gait_jog[i] = trajectory->gait_jog[i+1];
    trajectory->gait_crouch[i] = trajectory->gait_crouch[i+1];
    trajectory->gait_jump[i] = trajectory->gait_jump[i+1];
    trajectory->gait_bump[i] = trajectory->gait_bump[i+1];
  }
  
  /* Update Current Trajectory */
  
  float stand_amount = powf(1.0f-trajectory->gait_stand[Trajectory::LENGTH/2], 0.25f);
  
  glm::vec3 trajectory_update = (trajectory->rotations[Trajectory::LENGTH/2] * glm::vec3(pfnn->Yp(0), 0, pfnn->Yp(1)));
  trajectory->positions[Trajectory::LENGTH/2]  = trajectory->positions[Trajectory::LENGTH/2] + stand_amount * trajectory_update;
  trajectory->directions[Trajectory::LENGTH/2] = glm::mat3(glm::rotate(stand_amount * -pfnn->Yp(2), glm::vec3(0,1,0))) * trajectory->directions[Trajectory::LENGTH/2];
  trajectory->rotations[Trajectory::LENGTH/2] = glm::mat3(glm::rotate(atan2f(
      trajectory->directions[Trajectory::LENGTH/2].x,
      trajectory->directions[Trajectory::LENGTH/2].z), glm::vec3(0,1,0)));
      
  /* Collide with walls */
      
  for (int j = 0; j < areas->num_walls(); j++) {
    glm::vec2 trjpoint = glm::vec2(trajectory->positions[Trajectory::LENGTH/2].x, trajectory->positions[Trajectory::LENGTH/2].z);
    glm::vec2 segpoint = segment_nearest(areas->wall_start[j], areas->wall_stop[j], trjpoint);
    float segdist = glm::length(segpoint - trjpoint);
    if (segdist < areas->wall_width[j] + 100.0) {
      glm::vec2 prjpoint0 = (areas->wall_width[j] +   0.0f) * glm::normalize(trjpoint - segpoint) + segpoint; 
      glm::vec2 prjpoint1 = (areas->wall_width[j] + 100.0f) * glm::normalize(trjpoint - segpoint) + segpoint; 
      glm::vec2 prjpoint = glm::mix(prjpoint0, prjpoint1, glm::clamp((segdist - areas->wall_width[j]) / 100.0f, 0.0f, 1.0f));
      trajectory->positions[Trajectory::LENGTH/2].x = prjpoint.x;
      trajectory->positions[Trajectory::LENGTH/2].z = prjpoint.y;
    }
  }

  /* Update Future Trajectory */
  
  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    int w = (Trajectory::LENGTH/2)/10;
    float m = fmod(((float)i - (Trajectory::LENGTH/2)) / 10.0, 1.0);
    trajectory->positions[i].x  = (1-m) * pfnn->Yp(8+(w*0)+(i/10)-w) + m * pfnn->Yp(8+(w*0)+(i/10)-w+1);
    trajectory->positions[i].z  = (1-m) * pfnn->Yp(8+(w*1)+(i/10)-w) + m * pfnn->Yp(8+(w*1)+(i/10)-w+1);
    trajectory->directions[i].x = (1-m) * pfnn->Yp(8+(w*2)+(i/10)-w) + m * pfnn->Yp(8+(w*2)+(i/10)-w+1);
    trajectory->directions[i].z = (1-m) * pfnn->Yp(8+(w*3)+(i/10)-w) + m * pfnn->Yp(8+(w*3)+(i/10)-w+1);
    trajectory->positions[i]    = (trajectory->rotations[Trajectory::LENGTH/2] * trajectory->positions[i]) + trajectory->positions[Trajectory::LENGTH/2];
    trajectory->directions[i]   = glm::normalize((trajectory->rotations[Trajectory::LENGTH/2] * trajectory->directions[i]));
    trajectory->rotations[i]    = glm::mat3(glm::rotate(atan2f(trajectory->directions[i].x, trajectory->directions[i].z), glm::vec3(0,1,0)));
  }
  
  /* Update Phase */

  character->phase = fmod(character->phase + (stand_amount * 0.9f + 0.1f) * 2*M_PI * pfnn->Yp(3), 2*M_PI);
  
  /* Update Camera */

  camera->target = mix_vectors(camera->target, glm::vec3(
    trajectory->positions[Trajectory::LENGTH/2].x, 
    trajectory->heights[Trajectory::LENGTH/2] + 100, 
    trajectory->positions[Trajectory::LENGTH/2].z), 0.1);
  
}

void render() {
  
  /* Render Shadows */
  
  glm::mat4 light_view = glm::lookAt(camera->target + light->position, camera->target, glm::vec3(0,1,0));
#ifdef HIGH_QUALITY
  glm::mat4 light_proj = glm::ortho(-2000.0f, 2000.0f, -2000.0f, 2000.0f, 10.0f, 10000.0f);  
#else
  glm::mat4 light_proj = glm::ortho(-500.0f, 500.0f, -500.0f, 500.0f, 10.0f, 10000.0f);
#endif
  
  glBindFramebuffer(GL_FRAMEBUFFER, light->fbo);
  
#ifdef HIGH_QUALITY
  glViewport(0, 0, 2048, 2048);
#else
  glViewport(0, 0, 1024, 1024);
#endif
  glClearDepth(1.0f);  
  glClear(GL_DEPTH_BUFFER_BIT);
  
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  
  glUseProgram(shader_character_shadow->program);
  
  glUniformMatrix4fv(glGetUniformLocation(shader_character_shadow->program, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
  glUniformMatrix4fv(glGetUniformLocation(shader_character_shadow->program, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));
  glUniformMatrix4fv(glGetUniformLocation(shader_character_shadow->program, "joints"), Character::JOINT_NUM, GL_FALSE, (float*)character->joint_mesh_xform);

  glBindBuffer(GL_ARRAY_BUFFER, character->vbo);
  
  glEnableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vPosition"));  
  glEnableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vWeightVal"));
  glEnableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vWeightIds"));

  glVertexAttribPointer(glGetAttribLocation(shader_character_shadow->program, "vPosition"),  3, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  0));
  glVertexAttribPointer(glGetAttribLocation(shader_character_shadow->program, "vWeightVal"), 4, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  7));
  glVertexAttribPointer(glGetAttribLocation(shader_character_shadow->program, "vWeightIds"), 4, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) * 11));
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, character->tbo);
  glDrawElements(GL_TRIANGLES, character->ntri, GL_UNSIGNED_INT, (void*)0);
  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vPosition"));  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vWeightVal"));
  glDisableVertexAttribArray(glGetAttribLocation(shader_character_shadow->program, "vWeightIds"));

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  glUseProgram(0);
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
#ifdef HIGH_QUALITY
  glUseProgram(shader_terrain_shadow->program);
  
  glUniformMatrix4fv(glGetUniformLocation(shader_terrain_shadow->program, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
  glUniformMatrix4fv(glGetUniformLocation(shader_terrain_shadow->program, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));
  
  glBindBuffer(GL_ARRAY_BUFFER, heightmap->vbo);
  
  glEnableVertexAttribArray(glGetAttribLocation(shader_terrain_shadow->program, "vPosition"));  
  glVertexAttribPointer(glGetAttribLocation(shader_terrain_shadow->program, "vPosition"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 0));
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, heightmap->tbo);
  glDrawElements(GL_TRIANGLES, (heightmap->data.size()-1) * (heightmap->data[0].size()-1) * 2 * 3, GL_UNSIGNED_INT, (void*)0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  
  glDisableVertexAttribArray(glGetAttribLocation(shader_terrain_shadow->program, "vPosition"));  

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  glUseProgram(0);
#endif

  glCullFace(GL_BACK);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  
  glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  
  /* Render Terrain */
  
#ifdef HIGH_QUALITY
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
  
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_POLYGON_SMOOTH);
  glEnable(GL_POINT_SMOOTH);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_MULTISAMPLE);
#endif
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  glClearDepth(1.0);
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glm::vec3 light_direction = glm::normalize(light->target - light->position);
  
  glUseProgram(shader_terrain->program);
  
  glUniformMatrix4fv(glGetUniformLocation(shader_terrain->program, "view"), 1, GL_FALSE, glm::value_ptr(camera->view_matrix()));
  glUniformMatrix4fv(glGetUniformLocation(shader_terrain->program, "proj"), 1, GL_FALSE, glm::value_ptr(camera->proj_matrix()));
  glUniform3f(glGetUniformLocation(shader_terrain->program, "light_dir"), light_direction.x, light_direction.y, light_direction.z);

  glUniformMatrix4fv(glGetUniformLocation(shader_terrain->program, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
  glUniformMatrix4fv(glGetUniformLocation(shader_terrain->program, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));
  
  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_2D, light->tex);
  glUniform1i(glGetUniformLocation(shader_terrain->program, "shadows"), 0);
  
#ifdef HIGH_QUALITY
  glUniform3f(glGetUniformLocation(shader_terrain->program, "foot0"), character->joint_positions[ 4].x, character->joint_positions[ 4].y, character->joint_positions[ 4].z);
  glUniform3f(glGetUniformLocation(shader_terrain->program, "foot1"), character->joint_positions[ 5].x, character->joint_positions[ 5].y, character->joint_positions[ 5].z);
  glUniform3f(glGetUniformLocation(shader_terrain->program, "foot2"), character->joint_positions[ 9].x, character->joint_positions[ 9].y, character->joint_positions[ 9].z);
  glUniform3f(glGetUniformLocation(shader_terrain->program, "foot3"), character->joint_positions[10].x, character->joint_positions[10].y, character->joint_positions[10].z);
  glUniform3f(glGetUniformLocation(shader_terrain->program, "hip"),   character->joint_positions[ 0].x, character->joint_positions[ 0].y, character->joint_positions[ 0].z);
#endif
  
  glBindBuffer(GL_ARRAY_BUFFER, heightmap->vbo);
  
  glEnableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vPosition"));  
  glEnableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vNormal"));
  glEnableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vAO"));

  glVertexAttribPointer(glGetAttribLocation(shader_terrain->program, "vPosition"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 0));
  glVertexAttribPointer(glGetAttribLocation(shader_terrain->program, "vNormal"),   3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 3));
  glVertexAttribPointer(glGetAttribLocation(shader_terrain->program, "vAO"), 1, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 6));
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, heightmap->tbo);
#ifdef HIGH_QUALITY
  glDrawElements(GL_TRIANGLES, (heightmap->data.size()-1) * (heightmap->data[0].size()-1) * 2 * 3, GL_UNSIGNED_INT, (void*)0);
#else
  glDrawElements(GL_TRIANGLES, ((heightmap->data.size()-1)/2) * ((heightmap->data[0].size()-1)/2) * 2 * 3, GL_UNSIGNED_INT, (void*)0);  
#endif
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  
  glDisableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vPosition"));  
  glDisableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vNormal"));
  glDisableVertexAttribArray(glGetAttribLocation(shader_terrain->program, "vAO"));

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  glUseProgram(0);
  
  /* Render Character */
  
  glUseProgram(shader_character->program);
  
  glUniformMatrix4fv(glGetUniformLocation(shader_character->program, "view"), 1, GL_FALSE, glm::value_ptr(camera->view_matrix()));
  glUniformMatrix4fv(glGetUniformLocation(shader_character->program, "proj"), 1, GL_FALSE, glm::value_ptr(camera->proj_matrix()));
  glUniform3f(glGetUniformLocation(shader_character->program, "light_dir"), light_direction.x, light_direction.y, light_direction.z);

  glUniformMatrix4fv(glGetUniformLocation(shader_character->program, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
  glUniformMatrix4fv(glGetUniformLocation(shader_character->program, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));
  
  glUniformMatrix4fv(glGetUniformLocation(shader_character->program, "joints"), Character::JOINT_NUM, GL_FALSE, (float*)character->joint_mesh_xform);

  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_2D, light->tex);
  glUniform1i(glGetUniformLocation(shader_character->program, "shadows"), 0);
  
  glBindBuffer(GL_ARRAY_BUFFER, character->vbo);
  
  glEnableVertexAttribArray(glGetAttribLocation(shader_character->program, "vPosition"));  
  glEnableVertexAttribArray(glGetAttribLocation(shader_character->program, "vNormal"));
  glEnableVertexAttribArray(glGetAttribLocation(shader_character->program, "vAO"));
  glEnableVertexAttribArray(glGetAttribLocation(shader_character->program, "vWeightVal"));
  glEnableVertexAttribArray(glGetAttribLocation(shader_character->program, "vWeightIds"));

  glVertexAttribPointer(glGetAttribLocation(shader_character->program, "vPosition"),  3, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  0));
  glVertexAttribPointer(glGetAttribLocation(shader_character->program, "vNormal"),    3, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  3));
  glVertexAttribPointer(glGetAttribLocation(shader_character->program, "vAO"),        1, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  6));
  glVertexAttribPointer(glGetAttribLocation(shader_character->program, "vWeightVal"), 4, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) *  7));
  glVertexAttribPointer(glGetAttribLocation(shader_character->program, "vWeightIds"), 4, GL_FLOAT, GL_FALSE, sizeof(float) * 15, (void*)(sizeof(float) * 11));
  
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, character->tbo);
  glDrawElements(GL_TRIANGLES, character->ntri, GL_UNSIGNED_INT, (void*)0);
  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character->program, "vPosition"));  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character->program, "vNormal"));  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character->program, "vAO"));  
  glDisableVertexAttribArray(glGetAttribLocation(shader_character->program, "vWeightVal"));
  glDisableVertexAttribArray(glGetAttribLocation(shader_character->program, "vWeightIds"));

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  glUseProgram(0);
  
  /* Render the Rest */
  
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(glm::value_ptr(camera->view_matrix()));
  
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(glm::value_ptr(camera->proj_matrix()));
  
  /* Render Crouch Area */

  if (options->display_debug) {
    for (int i = 0; i < areas->num_crouches(); i++) {

      glm::vec3 c0 = areas->crouch_pos[i] + glm::vec3( areas->crouch_size[i].x/2, 0,  areas->crouch_size[i].y/2);
      glm::vec3 c1 = areas->crouch_pos[i] + glm::vec3(-areas->crouch_size[i].x/2, 0,  areas->crouch_size[i].y/2);
      glm::vec3 c2 = areas->crouch_pos[i] + glm::vec3( areas->crouch_size[i].x/2, 0, -areas->crouch_size[i].y/2);
      glm::vec3 c3 = areas->crouch_pos[i] + glm::vec3(-areas->crouch_size[i].x/2, 0, -areas->crouch_size[i].y/2);

      glColor3f(0.0, 0.0, 1.0);
      glLineWidth(options->display_scale * 2.0);
      glBegin(GL_LINES);
      glVertex3f(c0.x, c0.y, c0.z); glVertex3f(c1.x, c1.y, c1.z);
      glVertex3f(c0.x, c0.y, c0.z); glVertex3f(c2.x, c2.y, c2.z);
      glVertex3f(c3.x, c3.y, c3.z); glVertex3f(c1.x, c1.y, c1.z);
      glVertex3f(c3.x, c3.y, c3.z); glVertex3f(c2.x, c2.y, c2.z);
      
      for (float j = 0; j < 1.0; j+=0.05) {
        glm::vec3 cm_a = mix_vectors(c0, c1, j     );
        glm::vec3 cm_b = mix_vectors(c0, c1, j+0.05);
        glm::vec3 cm_c = mix_vectors(c3, c2, j     );
        glm::vec3 cm_d = mix_vectors(c3, c2, j+0.05);
        float cmh_a = ((sinf(cm_a.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;        
        float cmh_b = ((sinf(cm_b.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
        float cmh_c = ((sinf(cm_c.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;        
        float cmh_d = ((sinf(cm_d.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
        glVertex3f(cm_a.x, cmh_a, cm_a.z);
        glVertex3f(cm_b.x, cmh_b, cm_b.z);
        glVertex3f(cm_c.x, cmh_c, cm_c.z);
        glVertex3f(cm_d.x, cmh_d, cm_d.z);
        glVertex3f(cm_a.x, cmh_a, cm_a.z);
        glVertex3f(cm_a.x, cmh_a,   c2.z);
        if (j + 0.05 >= 1.0) {
          glVertex3f(cm_b.x, cmh_b, cm_b.z);
          glVertex3f(cm_b.x, cmh_b,   c2.z);
        }
      }
      
      float c0h = ((sinf(c0.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
      float c1h = ((sinf(c1.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
      float c2h = ((sinf(c2.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
      float c3h = ((sinf(c3.x/Areas::CROUCH_WAVE)+1.0)/2.0) * 50 + 130;
      
      glVertex3f(c0.x, c0.y + c0h, c0.z); glVertex3f(c0.x, c0.y, c0.z);
      glVertex3f(c1.x, c1.y + c1h, c1.z); glVertex3f(c1.x, c1.y, c1.z);
      glVertex3f(c2.x, c2.y + c2h, c2.z); glVertex3f(c2.x, c2.y, c2.z);
      glVertex3f(c3.x, c3.y + c3h, c3.z); glVertex3f(c3.x, c3.y, c3.z);
      
      glEnd();
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 1.0);
    }
  }

  /* Render Jump Areas */

  if (options->display_debug && options->display_areas_jump) {
    for (int i = 0; i < areas->num_jumps(); i++) {
      glColor3f(1.0, 0.0, 0.0);
      glLineWidth(options->display_scale * 2.0);
      glBegin(GL_LINES);
      for (float r = 0; r < 1.0; r+=0.01) {
        glVertex3f(areas->jump_pos[i].x + areas->jump_size[i] * sin((r+0.00)*2*M_PI), areas->jump_pos[i].y, areas->jump_pos[i].z + areas->jump_size[i] * cos((r+0.00)*2*M_PI));
        glVertex3f(areas->jump_pos[i].x + areas->jump_size[i] * sin((r+0.01)*2*M_PI), areas->jump_pos[i].y, areas->jump_pos[i].z + areas->jump_size[i] * cos((r+0.01)*2*M_PI));
      }
      glEnd();
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 1.0);
    }
  }

  /* Render Walls */
  
  if (options->display_debug && options->display_areas_walls) {
    for (int i = 0; i < areas->num_walls(); i++) {
      glColor3f(0.0, 1.0, 0.0);
      glLineWidth(options->display_scale * 2.0);
      glBegin(GL_LINES);
      for (float r = 0; r < 1.0; r+=0.1) {
        glm::vec2 p0 = mix_vectors(areas->wall_start[i], areas->wall_stop[i], r    );
        glm::vec2 p1 = mix_vectors(areas->wall_start[i], areas->wall_stop[i], r+0.1);
        glVertex3f(p0.x, heightmap->sample(p0) + 5, p0.y);
        glVertex3f(p1.x, heightmap->sample(p1) + 5, p1.y);
      }
      glEnd();
      glLineWidth(1.0);
      glColor3f(1.0, 1.0, 1.0);
    }

  }
  
  /* Render Trajectory */
  
  if (options->display_debug) {
    glPointSize(1.0 * options->display_scale);
    glBegin(GL_POINTS);
    for (int i = 0; i < Trajectory::LENGTH-10; i++) {
      glm::vec3 position_c = trajectory->positions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(position_c.x, position_c.y + 2.0, position_c.z);
    }
    glEnd();
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(1.0);

    
    glPointSize(4.0 * options->display_scale);
    glBegin(GL_POINTS);
    for (int i = 0; i < Trajectory::LENGTH; i+=10) {
      glm::vec3 position_c = trajectory->positions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(position_c.x, position_c.y + 2.0, position_c.z);
    }
    glEnd();
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(1.0);

    if (options->display_debug_heights) {
      glPointSize(2.0 * options->display_scale);
      glBegin(GL_POINTS);
      for (int i = 0; i < Trajectory::LENGTH; i+=10) {
        glm::vec3 position_r = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3( trajectory->width, 0, 0));
        glm::vec3 position_l = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(-trajectory->width, 0, 0));
        glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
        glVertex3f(position_r.x, heightmap->sample(glm::vec2(position_r.x, position_r.z)) + 2.0, position_r.z);
        glVertex3f(position_l.x, heightmap->sample(glm::vec2(position_l.x, position_l.z)) + 2.0, position_l.z);
      }
      glEnd();
      glColor3f(1.0, 1.0, 1.0);
      glPointSize(1.0);
    }
    
    glLineWidth(1.0 * options->display_scale);
    glBegin(GL_LINES);
    for (int i = 0; i < Trajectory::LENGTH; i+=10) {
      glm::vec3 base = trajectory->positions[i] + glm::vec3(0.0, 2.0, 0.0);
      glm::vec3 side = glm::normalize(glm::cross(trajectory->directions[i], glm::vec3(0.0, 1.0, 0.0)));
      glm::vec3 fwrd = base + 15.0f * trajectory->directions[i];
      fwrd.y = heightmap->sample(glm::vec2(fwrd.x, fwrd.z)) + 2.0;
      glm::vec3 arw0 = fwrd +  4.0f * side + 4.0f * -trajectory->directions[i];
      glm::vec3 arw1 = fwrd -  4.0f * side + 4.0f * -trajectory->directions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(base.x, base.y, base.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(arw0.x, fwrd.y, arw0.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(arw1.x, fwrd.y, arw1.z);
    }
    glEnd();
    glLineWidth(1.0);
    glColor3f(1.0, 1.0, 1.0);
  }
  
  /* Render Joints */
  
  if (options->display_debug && options->display_debug_joints) {
    glDisable(GL_DEPTH_TEST);
    glPointSize(3.0 * options->display_scale);
    glColor3f(0.6, 0.3, 0.4);      
    glBegin(GL_POINTS);
    for (int i = 0; i < Character::JOINT_NUM; i++) {
      glm::vec3 pos = character->joint_positions[i];
      glVertex3f(pos.x, pos.y, pos.z);
    }
    glEnd();
    glPointSize(1.0);

    glLineWidth(1.0 * options->display_scale);
    glColor3f(0.6, 0.3, 0.4);      
    glBegin(GL_LINES);
    for (int i = 0; i < Character::JOINT_NUM; i++) {
      glm::vec3 pos = character->joint_positions[i];
      glm::vec3 vel = pos - 5.0f * character->joint_velocities[i];
      glVertex3f(pos.x, pos.y, pos.z);
      glVertex3f(vel.x, vel.y, vel.z);
    }
    glEnd();
    glLineWidth(1.0);
    glEnable(GL_DEPTH_TEST);
  }
  
  /* UI Elements */

  glm::mat4 ui_view = glm::mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
  glm::mat4 ui_proj = glm::ortho(0.0f, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT, 0.0f, 0.0f, 1.0f);  
  
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(glm::value_ptr(ui_proj));

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(glm::value_ptr(ui_view));
  
  /* PFNN Visual */
  
  if (options->display_debug && options->display_debug_pfnn) {
    
    glColor3f(0.0, 0.0, 0.0);  

    glLineWidth(5);
    glBegin(GL_LINES);
    for (float i = 0; i < 2*M_PI; i+=0.01) {
      glVertex3f(WINDOW_WIDTH-125+50*cos(i     ),100+50*sin(i     ),0);    
      glVertex3f(WINDOW_WIDTH-125+50*cos(i+0.01),100+50*sin(i+0.01),0);
    }
    glEnd();
    glLineWidth(1);
    
    glPointSize(20);
    glBegin(GL_POINTS);
    glVertex3f(WINDOW_WIDTH-125+50*cos(character->phase), 100+50*sin(character->phase), 0);
    glEnd();
    glPointSize(1);
    
    glColor3f(1.0, 1.0, 1.0); 

    int pindex_1 = (int)((character->phase / (2*M_PI)) * 50);
    MatrixXf W0p = pfnn->W0[pindex_1];

    glPointSize(1);
    glBegin(GL_POINTS);
    
    for (int x = 0; x < W0p.rows(); x++)
    for (int y = 0; y < W0p.cols(); y++) {
      float v = (W0p(x, y)+0.5)/2.0;
      glm::vec3 col = v > 0.5 ? mix_vectors(glm::vec3(1,0,0), glm::vec3(0,0,1), v-0.5) : mix_vectors(glm::vec3(0,1,0), glm::vec3(0,0,1), v*2.0);
      glColor3f(col.x, col.y, col.z); 
      glVertex3f(WINDOW_WIDTH-W0p.cols()+y-25, x+175, 0);
    }

    glEnd();
    glPointSize(1);
    
  }
  
  /* Display UI */
  
  if (options->display_debug && options->display_hud_options) {
    glLineWidth(3);
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0); 
    glVertex3f(125+20,20,0); glVertex3f(125+20,40,0); /* I */
    glVertex3f(125+25,20,0); glVertex3f(125+25,40,0); /* K */
    glVertex3f(125+25,30,0); glVertex3f(125+30,40,0);
    glVertex3f(125+25,30,0); glVertex3f(125+30,20,0);
    glEnd();
    glLineWidth(1);
    
    if (options->enable_ik) { glColor3f(0.0, 1.0, 0.0); } else { glColor3f(1.0, 0.0, 0.0); }
    glPointSize(10);
    glBegin(GL_POINTS);
    glVertex3f(125+60, 30, 0);
    glEnd();
    glPointSize(1); 
    glColor3f(1.0, 1.0, 1.0); 
  }
  
  if (options->display_debug && options->display_hud_options) {
    glLineWidth(3);
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0); 
    glVertex3f(125+20,50,0); glVertex3f(125+20,70,0); /* D */
    glVertex3f(125+20,50,0); glVertex3f(125+25,55,0);
    glVertex3f(125+20,70,0); glVertex3f(125+25,65,0);
    glVertex3f(125+25,65,0); glVertex3f(125+25,55,0);
    glVertex3f(125+30,50,0); glVertex3f(125+30,70,0); /* I */
    glVertex3f(125+35,50,0); glVertex3f(125+35,70,0); /* R */
    glVertex3f(125+35,60,0); glVertex3f(125+40,70,0);
    glVertex3f(125+35,50,0); glVertex3f(125+40,55,0);
    glVertex3f(125+35,60,0); glVertex3f(125+40,55,0);
    glEnd();
    glLineWidth(1);
    
    glColor3f(1.0-character->strafe_amount, character->strafe_amount, 0.0);
    glPointSize(10);
    glBegin(GL_POINTS);
    glVertex3f(125+60, 60, 0);
    glEnd();
    glPointSize(1); 
    glColor3f(1.0, 1.0, 1.0); 
  }
  
  if (options->display_debug && options->display_hud_options) {
    glLineWidth(3);
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0); 
    glVertex3f(125+20,80,0); glVertex3f(125+20,100,0); /* R */
    glVertex3f(125+20,90,0); glVertex3f(125+25,100,0);
    glVertex3f(125+20,80,0); glVertex3f(125+25,85,0);
    glVertex3f(125+20,90,0); glVertex3f(125+25,85,0);
    glVertex3f(125+30,80,0); glVertex3f(125+30,100,0); /* E */
    glVertex3f(125+30,80,0); glVertex3f(125+35,80,0); 
    glVertex3f(125+30,90,0); glVertex3f(125+35,90,0); 
    glVertex3f(125+30,100,0); glVertex3f(125+35,100,0); 
    glVertex3f(125+40,80,0); glVertex3f(125+40,90,0); /* S */
    glVertex3f(125+45,90,0); glVertex3f(125+45,100,0);
    glVertex3f(125+40,80,0); glVertex3f(125+45,80,0); 
    glVertex3f(125+40,90,0); glVertex3f(125+45,90,0); 
    glVertex3f(125+40,100,0); glVertex3f(125+45,100,0); 
    glEnd();
    glLineWidth(1);
    
    glColor3f(1.0-character->responsive, character->responsive, 0.0);
    glPointSize(10);
    glBegin(GL_POINTS);
    glVertex3f(125+60, 90, 0);
    glEnd();
    glPointSize(1); 
    glColor3f(1.0, 1.0, 1.0); 
  }
  

  if (options->display_debug && options->display_hud_stick) {
    glColor3f(0.0, 0.0, 0.0); 
    glPointSize(1 * options->display_scale);
    glBegin(GL_POINTS);
    for (float i = 0; i < 1.0; i+=0.025) {
      glVertex3f(60+40*cos(2*M_PI*(i     )),60+40*sin(2*M_PI*(i     )), 0);    
    }
    glEnd();

    int x_vel = -X;
    int y_vel = -Y;
    glm::vec2 direction = glm::vec2((-x_vel) / 32768.0f, (-y_vel) / 32768.0f);
    glLineWidth(1 * options->display_scale);    
    glBegin(GL_LINES);
    glVertex3f(60, 60, 0);
    glVertex3f(60+direction.x*40, 60+direction.y*40, 0);
    glEnd();
    glLineWidth(1.0);    

    glPointSize(3 * options->display_scale);
    glBegin(GL_POINTS);
    glVertex3f(60, 60, 0);
    glVertex3f(60+direction.x*40, 60+direction.y*40, 0);
    glEnd();
    glPointSize(1);
    
    float speed0 = ((SDL_JoystickGetAxis(stick, 5) / 32768.0) + 1.0) / 2.0;

    glPointSize(1 * options->display_scale);
    glBegin(GL_POINTS);
    for (float i = 0; i < 1.0; i+=0.09) {
      glVertex3f(120, 100 - i * 80, 0);    
    }
    glEnd();
    
    glLineWidth(1 * options->display_scale);    
    glBegin(GL_LINES);
    glVertex3f(120, 100, 0);
    glVertex3f(120, 100 - speed0 * 80, 0);
    glEnd();
    glLineWidth(1.0); 
    
    glPointSize(3 * options->display_scale);
    glBegin(GL_POINTS);
    glVertex3f(120, 100, 0);
    glVertex3f(120, 100 - speed0 * 80, 0);
    glEnd();
    glPointSize(1);
    
    glColor3f(1.0, 1.0, 1.0); 
  }
  
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  
#ifdef HIGH_QUALITY
  glDisable(GL_MULTISAMPLE);

  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glDisable(GL_POINT_SMOOTH);
  
  glDisable(GL_BLEND);  
#endif
  
}

int main(int argc, char **argv) {
  
  /* Init SDL */
  
  SDL_Init(SDL_INIT_VIDEO|SDL_INIT_JOYSTICK);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
#ifdef HIGH_QUALITY
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8);
#endif
  SDL_Window *window = SDL_CreateWindow(
      "PFNN",
      SDL_WINDOWPOS_CENTERED, 
      SDL_WINDOWPOS_CENTERED,
      WINDOW_WIDTH, WINDOW_HEIGHT,
      SDL_WINDOW_OPENGL);
      
  if (!window) {
      printf("Could not create window: %s\n", SDL_GetError());
      return 1;
  }
  
  SDL_GLContext context = SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(1);
  
  stick = SDL_JoystickOpen(0);

  /* Init GLEW */
  
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    exit(1);
  }
  
  /* Resources */
  
  options = new Options();
  camera = new CameraOrbit();
  light = new LightDirectional();
  
  character = new Character();
  character->load(
    "./network/character_vertices.bin", 
    "./network/character_triangles.bin", 
    "./network/character_parents.bin", 
    "./network/character_xforms.bin");
  
  trajectory = new Trajectory();
  ik = new IK();
  
  shader_terrain = new Shader();
  shader_terrain_shadow = new Shader();
  shader_character = new Shader();
  shader_character_shadow = new Shader();
#ifdef HIGH_QUALITY
  shader_terrain->load("./shaders/terrain.vs", "./shaders/terrain.fs");
  shader_terrain_shadow->load("./shaders/terrain_shadow.vs", "./shaders/terrain_shadow.fs");
  shader_character->load("./shaders/character.vs", "./shaders/character.fs");
  shader_character_shadow->load("./shaders/character_shadow.vs", "./shaders/character_shadow.fs");
#else
  shader_terrain->load("./shaders/terrain.vs", "./shaders/terrain_low.fs");
  shader_terrain_shadow->load("./shaders/terrain_shadow.vs", "./shaders/terrain_shadow.fs");
  shader_character->load("./shaders/character.vs", "./shaders/character_low.fs");
  shader_character_shadow->load("./shaders/character_shadow.vs", "./shaders/character_shadow.fs");
#endif

  heightmap = new Heightmap();
  areas = new Areas();
  
  pfnn = new PFNN(PFNN::MODE_CONSTANT);
  //pfnn = new PFNN(PFNN::MODE_CUBIC);
  //pfnn = new PFNN(PFNN::MODE_LINEAR);
  pfnn->load();

  load_world0();
  
  /* Game Loop */
  
  static bool running = true;
  
  while (running) {
    
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
            
      if (event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
          case SDLK_w: W = true; break;
          case SDLK_a: A = true; break;
          case SDLK_s: S = true; break;
          case SDLK_d: D = true; break;
          case SDLK_ESCAPE: running = false; break;
          case SDLK_1: load_world0(); break;
          case SDLK_2: load_world1(); break;
          case SDLK_3: load_world2(); break;
          case SDLK_4: load_world3(); break;
          case SDLK_5: load_world4(); break;
          case SDLK_6: load_world5(); break;
          case SDLK_y: options->enable_ik = !options->enable_ik; break;
          case SDLK_x: options->display_debug = !options->display_debug; break;
          case SDLK_c: character->responsive = !character->responsive; break;
          case SDLK_v: character->crouched_target = character->crouched_target ? 0.0 : 1.0; break;
        }
      }
    
      if (event.type == SDL_KEYUP) {
        switch (event.key.keysym.sym) {
          case SDLK_w: W = false; break;
          case SDLK_a: A = false; break;
          case SDLK_s: S = false; break;
          case SDLK_d: D = false; break;
        }
      }
    }

    X = 0;
    Y = 0;
    if(W) {
      Y -= 1;
    }
    if(S) {
      Y += 1;
    }
    if(A) {
      X -= 1;
    }
    if(D) {
      X += 1;
    }
    
    pre_render();
    render();
    post_render();
    
    glFlush();
    glFinish();
    
    SDL_GL_SwapWindow(window);
  }

  /* Delete Resources */
  
  delete options;
  delete camera;
  delete light;
  delete character;
  delete trajectory;
  delete ik;
  delete shader_terrain;
  delete shader_terrain_shadow;
  delete shader_character;
  delete shader_character_shadow;
  delete heightmap;
  delete areas;
  delete pfnn;
  
  SDL_JoystickClose(stick);
  SDL_GL_DeleteContext(context);  
  SDL_DestroyWindow(window);
  SDL_Quit();
  
  return 0;
}