// ============================================================================
//  blackhole.cpp — трёхмерная симуляция чёрной дыры Шварцшильда
//  Релятивистская трассировка лучей на видеокарте (OpenGL 3.3 + GLSL)
// ============================================================================
//
//  Что моделируется
//   * Точные нулевые геодезические в метрике Шварцшильда (RK4, адаптивный шаг):
//     гравитационное линзирование, тень, фотонное кольцо, вторичные и третичные
//     изображения диска, линзированное звёздное небо (кольцо Эйнштейна).
//   * Камера — статичный наблюдатель: направления лучей заданы в его локальной
//     ортонормированной системе отсчёта (учтена аберрация вблизи дыры).
//   * Тонкий аккреционный диск от ISCO (r = 6M) с профилем температуры
//     Новикова–Торна и спектром абсолютно чёрного тела (закон Планка,
//     проинтегрированный с функциями цветового восприятия CIE 1931).
//   * Точный частотный множитель g = ν_набл / ν_изл: эффект Доплера,
//     релятивистское усиление и гравитационное красное смещение.
//   * Турбулентный диск (облака, волокна, спиральные волны), вращающийся
//     дифференциально — по Кеплеру, Ω = r^(-3/2).
//   * Звёздное небо и Млечный Путь. Мелкие детали фильтруются по реальному
//     «следу» пикселя (лучевые дифференциалы через dFdx/dFdy), поэтому линза
//     не превращает небо в «песок», а движение камеры — в разводы.
//   * HDR, многоуровневый bloom, ACES, накопительное сглаживание на паузе.
//
//  Каждый пиксель трассируется на видеокарте каждый кадр в полном разрешении.
//  Если видеокарта не успевает (< ~22 кадр/с), разрешение рендера плавно
//  снижается (до 50 %) и возвращается к 100 %, когда запас появляется (Q — выкл.).
//
//  Единицы: G = c = M = 1. Горизонт r = 2, фотонная сфера r = 3, ISCO r = 6.
//
//  Сборка в Visual Studio
//   1. Проект «Консольное приложение» (C++), заменить его .cpp этим файлом.
//   2. Конфигурация Release, платформа x64.
//   Из Developer Command Prompt: cl /O2 /EHsc blackhole.cpp
//   Требуется видеокарта с OpenGL 3.3 (любая за последние ~12 лет).
//
//  Запуск
//   blackhole.exe                         — интерактивное окно
//   blackhole.exe --render bh.bmp --w 3840 --h 2160 --spp 64
//                 [--dist 34 --pitch 7 --yaw 0 --fov 55 --time 0 --temp 7000
//                  --exposure 1 --no-doppler --no-redshift --no-lensing --no-bloom]
//                                         — кадр высокого качества в файл
//
//  Управление в окне
//   ЛКМ + мышь — вращение камеры, колесо / ПКМ + мышь — расстояние,
//   Пробел — пауза времени (включает сглаживание до 64 выборок), A — автоорбита,
//   D — эффект Доплера, G — гравитационное смещение, L — линзирование,
//   B — bloom, Q — адаптивное разрешение, ↑/↓ — температура диска,
//   ←/→ — скорость времени, PgUp/PgDn — экспозиция, [ / ] — угол обзора,
//   R — сброс камеры, P или F12 — скриншот (BMP), H — подсказки, Esc — выход.
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#include <GL/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "opengl32.lib")

// На ноутбуках с двумя видеокартами просим драйвер выбрать дискретную
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

namespace {

// ============================================================================
//  Математика и утилиты
// ============================================================================

constexpr double kPi = 3.14159265358979323846;

struct Vec3 {
    double x, y, z;
    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
};

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline double length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) { return v * (1.0 / length(v)); }
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

using Clock = std::chrono::steady_clock;
inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Субпиксельные смещения — низкодисперсная последовательность R2
inline double r2x(int n) { return std::fmod(0.5 + 0.7548776662466927 * n, 1.0); }
inline double r2y(int n) { return std::fmod(0.5 + 0.5698402909980532 * n, 1.0); }

// Вывод в консоль (UTF-16 в консоль, UTF-8 при перенаправлении в файл)
void say(const wchar_t* fmt, ...) {
    wchar_t buf[4096];
    va_list ap;
    va_start(ap, fmt);
    const int n = vswprintf(buf, 4096, fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == nullptr || h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    if (!WriteConsoleW(h, buf, (DWORD)n, &written, nullptr)) {
        std::vector<char> out((size_t)n * 4 + 4);
        const int m = WideCharToMultiByte(CP_UTF8, 0, buf, n, out.data(), (int)out.size(), nullptr, nullptr);
        if (m > 0) WriteFile(h, out.data(), (DWORD)m, &written, nullptr);
    }
}

// ============================================================================
//  Излучение абсолютно чёрного тела → цвет sRGB (таблица уходит в текстуру)
// ============================================================================

constexpr int    kBBN = 1024;
constexpr double kBBTmin = 300.0;
constexpr double kBBTmax = 60000.0;

// Аналитическая аппроксимация функций CIE 1931 (Wyman, Sloan, Shirley, 2013)
double cieLobe(double l, double mu, double s1, double s2) {
    const double t = (l - mu) / (l < mu ? s1 : s2);
    return std::exp(-0.5 * t * t);
}

class Blackbody {
public:
    void init() {
        logMin_ = std::log(kBBTmin);
        logMax_ = std::log(kBBTmax);
        table_.assign((size_t)kBBN * 4, 0.0f);
        lum_.assign(kBBN, 0.0);
        for (int i = 0; i < kBBN; ++i) {
            const double T = std::exp(logMin_ + (logMax_ - logMin_) * i / (kBBN - 1));
            double X = 0.0, Y = 0.0, Z = 0.0;
            for (double lam = 360.0; lam <= 830.0; lam += 2.0) {
                const double l = lam * 1e-3;                 // мкм
                const double e = 14387.77 / (l * T);         // hc / (λ k T)
                const double B = e > 700.0 ? 0.0 : 1.0 / (l * l * l * l * l * (std::exp(e) - 1.0));
                X += B * (1.056 * cieLobe(lam, 599.8, 37.9, 31.0) + 0.362 * cieLobe(lam, 442.0, 16.0, 26.7)
                        - 0.065 * cieLobe(lam, 501.1, 20.4, 26.2));
                Y += B * (0.821 * cieLobe(lam, 568.8, 46.9, 40.5) + 0.286 * cieLobe(lam, 530.9, 16.3, 31.1));
                Z += B * (1.217 * cieLobe(lam, 437.0, 11.8, 36.0) + 0.681 * cieLobe(lam, 459.0, 26.0, 13.8));
            }
            // XYZ → линейный sRGB (D65); цвета вне охвата обрезаются
            const double r =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
            const double g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
            const double b =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;
            table_[(size_t)i * 4 + 0] = (float)std::max(r, 0.0);
            table_[(size_t)i * 4 + 1] = (float)std::max(g, 0.0);
            table_[(size_t)i * 4 + 2] = (float)std::max(b, 0.0);
            table_[(size_t)i * 4 + 3] = (float)Y;
            lum_[i] = Y;
        }
    }

    double luminance(double T) const {
        if (T < kBBTmin) return 0.0;
        const double x = (std::log(clampd(T, kBBTmin, kBBTmax)) - logMin_) / (logMax_ - logMin_) * (kBBN - 1);
        const int i = std::min((int)x, kBBN - 2);
        const double f = x - i;
        double y = lum_[i] + (lum_[i + 1] - lum_[i]) * f;
        if (T > kBBTmax) y *= T / kBBTmax;
        return y;
    }

    const std::vector<float>& table() const { return table_; }   // RGB + яркость, ось — ln T

private:
    double logMin_ = 0.0, logMax_ = 1.0;
    std::vector<float> table_;
    std::vector<double> lum_;
};

Blackbody gBB;

// ============================================================================
//  Параметры сцены и камера
// ============================================================================

// Профиль температуры тонкого диска (Шакура–Сюняев / Новиков–Торн), без нормировки
inline double ntProfile(double r, double rIn) {
    if (r <= rIn) return 0.0;
    return std::pow(r, -0.75) * std::pow(1.0 - std::sqrt(rIn / r), 0.25);
}

struct Settings {
    double rIn = 6.0;            // внутренний край диска — ISCO
    double rOut = 26.0;          // внешний край диска
    double diskTmax = 7000.0;    // максимальная температура диска, K
    double tau0 = 1.3;           // оптическая толщина диска при нормальном падении
    double diskGain = 1.5;       // яркость диска
    double starGain = 1.0;       // яркость звёзд
    double exposure = 1.0;
    double bloom = 0.15;
    double timeSpeed = 12.0;     // скорость времени, M в секунду
    bool doppler = true;         // эффект Доплера + релятивистское усиление
    bool gravShift = true;       // гравитационное красное/синее смещение
    bool lensing = true;         // искривление лучей
    bool bloomOn = true;

    // производные величины — пересчитываются в prepare()
    double ntMax = 1.0;
    double lumNorm = 1.0;

    void prepare() {
        ntMax = ntProfile(49.0 / 36.0 * rIn, rIn);           // максимум профиля при r = 49/36·r_in
        lumNorm = 1.0 / std::max(gBB.luminance(diskTmax), 1e-300);
    }
};

struct Camera {
    double dist = 34.0;
    double yaw = 0.0;
    double pitch = 0.12;                    // угол над плоскостью диска, рад
    double fov = 55.0 * kPi / 180.0;        // вертикальный угол обзора
    Vec3 pos, fwd, right, up;

    Camera() { update(); }

    void update() {
        pos = Vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)) * dist;
        fwd = normalize(-pos);
        right = normalize(cross(fwd, Vec3(0.0, 1.0, 0.0)));
        up = cross(right, fwd);
    }

    double pixelAngle(int h) const { return 2.0 * std::tan(fov * 0.5) / h; }
};

// ============================================================================
//  Загрузка функций OpenGL 3.3 (без сторонних библиотек)
// ============================================================================

typedef char GLchar;

#define BH_GL_FUNCS(X)                                                                   \
    X(GLuint, CreateShader, (GLenum))                                                    \
    X(void, ShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*))         \
    X(void, CompileShader, (GLuint))                                                     \
    X(void, GetShaderiv, (GLuint, GLenum, GLint*))                                       \
    X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                      \
    X(GLuint, CreateProgram, (void))                                                     \
    X(void, AttachShader, (GLuint, GLuint))                                              \
    X(void, LinkProgram, (GLuint))                                                       \
    X(void, GetProgramiv, (GLuint, GLenum, GLint*))                                      \
    X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                     \
    X(void, UseProgram, (GLuint))                                                        \
    X(void, DeleteShader, (GLuint))                                                      \
    X(void, DeleteProgram, (GLuint))                                                     \
    X(GLint, GetUniformLocation, (GLuint, const GLchar*))                                \
    X(void, Uniform1i, (GLint, GLint))                                                   \
    X(void, Uniform1f, (GLint, GLfloat))                                                 \
    X(void, Uniform2f, (GLint, GLfloat, GLfloat))                                        \
    X(void, Uniform3f, (GLint, GLfloat, GLfloat, GLfloat))                               \
    X(void, GenVertexArrays, (GLsizei, GLuint*))                                         \
    X(void, BindVertexArray, (GLuint))                                                   \
    X(void, DeleteVertexArrays, (GLsizei, const GLuint*))                                \
    X(void, GenFramebuffers, (GLsizei, GLuint*))                                         \
    X(void, BindFramebuffer, (GLenum, GLuint))                                           \
    X(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))               \
    X(GLenum, CheckFramebufferStatus, (GLenum))                                          \
    X(void, DeleteFramebuffers, (GLsizei, const GLuint*))                                \
    X(void, ActiveTexture, (GLenum))                                                     \
    X(void, BlendColor, (GLfloat, GLfloat, GLfloat, GLfloat))

namespace gl3 {
#define BH_DECLARE(ret, name, args) typedef ret(APIENTRY* PFN_##name) args; PFN_##name name = nullptr;
BH_GL_FUNCS(BH_DECLARE)
#undef BH_DECLARE

constexpr GLenum FRAGMENT_SHADER = 0x8B30;
constexpr GLenum VERTEX_SHADER = 0x8B31;
constexpr GLenum COMPILE_STATUS = 0x8B81;
constexpr GLenum LINK_STATUS = 0x8B82;
constexpr GLenum FRAMEBUFFER = 0x8D40;
constexpr GLenum COLOR_ATTACHMENT0 = 0x8CE0;
constexpr GLenum FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr GLenum RGBA16F = 0x881A;
constexpr GLenum RGBA32F = 0x8814;
constexpr GLenum CLAMP_TO_EDGE = 0x812F;
constexpr GLenum BGRA = 0x80E1;
constexpr GLenum TEXTURE0 = 0x84C0;
constexpr GLenum CONSTANT_ALPHA = 0x8003;
constexpr GLenum ONE_MINUS_CONSTANT_ALPHA = 0x8004;
}  // namespace gl3

typedef HGLRC(WINAPI* PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);
typedef BOOL(WINAPI* PFN_wglSwapIntervalEXT)(int);

void* glProc(const char* name) {
    void* p = (void*)wglGetProcAddress(name);
    const intptr_t v = (intptr_t)p;
    if (v == 0 || v == 1 || v == 2 || v == 3 || v == -1) {
        HMODULE lib = GetModuleHandleW(L"opengl32.dll");
        p = lib ? (void*)GetProcAddress(lib, name) : nullptr;
    }
    return p;
}

bool loadGL() {
    bool ok = true;
#define BH_LOAD(ret, name, args) gl3::name = (gl3::PFN_##name)glProc("gl" #name); if (!gl3::name) ok = false;
    BH_GL_FUNCS(BH_LOAD)
#undef BH_LOAD
    return ok;
}

// Контекст OpenGL 3.3 core для окна
bool createGLContext(HWND hwnd, HDC& dc, HGLRC& rc) {
    dc = GetDC(hwnd);
    rc = nullptr;
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    const int pf = ChoosePixelFormat(dc, &pfd);
    if (pf == 0 || !SetPixelFormat(dc, pf, &pfd)) return false;

    HGLRC legacy = wglCreateContext(dc);
    if (!legacy || !wglMakeCurrent(dc, legacy)) return false;
    PFN_wglCreateContextAttribsARB createAttribs =
        (PFN_wglCreateContextAttribsARB)glProc("wglCreateContextAttribsARB");
    if (createAttribs) {
        const int attribs[] = {
            0x2091, 3,          // WGL_CONTEXT_MAJOR_VERSION_ARB
            0x2092, 3,          // WGL_CONTEXT_MINOR_VERSION_ARB
            0x9126, 0x0001,     // WGL_CONTEXT_PROFILE_MASK_ARB = CORE
            0
        };
        rc = createAttribs(dc, nullptr, attribs);
    }
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(legacy);
    if (!rc || !wglMakeCurrent(dc, rc)) return false;
    return loadGL();
}

// ============================================================================
//  Шейдеры (GLSL 3.30). Текст шейдеров — только ASCII.
// ============================================================================

const char* kVertexShader = R"GLSL(#version 330 core
out vec2 vUV;
void main() {
    // fullscreen triangle without vertex buffers
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

// ---------------------------------------------------------------- сцена
const char* kSceneShader =
R"GLSL(#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform vec2  uRes;
uniform vec2  uJitter;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;
uniform float uTanHalfFov;
uniform float uPixAngle;      // angular size of one render pixel
uniform float uRefPixAngle;   // angular size of one window pixel (star flux reference)
uniform float uTime;
uniform float uRin;
uniform float uRout;
uniform float uTau0;
uniform float uDiskTmax;
uniform float uNtMax;
uniform float uLumNorm;
uniform float uDiskGain;
uniform float uStarGain;
uniform int   uDoppler;
uniform int   uGrav;
uniform int   uLensing;
uniform sampler2D uBB;        // blackbody: rgb radiance + luminance, axis = log(T)
uniform float uBBLogMin;
uniform float uBBLogMax;

const float PI = 3.14159265;
const int   MAX_STEPS = 420;

// ------------------------------------------------------------ hashing, noise
uint hashU(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
uint hash3(ivec3 c) {
    uvec3 u = uvec3(c + ivec3(32768));
    return hashU(u.x ^ hashU(u.y ^ hashU(u.z + 0x9e3779b9u)));
}
float unitF(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }

vec3 grad3(ivec3 c) {
    uint h = hash3(c);
    return vec3(float(h & 1023u), float((h >> 10u) & 1023u), float((h >> 20u) & 1023u)) * (2.0 / 1023.0) - 1.0;
}

float gnoise(vec3 p) {
    vec3 fl = floor(p);
    ivec3 i = ivec3(fl);
    vec3 f = p - fl;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = dot(grad3(i),                  f);
    float n100 = dot(grad3(i + ivec3(1, 0, 0)), f - vec3(1.0, 0.0, 0.0));
    float n010 = dot(grad3(i + ivec3(0, 1, 0)), f - vec3(0.0, 1.0, 0.0));
    float n110 = dot(grad3(i + ivec3(1, 1, 0)), f - vec3(1.0, 1.0, 0.0));
    float n001 = dot(grad3(i + ivec3(0, 0, 1)), f - vec3(0.0, 0.0, 1.0));
    float n101 = dot(grad3(i + ivec3(1, 0, 1)), f - vec3(1.0, 0.0, 1.0));
    float n011 = dot(grad3(i + ivec3(0, 1, 1)), f - vec3(0.0, 1.0, 1.0));
    float n111 = dot(grad3(i + ivec3(1, 1, 1)), f - vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// Band-limited fbm: octaves finer than the pixel footprint fade out
// (freq * sigma = cycles per footprint for the first octave).
float fbmAA(vec3 p, int oct, float freq, float sigma) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < oct; ++i) {
        float fade = 1.0 - smoothstep(0.15, 0.5, freq * sigma);
        if (fade <= 0.0) break;
        sum += amp * fade * gnoise(p);
        p = p * 2.03 + vec3(1.7, -3.1, 0.9);
        amp *= 0.5;
        freq *= 2.03;
    }
    return sum / (1.0 - exp2(-float(oct)));
}

// ------------------------------------------------------------ blackbody
vec4 blackbody(float T) {
    if (T < 300.0) return vec4(0.0);
    float x = (log(min(T, 60000.0)) - uBBLogMin) / (uBBLogMax - uBBLogMin) * 1023.0;
    int i = int(min(floor(x), 1022.0));
    vec4 c = mix(texelFetch(uBB, ivec2(i, 0), 0), texelFetch(uBB, ivec2(i + 1, 0), 0), x - float(i));
    if (T > 60000.0) c *= T / 60000.0;   // Rayleigh-Jeans tail
    return c;
}
)GLSL"
R"GLSL(
// ------------------------------------------------------------ sky
void dirToFace(vec3 d, out int face, out vec2 uv) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) { face = d.x > 0.0 ? 0 : 1; uv = vec2(d.z, d.y) / a.x; }
    else if (a.y >= a.z)          { face = d.y > 0.0 ? 2 : 3; uv = vec2(d.x, d.z) / a.y; }
    else                          { face = d.z > 0.0 ? 4 : 5; uv = vec2(d.x, d.y) / a.z; }
}

vec3 faceToDir(int face, vec2 uv) {
    vec3 d;
    if (face == 0)      d = vec3(1.0, uv.y, uv.x);
    else if (face == 1) d = vec3(-1.0, uv.y, uv.x);
    else if (face == 2) d = vec3(uv.x, 1.0, uv.y);
    else if (face == 3) d = vec3(uv.x, -1.0, uv.y);
    else if (face == 4) d = vec3(uv.x, uv.y, 1.0);
    else                d = vec3(uv.x, uv.y, -1.0);
    return normalize(d);
}

const int   L_CELLS[3] = int[3](40, 120, 320);
const float L_PROB[3]  = float[3](0.75, 0.60, 0.35);
const float L_GAIN[3]  = float[3](14.0, 2.5, 0.4);
const float L_SHARP[3] = float[3](12.0, 6.0, 3.0);

// Stars are gaussian splats of width sigma (pixel footprint on the sky).
// Flux is conserved: a wider splat gets dimmer as (sigmaRef/sigma)^2.
// Layers whose stars are smaller than the footprint become their mean glow.
vec3 starField(vec3 d, float sigma, float sigmaRef) {
    int face;
    vec2 uv;
    dirToFace(d, face, uv);
    float inv = 1.0 / (2.0 * sigma * sigma);
    float gain = min(sigmaRef * sigmaRef / (sigma * sigma), 6.0);
    vec3 sum = vec3(0.0);
    for (int layer = 0; layer < 3; ++layer) {
        float cells = float(L_CELLS[layer]);
        float cell = 2.0 / cells;
        float resolved = 1.0 - smoothstep(0.3, 0.6, sigma / cell);
        if (resolved < 1.0) {
            float mean = L_PROB[layer] * L_GAIN[layer] / (L_SHARP[layer] + 1.0)
                       * 2.0 * PI * sigmaRef * sigmaRef / (cell * cell);
            sum += vec3(1.0, 0.86, 0.72) * (mean * (1.0 - resolved));
        }
        if (resolved <= 0.0) continue;
        ivec2 ic = ivec2((uv * 0.5 + 0.5) * cells);
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                ivec2 c = ic + ivec2(i, j);
                if (c.x < 0 || c.y < 0 || c.x >= L_CELLS[layer] || c.y >= L_CELLS[layer]) continue;
                uint h = hash3(ivec3(c, face * 4 + layer));
                if (unitF(h) > L_PROB[layer]) continue;
                uint h2 = hashU(h ^ 0x68bc21ebu);
                uint h3 = hashU(h2);
                uint h4 = hashU(h3);
                vec2 suv = (vec2(c) + 0.15 + 0.7 * vec2(unitF(h2), unitF(h3))) / cells * 2.0 - 1.0;
                vec3 dd = d - faceToDir(face, suv);
                float a2 = dot(dd, dd) * inv;
                if (a2 > 16.0) continue;
                float b = L_GAIN[layer] * pow(unitF(h4), L_SHARP[layer]);
                float T = 2800.0 + 9000.0 * pow(unitF(hashU(h4)), 2.0);
                vec4 bb = blackbody(T);
                sum += bb.rgb / max(bb.a, 1e-20) * (b * gain * resolved * exp(-a2));
            }
        }
    }
    return sum;
}

vec3 milkyWay(vec3 d, float sigma) {
    vec3 pole = normalize(vec3(0.28, 1.0, -0.42));
    vec3 center = normalize(cross(pole, vec3(0.3, 0.1, 1.0)));
    float lat = dot(d, pole);
    float band = exp(-lat * lat / (2.0 * 0.13 * 0.13));
    float bulge = pow(0.5 + 0.5 * dot(d, center), 6.0);
    float n1 = fbmAA(d * 3.0 + vec3(11.3, 2.7, -5.1), 5, 3.0, sigma);
    float n2 = fbmAA(d * 9.0 + vec3(-3.3, 8.1, 1.9), 4, 9.0, sigma);
    float clouds = clamp(0.6 + 2.2 * n1, 0.0, 2.0);
    float lane = smoothstep(0.0, 0.12, 0.10 - abs(lat - 0.02 + 0.10 * n1));
    float dust = clamp(lane * (0.6 + 2.5 * n2), 0.0, 0.92);
    vec3 col = mix(vec3(0.62, 0.72, 1.0), vec3(1.0, 0.80, 0.58), 0.35 + 0.65 * bulge)
             * (band * clouds * (0.5 + 2.5 * bulge) * (1.0 - dust));
    float neb1 = max(0.0, fbmAA(d * 1.7 + vec3(4.1, -7.7, 3.3), 4, 1.7, sigma));
    float neb2 = max(0.0, fbmAA(d * 2.3 + vec3(-9.2, 1.4, 6.6), 4, 2.3, sigma));
    col += vec3(0.75, 0.22, 0.45) * (neb1 * neb1 * 1.2 * (0.3 + band));
    col += vec3(0.15, 0.45, 0.65) * (neb2 * neb2 * 0.8);
    return col * 0.025;
}

vec3 background(vec3 d, float sigma, float sigmaRef) {
    return milkyWay(d, sigma) + starField(d, sigma, sigmaRef) * uStarGain;
}
)GLSL"
R"GLSL(
// ------------------------------------------------------------ accretion disk
float ntProfile(float r) {
    if (r <= uRin) return 0.0;
    return pow(r, -0.75) * pow(1.0 - sqrt(uRin / r), 0.25);
}

// One flow layer: clouds, thin ring filaments, two-armed spiral wave.
// psi is the azimuth in the frame co-rotating with the gas, lod = footprint * freq.
void diskLayer(float lr, float psi, uint cycle, float lod, out float clouds, out float fil) {
    uint h = hashU(cycle * 0x9E3779B1u + 12345u);
    vec3 off = vec3(unitF(h), unitF(hashU(h)), unitF(hashU(h + 1u))) * 97.0;
    float c = cos(psi);
    float s = sin(psi);
    clouds = fbmAA(vec3(c * 2.2, s * 2.2, lr * 6.0) + off, 5, 1.0, lod);
    clouds += 0.18 * cos(2.0 * psi - 7.0 * lr);
    float w = gnoise(vec3(c * 1.3, s * 1.3, lr * 26.0) + off * 1.31);
    fil = 1.0 - abs(w);
    fil *= fil;
    fil *= fil;
    fil = mix(0.4, fil, 1.0 - smoothstep(0.15, 0.5, 4.3 * lod));
}

// Keplerian differential rotation winds the pattern up; two phase-shifted layers
// are periodically restarted and cross-faded so it never shears into noise.
float diskDensity(float r, float phi, float t, float lod, out float heat) {
    heat = 1.0;
    float edge = smoothstep(uRin, uRin + 1.5, r) * (1.0 - smoothstep(uRout - 8.0, uRout, r));
    if (edge <= 0.0) return 0.0;
    const float P = 100.0;
    float om = 1.0 / (r * sqrt(r));
    float lr = log(r);
    float ca = t / P;
    float cb = ca + 0.5;
    float fa = fract(ca);
    float fb = fract(cb);
    float wa = 1.0 - abs(2.0 * fa - 1.0);
    float wb = 1.0 - wa;
    float cA = 0.0, fA = 0.0, cB = 0.0, fB = 0.0;
    // gas moves towards decreasing phi (disk angular momentum along +y)
    if (wa > 0.001) diskLayer(lr, phi + om * fa * P, uint(floor(ca)), lod, cA, fA);
    if (wb > 0.001) diskLayer(lr, phi + om * fb * P, uint(floor(cb)) + 7777u, lod, cB, fB);
    float nrm = 1.0 / sqrt(wa * wa + wb * wb);
    float clouds = (cA * wa + cB * wb) * nrm;
    float fil = (fA * wa + fB * wb) * nrm;
    heat = 1.0 + 0.35 * clouds + 0.12 * (fil - 0.4);
    return edge * clamp(0.55 + 1.6 * clouds + 0.5 * fil, 0.03, 2.0);
}

// ------------------------------------------------------------ geodesics
// The spatial path of light in Schwarzschild is that of a particle in the
// central field a = -3 M h^2 r / |r|^5, h = |r x v| (gives u'' + u = 3 M u^2).
vec3 gravity(vec3 x, float k) {
    float r2 = dot(x, x);
    float r = sqrt(r2);
    return x * (-k / (r2 * r2 * r));
}

void rk4(inout vec3 x, inout vec3 v, float h, float k) {
    vec3 a1 = gravity(x, k);
    vec3 x2 = x + v * (0.5 * h);
    vec3 v2 = v + a1 * (0.5 * h);
    vec3 a2 = gravity(x2, k);
    vec3 x3 = x + v2 * (0.5 * h);
    vec3 v3 = v + a2 * (0.5 * h);
    vec3 a3 = gravity(x3, k);
    vec3 x4 = x + v3 * h;
    vec3 v4 = v + a3 * h;
    vec3 a4 = gravity(x4, k);
    x += (v + 2.0 * v2 + 2.0 * v3 + v4) * (h / 6.0);
    v += (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0);
}

void main() {
    vec2 frag = gl_FragCoord.xy - 0.5 + uJitter;
    vec2 ndc = frag / uRes * 2.0 - 1.0;
    vec3 dir = normalize(uCamFwd + uCamRight * (ndc.x * uTanHalfFov * uRes.x / uRes.y)
                                 + uCamUp * (ndc.y * uTanHalfFov));

    float rc = length(uCamPos);
    bool lens = uLensing != 0;
    float alpha = lens ? sqrt(1.0 - 2.0 / rc) : 1.0;
    // L_y / E of the physical photon (it travels towards the camera, along -dir)
    float lambda = cross(uCamPos, -dir).y / alpha;

    // Local static-observer frame -> model velocity: radial part scaled by
    // sqrt(1 - 2M/r), which makes the impact parameter exact.
    vec3 x = uCamPos;
    vec3 v = dir;
    if (lens) {
        vec3 rh = uCamPos / rc;
        v -= rh * (dot(v, rh) * (1.0 - alpha));
    }
    vec3 Lm = cross(x, v);
    float k = lens ? 3.0 * dot(Lm, Lm) : 0.0;
    float escR = max(120.0, 2.0 * rc);
    float rFar = uRout + 4.0;

    int nHits = 0;
    vec3 hp[3];
    float hmu[3];
    hp[0] = vec3(0.0); hp[1] = vec3(0.0); hp[2] = vec3(0.0);
    hmu[0] = 1.0; hmu[1] = 1.0; hmu[2] = 1.0;
    bool escaped = false;
    vec3 escDir = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; ++i) {
        float r = length(x);
        if (r < 2.0) break;                                   // event horizon
        bool outward = dot(x, v) > 0.0;
        if (outward && r > escR) { escaped = true; escDir = normalize(v); break; }
        float len = r > rFar ? (outward ? 0.25 : 0.12) * r : 0.05 * r;
        float h = max(len, 0.02) / length(v);
        vec3 x0 = x;
        vec3 v0 = v;
        rk4(x, v, h, k);
        if ((x0.y > 0.0) != (x.y > 0.0)) {                    // crossed the disk plane
            float f = x0.y / (x0.y - x.y);
            vec3 p = mix(x0, x, f);
            float rh = length(p.xz);
            if (rh >= uRin && rh <= uRout && nHits < 3) {
                vec3 vv = mix(v0, v, f);
                hp[nHits] = p;
                hmu[nHits] = abs(vv.y) / length(vv);
                nHits++;
            }
        }
    }

    // Ray differentials (uniform control flow): pixel footprint on the sky and
    // on the disk. Lensing-compressed regions get a large footprint and are
    // band-limited instead of turning into sparkling noise.
    vec3 ddx = dFdx(escDir);
    vec3 ddy = dFdy(escDir);
    float fx = length(ddx);
    float fy = length(ddy);
    float skyFoot = clamp(max(sqrt(fx * fy), 0.5 * max(fx, fy)), 0.25 * uPixAngle, 0.25);
    float foot0 = min(length(fwidth(hp[0])), 4.0);
    float foot1 = min(length(fwidth(hp[1])), 4.0);
    float foot2 = min(length(fwidth(hp[2])), 4.0);

    vec3 col = vec3(0.0);
    float trans = 1.0;
    float gCam = uGrav != 0 ? 1.0 / sqrt(1.0 - 2.0 / rc) : 1.0;
    for (int i = 0; i < 3; ++i) {
        if (i >= nHits || trans < 1e-3) break;
        vec3 p = hp[i];
        float r = length(p.xz);
        float foot = i == 0 ? foot0 : (i == 1 ? foot1 : foot2);
        float heat;
        float dens = diskDensity(r, atan(p.z, p.x), uTime, 6.0 * foot / r, heat);
        if (dens <= 0.0) continue;
        float a = 1.0 - exp(-uTau0 * dens / max(hmu[i], 0.04));
        // g = nu_obs / nu_emit = [sqrt(1-2/r) / sqrt(1-2/r_cam)] * [1 / (gamma (1 - Omega lambda))]
        float g = 1.0;
        if (uDoppler != 0) {
            float om = 1.0 / (r * sqrt(r));
            float gamma = sqrt((1.0 - 2.0 / r) / (1.0 - 3.0 / r));
            g /= gamma * max(1.0 - om * lambda, 0.02);
        }
        if (uGrav != 0) g *= sqrt(1.0 - 2.0 / r) * gCam;
        // a Planck spectrum at T is seen as a Planck spectrum at g*T
        float T = uDiskTmax * ntProfile(r) / uNtMax * heat * g;
        col += blackbody(T).rgb * (uLumNorm * uDiskGain * trans * a);
        trans *= 1.0 - a;
    }
    if (escaped && trans > 1e-4) {
        float g2 = gCam * gCam;
        col += background(escDir, 0.75 * skyFoot, 0.75 * uRefPixAngle) * (trans * g2 * g2);
    }
    if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);
    fragColor = vec4(max(col, vec3(0.0)), 1.0);
}
)GLSL";

// ---------------------------------------------------------------- bloom
const char* kDownShader = R"GLSL(#version 330 core
in vec2 vUV;
out vec4 o;
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
uniform int uThreshold;
void main() {
    vec3 c = 0.25 * (texture(uSrc, vUV + uSrcTexel * vec2(-1.0, -1.0)).rgb +
                     texture(uSrc, vUV + uSrcTexel * vec2( 1.0, -1.0)).rgb +
                     texture(uSrc, vUV + uSrcTexel * vec2(-1.0,  1.0)).rgb +
                     texture(uSrc, vUV + uSrcTexel * vec2( 1.0,  1.0)).rgb);
    if (uThreshold != 0) {
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c *= max(l - 0.9, 0.0) / max(l, 1e-6);
    }
    o = vec4(c, 1.0);
}
)GLSL";

const char* kBlurShader = R"GLSL(#version 330 core
in vec2 vUV;
out vec4 o;
uniform sampler2D uSrc;
uniform vec2 uStep;
void main() {
    vec3 c = texture(uSrc, vUV).rgb * 0.20416;
    c += (texture(uSrc, vUV + uStep).rgb       + texture(uSrc, vUV - uStep).rgb)       * 0.18017;
    c += (texture(uSrc, vUV + 2.0 * uStep).rgb + texture(uSrc, vUV - 2.0 * uStep).rgb) * 0.12383;
    c += (texture(uSrc, vUV + 3.0 * uStep).rgb + texture(uSrc, vUV - 3.0 * uStep).rgb) * 0.06629;
    c += (texture(uSrc, vUV + 4.0 * uStep).rgb + texture(uSrc, vUV - 4.0 * uStep).rgb) * 0.02763;
    o = vec4(c, 1.0);
}
)GLSL";

// ---------------------------------------------------------------- финальный кадр
const char* kCompositeShader = R"GLSL(#version 330 core
in vec2 vUV;
out vec4 o;
uniform sampler2D uHdr;
uniform sampler2D uB0;
uniform sampler2D uB1;
uniform sampler2D uB2;
uniform sampler2D uB3;
uniform sampler2D uB4;
uniform sampler2D uB5;
uniform float uW[6];
uniform float uBloom;
uniform float uExposure;
uniform int uDither;
uint hashU(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
void main() {
    vec3 c = texture(uHdr, vUV).rgb;
    vec3 b = texture(uB0, vUV).rgb * uW[0] + texture(uB1, vUV).rgb * uW[1] + texture(uB2, vUV).rgb * uW[2]
           + texture(uB3, vUV).rgb * uW[3] + texture(uB4, vUV).rgb * uW[4] + texture(uB5, vUV).rgb * uW[5];
    vec3 x = max((c + b * uBloom) * uExposure, vec3(0.0));
    vec3 a = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);   // ACES
    vec3 s = mix(a * 12.92, 1.055 * pow(a, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), a));
    if (uDither != 0) {
        uvec2 p = uvec2(gl_FragCoord.xy);
        float n = float(hashU(p.x * 1973u + p.y * 9277u) >> 8u) / 16777216.0;
        s += (n - 0.5) / 255.0;
    }
    o = vec4(s, 1.0);
}
)GLSL";

// ---------------------------------------------------------------- подсказки
const char* kHudShader = R"GLSL(#version 330 core
out vec4 o;
uniform sampler2D uTex;
uniform vec2 uHudSize;
uniform float uWinH;
void main() {
    vec2 p = vec2(gl_FragCoord.x, uWinH - gl_FragCoord.y);
    if (p.x > uHudSize.x + 2.0 || p.y > uHudSize.y + 2.0) discard;
    float a = texture(uTex, p / uHudSize).r;
    float sh = texture(uTex, (p - vec2(1.0)) / uHudSize).r;    // drop shadow
    o = vec4(vec3(a), max(a, 0.85 * sh));
}
)GLSL";

// ============================================================================
//  Обёртки OpenGL
// ============================================================================

void reportShaderError(const char* what, const char* log) {
    say(L"Ошибка шейдера «%hs»:\n%hs\n", what, log);
    const std::string msg = std::string("Shader error (") + what + "):\n\n" + log;
    MessageBoxA(nullptr, msg.c_str(), "Black hole: OpenGL", MB_ICONERROR);
}

GLuint compileStage(GLenum type, const char* src, const char* name) {
    const GLuint s = gl3::CreateShader(type);
    gl3::ShaderSource(s, 1, &src, nullptr);
    gl3::CompileShader(s);
    GLint ok = 0;
    gl3::GetShaderiv(s, gl3::COMPILE_STATUS, &ok);
    if (!ok) {
        std::vector<char> log(16384, 0);
        gl3::GetShaderInfoLog(s, (GLsizei)log.size(), nullptr, log.data());
        reportShaderError(name, log.data());
        gl3::DeleteShader(s);
        return 0;
    }
    return s;
}

GLuint buildProgram(const char* fragmentSrc, const char* name) {
    const GLuint vs = compileStage(gl3::VERTEX_SHADER, kVertexShader, "fullscreen.vert");
    const GLuint fs = compileStage(gl3::FRAGMENT_SHADER, fragmentSrc, name);
    if (!vs || !fs) {
        if (vs) gl3::DeleteShader(vs);
        if (fs) gl3::DeleteShader(fs);
        return 0;
    }
    const GLuint p = gl3::CreateProgram();
    gl3::AttachShader(p, vs);
    gl3::AttachShader(p, fs);
    gl3::LinkProgram(p);
    gl3::DeleteShader(vs);
    gl3::DeleteShader(fs);
    GLint ok = 0;
    gl3::GetProgramiv(p, gl3::LINK_STATUS, &ok);
    if (!ok) {
        std::vector<char> log(16384, 0);
        gl3::GetProgramInfoLog(p, (GLsizei)log.size(), nullptr, log.data());
        reportShaderError(name, log.data());
        gl3::DeleteProgram(p);
        return 0;
    }
    return p;
}

inline void setI(GLuint p, const char* n, int v) { gl3::Uniform1i(gl3::GetUniformLocation(p, n), v); }
inline void setF(GLuint p, const char* n, double v) { gl3::Uniform1f(gl3::GetUniformLocation(p, n), (GLfloat)v); }
inline void setV2(GLuint p, const char* n, double x, double y) {
    gl3::Uniform2f(gl3::GetUniformLocation(p, n), (GLfloat)x, (GLfloat)y);
}
inline void setV3(GLuint p, const char* n, const Vec3& v) {
    gl3::Uniform3f(gl3::GetUniformLocation(p, n), (GLfloat)v.x, (GLfloat)v.y, (GLfloat)v.z);
}

inline void bindTex(int unit, GLuint tex) {
    gl3::ActiveTexture(gl3::TEXTURE0 + (GLenum)unit);
    glBindTexture(GL_TEXTURE_2D, tex);
}

// Текстура + framebuffer для рендера в неё
struct Target {
    GLuint tex = 0, fbo = 0;
    int w = 0, h = 0;
};

bool createTarget(Target& t, int w, int h, GLenum internalFormat, GLenum format, GLenum type) {
    t.w = w;
    t.h = h;
    glGenTextures(1, &t.tex);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)internalFormat, w, h, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)gl3::CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)gl3::CLAMP_TO_EDGE);
    gl3::GenFramebuffers(1, &t.fbo);
    gl3::BindFramebuffer(gl3::FRAMEBUFFER, t.fbo);
    gl3::FramebufferTexture2D(gl3::FRAMEBUFFER, gl3::COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    const bool ok = gl3::CheckFramebufferStatus(gl3::FRAMEBUFFER) == gl3::FRAMEBUFFER_COMPLETE;
    glViewport(0, 0, w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    gl3::BindFramebuffer(gl3::FRAMEBUFFER, 0);
    return ok;
}

void destroyTarget(Target& t) {
    if (t.fbo) gl3::DeleteFramebuffers(1, &t.fbo);
    if (t.tex) glDeleteTextures(1, &t.tex);
    t = Target();
}

// ============================================================================
//  Рендерер: сцена → HDR → bloom → ACES
// ============================================================================

class Renderer {
public:
    bool init(const Blackbody& bb) {
        progScene_ = buildProgram(kSceneShader, "scene.frag");
        progDown_ = buildProgram(kDownShader, "downsample.frag");
        progBlur_ = buildProgram(kBlurShader, "blur.frag");
        progComp_ = buildProgram(kCompositeShader, "composite.frag");
        progHud_ = buildProgram(kHudShader, "hud.frag");
        if (!progScene_ || !progDown_ || !progBlur_ || !progComp_ || !progHud_) return false;

        gl3::GenVertexArrays(1, &vao_);

        glGenTextures(1, &bbTex_);
        glBindTexture(GL_TEXTURE_2D, bbTex_);
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint)gl3::RGBA32F, kBBN, 1, 0, GL_RGBA, GL_FLOAT, bb.table().data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)gl3::CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)gl3::CLAMP_TO_EDGE);
        return true;
    }

    void release() {
        destroyTargets();
        if (bbTex_) glDeleteTextures(1, &bbTex_);
        if (vao_) gl3::DeleteVertexArrays(1, &vao_);
        const GLuint progs[5] = { progScene_, progDown_, progBlur_, progComp_, progHud_ };
        for (int i = 0; i < 5; ++i)
            if (progs[i]) gl3::DeleteProgram(progs[i]);
        bbTex_ = vao_ = progScene_ = progDown_ = progBlur_ = progComp_ = progHud_ = 0;
    }

    // Пересоздаёт буферы под размер рендера; true — если размер изменился
    bool ensureSize(int w, int h) {
        if (scene_.w == w && scene_.h == h) return false;
        destroyTargets();
        createTarget(scene_, w, h, gl3::RGBA16F, GL_RGBA, GL_FLOAT);
        int bw = w / 2, bh = h / 2;
        while (bloomA_.size() < 6 && bw >= 8 && bh >= 8) {
            Target a, b;
            createTarget(a, bw, bh, gl3::RGBA16F, GL_RGBA, GL_FLOAT);
            createTarget(b, bw, bh, gl3::RGBA16F, GL_RGBA, GL_FLOAT);
            bloomA_.push_back(a);
            bloomB_.push_back(b);
            bw /= 2;
            bh /= 2;
        }
        return true;
    }

    int width() const { return scene_.w; }
    int height() const { return scene_.h; }

    // Трассировка всех пикселей. sample > 0 — очередная субпиксельная выборка,
    // смешиваемая с накопленным кадром с весом 1/(sample+1).
    void renderScene(const Settings& st, const Camera& cam, double time, int sample, double refPixAngle) {
        gl3::BindFramebuffer(gl3::FRAMEBUFFER, scene_.fbo);
        glViewport(0, 0, scene_.w, scene_.h);
        if (sample == 0) {
            glDisable(GL_BLEND);
        } else {
            glEnable(GL_BLEND);
            glBlendFunc(gl3::CONSTANT_ALPHA, gl3::ONE_MINUS_CONSTANT_ALPHA);
            gl3::BlendColor(0.0f, 0.0f, 0.0f, 1.0f / (float)(sample + 1));
        }
        const GLuint p = progScene_;
        gl3::UseProgram(p);
        setV2(p, "uRes", scene_.w, scene_.h);
        setV2(p, "uJitter", r2x(sample), r2y(sample));
        setV3(p, "uCamPos", cam.pos);
        setV3(p, "uCamRight", cam.right);
        setV3(p, "uCamUp", cam.up);
        setV3(p, "uCamFwd", cam.fwd);
        setF(p, "uTanHalfFov", std::tan(cam.fov * 0.5));
        setF(p, "uPixAngle", cam.pixelAngle(scene_.h));
        setF(p, "uRefPixAngle", refPixAngle);
        setF(p, "uTime", time);
        setF(p, "uRin", st.rIn);
        setF(p, "uRout", st.rOut);
        setF(p, "uTau0", st.tau0);
        setF(p, "uDiskTmax", st.diskTmax);
        setF(p, "uNtMax", st.ntMax);
        setF(p, "uLumNorm", st.lumNorm);
        setF(p, "uDiskGain", st.diskGain);
        setF(p, "uStarGain", st.starGain);
        setI(p, "uDoppler", st.doppler ? 1 : 0);
        setI(p, "uGrav", st.gravShift ? 1 : 0);
        setI(p, "uLensing", st.lensing ? 1 : 0);
        setF(p, "uBBLogMin", std::log(kBBTmin));
        setF(p, "uBBLogMax", std::log(kBBTmax));
        bindTex(0, bbTex_);
        setI(p, "uBB", 0);
        drawTriangle();
        glDisable(GL_BLEND);
    }

    // Bloom + тональная компрессия в outFbo (0 — окно)
    void composite(const Settings& st, GLuint outFbo, int outW, int outH, bool dither) {
        glDisable(GL_BLEND);
        const bool bloom = st.bloomOn && st.bloom > 0.0 && !bloomA_.empty();
        if (bloom) {
            const Target* src = &scene_;
            for (size_t i = 0; i < bloomA_.size(); ++i) {
                Target& a = bloomA_[i];
                Target& b = bloomB_[i];
                gl3::BindFramebuffer(gl3::FRAMEBUFFER, a.fbo);
                glViewport(0, 0, a.w, a.h);
                gl3::UseProgram(progDown_);
                bindTex(0, src->tex);
                setI(progDown_, "uSrc", 0);
                setV2(progDown_, "uSrcTexel", 1.0 / src->w, 1.0 / src->h);
                setI(progDown_, "uThreshold", i == 0 ? 1 : 0);
                drawTriangle();

                gl3::UseProgram(progBlur_);
                setI(progBlur_, "uSrc", 0);
                gl3::BindFramebuffer(gl3::FRAMEBUFFER, b.fbo);
                bindTex(0, a.tex);
                setV2(progBlur_, "uStep", 1.0 / a.w, 0.0);
                drawTriangle();
                gl3::BindFramebuffer(gl3::FRAMEBUFFER, a.fbo);
                bindTex(0, b.tex);
                setV2(progBlur_, "uStep", 0.0, 1.0 / a.h);
                drawTriangle();
                src = &a;
            }
        }

        static const char* kBNames[6] = { "uB0", "uB1", "uB2", "uB3", "uB4", "uB5" };
        static const char* kWNames[6] = { "uW[0]", "uW[1]", "uW[2]", "uW[3]", "uW[4]", "uW[5]" };
        static const double kWeights[6] = { 0.30, 0.25, 0.18, 0.12, 0.09, 0.06 };
        gl3::BindFramebuffer(gl3::FRAMEBUFFER, outFbo);
        glViewport(0, 0, outW, outH);
        const GLuint p = progComp_;
        gl3::UseProgram(p);
        bindTex(0, scene_.tex);
        setI(p, "uHdr", 0);
        for (int i = 0; i < 6; ++i) {
            const bool has = bloom && i < (int)bloomA_.size();
            bindTex(i + 1, has ? bloomA_[(size_t)i].tex : scene_.tex);
            setI(p, kBNames[i], i + 1);
            setF(p, kWNames[i], has ? kWeights[i] : 0.0);
        }
        setF(p, "uBloom", st.bloom);
        setF(p, "uExposure", st.exposure);
        setI(p, "uDither", dither ? 1 : 0);
        drawTriangle();
        gl3::ActiveTexture(gl3::TEXTURE0);
    }

    void drawHud(GLuint tex, int hudW, int hudH, int winW, int winH) {
        gl3::BindFramebuffer(gl3::FRAMEBUFFER, 0);
        glViewport(0, 0, winW, winH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        gl3::UseProgram(progHud_);
        bindTex(0, tex);
        setI(progHud_, "uTex", 0);
        setV2(progHud_, "uHudSize", hudW, hudH);
        setF(progHud_, "uWinH", winH);
        drawTriangle();
        glDisable(GL_BLEND);
    }

private:
    void drawTriangle() {
        gl3::BindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    void destroyTargets() {
        destroyTarget(scene_);
        for (size_t i = 0; i < bloomA_.size(); ++i) {
            destroyTarget(bloomA_[i]);
            destroyTarget(bloomB_[i]);
        }
        bloomA_.clear();
        bloomB_.clear();
    }

    GLuint vao_ = 0, bbTex_ = 0;
    GLuint progScene_ = 0, progDown_ = 0, progBlur_ = 0, progComp_ = 0, progHud_ = 0;
    Target scene_;
    std::vector<Target> bloomA_, bloomB_;
};

// ============================================================================
//  Сохранение BMP
// ============================================================================

// px — строки сверху вниз, пиксели 0x00RRGGBB
bool writeBMP(const std::wstring& path, const std::vector<uint32_t>& px, int w, int h) {
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"wb") != 0 || f == nullptr) return false;
    const int row = (w * 3 + 3) & ~3;
    BITMAPFILEHEADER fh = {};
    BITMAPINFOHEADER ih = {};
    fh.bfType = 0x4D42;  // "BM"
    fh.bfOffBits = sizeof(fh) + sizeof(ih);
    fh.bfSize = fh.bfOffBits + (DWORD)row * h;
    ih.biSize = sizeof(ih);
    ih.biWidth = w;
    ih.biHeight = h;     // строки снизу вверх
    ih.biPlanes = 1;
    ih.biBitCount = 24;
    ih.biCompression = BI_RGB;
    ih.biSizeImage = (DWORD)row * h;
    fwrite(&fh, sizeof(fh), 1, f);
    fwrite(&ih, sizeof(ih), 1, f);
    std::vector<uint8_t> line((size_t)row, 0);
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            const uint32_t c = px[(size_t)y * w + x];
            line[(size_t)x * 3 + 0] = (uint8_t)(c & 255);
            line[(size_t)x * 3 + 1] = (uint8_t)((c >> 8) & 255);
            line[(size_t)x * 3 + 2] = (uint8_t)((c >> 16) & 255);
        }
        fwrite(line.data(), 1, (size_t)row, f);
    }
    const bool ok = ferror(f) == 0;
    fclose(f);
    return ok;
}

// Чтение текущего framebuffer (OpenGL отдаёт строки снизу вверх)
std::vector<uint32_t> readPixelsTopDown(int w, int h) {
    std::vector<uint32_t> px((size_t)w * h);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, w, h, gl3::BGRA, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h / 2; ++y)
        std::swap_ranges(px.begin() + (ptrdiff_t)y * w, px.begin() + (ptrdiff_t)(y + 1) * w,
                         px.begin() + (ptrdiff_t)(h - 1 - y) * w);
    return px;
}

// ============================================================================
//  Офлайн-рендер одного кадра
// ============================================================================

int runOffline(int argc, wchar_t** argv) {
    Settings st;
    Camera cam;
    std::wstring out = L"blackhole.bmp";
    int W = 1920, H = 1080, spp = 32;
    double t = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const bool more = i + 1 < argc;
        if (a == L"--render" && more)        out = argv[++i];
        else if (a == L"--w" && more)        W = _wtoi(argv[++i]);
        else if (a == L"--h" && more)        H = _wtoi(argv[++i]);
        else if (a == L"--spp" && more)      spp = _wtoi(argv[++i]);
        else if (a == L"--dist" && more)     cam.dist = _wtof(argv[++i]);
        else if (a == L"--pitch" && more)    cam.pitch = _wtof(argv[++i]) * kPi / 180.0;
        else if (a == L"--yaw" && more)      cam.yaw = _wtof(argv[++i]) * kPi / 180.0;
        else if (a == L"--fov" && more)      cam.fov = _wtof(argv[++i]) * kPi / 180.0;
        else if (a == L"--time" && more)     t = _wtof(argv[++i]);
        else if (a == L"--temp" && more)     st.diskTmax = _wtof(argv[++i]);
        else if (a == L"--exposure" && more) st.exposure = _wtof(argv[++i]);
        else if (a == L"--no-doppler")       st.doppler = false;
        else if (a == L"--no-redshift")      st.gravShift = false;
        else if (a == L"--no-lensing")       st.lensing = false;
        else if (a == L"--no-bloom")         st.bloomOn = false;
    }
    spp = std::min(std::max(spp, 1), 4096);
    cam.dist = clampd(cam.dist, 3.5, 1000.0);
    cam.pitch = clampd(cam.pitch, -1.5, 1.5);
    cam.fov = clampd(cam.fov, 5.0 * kPi / 180.0, 150.0 * kPi / 180.0);
    st.diskTmax = clampd(st.diskTmax, 1000.0, 40000.0);
    st.prepare();
    cam.update();

    // скрытое окно — только чтобы получить контекст OpenGL
    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = inst;
    wc.lpszClassName = L"BlackHoleOffscreen";
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64,
                                nullptr, nullptr, inst, nullptr);
    HDC dc = nullptr;
    HGLRC rc = nullptr;
    if (!hwnd || !createGLContext(hwnd, dc, rc)) {
        say(L"Не удалось создать контекст OpenGL 3.3 — обновите драйвер видеокарты.\n");
        return 1;
    }
    GLint maxTex = 4096;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    W = std::min(std::max(W, 16), (int)maxTex);
    H = std::min(std::max(H, 16), (int)maxTex);
    say(L"Видеокарта: %hs\nРендер %dx%d, %d выборок на пиксель → %ls\n",
        (const char*)glGetString(GL_RENDERER), W, H, spp, out.c_str());

    int code = 0;
    Renderer ren;
    if (!ren.init(gBB)) {
        code = 1;
    } else {
        const Clock::time_point t0 = Clock::now();
        ren.ensureSize(W, H);
        for (int s = 0; s < spp; ++s) {
            ren.renderScene(st, cam, t, s, cam.pixelAngle(H));
            glFinish();
            say(L"\r  %3d%%", (s + 1) * 100 / spp);
        }
        Target outTarget;
        createTarget(outTarget, W, H, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
        ren.composite(st, outTarget.fbo, W, H, true);
        gl3::BindFramebuffer(gl3::FRAMEBUFFER, outTarget.fbo);
        const std::vector<uint32_t> px = readPixelsTopDown(W, H);
        gl3::BindFramebuffer(gl3::FRAMEBUFFER, 0);
        destroyTarget(outTarget);
        if (writeBMP(out, px, W, H)) {
            say(L"\nГотово за %.1f с: %ls\n", msSince(t0) / 1000.0, out.c_str());
        } else {
            say(L"\nНе удалось записать %ls\n", out.c_str());
            code = 1;
        }
    }
    ren.release();
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(rc);
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return code;
}

// ============================================================================
//  Интерактивное приложение
// ============================================================================

constexpr int kMaxSamples = 64;     // выборок сглаживания на паузе

class App {
public:
    App() { st_.prepare(); }
    ~App() {
        if (hudDC_) {
            SelectObject(hudDC_, hudOldFont_);
            SelectObject(hudDC_, hudOldBmp_);
            DeleteObject(hudBmp_);
            DeleteDC(hudDC_);
        }
        if (font_) DeleteObject(font_);
    }
    App(const App&) = delete;
    App& operator=(const App&) = delete;

    int run();

private:
    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    LRESULT handle(UINT msg, WPARAM wp, LPARAM lp);
    void onKey(WPARAM key);
    bool idle();
    void frame(double dt);
    void initHud();
    void updateHud(bool force);
    void screenshot();
    void releaseGL();
    void setStatus(const std::wstring& s) {
        status_ = s;
        statusUntil_ = Clock::now() + std::chrono::seconds(4);
        redraw_ = true;
    }
    void cameraChanged() {
        cam_.update();
        dirty_ = true;
    }

    Settings st_;
    Camera cam_;
    Renderer ren_;

    HWND hwnd_ = nullptr;
    HDC dc_ = nullptr;
    HGLRC glrc_ = nullptr;
    bool glReady_ = false;
    int cw_ = 1280, ch_ = 720;          // размер клиентской области окна

    int scaleTenths_ = 10;              // разрешение рендера, десятые доли окна
    bool adaptive_ = true;
    double avgMs_ = 16.0;
    Clock::time_point lastScaleChange_;

    int sample_ = 0;
    bool dirty_ = true, redraw_ = true, wantShot_ = false;
    double time_ = 0.0, fps_ = 0.0;
    bool paused_ = false, autoOrbit_ = false, showHelp_ = true, minimized_ = false, running_ = true;
    int drag_ = 0, lastX_ = 0, lastY_ = 0;
    std::wstring status_;
    Clock::time_point statusUntil_;

    // подсказки: текст рисуется GDI в DIB и загружается в текстуру
    HDC hudDC_ = nullptr;
    HBITMAP hudBmp_ = nullptr, hudOldBmp_ = nullptr;
    HGDIOBJ hudOldFont_ = nullptr;
    HFONT font_ = nullptr;
    void* hudBits_ = nullptr;
    GLuint hudTex_ = 0;
    int hudW_ = 1150, hudH_ = 240, lineH_ = 22;
    Clock::time_point lastHud_;
};

int App::run() {
    SetProcessDPIAware();
    HINSTANCE inst = GetModuleHandleW(nullptr);

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = &App::wndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.lpszClassName = L"BlackHoleSimulation";
    RegisterClassExW(&wc);

    const DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rc = { 0, 0, cw_, ch_ };
    AdjustWindowRect(&rc, style, FALSE);
    CreateWindowExW(0, wc.lpszClassName, L"Чёрная дыра Шварцшильда — релятивистская трассировка лучей (GPU)",
                    style, CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
                    nullptr, nullptr, inst, this);
    if (!hwnd_) return 1;

    if (!createGLContext(hwnd_, dc_, glrc_)) {
        MessageBoxW(hwnd_, L"Не удалось создать контекст OpenGL 3.3.\nОбновите драйвер видеокарты.",
                    L"Чёрная дыра", MB_ICONERROR);
        DestroyWindow(hwnd_);
        return 1;
    }
    say(L"Видеокарта: %hs (OpenGL %hs)\n", (const char*)glGetString(GL_RENDERER),
        (const char*)glGetString(GL_VERSION));
    PFN_wglSwapIntervalEXT swapInterval = (PFN_wglSwapIntervalEXT)glProc("wglSwapIntervalEXT");
    if (swapInterval) swapInterval(1);    // вертикальная синхронизация — без разрывов кадра

    if (!ren_.init(gBB)) {
        DestroyWindow(hwnd_);
        return 1;
    }
    initHud();
    glReady_ = true;
    lastScaleChange_ = lastHud_ = Clock::now();
    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    say(L"Управление — в окне (H — подсказки).\n");

    MSG msg;
    Clock::time_point last = Clock::now();
    while (running_) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running_ = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running_ || !glReady_) break;
        if (minimized_ || idle()) {           // ничего не меняется — не нагружаем видеокарту
            WaitMessage();
            last = Clock::now();
            continue;
        }
        const Clock::time_point now = Clock::now();
        const double dt = std::min(std::chrono::duration<double>(now - last).count(), 0.1);
        last = now;

        frame(dt);
        SwapBuffers(dc_);

        const double instFps = 1.0 / std::max(dt, 1e-3);
        fps_ = fps_ <= 0.0 ? instFps : fps_ * 0.9 + instFps * 0.1;
    }
    return 0;
}

LRESULT CALLBACK App::wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTW* cs = reinterpret_cast<const CREATESTRUCTW*>(lp);
        App* self = static_cast<App*>(cs->lpCreateParams);
        self->hwnd_ = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    }
    App* app = reinterpret_cast<App*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    return app ? app->handle(msg, wp, lp) : DefWindowProcW(hwnd, msg, wp, lp);
}

LRESULT App::handle(UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp == SIZE_MINIMIZED) {
            minimized_ = true;
            return 0;
        }
        minimized_ = false;
        cw_ = std::max((int)LOWORD(lp), 1);
        ch_ = std::max((int)HIWORD(lp), 1);
        dirty_ = true;
        return 0;

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT:
        ValidateRect(hwnd_, nullptr);
        redraw_ = true;
        return 0;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
        drag_ = msg == WM_LBUTTONDOWN ? 1 : 2;
        lastX_ = (short)LOWORD(lp);
        lastY_ = (short)HIWORD(lp);
        SetCapture(hwnd_);
        return 0;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        drag_ = 0;
        ReleaseCapture();
        return 0;

    case WM_CAPTURECHANGED:
        drag_ = 0;
        return 0;

    case WM_MOUSEMOVE:
        if (drag_ != 0) {
            const int x = (short)LOWORD(lp), y = (short)HIWORD(lp);
            const int dx = x - lastX_, dy = y - lastY_;
            lastX_ = x;
            lastY_ = y;
            if (dx != 0 || dy != 0) {
                if (drag_ == 1) {
                    cam_.yaw -= dx * 0.005;
                    cam_.pitch = clampd(cam_.pitch + dy * 0.005, -1.5, 1.5);
                } else {
                    cam_.dist = clampd(cam_.dist * std::exp(dy * 0.005), 3.5, 400.0);
                }
                cameraChanged();
            }
        }
        return 0;

    case WM_MOUSEWHEEL:
        cam_.dist = clampd(cam_.dist * std::pow(0.9, GET_WHEEL_DELTA_WPARAM(wp) / 120.0), 3.5, 400.0);
        cameraChanged();
        return 0;

    case WM_KEYDOWN:
        onKey(wp);
        return 0;

    case WM_DESTROY:
        releaseGL();
        running_ = false;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd_, msg, wp, lp);
}

void App::onKey(WPARAM key) {
    const double deg = kPi / 180.0;
    switch (key) {
    case VK_ESCAPE: DestroyWindow(hwnd_); break;
    case VK_SPACE:  paused_ = !paused_; dirty_ = true; break;
    case 'D':       st_.doppler = !st_.doppler; dirty_ = true; break;
    case 'G':       st_.gravShift = !st_.gravShift; dirty_ = true; break;
    case 'L':       st_.lensing = !st_.lensing; dirty_ = true; break;
    case 'B':       st_.bloomOn = !st_.bloomOn; redraw_ = true; break;
    case 'A':       autoOrbit_ = !autoOrbit_; redraw_ = true; break;
    case 'Q':       adaptive_ = !adaptive_; redraw_ = true; break;
    case 'H':       showHelp_ = !showHelp_; redraw_ = true; break;
    case 'R':       cam_ = Camera(); dirty_ = true; break;
    case 'P':
    case VK_F12:    wantShot_ = true; redraw_ = true; break;
    case VK_UP:
        st_.diskTmax = std::min(st_.diskTmax + 500.0, 30000.0);
        st_.prepare();
        dirty_ = true;
        break;
    case VK_DOWN:
        st_.diskTmax = std::max(st_.diskTmax - 500.0, 2000.0);
        st_.prepare();
        dirty_ = true;
        break;
    case VK_RIGHT:  st_.timeSpeed = std::min(st_.timeSpeed * 1.5, 400.0); redraw_ = true; break;
    case VK_LEFT:   st_.timeSpeed = std::max(st_.timeSpeed / 1.5, 0.5); redraw_ = true; break;
    case VK_PRIOR:  st_.exposure *= 1.25; redraw_ = true; break;
    case VK_NEXT:   st_.exposure /= 1.25; redraw_ = true; break;
    case VK_OEM_4:  cam_.fov = std::max(cam_.fov - 5.0 * deg, 15.0 * deg); cameraChanged(); break;   // [
    case VK_OEM_6:  cam_.fov = std::min(cam_.fov + 5.0 * deg, 120.0 * deg); cameraChanged(); break;  // ]
    }
}

bool App::idle() {
    if (!status_.empty()) {
        if (Clock::now() < statusUntil_) return false;
        status_.clear();
        redraw_ = true;
    }
    return paused_ && !autoOrbit_ && !dirty_ && !redraw_ && !wantShot_ && drag_ == 0 && sample_ >= kMaxSamples;
}

void App::frame(double dt) {
    if (!paused_) time_ += dt * st_.timeSpeed;
    if (autoOrbit_) {
        cam_.yaw += dt * 0.10;
        cameraChanged();
    }

    // Адаптивное разрешение: держим не ниже ~22 кадр/с, на паузе — всегда 100 %
    avgMs_ = avgMs_ * 0.85 + dt * 1000.0 * 0.15;
    int target = scaleTenths_;
    if (!adaptive_ || paused_) {
        target = 10;
    } else if (msSince(lastScaleChange_) > 400.0) {
        if (avgMs_ > 45.0 && scaleTenths_ > 5) target = scaleTenths_ - 1;
        else if (avgMs_ < 28.0 && scaleTenths_ < 10) target = scaleTenths_ + 1;
    }
    if (target != scaleTenths_) {
        scaleTenths_ = target;
        lastScaleChange_ = Clock::now();
    }
    const int rw = std::max(16, (cw_ * scaleTenths_ + 5) / 10);
    const int rh = std::max(16, (ch_ * scaleTenths_ + 5) / 10);
    if (ren_.ensureSize(rw, rh)) dirty_ = true;

    // Сцена: пока время идёт — новый кадр каждый раз; на паузе — накопление выборок
    if (dirty_ || !paused_) sample_ = 0;
    dirty_ = false;
    if (sample_ < kMaxSamples) {
        ren_.renderScene(st_, cam_, time_, sample_, cam_.pixelAngle(ch_));
        ++sample_;
    }

    ren_.composite(st_, 0, cw_, ch_, true);
    if (wantShot_) {
        wantShot_ = false;
        screenshot();
    }
    updateHud(redraw_ || sample_ >= kMaxSamples);
    ren_.drawHud(hudTex_, hudW_, hudH_, cw_, ch_);
    redraw_ = false;
}

void App::initHud() {
    HDC screen = GetDC(nullptr);
    const double dpi = GetDeviceCaps(screen, LOGPIXELSY) / 96.0;
    hudDC_ = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);

    lineH_ = (int)(22 * dpi);
    hudW_ = (int)(1150 * dpi);
    hudH_ = lineH_ * 10 + (int)(16 * dpi);
    font_ = CreateFontW(-(int)(16 * dpi), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN,
                        L"Consolas");

    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = hudW_;
    bi.bmiHeader.biHeight = -hudH_;               // строки сверху вниз
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    hudBmp_ = CreateDIBSection(hudDC_, &bi, DIB_RGB_COLORS, &hudBits_, nullptr, 0);
    hudOldBmp_ = (HBITMAP)SelectObject(hudDC_, hudBmp_);
    hudOldFont_ = SelectObject(hudDC_, font_);
    SetBkMode(hudDC_, TRANSPARENT);
    SetTextColor(hudDC_, RGB(255, 255, 255));

    glGenTextures(1, &hudTex_);
    glBindTexture(GL_TEXTURE_2D, hudTex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, hudW_, hudH_, 0, gl3::BGRA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)gl3::CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)gl3::CLAMP_TO_EDGE);
    updateHud(true);
}

void App::updateHud(bool force) {
    if (!hudBits_ || (!force && msSince(lastHud_) < 150.0)) return;
    lastHud_ = Clock::now();

    auto onoff = [](bool b) { return b ? L"вкл" : L"выкл"; };
    std::vector<std::wstring> lines;
    wchar_t buf[512];

    wchar_t mode[96] = L"";
    if (paused_) swprintf(mode, 96, L"   сглаживание %d/%d", std::min(sample_, kMaxSamples), kMaxSamples);
    swprintf(buf, 512, L"%.0f кадр/с   рендер %dx%d (%d%%, %ls)%ls", fps_, ren_.width(), ren_.height(),
             scaleTenths_ * 10, adaptive_ ? L"авто [Q]" : L"фикс. [Q]", mode);
    lines.push_back(buf);

    swprintf(buf, 512, L"r = %.1f M   высота над диском %.1f°   обзор %.0f°   t = %.0f M (x%.1f)%ls",
             cam_.dist, cam_.pitch * 180.0 / kPi, cam_.fov * 180.0 / kPi, time_, st_.timeSpeed,
             paused_ ? L"   [пауза]" : L"");
    lines.push_back(buf);

    swprintf(buf, 512, L"Доплер [D] %ls   Грав. смещение [G] %ls   Линзирование [L] %ls   Bloom [B] %ls",
             onoff(st_.doppler), onoff(st_.gravShift), onoff(st_.lensing), onoff(st_.bloomOn));
    lines.push_back(buf);

    swprintf(buf, 512, L"T диска %.0f K [↑↓]   экспозиция %.2f [PgUp/PgDn]%ls",
             st_.diskTmax, st_.exposure, autoOrbit_ ? L"   автоорбита [A]" : L"");
    lines.push_back(buf);

    if (showHelp_) {
        lines.push_back(L"");
        lines.push_back(L"ЛКМ — вращать камеру   колесо / ПКМ — расстояние   Пробел — пауза и сглаживание");
        lines.push_back(L"←/→ — скорость времени   [ ] — угол обзора   A — автоорбита   R — сброс камеры");
        lines.push_back(L"P — скриншот   Q — адаптивное разрешение   H — скрыть подсказки   Esc — выход");
    }
    if (!status_.empty()) {
        lines.push_back(L"");
        lines.push_back(status_);
    }

    std::memset(hudBits_, 0, (size_t)hudW_ * hudH_ * 4);
    int y = lineH_ / 3;
    for (size_t i = 0; i < lines.size(); ++i) {
        TextOutW(hudDC_, lineH_ / 2, y, lines[i].c_str(), (int)lines[i].size());
        y += lineH_;
    }
    GdiFlush();
    glBindTexture(GL_TEXTURE_2D, hudTex_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, hudW_, hudH_, gl3::BGRA, GL_UNSIGNED_BYTE, hudBits_);
}

void App::screenshot() {
    gl3::BindFramebuffer(gl3::FRAMEBUFFER, 0);
    glReadBuffer(GL_BACK);
    const std::vector<uint32_t> px = readPixelsTopDown(cw_, ch_);
    SYSTEMTIME t;
    GetLocalTime(&t);
    wchar_t name[128];
    swprintf(name, 128, L"blackhole_%04d%02d%02d_%02d%02d%02d.bmp",
             (int)t.wYear, (int)t.wMonth, (int)t.wDay, (int)t.wHour, (int)t.wMinute, (int)t.wSecond);
    if (writeBMP(name, px, cw_, ch_))
        setStatus(std::wstring(L"Скриншот сохранён: ") + name);
    else
        setStatus(L"Не удалось сохранить скриншот");
}

void App::releaseGL() {
    if (!glrc_) return;
    if (glReady_) {
        ren_.release();
        if (hudTex_) glDeleteTextures(1, &hudTex_);
        hudTex_ = 0;
    }
    glReady_ = false;
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(glrc_);
    glrc_ = nullptr;
}

}  // namespace

// ============================================================================

int main() {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    gBB.init();

    bool offline = false;
    for (int i = 1; i < argc; ++i)
        if (std::wcscmp(argv[i], L"--render") == 0) offline = true;

    int code = 0;
    if (offline) {
        code = runOffline(argc, argv);
    } else {
        App app;
        code = app.run();
    }
    if (argv) LocalFree(argv);
    return code;
}
