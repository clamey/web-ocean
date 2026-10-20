// Physically based atmosphere (Hillaire 2020 style LUTs) + procedural cloud deck.
import { FRAME_WGSL, ATMO_CORE_WGSL, NOISE_WGSL, SKY_W, SKY_H, ENV_SIZE } from './common.js';

// Sky luminance (renderer units) below which the eye adaptation starts: the measured
// average at sun elevation 0 deg, so sunset and daytime scenes keep their exposure.
const ADAPT_REF = '0.046';

const ATMO_UNIFORM = /* wgsl */ `
struct AtmoP { haze: vec4f };   // x = mie multiplier
@group(0) @binding(0) var<uniform> A: AtmoP;
`;

export const TRANS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= TRANS_W || f32(id.y) >= TRANS_H) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANS_W, TRANS_H);
  let rm = transUVToRMu(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(0.0, 1.0 - rm.y * rm.y)), rm.y, 0.0);
  let tMax = raySphere(ro, rd, RT).y;
  var od = vec3f(0.0);
  let STEPS = 64u;
  let dt = tMax / f32(STEPS);
  for (var i = 0u; i < STEPS; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    od += medium(length(p) - RG, A.haze.x).ext * dt;
  }
  textureStore(dst, id.xy, vec4f(exp(-od), 1.0));
}
`;

export const MS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var sampC: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= MS_SIZE || f32(id.y) >= MS_SIZE) { return; }
  var uv = (vec2f(id.xy) + 0.5) / MS_SIZE;
  uv = vec2f(fromSubUvsToUnit(uv.x, MS_SIZE), fromSubUvsToUnit(uv.y, MS_SIZE));
  let cosSun = uv.x * 2.0 - 1.0;
  let h = max(uv.y * (RT - RG), 0.01);
  let r = RG + h;
  let ro = vec3f(0.0, r, 0.0);
  let sun = vec3f(sqrt(max(0.0, 1.0 - cosSun * cosSun)), cosSun, 0.0);
  var Lsum = vec3f(0.0);
  var Fsum = vec3f(0.0);
  let SQ = 8u;
  let STEPS = 20u;
  for (var a = 0u; a < SQ; a++) {
    for (var b = 0u; b < SQ; b++) {
      let th = acos(1.0 - 2.0 * (f32(b) + 0.5) / f32(SQ));
      let ph = 2.0 * PI * (f32(a) + 0.5) / f32(SQ);
      let rd = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
      let tTop = raySphere(ro, rd, RT);
      let tGr = raySphere(ro, rd, RG);
      var tMax = tTop.y;
      let hitG = tGr.x > 0.0;
      if (hitG) { tMax = tGr.x; }
      let dt = tMax / f32(STEPS);
      var T = vec3f(1.0);
      var L = vec3f(0.0);
      var Fms = vec3f(0.0);
      for (var s = 0u; s < STEPS; s++) {
        let p = ro + rd * ((f32(s) + 0.5) * dt);
        let pr = length(p);
        let m = medium(pr - RG, A.haze.x);
        let scat = m.rayS + vec3f(m.mieS);
        let Ts = exp(-m.ext * dt);
        let smu = dot(p, sun) / pr;
        var sunT = sampleTrans(pr, smu);
        if (sunBelowHorizon(pr, smu)) { sunT = vec3f(0.0); }
        let ext = max(m.ext, vec3f(1e-9));
        let S = scat * sunT / (4.0 * PI);
        L += T * (S - S * Ts) / ext;
        Fms += T * (scat - scat * Ts) / ext;
        T *= Ts;
      }
      if (hitG) {
        let pg = ro + rd * tMax;
        let ng = normalize(pg);
        let smu = dot(ng, sun);
        L += T * sampleTrans(RG, smu) * max(smu, 0.0) * 0.06 / PI;
      }
      Lsum += L;
      Fsum += Fms;
    }
  }
  let invN = 1.0 / f32(SQ * SQ);
  let L2 = Lsum * invN;
  let fms = Fsum * invN;
  textureStore(dst, id.xy, vec4f(L2 / (1.0 - fms), 1.0));
}
`;

// Sky-view LUT, recomputed every frame (sun / camera altitude may change).
export const SKYVIEW_LUT = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var msLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;
// 1 = compute the moonlit sky (layer 1) instead of the sunlit one (layer 0)
override MOON: f32 = 0.0;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= SKY_W || f32(id.y) >= SKY_H) { return; }
  var uv = (vec2f(id.xy) + 0.5) / vec2f(SKY_W, SKY_H);
  uv = vec2f(fromSubUvsToUnit(uv.x, SKY_W), fromSubUvsToUnit(uv.y, SKY_H));
  let haze = F.atmo.x;
  let r = RG + F.atmo.y;
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  var vza: f32;
  if (uv.y < 0.5) {
    var coord = 2.0 * uv.y;
    coord = 1.0 - coord;
    coord *= coord;
    coord = 1.0 - coord;
    vza = zha * coord;
  } else {
    var coord = uv.y * 2.0 - 1.0;
    coord *= coord;
    vza = zha + beta * coord;
  }
  let vzc = cos(vza);
  let vzs = sin(vza);
  let cx = uv.x * uv.x;
  let lvc = -(cx * 2.0 - 1.0);
  let lvs = sqrt(max(0.0, 1.0 - lvc * lvc));
  let rd = vec3f(vzs * lvc, vzc, vzs * lvs);
  // the light: the sun, or the moon for the moonlit-sky layer
  let smu = select(F.sunDir.y, F.moonDir.y, MOON > 0.5);
  let sun = vec3f(sqrt(max(0.0, 1.0 - smu * smu)), smu, 0.0);
  let ro = vec3f(0.0, r, 0.0);

  let tTop = raySphere(ro, rd, RT);
  let tGr = raySphere(ro, rd, RG);
  var tMax = tTop.y;
  if (tGr.x > 0.0) { tMax = tGr.x; }
  let cosT = dot(rd, sun);
  let pR = phaseRayleigh(cosT);
  let pM = phaseMie(cosT, 0.84);
  var L = vec3f(0.0);
  var T = vec3f(1.0);
  let STEPS = 48u;
  var tPrev = 0.0;
  for (var i = 0u; i < STEPS; i++) {
    let fa = (f32(i) + 1.0) / f32(STEPS);
    let tn = tMax * fa * fa;
    let dt = tn - tPrev;
    let t = tPrev + dt * 0.3;
    tPrev = tn;
    let p = ro + rd * t;
    let pr = length(p);
    let h = pr - RG;
    let m = medium(h, haze);
    let pmu = dot(p, sun) / pr;
    var sunT = sampleTrans(pr, pmu);
    if (sunBelowHorizon(pr, pmu)) { sunT = vec3f(0.0); }
    let msuv = vec2f(fromUnitToSubUvs(pmu * 0.5 + 0.5, MS_SIZE), fromUnitToSubUvs(clamp(h / (RT - RG), 0.0, 1.0), MS_SIZE));
    let ms = textureSampleLevel(msLUT, sampC, msuv, 0.0).rgb;
    let S = sunT * (m.rayS * pR + vec3f(m.mieS * pM)) + ms * (m.rayS + vec3f(m.mieS));
    let Ts = exp(-m.ext * dt);
    let ext = max(m.ext, vec3f(1e-9));
    L += T * (S - S * Ts) / ext;
    T *= Ts;
  }
  // pre-exposed, so deep twilight stays well inside half-float range
  // pre-exposed, so deep twilight stays well inside half-float range; the moon layer also
  // carries the moonlight illuminance (per unit it would overflow under the night gain)
  textureStore(dst, id.xy, vec4f(L * select(F.astro.w, F.astro.w * F.moonDir.w, MOON > 0.5), 1.0));
}
`;

// Functions for rendering the sky (background, env cube, ocean fog).
// Requires: F (Frame), transLUT, skyLUT, sampC declared by the including module.
export const SKY_RENDER_FUNCS = NOISE_WGSL + /* wgsl */ `
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn camR() -> f32 { return RG + F.atmo.y; }

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

fn sunIlluminance() -> vec3f {
  return sampleTrans(camR(), F.sunDir.y) * (F.sunDir.w * F.astro.w * earthShadow(camR(), F.sunDir.y));
}

// moonlight reaching the camera (pre-exposed), zero once the moon has set
fn moonIlluminance() -> vec3f {
  return sampleTrans(camR(), F.moonDir.y) * (F.moonDir.w * F.astro.w * earthShadow(camR(), F.moonDir.y));
}

fn skyViewUV(dir: vec3f, light: vec3f) -> vec2f {
  let r = camR();
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  let vzc = clamp(dir.y, -1.0, 1.0);
  var uv: vec2f;
  if (vzc >= -cosBeta) {
    var coord = acos(vzc) / zha;
    coord = 1.0 - coord;
    coord = sqrt(max(coord, 0.0));
    coord = 1.0 - coord;
    uv.y = coord * 0.5;
  } else {
    var coord = (acos(vzc) - zha) / beta;
    coord = sqrt(max(coord, 0.0));
    uv.y = coord * 0.5 + 0.5;
  }
  let dl = length(dir.xz);
  let sl = length(light.xz);
  var lvc = 1.0;
  if (dl > 1e-6 && sl > 1e-6) { lvc = clamp(dot(dir.xz, light.xz) / (dl * sl), -1.0, 1.0); }
  uv.x = sqrt(max(-lvc * 0.5 + 0.5, 0.0));
  return vec2f(fromUnitToSubUvs(uv.x, SKY_W), fromUnitToSubUvs(uv.y, SKY_H));
}

fn skyAtmo(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyLUT, sampC, skyViewUV(dir, F.sunDir.xyz), 0, 0.0).rgb * F.sunDir.w
       + textureSampleLevel(skyLUT, sampC, skyViewUV(dir, F.moonDir.xyz), 1, 0.0).rgb + nightSky(dir);
}

// Moonless night sky: airglow, zodiacal light and unresolved starlight, about 21.5 mag per
// square arcsecond (2e-4 cd/m2) at the zenith and brighter towards the horizon. Far below
// anything visible by day.
const NIGHT_SKY: vec3f = vec3f(1.7e-8, 2.2e-8, 2.9e-8);
fn nightSky(dir: vec3f) -> vec3f {
  return NIGHT_SKY * (1.0 + 2.0 * (1.0 - smoothstep(0.0, 0.35, dir.y))) * F.astro.w;
}
`;

// Clouds + full sky radiance (needs the same bindings as SKY_RENDER_FUNCS).
export const SKY_CLOUD_FUNCS = /* wgsl */ `

// ---- clouds -------------------------------------------------------------
// Two thin, light-aware layers: broken low cumulus with cauliflower edges (2 km) and
// high cirrus wisps (7 km). Each is shaded with self-shadowing towards the sun, bright
// sun-facing rims, silver linings on thin back-lit parts and warm multiple scattering.
const CLOUD_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);
// fbm whose octaves fade once smaller than the pixel footprint fp (noise units);
// 'billow' turns the small octaves into bulgy |noise| for cauliflower-like edges
fn cloudFbm(p: vec2f, fp: f32, oct: i32, billow: f32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  var f = 1.0;
  for (var i = 0; i < oct; i++) {
    // fade octaves well before they reach pixel size (half-res buffer) -> no shimmer in motion
    let w = 1.0 - smoothstep(0.08, 0.28, fp * f);
    var n = gnoise(q);
    if (i >= 2) { n = mix(n, abs(n) * 2.0 - 0.38, billow); }
    s += a * n * w;
    q = CLOUD_ROT * q * 2.03 + vec2f(13.1, 7.7);
    f *= 2.03;
    a *= 0.5;
  }
  return s;
}
// Signed "edge" functions (cloud where > 0) and their density ramps. The edge value is
// what the sky pass differentiates (fwidth) to anti-alias the cloud silhouettes analytically.
fn lowEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  var p = q * 0.38 + vec2f(t * 0.004, t * 0.0012);
  let w = vec2f(gnoise(p * 0.3 + vec2f(1.7, 9.2)), gnoise(p * 0.3 + vec2f(8.3, 2.8)));
  p += w * 1.4;
  // slightly elongated cells (cloud streets), mildly billowy small scales for ragged edges
  let base = cloudFbm(p * vec2f(1.0, 1.7), fp * 0.65, oct, 0.35) * 1.3 + 0.5;
  let large = gnoise(q * 0.05 + vec2f(3.1, 1.3)) * 0.7;
  return base + large - (1.0 - F.cloudP.y);
}
fn lowSoft(fp: f32) -> f32 { return 0.38 + min(fp * 0.65, 1.0) * 0.3; }
fn lowDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(lowEdge(q, fp, oct) / lowSoft(fp), 0.0, 1.0);
}
fn highEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  // altostratus deck: smooth, slightly fibrous
  let p = CLOUD_ROT * (q * vec2f(0.09, 0.20)) + vec2f(t * 0.0015, 0.0);
  let n = cloudFbm(p, fp * 0.2, oct, 0.0) * 1.25 + 0.5;
  let big = gnoise(q * 0.012 + vec2f(7.0, 3.0)) * 0.6;
  return n + big - (1.0 - F.atmo.z);
}
fn highSoft(fp: f32) -> f32 { return 0.85 + min(fp * 0.2, 1.0) * 0.3; }
fn highDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(highEdge(q, fp, oct) / highSoft(fp), 0.0, 1.0);
}

struct CloudHit { t: f32, q: vec2f, mu: f32, up: vec3f };
fn hitLayer(dir: vec3f, hc: f32) -> CloudHit {
  var h: CloudHit;
  let ro = vec3f(0.0, camR(), 0.0);
  h.t = raySphere(ro, dir, RG + hc).y;
  let p = ro + dir * h.t;
  h.up = p / length(p);
  h.q = p.xz + F.camPos.xz * 0.001;
  h.mu = max(dot(h.up, dir), 0.03);
  return h;
}

// d: density at the hit; s1..s3: densities 80 m / 250 m / 600 m towards the sun
fn shadeLayer(dir: vec3f, h: CloudHit, hc: f32, d: f32, s1: f32, s2: f32, s3: f32, sigma: f32, g: f32) -> vec4f {
  let tau = d * sigma / h.mu;
  let alpha = 1.0 - exp(-tau);
  let tauS = (d * 0.5 + s1 * 0.6 + s2 + s3 * 0.7) * sigma * 0.55;
  let Ts = exp(-tauS);
  let smu = dot(h.up, F.sunDir.xyz);
  // clouds keep the sun a little after sunset at sea level, until the Earth's shadow reaches them
  let sunC = sampleTrans(RG + hc, smu) * (F.sunDir.w * F.astro.w * earthShadow(RG + hc, smu));
  let cosT = dot(dir, F.sunDir.xyz);
  let ph = mix(phaseHG(cosT, g), phaseHG(cosT, -0.2), 0.25);
  let fwd = smoothstep(-0.2, 0.7, cosT);
  // single scattering: reflected when front-lit, transmitted through thin parts when back-lit
  let single = sunC * Ts * ph * mix(alpha * 0.55, tau * exp(-tau) * 2.2, fwd) * 3.0;
  // edges where density falls off towards the sun catch direct light
  let rim = clamp((d - s1) * 2.5, 0.0, 1.0);
  // multiple scattering: a low sun lights cloud tops and sun-facing flanks, not the bases we
  // see from below, so diffused sunlight shows mostly when looking towards the sun
  let toward = smoothstep(-0.3, 0.9, cosT);
  let ms = sunC * (0.025 + 0.22 * rim) * mix(1.0, Ts, 0.5) * alpha * (0.25 + 0.75 * toward);
  // bases are lit by the sky from the sides/below (horizon band + sea), cooler than the tops.
  // Once the cloud is in the Earth's shadow the twilight glow is concentrated towards the
  // sun, so take the horizon at right angles to it instead (the band's azimuthal average)
  let hz = mix(skyAtmo(normalize(vec3f(dir.x, 0.03, dir.z))),
               skyAtmo(normalize(vec3f(-F.sunDir.z, 0.03, F.sunDir.x))), 1.0 - earthShadow(RG + hc, smu));
  let amb = (hz * 0.26 + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 0.42) * alpha * (1.0 - 0.35 * d);
  // moonlight: forward-scattered through thin parts, a dull grey glow elsewhere
  let mmu = dot(h.up, F.moonDir.xyz);
  let moonC = sampleTrans(RG + hc, mmu) * (F.moonDir.w * F.astro.w * earthShadow(RG + hc, mmu));
  let cosM = dot(dir, F.moonDir.xyz);
  let moonLit = moonC * alpha * (0.05 + 1.5 * mix(phaseHG(cosM, g), phaseHG(cosM, -0.2), 0.25)) * (1.0 - 0.4 * d);
  return vec4f(single + ms + amb + moonLit, alpha);
}

// returns (premultiplied radiance, fraction of the background sky hidden)
// aa: per-layer edge change per pixel (fwidth of the edge function); negative = estimate it
// from the pixel footprint (compute passes, which have no derivatives)
fn cloudLayers(dir: vec3f, pixAng: f32, aa: vec2f, sunT: ptr<function, f32>) -> vec4f {
  *sunT = 1.0;
  if (dir.y <= 0.0) { return vec4f(0.0); }
  let sxz = F.sunDir.xz / max(length(F.sunDir.xz), 1e-4);
  var lo = vec4f(0.0);
  var loVis = 1.0;
  if (F.cloudP.y > 0.001) {
    let hc = F.cloudP.x;
    let h = hitLayer(dir, hc);
    // clear band near the horizon, where the low sun shines under the clouds
    let far = 1.0 - smoothstep(18.0, 36.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = lowEdge(h.q, fp, 7);
      let d = clamp(e / lowSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = lowDensity(h.q + sxz * 0.08, fp, 5) * far;
        let s2 = lowDensity(h.q + sxz * 0.25, fp, 4) * far;
        let s3 = lowDensity(h.q + sxz * 0.6, fp, 3) * far;
        lo = shadeLayer(dir, h, hc, d, s1, s2, s3, F.cloudP.z, 0.75);
        // opacity saturates within a sliver of the density ramp, making a hard silhouette:
        // spread it over ~1.5 pixels
        let aw = select(fp * 0.38 * 2.5, aa.x, aa.x >= 0.0);
        lo *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        loVis = exp(-h.t * 0.02 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  var hi = vec4f(0.0);
  var hiVis = 1.0;
  if (F.atmo.z > 0.001) {
    let hc = 4.5;
    let h = hitLayer(dir, hc);
    let far = 1.0 - smoothstep(30.0, 70.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = highEdge(h.q, fp, 6);
      let d = clamp(e / highSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = highDensity(h.q + sxz * 0.15, fp, 4) * far;
        let s2 = highDensity(h.q + sxz * 0.5, fp, 3) * far;
        hi = shadeLayer(dir, h, hc, d, s1, s2, s2, F.atmo.w, 0.80);
        let aw = select(fp * 0.2 * 2.5, aa.y, aa.y >= 0.0);
        hi *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        hiVis = exp(-h.t * 0.010 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  *sunT = (1.0 - lo.a) * (1.0 - hi.a);
  // low layer in front; aerial perspective fades each layer into the haze without making
  // it transparent to the layer behind
  let col = lo.rgb * loVis + hi.rgb * hiVis * (1.0 - lo.a);
  let hidden = lo.a * loVis + hi.a * hiVis * (1.0 - lo.a);
  return vec4f(col, hidden);
}

fn sunDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, F.sunDir.xyz);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(SUN_ANG_R - pix, SUN_ANG_R + pix, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  let x = min(ang / SUN_ANG_R, 1.0);
  let mu = sqrt(max(1.0 - x * x, 0.0));
  let limb = vec3f(1.0) - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  let rad = F.sunDir.w * F.astro.w / (PI * SUN_ANG_R * SUN_ANG_R);
  return rad * limb * edge * sampleTrans(camR(), max(dir.y, -0.01));
}

// ---- the Moon ------------------------------------------------------------
// Albedo relative to the highlands: the maria at their selenographic positions
// (lat, lon east towards Mare Crisium, radius; degrees), with ragged edges.
var<private> MARIA: array<vec3f, 22> = array<vec3f, 22>(
  vec3f(8.0, -57.0, 18.0), vec3f(28.0, -52.0, 15.0), vec3f(-8.0, -48.0, 14.0),   // Oceanus Procellarum
  vec3f(32.8, -15.6, 19.0),   // Imbrium
  vec3f(28.0, 17.5, 11.0),    // Serenitatis
  vec3f(8.5, 31.4, 14.0),     // Tranquillitatis
  vec3f(17.0, 59.1, 9.0),     // Crisium
  vec3f(-7.8, 51.3, 12.0),    // Fecunditatis
  vec3f(-15.2, 35.5, 5.5),    // Nectaris
  vec3f(-21.3, -16.6, 11.0),  // Nubium
  vec3f(-24.4, -38.6, 6.5),   // Humorum
  vec3f(-10.0, -23.0, 6.0),   // Cognitum
  vec3f(7.5, -30.9, 8.0),     // Insularum
  vec3f(13.3, 3.6, 4.0),      // Vaporum
  vec3f(2.0, 1.0, 3.0),       // Sinus Medii
  vec3f(56.0, -38.0, 5.0), vec3f(57.0, -15.0, 6.0), vec3f(57.0, 8.0, 6.0), vec3f(56.0, 28.0, 5.0), vec3f(55.0, 45.0, 4.0),   // Frigoris
  vec3f(1.3, 87.0, 6.0),      // Smythii
  vec3f(13.0, 86.0, 5.0)      // Marginis
);
fn selVec(latDeg: f32, lonDeg: f32) -> vec3f {
  let la = radians(latDeg);
  let lo = radians(lonDeg);
  return vec3f(cos(la) * sin(lo), sin(la), cos(la) * cos(lo));
}
fn moonAlbedo(p: vec3f) -> f32 {
  var mare = 0.0;
  for (var i = 0; i < 22; i++) {
    let m = MARIA[i];
    let d = degrees(acos(clamp(dot(p, selVec(m.x, m.y)), -1.0, 1.0)));
    let wob = (gnoise(p.xy * 3.5 + vec2f(f32(i) * 3.7, 1.3)) + 0.5 * gnoise(p.yz * 8.0 + f32(i))) * 0.3 * m.z;
    // overlapping margins add up, so neighbouring maria flow into each other
    mare += smoothstep(m.z * 1.3 + wob, m.z * 0.7 + wob, d);
  }
  mare = smoothstep(0.0, 1.0, min(mare, 1.0));
  // maria vary in tone; the highlands are mottled with craters
  let tone = 0.42 + 0.08 * gnoise(p.xy * 5.0 + vec2f(7.0, 2.0));
  var a = mix(1.0, tone, mare) * (1.0 + 0.10 * gnoise(p.xy * 23.0) + 0.06 * gnoise(p.xy * 61.0 + 4.0));
  // young bright craters: Tycho with its ray system, Copernicus, Kepler, Aristarchus
  let dt = degrees(acos(clamp(dot(p, selVec(-43.3, -11.2)), -1.0, 1.0)));
  let tv = selVec(-43.3, -11.2);
  let ray = gnoise(vec2f(atan2(dot(p, cross(tv, vec3f(0.0, 1.0, 0.0))), dot(p, vec3f(0.0, 1.0, 0.0))) * 7.0, 0.5));
  a += 0.9 * exp(-dt / 1.2) + 0.35 * smoothstep(0.1, 0.6, ray) * exp(-dt / 22.0);
  a += 0.6 * exp(-degrees(acos(clamp(dot(p, selVec(9.6, -20.1)), -1.0, 1.0))) / 1.0);
  a += 0.4 * exp(-degrees(acos(clamp(dot(p, selVec(8.1, -38.0)), -1.0, 1.0))) / 0.8);
  a += 0.8 * exp(-degrees(acos(clamp(dot(p, selVec(23.7, -47.4)), -1.0, 1.0))) / 0.6);
  return a;
}
// full-moon disc radiance per unit solar illuminance (~2500 cd/m2 with the mean albedo)
const MOON_L: f32 = 0.0195;
// largest pre-exposed radiance the lit disc may reach: the night gain would burn it to a
// white blob, while the eye still sees the maria (a local adaptation)
const MOON_DISC_MAX: f32 = 1.5;
fn moonDisk(dir: vec3f) -> vec3f {
  let m = F.moonDir.xyz;
  let R = F.moonUp.w;
  let c = dot(dir, m);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(R - pix, R + pix, ang);
  if (edge <= 0.0 || F.moonDir.w <= 0.0) { return vec3f(0.0); }
  // point on the visible hemisphere and its selenographic direction
  let q = (dir - m * c) / max(R, 1e-6);
  let n = normalize(q - m * sqrt(max(1.0 - dot(q, q), 0.0)));
  let toE = -m;
  let north = normalize(F.moonUp.xyz - m * dot(F.moonUp.xyz, m));
  let east = cross(m, north);   // lunar east, towards Mare Crisium = celestial west
  let p = vec3f(dot(n, east), dot(n, north), dot(n, toE));
  let alb = moonAlbedo(p);
  // sunlight: Lommel-Seeliger (no limb darkening) with the opposition surge; the sun lights
  // it from the scene's own sun direction, so phase and terminator tilt follow geometrically
  let Ld = F.sunDir.xyz;
  let mu0 = max(dot(n, Ld), 0.0);
  let mu = max(dot(n, toE), 1e-3);
  let cosA = dot(Ld, toE);
  let surge = (1.0 + 3.5 * exp(-acos(clamp(cosA, -1.0, 1.0)) / 0.12)) / 4.5;
  var lit = F.sunDir.w * MOON_L * 2.0 / 0.8 * alb * mu0 / (mu0 + mu) * surge;
  lit *= min(1.0, MOON_DISC_MAX / (F.sunDir.w * MOON_L * F.astro.w));
  // earthshine: the sunlit Earth (full when the moon is new) lights the night side
  let es = F.sunDir.w * 2e-4 * (0.5 - 0.5 * cosA) * 0.13 / PI * alb * mu;
  let col = vec3f(1.04, 1.0, 0.93) * lit + vec3f(0.85, 0.95, 1.15) * es;
  return col * F.astro.w * edge * sampleTrans(camR(), max(dir.y, -0.01));
}

fn skyRadiance(dirIn: vec3f, withSun: bool, pixAng: f32) -> vec3f {
  var dir = dirIn;
  dir.y = max(dir.y, 0.0005);
  dir = normalize(dir);
  var L = skyAtmo(dir);
  var sunT = 1.0;
  let cl = cloudLayers(dir, pixAng, vec2f(-1.0), &sunT);
  L = L * (1.0 - cl.a) + cl.rgb;
  if (withSun) { L += sunDisk(dir) * sunT; }
  return L;
}
`;

export const ENV_CUBE = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d_array<f32>;   // layer 0: sun, 1: moon
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d_array<rgba16float, write>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
const ENV: f32 = ${ENV_SIZE}.0;
fn faceDir(face: u32, uv: vec2f) -> vec3f {
  let u = uv.x * 2.0 - 1.0;
  let v = uv.y * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -v, -u); }
    case 1u: { return vec3f(-1.0, -v, u); }
    case 2u: { return vec3f(u, 1.0, v); }
    case 3u: { return vec3f(u, -1.0, -v); }
    case 4u: { return vec3f(u, -v, 1.0); }
    default: { return vec3f(-u, -v, -1.0); }
  }
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= ENV || f32(id.y) >= ENV) { return; }
  // faces are refreshed round-robin: face = (frame + z) % 6
  let face = (u32(F.post.z) + id.z) % 6u;
  let dir = normalize(faceDir(face, (vec2f(id.xy) + 0.5) / ENV));
  var L: vec3f;
  if (dir.y >= 0.0) {
    L = skyRadiance(dir, false, 1.5708 / ENV * 1.5);
  } else {
    // below the horizon: what a reflected ray would see -> far, hazy ocean
    let hz = skyRadiance(normalize(vec3f(dir.x, 0.004, dir.z)), false, 1.5708 / ENV * 1.5);
    let body = F.waterCol.rgb * (sunIlluminance() * max(F.sunDir.y, 0.0) + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 2.0);
    L = mix(hz * 0.75, body, clamp(-dir.y * 6.0, 0.0, 1.0));
  }
  textureStore(dst, id.xy, face, vec4f(L, 1.0));
}
`;

export const CUBE_MIPS = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let srcSize = vec2i(textureDimensions(src));
  let p = vec2i(id.xy) * 2;
  // 4x4 tent-ish filter for smoother roughness blur
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 2; j++) {
    for (var i = -1; i <= 2; i++) {
      let q = clamp(p + vec2i(i, j), vec2i(0), srcSize - 1);
      let w = select(1.0, 3.0, i == 0 || i == 1) * select(1.0, 3.0, j == 0 || j == 1);
      acc += textureLoad(src, q, id.z, 0) * w;
      wsum += w;
    }
  }
  textureStore(dst, id.xy, id.z, acc / wsum);
}
`;

// Cloud transmittance towards the sun (averaged over the solar disc), used to dim
// direct sunlight on the water when the sun is behind clouds.
export const SUN_VIS = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d_array<f32>;   // layer 0: sun, 1: moon
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var<storage, read_write> sunVis: array<f32, 4>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
var<workgroup> part: array<f32, 8>;
var<workgroup> partL: array<f32, 8>;
// Eye adaptation: the gain applied as pre-exposure when the sky gets darker than at sunset
// (partial adaptation, so night still looks like night). Exactly 1 for brighter skies.
const ADAPT_REF: f32 = ${ADAPT_REF};
const ADAPT_POW: f32 = 0.85;
// one ray per thread (7 rays across the disc), then a tiny reduction; each thread also
// measures the bare sky luminance in one direction (zenith + a ring at 24 deg elevation)
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_index) k: u32) {
  let la = f32(k) * 0.8975979;
  let ld = select(normalize(vec3f(cos(la), 0.45, sin(la))), vec3f(0.0, 1.0, 0.0), k == 7u);
  partL[k] = dot(skyAtmo(ld), vec3f(0.2126, 0.7152, 0.0722)) / F.astro.w;
  let sd = F.sunDir.xyz;
  let tx = normalize(cross(sd, vec3f(0.0, 1.0, 0.0)));
  let ty = cross(tx, sd);
  var t = 1.0;
  if (k < 7u) {
    let a = f32(k) * 0.8975979;
    let o = select(vec2f(cos(a), sin(a)) * 0.65, vec2f(0.0), k == 0u);
    let dir = normalize(sd + (tx * o.x + ty * o.y) * SUN_ANG_R);
    let cl = cloudLayers(dir, 0.0, vec2f(-1.0), &t);
  }
  part[k] = t;
  workgroupBarrier();
  if (k == 0u) {
    var acc = 0.0;
    for (var i = 0u; i < 7u; i++) { acc += part[i]; }
    sunVis[0] = acc / 7.0;
    var lum = 0.0;
    for (var i = 0u; i < 8u; i++) { lum += partL[i]; }
    lum *= 0.125;
    sunVis[1] = clamp(pow(ADAPT_REF / max(lum, 1e-12), ADAPT_POW), 1.0, 1e6);
    sunVis[2] = lum;
  }
}
`;
