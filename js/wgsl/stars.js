// Stars from the Yale Bright Star Catalogue: one instanced point glow per star, placed for
// the observer's sidereal time and latitude, drawn additively where only sky is behind them
// (depth still at the far plane). Physically scaled: the illuminance of each star's V
// magnitude, dimmed by atmospheric extinction and clouds, so they emerge on their own as
// twilight darkens the sky. They twinkle more through the long air path near the horizon.
import { FRAME_WGSL, ATMO_CORE_WGSL } from './common.js';
import { SKY_RENDER_FUNCS, SKY_CLOUD_FUNCS } from './sky.js';
import { RENDER_BINDINGS } from './ocean.js';

export const STARS = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
// illuminance of a V = 0 star above the atmosphere, 10^(-0.4 * 14.18) lux, in renderer
// units (solar illuminance 1.28e5 lux = 14)
const STAR_E0: f32 = 2.34e-10;
const PSF: f32 = 0.8;   // point spread, gaussian sigma in pixels

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) off: vec2f,
  @location(1) col: vec3f,
};

// s0 = RA, Dec (rad), relative illuminance 10^(-0.4 V), twinkle seed; s1 = linear rgb
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) s0: vec4f, @location(1) s1: vec4f) -> VSOut {
  var o: VSOut;
  o.pos = vec4f(2.0, 2.0, 0.0, 1.0);   // off screen unless the star is up
  o.off = vec2f(0.0);
  o.col = vec3f(0.0);
  // equatorial -> local east / north / up
  let H = F.astro.x - s0.x;
  let cd = cos(s0.y);
  let sd = sin(s0.y);
  let cl = cos(F.astro.y);
  let sl = sin(F.astro.y);
  let e = -cd * sin(H);
  let n = sd * cl - cd * sl * cos(H);
  let u = sd * sl + cd * cl * cos(H);
  if (u < -0.01) { return o; }
  // true north lies at scene azimuth F.astro.z (scene azimuth a: direction (sin a, 0, -cos a))
  let co = cos(F.astro.z);
  let so = sin(F.astro.z);
  let dir = vec3f(e * co + n * so, u, e * so - n * co);
  if (dot(dir, F.moonDir.xyz) > cos(F.moonUp.w)) { return o; }   // behind the moon
  var clip = F.viewProj * vec4f(dir, 0.0);   // at infinity: depth 0 (reverse-Z far plane)
  if (clip.w <= 0.0) { return o; }

  var d2 = dir;
  d2.y = max(d2.y, 0.0005);
  d2 = normalize(d2);
  var sunT = 1.0;
  let cloud = cloudLayers(d2, 0.0, vec2f(-1.0), &sunT);
  let t = F.camPos.w;
  let amp = 0.06 + 0.45 * (1.0 - smoothstep(0.03, 0.5, u));
  let tw = max(0.15, 1.0 + amp * (0.6 * sin(t * 13.7 + s0.w * 61.0) + 0.4 * sin(t * 29.3 + s0.w * 17.0)));
  let E = STAR_E0 * s0.z * tw * F.astro.w * (1.0 - cloud.a);
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  // peak radiance of the gaussian spot carrying the star's illuminance
  o.col = s1.rgb * sampleTrans(camR(), max(u, 0.0)) * (E / (2.0 * PI * PSF * PSF * pixAng * pixAng));

  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi] * (3.0 * PSF);
  clip = vec4f(clip.xy + c * 2.0 * F.screen.zw * clip.w, 0.0, clip.w);
  o.pos = clip;
  o.off = c;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.col * exp(-0.5 * dot(in.off, in.off) / (PSF * PSF)), 0.0);
}
`;
