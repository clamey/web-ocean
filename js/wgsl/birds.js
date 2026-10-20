// Seagulls: instanced low-poly birds with articulated wings (shoulder + wrist), lit by the
// sun (dimmed by clouds), the sky hemisphere above and the darker sea below, with feather
// translucency when backlit, plus the same aerial perspective as the water.
import { FRAME_WGSL, ATMO_CORE_WGSL, ENV_MIPS } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';
import { RENDER_BINDINGS } from './ocean.js';

export const BIRDS = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + /* wgsl */ `
@group(0) @binding(4) var envCube: texture_cube<f32>;
@group(0) @binding(5) var<storage, read> sunVis: array<f32, 4>;
` + SKY_RENDER_FUNCS + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;
const SHOULDER: f32 = 0.05;
const WRIST: f32 = 0.32;
const SPAN: f32 = 1.36;
// 1 = render the mirror image below sea level (sampled by the ocean as the birds' reflection)
override MIRROR: f32 = 0.0;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) nrm: vec3f,
  @location(2) @interpolate(flat) mat: f32,
};

fn rot(p: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(p.x * c - p.y * s, p.x * s + p.y * c);
}

// i0 = camera-relative position, roll (+ = right wing down)
// i1 = flight direction (unit), wing beat phase
// i2 = flap amount (0 glide .. 1 full beat), size
@vertex fn vs(@location(0) lp: vec3f, @location(1) ln: vec3f, @location(2) pm: vec2f,
              @location(3) i0: vec4f, @location(4) i1: vec4f, @location(5) i2: vec4f) -> VSOut {
  let side = select(-1.0, 1.0, lp.x >= 0.0);
  let ph = i1.w;
  let amp = i2.x;
  // gliding: arms raised, hands drooped (the gull "M"); flapping: big shoulder stroke with
  // the hand lagging behind it
  let a1 = mix(0.12, 0.10 + 0.62 * sin(ph), amp) * side;
  let a2 = mix(-0.24, 0.42 * sin(ph - 0.7), amp) * side;
  var p = lp;
  var n = ln;
  if (pm.x > 1.5) {
    let q = rot(vec2f(p.x - side * WRIST, p.y), a2);
    p = vec3f(q.x + side * WRIST, q.y, p.z);
    n = vec3f(rot(n.xy, a2), n.z);
  }
  if (pm.x > 0.5) {
    let q = rot(vec2f(p.x - side * SHOULDER, p.y), a1);
    p = vec3f(q.x + side * SHOULDER, q.y, p.z);
    n = vec3f(rot(n.xy, a1), n.z);
  }
  // the body rides up on the downstroke
  p.y += 0.03 * cos(ph) * amp;
  p = vec3f(rot(p.xy, -i0.w), p.z);
  n = vec3f(rot(n.xy, -i0.w), n.z);
  let f = i1.xyz;
  let r = normalize(vec3f(-f.z, 0.0, f.x));
  let u = cross(r, f);
  // keep far birds at least a few pixels wide so they don't shimmer away
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  let s = i2.y * clamp(5.0 * pixAng * length(i0.xyz) / SPAN, 1.0, 2.5);
  var o: VSOut;
  o.rel = i0.xyz + (r * p.x + u * p.y + f * p.z) * s;
  o.nrm = r * n.x + u * n.y + f * n.z;
  if (MIRROR > 0.5) {
    o.rel.y = -o.rel.y - 2.0 * F.camPos.y;
    o.nrm.y = -o.nrm.y;
  }
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.mat = pm.y;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  // the mirror image is shaded as the real bird seen from the mirrored viewpoint; dist is the
  // full camera -> water -> bird path either way
  let flip = select(vec3f(1.0), vec3f(1.0, -1.0, 1.0), MIRROR > 0.5);
  let dist = length(in.rel);
  let V = -in.rel / dist * flip;
  var n = normalize(in.nrm) * flip;
  // thin surfaces: light the side facing the viewer
  let under = dot(n, V) < 0.0;
  if (under) { n = -n; }
  let white = vec3f(0.80, 0.80, 0.78);
  var alb = white;
  var trans = 0.35;
  if (in.mat > 0.5 && in.mat < 1.5) {
    alb = select(vec3f(0.42, 0.45, 0.48), white, under);
    trans = select(0.18, 0.35, under);
  } else if (in.mat > 1.5 && in.mat < 2.5) {
    alb = vec3f(0.035, 0.035, 0.04);
    trans = 0.02;
  } else if (in.mat > 2.5) {
    alb = vec3f(0.72, 0.52, 0.10);
    trans = 0.0;
  }

  let Ld = F.sunDir.xyz;
  let sunE = sunIlluminance() * sunVis[0] * smoothstep(-0.01, 0.01, Ld.y);
  let skyUp = textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb;
  let skyN = textureSampleLevel(envCube, sampC, normalize(vec3f(n.x, max(n.y, 0.05), n.z)), ENV_MAXMIP - 1.0).rgb;
  // from below: the sea is much darker than the sky it reflects
  let amb = mix(skyUp * 0.15, skyN, smoothstep(-0.5, 0.5, n.y));
  let NdL = dot(n, Ld);
  let direct = (sunE * max(NdL, 0.0) + moonIlluminance() * max(dot(n, F.moonDir.xyz), 0.0)) * 0.318;
  // sunlight diffusing through thin feathers when the bird is backlit
  let back = sunE * max(-NdL, 0.0) * 0.318 * trans;
  var col = alb * (direct + amb) + white * back;

  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let fogCol = skyAtmo(normalize(vec3f(-V.x, max(-V.y, 0.002), -V.z)));
  col = col * T + fogCol * (1.0 - T);
  return vec4f(col, 1.0);
}
`;
