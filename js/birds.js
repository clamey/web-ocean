// Seagulls. A few small flocks simulated on the CPU: boids steering towards waypoints that
// circle an anchor lazily following the camera, so the birds stay around wherever you fly.
// Drawn as instanced low-poly gulls with articulated wings by wgsl/birds.js.
// All state lives in preallocated typed arrays (no per-frame garbage).
export const MAX_BIRDS = 64;
export const BIRD_FLOATS = 12;   // per instance: rel pos + roll, forward + flap phase, flap amount + size
// per bird for the ocean's shadow test: rel pos + projected half-span, right axis + size, forward
export const SHADOW_FLOATS = 12;
const FLOCKS = 4;
const TAU = Math.PI * 2;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gull mesh in bird space (x right, y up, z forward; 1.36 m wingspan). 8 floats per vertex:
// position, normal, part (0 body, 1 inner wing, 2 hand), material (0 white, 1 wing: grey above
// and white below, 2 black wingtip, 3 beak). Non-indexed triangle list.
export function buildBirdMesh() {
  const out = [];
  const vtx = (x, y, z, nx, ny, nz, part, mat) => [x, y, z, nx, ny, nz, part, mat];
  const tri = (a, b, c) => out.push(...a, ...b, ...c);
  const quad = (a, b, c, d) => { tri(a, b, c); tri(a, c, d); };

  // body: elliptical rings lofted along z, wings attach at y = 0 (upper body)
  const RINGS = [
    // z, half width, half height, centre y
    [-0.30, 0.010, 0.008, -0.010],
    [-0.20, 0.045, 0.040, -0.018],
    [-0.04, 0.068, 0.066, -0.022],
    [0.10, 0.058, 0.058, -0.012],
    [0.18, 0.034, 0.038, 0.012],
    [0.235, 0.042, 0.042, 0.028],
    [0.285, 0.016, 0.018, 0.028],
  ];
  const SEG = 8;
  const ringV = (r, k, mat) => {
    const a = (k / SEG) * TAU, c = Math.cos(a), s = Math.sin(a);
    const nl = Math.hypot(c * r[2], s * r[1]);
    return vtx(c * r[1], r[3] + s * r[2], r[0], (c * r[2]) / nl, (s * r[1]) / nl, 0, 0, mat);
  };
  for (let i = 0; i + 1 < RINGS.length; i++) {
    for (let k = 0; k < SEG; k++) {
      quad(ringV(RINGS[i], k, 0), ringV(RINGS[i], k + 1, 0), ringV(RINGS[i + 1], k + 1, 0), ringV(RINGS[i + 1], k, 0));
    }
  }
  const head = RINGS[RINGS.length - 1];
  const tip = vtx(0, 0.022, 0.36, 0, 0, 1, 0, 3);
  for (let k = 0; k < SEG; k++) tri(ringV(head, k, 3), ringV(head, k + 1, 3), tip);

  const flat = (x, y, z, part, mat) => vtx(x, y, z, 0, 1, 0, part, mat);
  // tail
  quad(flat(-0.035, -0.012, -0.20, 0, 0), flat(0.035, -0.012, -0.20, 0, 0),
    flat(0.075, -0.016, -0.38, 0, 0), flat(-0.075, -0.016, -0.38, 0, 0));
  // wings: inner arm, grey hand, black tip (swept back)
  const WING = [
    // part, material, inner x, leading z, trailing z, outer x, leading z, trailing z
    [1, 1, 0.05, 0.07, -0.13, 0.32, 0.06, -0.12],
    [2, 1, 0.32, 0.06, -0.12, 0.54, -0.01, -0.13],
    [2, 2, 0.54, -0.01, -0.13, 0.68, -0.13, -0.16],
  ];
  for (const s of [1, -1]) {
    for (const [part, mat, x0, l0, t0, x1, l1, t1] of WING) {
      quad(flat(s * x0, 0, l0, part, mat), flat(s * x1, 0, l1, part, mat),
        flat(s * x1, 0, t1, part, mat), flat(s * x0, 0, t0, part, mat));
    }
  }
  return new Float32Array(out);
}

export function createBirds(seed = 1) {
  const rnd = mulberry32(seed);
  const range = (a, b) => a + (b - a) * rnd();
  const n = MAX_BIRDS;
  const pos = new Float64Array(n * 3), vel = new Float32Array(n * 3);
  const form = new Float32Array(n * 3);   // slot offset from the flock waypoint
  const cruise = new Float32Array(n), freq = new Float32Array(n), size = new Float32Array(n);
  const phase = new Float32Array(n), amp = new Float32Array(n), bank = new Float32Array(n);
  const burst = new Uint8Array(n), burstT = new Float32Array(n);
  const inst = new Float32Array(n * BIRD_FLOATS);
  const shadow = new Float32Array(n * SHADOW_FLOATS);

  // Flocks start at fixed bearings around the camera (angle 0 = +x, -pi/2 = straight ahead
  // of the start view) and slowly drift around it.
  const FLOCK_INIT = [
    // bearing, distance (m), altitude (m), orbit radius (m)
    [-1.75, 70, 12, 30],
    [-1.20, 220, 28, 60],
    [0.90, 120, 18, 45],
    [2.60, 65, 9, 28],
  ];
  const fAng = new Float64Array(FLOCKS), fRad = new Float32Array(FLOCKS), fDrift = new Float32Array(FLOCKS);
  const fAx = new Float64Array(FLOCKS), fAz = new Float64Array(FLOCKS);
  const fAlt = new Float32Array(FLOCKS), fOrbR = new Float32Array(FLOCKS);
  const fTh = new Float64Array(FLOCKS), fDir = new Float32Array(FLOCKS);
  const wx = new Float64Array(FLOCKS), wy = new Float64Array(FLOCKS), wz = new Float64Array(FLOCKS);
  for (let f = 0; f < FLOCKS; f++) {
    const [a, d, h, r] = FLOCK_INIT[f];
    fAng[f] = a; fRad[f] = d; fAlt[f] = h; fOrbR[f] = r;
    fDrift[f] = range(-0.006, 0.006);
    fTh[f] = range(0, TAU);
    fDir[f] = rnd() < 0.5 ? -1 : 1;
  }
  for (let i = 0; i < n; i++) {
    const a = range(0, TAU), r = 12 * Math.sqrt(rnd());
    form[i * 3] = Math.cos(a) * r; form[i * 3 + 1] = range(-3, 3); form[i * 3 + 2] = Math.sin(a) * r;
    cruise[i] = range(8.5, 11.5);
    freq[i] = range(2.6, 3.4);
    size[i] = range(0.88, 1.1);
    phase[i] = range(0, TAU);
    burstT[i] = range(0, 6);
  }

  function spawn(i, f) {
    const o = i * 3;
    pos[o] = wx[f] + form[o] + range(-4, 4);
    pos[o + 1] = Math.max(wy[f] + form[o + 1], 6);
    pos[o + 2] = wz[f] + form[o + 2] + range(-4, 4);
    // fly along the orbit
    const s = fDir[f] * cruise[i];
    vel[o] = -Math.sin(fTh[f]) * s; vel[o + 1] = 0; vel[o + 2] = Math.cos(fTh[f]) * s;
    amp[i] = 0; bank[i] = 0;
  }

  let time = 0, first = true;

  function update(dt, cx, cy, cz) {
    time += dt;
    for (let f = 0; f < FLOCKS; f++) {
      fAng[f] += fDrift[f] * dt;
      const tx = cx + Math.cos(fAng[f]) * fRad[f], tz = cz + Math.sin(fAng[f]) * fRad[f];
      let respawn = first;
      const ox = cx - fAx[f], oz = cz - fAz[f], od = Math.hypot(ox, oz);
      if (first) {
        fAx[f] = tx; fAz[f] = tz;
      } else if (od > 900) {
        // left far behind (fast flight): reappear far ahead, then drift back into place
        fAx[f] = cx + (ox / od) * 600; fAz[f] = cz + (oz / od) * 600;
        respawn = true;
      } else {
        const k = Math.min(1, dt * 0.08);
        fAx[f] += (tx - fAx[f]) * k; fAz[f] += (tz - fAz[f]) * k;
      }
      const R = fOrbR[f] * (1 + 0.3 * Math.sin(time * 0.05 + f * 1.7));
      fTh[f] += dt * fDir[f] * 7.5 / R;
      wx[f] = fAx[f] + Math.cos(fTh[f]) * R;
      wy[f] = fAlt[f] + 5 * Math.sin(fTh[f] * 0.37 + f);
      wz[f] = fAz[f] + Math.sin(fTh[f]) * R;
      if (respawn) for (let i = f; i < n; i += FLOCKS) spawn(i, f);
    }
    first = false;

    for (let i = 0; i < n; i++) {
      const f = i % FLOCKS, o = i * 3;
      const px = pos[o], py = pos[o + 1], pz = pos[o + 2];
      const vx = vel[o], vy = vel[o + 1], vz = vel[o + 2];
      // seek this bird's slot around the waypoint
      const gx = wx[f] + form[o] - px, gy = Math.max(wy[f] + form[o + 1], 6) - py, gz = wz[f] + form[o + 2] - pz;
      const gk = cruise[i] / (Math.hypot(gx, gy, gz) + 1e-6);
      let dx = gx * gk, dy = gy * gk, dz = gz * gk;
      // flockmates: separation, alignment, cohesion
      let cnt = 0, ax = 0, ay = 0, az = 0, mx = 0, my = 0, mz = 0;
      for (let j = f; j < n; j += FLOCKS) {
        if (j === i) continue;
        const q = j * 3;
        const rx = pos[q] - px, ry = pos[q + 1] - py, rz = pos[q + 2] - pz;
        const d2 = rx * rx + ry * ry + rz * rz;
        if (d2 > 1600) continue;
        cnt++;
        ax += vel[q]; ay += vel[q + 1]; az += vel[q + 2];
        mx += rx; my += ry; mz += rz;
        if (d2 < 16) {
          const k = 6 / Math.max(d2, 0.25);
          dx -= rx * k; dy -= ry * k; dz -= rz * k;
        }
      }
      if (cnt) {
        const ic = 1 / cnt;
        dx += (ax * ic - vx) * 0.3 + mx * ic * 0.04;
        dy += (ay * ic - vy) * 0.3 + my * ic * 0.04;
        dz += (az * ic - vz) * 0.3 + mz * ic * 0.04;
      }
      // stay clear of the waves and out of the camera's face
      if (py < 5) dy += (5 - py) * 2;
      const ex = px - cx, ey = py - cy, ez = pz - cz, e2 = ex * ex + ey * ey + ez * ez;
      if (e2 < 100) {
        const k = 40 / Math.max(e2, 1);
        dx += ex * k; dy += ey * k; dz += ez * k;
      }
      dy = Math.max(-3.5, Math.min(3.5, dy));
      let accx = (dx - vx) * 1.2, accy = (dy - vy) * 1.2, accz = (dz - vz) * 1.2;
      const al = Math.hypot(accx, accy, accz);
      if (al > 6) { const k = 6 / al; accx *= k; accy *= k; accz *= k; }
      let nvx = vx + accx * dt, nvy = vy + accy * dt, nvz = vz + accz * dt;
      let sp = Math.hypot(nvx, nvy, nvz);
      const cs = Math.max(6, Math.min(15, sp));
      nvx *= cs / sp; nvy *= cs / sp; nvz *= cs / sp; sp = cs;
      vel[o] = nvx; vel[o + 1] = nvy; vel[o + 2] = nvz;
      pos[o] = px + nvx * dt; pos[o + 1] = py + nvy * dt; pos[o + 2] = pz + nvz * dt;

      // bank into turns (right vector of the flight direction is (-fz, 0, fx))
      const hs = Math.hypot(nvx, nvz) + 1e-6;
      const lat = (-accx * nvz + accz * nvx) / hs;
      const tb = Math.max(-0.9, Math.min(0.9, Math.atan2(lat, 9.81) * 1.4));
      bank[i] += (tb - bank[i]) * Math.min(1, dt * 3);

      // wing beat: flap when climbing or slow and in occasional bursts, glide otherwise
      burstT[i] -= dt;
      if (burstT[i] <= 0) {
        burst[i] ^= 1;
        burstT[i] = burst[i] ? range(0.8, 2.2) : range(3, 10);
      }
      const tgt = Math.max(0, Math.min(1, nvy * 0.45 + (cruise[i] - sp) * 0.35 + burst[i]));
      amp[i] += (tgt - amp[i]) * Math.min(1, dt * 2.5);
      phase[i] = (phase[i] + dt * TAU * freq[i]) % TAU;

      const b = i * BIRD_FLOATS, is = 1 / sp;
      inst[b] = pos[o] - cx; inst[b + 1] = pos[o + 1] - cy; inst[b + 2] = pos[o + 2] - cz; inst[b + 3] = bank[i];
      inst[b + 4] = nvx * is; inst[b + 5] = nvy * is; inst[b + 6] = nvz * is; inst[b + 7] = phase[i];
      inst[b + 8] = amp[i]; inst[b + 9] = size[i]; inst[b + 10] = 0; inst[b + 11] = 0;

      // shadow caster: wing plane axes with the roll applied (matches the vertex shader) and
      // the half-span shortened by the current wing angles
      const fx = inst[b + 4], fy = inst[b + 5], fz = inst[b + 6];
      const rx0 = -fz / hs * sp, rz0 = fx / hs * sp;
      const ux = -rz0 * fy, uy = rz0 * fx - rx0 * fz, uz = rx0 * fy;
      const cr = Math.cos(bank[i]), sr = Math.sin(bank[i]);
      const a1 = 0.12 + (0.10 + 0.62 * Math.sin(phase[i]) - 0.12) * amp[i];
      const a2 = -0.24 + (0.42 * Math.sin(phase[i] - 0.7) + 0.24) * amp[i];
      const s = i * SHADOW_FLOATS;
      shadow[s] = inst[b]; shadow[s + 1] = inst[b + 1]; shadow[s + 2] = inst[b + 2];
      shadow[s + 3] = (0.05 + 0.27 * Math.cos(a1) + 0.36 * Math.cos(a1 + a2)) * size[i];
      shadow[s + 4] = rx0 * cr - ux * sr; shadow[s + 5] = -uy * sr; shadow[s + 6] = rz0 * cr - uz * sr;
      shadow[s + 7] = size[i];
      shadow[s + 8] = fx; shadow[s + 9] = fy; shadow[s + 10] = fz; shadow[s + 11] = 0;
    }
  }

  return { inst, shadow, update };
}
