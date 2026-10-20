// Where and when: observer locations, the sun's position (low-precision Meeus / NOAA
// formulae, ~0.01 deg) and local sidereal time, plus decoding of the packed star catalogue.
// The scene's sun elevation slider picks the time: the evening moment on the chosen date
// when the sun reaches that elevation.
const DEG = Math.PI / 180;

export const LOCATIONS = {
  lajolla: { name: 'La Jolla', lat: 32.8328, lon: -117.2713, tz: 'America/Los_Angeles' },
  santorini: { name: 'Santorini', lat: 36.4167, lon: 25.4318, tz: 'Europe/Athens' },
  split: { name: 'Split', lat: 43.5081, lon: 16.4402, tz: 'Europe/Zagreb' },
  keywest: { name: 'Key West', lat: 24.5551, lon: -81.7800, tz: 'America/New_York' },
};

// today's calendar date (YYYY-MM-DD) in a time zone
export function todayIn(tz) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: tz }).format(new Date());
}

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
// d: days since J2000.0 (2000-01-01 12:00 UT)
function sunRaDec(d) {
  const g = (357.529 + 0.98560028 * d) * DEG;
  const L = (280.459 + 0.98564736 * d + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * DEG;
  const e = (23.439 - 0.00000036 * d) * DEG;
  return [Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L)), Math.asin(Math.sin(e) * Math.sin(L)), L];
}
const gmst = (d) => ((280.46061837 + 360.98564736629 * d) % 360) * DEG;
const J2000_MS = Date.UTC(2000, 0, 1, 12);

// The evening moment on the local calendar date dateStr when the sun stands at elevDeg.
// Returns local sidereal time and latitude (rad), the sun's true azimuth (rad from north
// through east) and the UTC time in ms. Elevations above the day's maximum give local noon.
export function solveSky(loc, dateStr, elevDeg) {
  const [y, m, day] = dateStr.split('-').map(Number);
  const lat = loc.lat * DEG, lon = loc.lon * DEG;
  // start at local mean noon and step along the hour angle
  let d = (Date.UTC(y, m - 1, day, 12) - J2000_MS) / 864e5 - loc.lon / 360;
  const sinE = Math.sin(elevDeg * DEG);
  for (let i = 0; i < 5; i++) {
    const [ra, dec] = sunRaDec(d);
    const c = (sinE - Math.sin(lat) * Math.sin(dec)) / (Math.cos(lat) * Math.cos(dec));
    const H = Math.acos(Math.max(-1, Math.min(1, c)));
    d += wrapAngle(H - (gmst(d) + lon - ra)) / (2 * Math.PI * 1.00273790935);
  }
  const [ra, dec] = sunRaDec(d);
  const lst = gmst(d) + lon;
  const [sunAz] = altAz(ra, dec, lst, lat);
  return { d, lst: wrapAngle(lst), lat, sunAz, utcMs: J2000_MS + d * 864e5 };
}

// equatorial (rad) -> [azimuth from north through east, altitude] for sidereal time lst
export function altAz(ra, dec, lst, lat) {
  const ha = lst - ra;
  const east = -Math.cos(dec) * Math.sin(ha);
  const north = Math.sin(dec) * Math.cos(lat) - Math.cos(dec) * Math.sin(lat) * Math.cos(ha);
  const up = Math.sin(dec) * Math.sin(lat) + Math.cos(dec) * Math.cos(lat) * Math.cos(ha);
  return [Math.atan2(east, north), Math.asin(Math.max(-1, Math.min(1, up)))];
}

// The Moon: main periodic terms of Meeus' lunar theory (Astronomical Algorithms ch. 47),
// good to ~0.2 deg. Returns geocentric RA, Dec (rad), distance (km) and ecliptic longitude.
export function moonRaDec(d) {
  const T = d / 36525;
  const r = (x) => (x % 360) * DEG;
  const Lp = r(218.3164477 + 481267.88123421 * T);
  const D = r(297.8501921 + 445267.1114034 * T);
  const M = r(357.5291092 + 35999.0502909 * T);
  const Mp = r(134.9633964 + 477198.8675055 * T);
  const F = r(93.2720950 + 483202.0175233 * T);
  const s = Math.sin, c = Math.cos;
  const lon = Lp + DEG * (6.289 * s(Mp) + 1.274 * s(2 * D - Mp) + 0.658 * s(2 * D) + 0.214 * s(2 * Mp)
    - 0.186 * s(M) - 0.114 * s(2 * F) + 0.059 * s(2 * D - 2 * Mp) + 0.057 * s(2 * D - M - Mp)
    + 0.053 * s(2 * D + Mp) + 0.046 * s(2 * D - M) - 0.041 * s(M - Mp) - 0.035 * s(D) - 0.030 * s(M + Mp));
  const lat = DEG * (5.128 * s(F) + 0.281 * s(Mp + F) + 0.278 * s(Mp - F) + 0.173 * s(2 * D - F)
    + 0.055 * s(2 * D - Mp + F) - 0.046 * s(2 * D - Mp - F) + 0.033 * s(2 * D + F) + 0.017 * s(2 * Mp + F));
  const dist = 385000.56 - 20905.355 * c(Mp) - 3699.111 * c(2 * D - Mp) - 2955.968 * c(2 * D)
    - 569.925 * c(2 * Mp) + 48.888 * c(M) + 246.158 * c(2 * D - 2 * Mp) - 152.138 * c(2 * D - M - Mp)
    - 170.733 * c(2 * D + Mp) - 204.586 * c(2 * D - M) - 129.620 * c(M - Mp) + 108.743 * c(D) + 104.755 * c(M + Mp);
  const e = (23.439 - 0.00000036 * d) * DEG;
  const ra = Math.atan2(s(lon) * c(e) - Math.tan(lat) * s(e), c(lon));
  const dec = Math.asin(s(lat) * c(e) + c(lat) * s(e) * s(lon));
  return { ra, dec, dist, lon };
}

// The Moon seen from the observer at day d: true azimuth and topocentric altitude (rad),
// angular radius, illuminated fraction, waxing flag, and the moonlight illuminance above
// the atmosphere (lux) from its phase (Allen's magnitude law) and distance.
export function moonFor(d, lst, lat) {
  const m = moonRaDec(d);
  const [az, altGeo] = altAz(m.ra, m.dec, lst, lat);
  const alt = altGeo - Math.asin(6378.14 / m.dist) * Math.cos(altGeo);   // parallax, up to ~1 deg
  const [sra, sdec, sunLon] = sunRaDec(d);
  const cosElong = Math.sin(sdec) * Math.sin(m.dec) + Math.cos(sdec) * Math.cos(m.dec) * Math.cos(m.ra - sra);
  const phase = Math.PI - Math.acos(Math.max(-1, Math.min(1, cosElong)));   // sun-moon-earth angle
  const pDeg = phase / DEG;
  const mag = -12.73 + 0.026 * pDeg + 4e-9 * pDeg ** 4 + 5 * Math.log10(m.dist / 384400);
  return {
    az, alt, dist: m.dist, radius: Math.asin(1737.4 / m.dist),
    illum: (1 + Math.cos(phase)) / 2, waxing: Math.sin(m.lon - sunLon) > 0,
    lux: Math.pow(10, -0.4 * (mag + 14.18)),
  };
}

// The north ecliptic pole (~ the Moon's north, within 1.5 deg) for orienting the lunar face
export const ECLIPTIC_POLE = [18 * 15 * DEG, (90 - 23.439) * DEG];

// Linear sRGB (luminance 1) of a blackbody star with colour index B-V: Ballesteros (2012)
// temperature, Kim et al. (2002) Planckian locus chromaticity.
function starColor(bv, out, o) {
  const T = Math.max(1667, Math.min(25000, 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))));
  const t = 1e3 / T, t2 = t * t, t3 = t2 * t;
  const x = T <= 4000 ? -0.2661239 * t3 - 0.2343589 * t2 + 0.8776956 * t + 0.179910
    : -3.0258469 * t3 + 2.1070379 * t2 + 0.2226347 * t + 0.240390;
  const y = T <= 2222 ? -1.1063814 * x ** 3 - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
    : T <= 4000 ? -0.9549476 * x ** 3 - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
      : 3.0817580 * x ** 3 - 5.87338670 * x * x + 3.75112997 * x - 0.37001483;
  const X = x / y, Z = (1 - x - y) / y;
  const r = Math.max(0, 3.2406 * X - 1.5372 - 0.4986 * Z);
  const g = Math.max(0, -0.9689 * X + 1.8758 + 0.0415 * Z);
  const b = Math.max(0, 0.0557 * X - 0.2040 + 1.0570 * Z);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  out[o] = r / lum; out[o + 1] = g / lum; out[o + 2] = b / lum;
}

// Packed catalogue (see stars.js) -> 8 floats per star:
// RA, Dec (rad), relative illuminance 10^(-0.4 V), twinkle seed, linear rgb, 0
export function decodeStars(b64, count) {
  const bin = atob(b64);
  const dv = new DataView(new Uint8Array(Array.from(bin, (ch) => ch.charCodeAt(0))).buffer);
  const out = new Float32Array(count * 8);
  for (let i = 0; i < count; i++) {
    const p = i * 6, o = i * 8;
    out[o] = (dv.getUint16(p, true) / 65536) * 2 * Math.PI;
    out[o + 1] = (dv.getInt16(p + 2, true) / 32767) * (Math.PI / 2);
    out[o + 2] = Math.pow(10, -0.4 * (dv.getUint8(p + 4) / 20 - 2));
    out[o + 3] = ((i * 0.618034) % 1) * 10;
    starColor(dv.getInt8(p + 5) / 50, out, o + 4);
  }
  return out;
}
