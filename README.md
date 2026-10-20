# web-ocean

A real-time ocean renderer written from scratch on WebGPU, with no engine or rendering library. It draws a sunset sea with simulated waves, whitecaps, sun glitter, a physically based sky with clouds, and flocks of seagulls. You fly around it with a free camera.

![Looking toward the sun from 2.4 m above the water](docs/front.jpg)

| 75 degrees off the sun, 2.4 m up | Looking toward the sun from 30 m up |
|---|---|
| ![](docs/side.jpg) | ![](docs/high.jpg) |

## Run it

You need a recent Chrome with WebGPU and a reasonably capable GPU. On a laptop, force Chrome onto the dedicated GPU in the Windows graphics settings, or it will fall back to the integrated one.

Open `ocean.html` directly in the browser. `ocean.min.html` is the same page minified (124 KB instead of 187 KB, 40 KB of which is the star catalogue). Both are self-contained and work from `file://`.

To run from source:

```
npm install
npm start        # serves http://localhost:8080/
```

After editing anything under `js/`, run `npm run build` to regenerate both bundles. The minified build strips WGSL comments and whitespace and minifies the JavaScript with esbuild.

## Controls

| Input | Action |
|---|---|
| Click the canvas | Capture the mouse (Esc releases it). Holding the right mouse button also works. |
| Mouse | Look around |
| W A S D | Move |
| Space or E / Q or C | Up / down |
| Shift | 5x speed |
| Mouse wheel | Change base speed |
| H | Settings panel: location, date, wind, swell, choppiness, turbidity, foam, sun, haze, clouds, exposure, bloom, seagulls |

The camera keeps itself above the displaced water surface, so waves never pass through it.

URL parameters are handy for testing. `?w=2560&h=1440` forces the render resolution, `cam=x,y,z,yawDeg,pitchDeg` sets the camera, `fov=deg` sets the field of view, `ui=0` hides the HUD, and any settings key works too, for example `windSpeed=12&swellHs=2&sunElev=8`. `loc=lajolla|santorini|split|keywest` and `date=YYYY-MM-DD` pick the night sky, for example `?loc=santorini&date=2026-08-12&sunElev=-18`, or a crescent moon with earthshine at `?loc=santorini&date=2026-08-16&sunElev=-8&cam=0,2.4,0,-42.7,13&fov=3`.

## How it works

### Waves

The sea is a Tessendorf FFT simulation with six cascades of 256 by 256 samples. The patches range from 1103 m down to 0.26 m, so wavelengths run from about a kilometre to a few millimetres. The smallest cascade produces the sharp sun sparkles near the camera.

The spectrum is a fetch-limited JONSWAP wind sea with Donelan-Banner directional spreading, plus a narrow long-crested swell. Dispersion includes the capillary term, and choppy horizontal displacement sharpens the crests. The inverse FFTs are radix-2 compute shaders working in workgroup memory.

The surface mesh is a CDLOD quadtree with 17 levels. Vertices are 6.25 cm apart near the camera and the mesh reaches 160 km, with continuous vertex morphing between levels.

### Shading

- Mip-mapped first and second moments of the wave slopes (LEAN mapping) give the right anisotropic roughness at every distance. The sun glitter widens with distance and the far sea settles into a sheen without aliasing. Screen-space slope derivatives add specular anti-aliasing.
- Sun glitter uses an anisotropic Gaussian slope distribution with Smith masking and the size of the solar disc.
- The sky reflection comes from an environment cube map rebuilt every frame, with roughness-filtered mips. At grazing angles the reflection direction is tilted to account for which wave facets the viewer can actually see.
- The water body is turbid blue-green, lit by the whole sky hemisphere. Backlit crests get a subsurface glow, and wind-gust patches modulate the short ripples.
- Whitecaps start where the total surface Jacobian folds. Foam is stored in two camera-centred world-space maps (128 m and 1 km wide), decays over time, drifts downwind, and breaks into streaks as it ages. A procedural bubble texture gives it its structure.

### Sky

The atmosphere follows Hillaire's approach, with transmittance, multiple-scattering, and sky-view lookup tables plus aerosol haze. There are two cloud layers, a broken low cumulus and a fibrous altostratus deck. Each is shaded with self-shadowing toward the sun, lit rims, silver linings when backlit, and direction-dependent multiple scattering.

Cloud noise octaves are filtered by pixel footprint, and silhouettes are anti-aliased analytically: the screen-space derivative of each layer's edge function sets the width of its opacity edge. Cloud transmittance along the sun direction dims the sunlight that reaches the water, and aerial perspective is applied over the sea.

### Seagulls

Four small flocks fly around the camera, 24 birds by default (up to 64 with the "Seagulls" slider or `birds=` in the URL). The flight model runs on the CPU, allocation free: each bird steers toward its own slot around a waypoint that circles a slowly drifting anchor, with boids separation, alignment, and cohesion inside its flock. Birds bank into turns, flap when climbing or slow and in occasional bursts, and glide with the gull's bent "M" wing pose the rest of the time. The anchors follow the camera lazily, so the flocks stay around you, and a flock that is left far behind reappears far ahead.

Each gull is a low-poly mesh drawn in one instanced call, with the wings articulated at the shoulder and wrist in the vertex shader. Upper wings are grey, the body and underwings white, the wingtips black. They are lit by the same cloud-dimmed sun and sky as the water, with darker light from the sea below, sunlight diffusing through the feathers when backlit, and the same aerial perspective. Distant birds are kept at least a few pixels wide so they don't shimmer away.

The birds cast soft shadows on the water: the ocean shader intersects the sun ray from each water point with every bird's wing plane and estimates how much of the solar disc the wings and body cover, which dims the sun glitter, the water body, and the foam. The penumbra grows with the distance to the bird, so shadows are crisp under a high sun and practically vanish at sunset, as they do in reality. Reflections come from a half-resolution render of the birds mirrored about sea level, looked up along each pixel's reflection ray, displaced by the resolved waves and blurred by the unresolved slope spread, like the sky reflection. A reflected bird also hides the sun glitter behind it. On a wind-rippled sea this smears a small bird's reflection out until it is barely visible, which is also what happens in reality.

### Stars and night

Lower the sun below the horizon (the slider goes down to -18 degrees, astronomical night) and the stars come out. They are the 5,080 stars of the Yale Bright Star Catalogue down to magnitude 6, the naked-eye limit, packed into 40 KB (`js/stars.js`, regenerated by `tools/make-stars.mjs`). Each is drawn as a small point glow carrying the real illuminance of its magnitude, colored from its B-V index via its blackbody temperature, dimmed by atmospheric extinction and hidden by clouds. Nothing fades them in artificially: they appear when the sky behind them gets dark enough, so the brightest show up around -6 to -10 degrees and the full sky by about -15. They twinkle, more so near the horizon.

The location (La Jolla, Santorini, Split or Key West) and date set where the stars are. The sun elevation slider picks the time: the evening moment on that date when the sun stands at that elevation, computed with the standard low-precision solar position formulae (good to about 0.01 degrees). The HUD shows the resulting local time. The compass is turned so the real sun's azimuth lines up with the scene's sun azimuth, so the stars stay consistent with the sun and the glow it leaves on the horizon.

Night needed a few more pieces:

- A moonless night sky floor (airglow, zodiacal light, unresolved starlight, about 21.5 mag per square arcsecond), slightly brighter towards the horizon.
- The Earth's shadow: direct sunlight on the clouds and the sea stops when the sun drops below their horizon, so clouds keep their afterglow briefly after sunset and then go dark.
- Eye adaptation. The sky's luminance is measured on the GPU each frame. Once it drops below its value at sunset, an exposure gain (partial, to the 0.85 power, so night still reads as night) is applied as pre-exposure in every pass, which keeps night radiances inside half-float range. It is exactly 1 for brighter skies, so daytime and sunset scenes are unchanged.

### The Moon

The Moon's position comes from the main terms of Meeus' lunar theory with topocentric parallax, good to about 0.2 degrees; it reproduces the 2026 eclipses, down to the Moon sitting on the sun from Oviedo during the total solar eclipse of 12 August 2026. Its phase is not looked up: the disc is a sphere lit from the scene's own sun direction, so the lit fraction and the tilt of the terminator follow from the geometry. The surface uses a Lommel-Seeliger reflectance (no limb darkening, as on the real full moon) with the opposition surge that makes the full moon so much brighter than a half moon, and its face is a procedural albedo map of the maria at their selenographic positions plus a few bright young craters (Tycho with rays, Copernicus, Kepler, Aristarchus). It is oriented by the ecliptic pole, so it tilts the way it does for the chosen location and time. The night side glows faintly with earthshine. Only the lit disc is compressed at night, the way the eye still sees the maria on a full moon that a camera would burn out.

Moonlight is physical, from the phase-dependent magnitude of the Moon (0.26 lux at full moon). A second sky-view LUT computes the moonlit sky, so a bright moon turns the night sky deep blue and washes out the faint stars (the eye adaptation follows it automatically). The Moon lays its own glitter path on the sea and lights the water, foam, clouds and birds.

### Post-processing

The scene renders in HDR with 4x MSAA. Post-processing adds an 8-level bloom, AgX tone mapping, a slight white balance shift, and dithering.

## Performance

At 2560 by 1440 on an RTX 5070 Ti, the GPU takes about 6 to 8 ms per frame, measured while other programs were also using the GPU. The HUD shows FPS and GPU time when the browser supports `timestamp-query`.

The per-frame code reuses preallocated typed arrays and descriptors, so the garbage collector has almost nothing to do. The only recurring allocations are the objects WebGPU itself requires, such as the swap-chain view and command buffers.

## Known limitations

- Far water is still somewhat darker than the bright horizon sky above it.
- The water is tuned toward turbid coastal water. The "Water turbidity" slider gives darker, bluer deep water when turned down.
- The scene is open ocean only. There is no shoreline, surf, or objects in the water.
- Clouds are two flat layers, not volumetric.
- No planets or Milky Way, and no lunar eclipses. The Moon ignores libration. Star positions are J2000 without precession (about 0.36 degrees off by now) and without refraction near the horizon.
- Seagull reflections are physically based, so they are barely visible on a rippled sea. Bird shadows only show clearly when the sun is high.

## How it was made

I built this with Claude Code (Opus 5.5). I asked for the most realistic water it could produce, in WebGPU, with a floating camera, and left the architecture and implementation to it. It checked its own work by rendering screenshots in headless Chrome and having separate critic subagents review them, so it was not grading its own output. I gave feedback along the way, mainly on the clouds, which led to replacing the volumetric clouds with the flat layers and later fixing their edge aliasing.

## Repository layout

| Path | Contents |
|---|---|
| `index.html` | Page shell, HUD, and settings panel styles |
| `js/main.js` | Device setup, resources, camera, per-frame loop |
| `js/birds.js` | Seagull mesh and flock simulation |
| `js/astro.js` | Locations, solar and lunar positions, sidereal time, star catalogue decoding |
| `js/stars.js` | Packed star catalogue (generated) |
| `js/wgsl/` | Shaders as JavaScript modules: `sim.js` (waves and foam), `sky.js` (atmosphere and clouds), `ocean.js` (surface and sky pass), `birds.js` (seagulls), `stars.js` (stars), `post.js` (bloom and tone mapping), `common.js` (shared structs) |
| `server.mjs` | Tiny static server for development |
| `build.mjs` | Builds `ocean.html` and `ocean.min.html` |
| `tools/make-stars.mjs` | Regenerates `js/stars.js` from the Yale Bright Star Catalogue |
| `docs/` | README images |

## License

MIT. See [LICENSE](LICENSE).
