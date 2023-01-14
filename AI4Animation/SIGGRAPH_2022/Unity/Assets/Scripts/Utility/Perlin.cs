//
// Perlin noise generator for Unity
// Keijiro Takahashi, 2013, 2015
// https://github.com/keijiro/PerlinNoise
//
// Based on the original implementation by Ken Perlin
// http://mrl.nyu.edu/~perlin/noise/
//
using UnityEngine;

public static class Perlin {
    public static float Noise(float x)
    {
        var X = Mathf.FloorToInt(x) & 0xff;
        x -= Mathf.Floor(x);
        var u = Fade(x);
        return Lerp(u, Grad(perm[X], x), Grad(perm[X+1], x-1)) * 2;
    }

    static float Fade(float t)
    {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    static float Lerp(float t, float a, float b)
    {
        return a + t * (b - a);
    }

    static float Grad(int hash, float x)
    {
        return (hash & 1) == 0 ? x : -x;
    }

    static int[] perm = {
        151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        151
    };

    public static float[] Sample(float[] coordinates, int seed, int octaves, float amplitude, float frequency, float offset) {
		float[] noise = new float[coordinates.Length];

		System.Random prng = new System.Random (seed);
		float[] offsets = new float[octaves];
		for(int i=0; i<octaves; i++) {
            offsets[i] = prng.Next (-100000, 100000) + offset;
		}

		for(int x=0; x<coordinates.Length; x++) {
            float freq = 1f;
            float value = 0f;
            for(int o=0; o<octaves; o++) {
                float sample = (coordinates[x]+offsets[o]) * freq;
                float perlin = Mathf.PerlinNoise(sample, 0f).Normalize(0f, 1f, -amplitude, amplitude);
                value += perlin;
                freq *= frequency;
            }
            noise[x] = value;
		}

		return noise;
	}
}























// //
// // Perlin noise generator for Unity
// // Keijiro Takahashi, 2013, 2015
// // https://github.com/keijiro/PerlinNoise
// //
// // Based on the original implementation by Ken Perlin
// // http://mrl.nyu.edu/~perlin/noise/
// //
// using UnityEngine;

// public static class Perlin
// {
//     #region Noise functions

//     public static float Noise(float x)
//     {
//         var X = Mathf.FloorToInt(x) & 0xff;
//         x -= Mathf.Floor(x);
//         var u = Fade(x);
//         return Lerp(u, Grad(perm[X], x), Grad(perm[X+1], x-1)) * 2;
//     }

//     public static float Noise(float x, float y)
//     {
//         var X = Mathf.FloorToInt(x) & 0xff;
//         var Y = Mathf.FloorToInt(y) & 0xff;
//         x -= Mathf.Floor(x);
//         y -= Mathf.Floor(y);
//         var u = Fade(x);
//         var v = Fade(y);
//         var A = (perm[X  ] + Y) & 0xff;
//         var B = (perm[X+1] + Y) & 0xff;
//         return Lerp(v, Lerp(u, Grad(perm[A  ], x, y  ), Grad(perm[B  ], x-1, y  )),
//                        Lerp(u, Grad(perm[A+1], x, y-1), Grad(perm[B+1], x-1, y-1)));
//     }

//     public static float Noise(Vector2 coord)
//     {
//         return Noise(coord.x, coord.y);
//     }

//     public static float Noise(float x, float y, float z)
//     {
//         var X = Mathf.FloorToInt(x) & 0xff;
//         var Y = Mathf.FloorToInt(y) & 0xff;
//         var Z = Mathf.FloorToInt(z) & 0xff;
//         x -= Mathf.Floor(x);
//         y -= Mathf.Floor(y);
//         z -= Mathf.Floor(z);
//         var u = Fade(x);
//         var v = Fade(y);
//         var w = Fade(z);
//         var A  = (perm[X  ] + Y) & 0xff;
//         var B  = (perm[X+1] + Y) & 0xff;
//         var AA = (perm[A  ] + Z) & 0xff;
//         var BA = (perm[B  ] + Z) & 0xff;
//         var AB = (perm[A+1] + Z) & 0xff;
//         var BB = (perm[B+1] + Z) & 0xff;
//         return Lerp(w, Lerp(v, Lerp(u, Grad(perm[AA  ], x, y  , z  ), Grad(perm[BA  ], x-1, y  , z  )),
//                                Lerp(u, Grad(perm[AB  ], x, y-1, z  ), Grad(perm[BB  ], x-1, y-1, z  ))),
//                        Lerp(v, Lerp(u, Grad(perm[AA+1], x, y  , z-1), Grad(perm[BA+1], x-1, y  , z-1)),
//                                Lerp(u, Grad(perm[AB+1], x, y-1, z-1), Grad(perm[BB+1], x-1, y-1, z-1))));
//     }

//     public static float Noise(Vector3 coord)
//     {
//         return Noise(coord.x, coord.y, coord.z);
//     }

//     #endregion

//     #region fBm functions

//     public static float Fbm(float x, int octave)
//     {
//         var f = 0.0f;
//         var w = 0.5f;
//         for (var i = 0; i < octave; i++) {
//             f += w * Noise(x);
//             x *= 2.0f;
//             w *= 0.5f;
//         }
//         return f;
//     }

//     public static float Fbm(Vector2 coord, int octave)
//     {
//         var f = 0.0f;
//         var w = 0.5f;
//         for (var i = 0; i < octave; i++) {
//             f += w * Noise(coord);
//             coord *= 2.0f;
//             w *= 0.5f;
//         }
//         return f;
//     }

//     public static float Fbm(float x, float y, int octave)
//     {
//         return Fbm(new Vector2(x, y), octave);
//     }

//     public static float Fbm(Vector3 coord, int octave)
//     {
//         var f = 0.0f;
//         var w = 0.5f;
//         for (var i = 0; i < octave; i++) {
//             f += w * Noise(coord);
//             coord *= 2.0f;
//             w *= 0.5f;
//         }
//         return f;
//     }

//     public static float Fbm(float x, float y, float z, int octave)
//     {
//         return Fbm(new Vector3(x, y, z), octave);
//     }

//     #endregion

//     #region Private functions

//     static float Fade(float t)
//     {
//         return t * t * t * (t * (t * 6 - 15) + 10);
//     }

//     static float Lerp(float t, float a, float b)
//     {
//         return a + t * (b - a);
//     }

//     static float Grad(int hash, float x)
//     {
//         return (hash & 1) == 0 ? x : -x;
//     }

//     static float Grad(int hash, float x, float y)
//     {
//         return ((hash & 1) == 0 ? x : -x) + ((hash & 2) == 0 ? y : -y);
//     }

//     static float Grad(int hash, float x, float y, float z)
//     {
//         var h = hash & 15;
//         var u = h < 8 ? x : y;
//         var v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
//         return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
//     }

//     static int[] perm = {
//         151,160,137,91,90,15,
//         131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
//         190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
//         88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
//         77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
//         102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
//         135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
//         5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
//         223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
//         129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
//         251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
//         49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
//         138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
//         151
//     };

//     public static float[] Sample(float[] coordinates, int seed, int octaves, float amplitude, float frequency, float offset) {
// 		float[] noise = new float[coordinates.Length];

// 		System.Random prng = new System.Random (seed);
// 		float[] offsets = new float[octaves];
// 		for(int i=0; i<octaves; i++) {
//             offsets[i] = prng.Next (-100000, 100000) + offset;
// 		}

// 		for(int x=0; x<coordinates.Length; x++) {
//             float freq = 1f;
//             float value = 0f;
//             for(int o=0; o<octaves; o++) {
//                 float sample = (coordinates[x]+offsets[o]) * freq;
//                 float perlin = Mathf.PerlinNoise(sample, 0f).Normalize(0f, 1f, -amplitude, amplitude);
//                 value += perlin;
//                 freq *= frequency;
//             }
//             noise[x] = value;
// 		}

// 		return noise;
// 	}

//     #endregion
// }