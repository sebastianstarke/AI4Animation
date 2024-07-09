using System;
using UnityEngine;

namespace AI4Animation {
    public class AudioSpectrum : ScriptableObject {
        public AudioClip Clip;
        public int Framerate;
        public Sample[] Samples;

        private float LowPassHz = 0f;
        private Sample[] Filtered;
        
        private AudioSource AudioSource;

        public float GetLength() {
			return (Samples.Length-1) / (float)Framerate;
        }

        public bool IsPlaying() {
            return AudioSource == null ? false : AudioSource.isPlaying;
        }

        public float GetTimestamp() {
            return AudioSource == null ? 0f : AudioSource.time;
        }

        public void PlayMusic(float timestamp, bool reset) {
            if(AudioSource == null) {
                string id = "Audio Source " + name;
                GameObject instance = GameObject.Find(id);
                if(instance == null) {
                    instance = new GameObject(id);
                    AudioSource = instance.gameObject.AddComponent<AudioSource>();
                } else {
                    AudioSource = instance.gameObject.GetComponent<AudioSource>();
                }
            }
            if(!AudioSource.isPlaying || reset) {
                AudioSource.clip = Clip;
                AudioSource.time = timestamp;
                AudioSource.loop = false;
                AudioSource.Play();
            }
        }

        public void StopMusic() {
            if(AudioSource != null) {
                Utility.Destroy(AudioSource.gameObject);
            }
        }

        public void ApplyPitch(float pitch) {
            if(AudioSource != null) {
                AudioSource.pitch = pitch;
            }
        }

        public Sample GetSample(float timestamp) {
            return Samples[Mathf.Clamp(Mathf.RoundToInt(timestamp * Framerate), 0, Samples.Length-1)];
        }

        public Sample GetFiltered(float timestamp, float lowPassHz=0f) {
            if(LowPassHz == lowPassHz) {
                return Filtered[Mathf.Clamp(Mathf.RoundToInt(timestamp * Framerate), 0, Filtered.Length-1)];
            }
            // Debug.Log("Recomputing filtered music data for " + lowPassHz + "Hz.");
            LowPassHz = lowPassHz;
            float[][] Filter(int var) {
                float[][] values = new float[Samples.Length][];
                for(int i=0; i<Samples.Length; i++) {
                    if(var == 1) {
                        values[i] = Samples[i].Spectogram;
                    }
                    if(var == 2) {
                        values[i] = Samples[i].Beats;
                    }
                    if(var == 3) {
                        values[i] = Samples[i].Flux;
                    }
                    if(var == 4) {
                        values[i] = Samples[i].MFCC;
                    }
                    if(var == 5) {
                        values[i] = Samples[i].Chroma;
                    }
                    if(var == 6) {
                        values[i] = Samples[i].ZeroCrossing;
                    }
                }
                values = values.GetTranspose();
                for(int i=0; i<values.Length; i++) {
                    values[i] = Utility.Butterworth(values[i], 1f/Framerate, LowPassHz);
                }
                values = values.GetTranspose();
                return values;
            }
            float[][] spectogram = Filter(1);
            float[][] beats = Filter(2);
            float[][] flux = Filter(3);
            float[][] mfcc = Filter(4);
            float[][] chroma = Filter(5);
            float[][] zeroCrossing = Filter(6);
            Filtered = new Sample[Samples.Length];
            for(int i=0; i<Samples.Length; i++) {
                Filtered[i] = new Sample(spectogram[i], beats[i], flux[i], mfcc[i], chroma[i], zeroCrossing[i]);
            }
            return GetFiltered(timestamp, lowPassHz);
        }

        [Serializable]
        public class Sample {
            public float[] Spectogram;
            public float[] Beats;
            public float[] Flux;
            public float[] MFCC;
            public float[] Chroma;
            public float[] ZeroCrossing;
            public Sample(float[] spectogram, float[] beats, float[] flux, float[] mfcc, float[] chroma, float[] zeroCrossing) {
                Spectogram = spectogram;
                Beats = beats;
                Flux = flux;
                MFCC = mfcc;
                Chroma = chroma;
                ZeroCrossing = zeroCrossing;
            }
        }

        // public static float[] GetLevels(AudioSource source, BAND band, int sampleRate) {
        //     int FreqToSpecIndex (float[] raw, float f) {
        //         int i = Mathf.FloorToInt (f / AudioSettings.outputSampleRate * 2.0f * raw.Length);
        //         return Mathf.Clamp (i, 0, raw.Length - 1);
        //     }
        //     int length = MiddleFrequencies[(int)band].Length;
        //     float[] _band_ = MiddleFrequencies[(int)band];
        //     float _bandwidth_ = Bandwidths[(int)band];
        //     float[] levels = new float[length];
        //     float[] raw = new float[length];
        //     source.GetSpectrumData(raw, sampleRate, FFTWindow.BlackmanHarris);
        //     for (int bi=0; bi<length; bi++) {
        //         int imin = FreqToSpecIndex(raw, _band_[bi] / _bandwidth_);
        //         int imax = FreqToSpecIndex(raw, _band_[bi] * _bandwidth_);
        //         float bandMax = 0.0f;
        //         for (int fi=imin; fi<=imax; fi++) {
        //             bandMax = Mathf.Max(bandMax, raw[fi]);
        //         }
        //         levels[bi] = bandMax;
        //     }
        //     return levels;
        // }
        // public enum BAND {
        //     FourBand,
        //     FourBandVisual,
        //     EightBand,
        //     TenBand,
        //     TwentySixBand,
        //     ThirtyOneBand
        // };
        // public static int[] SampleResolution = {
        //     256, 512, 1024, 2048, 4096
        // };
        // public static float[][] MiddleFrequencies = {
        //     new float[]{ 125.0f, 500, 1000, 2000 },
        //     new float[]{ 250.0f, 400, 600, 800 },
        //     new float[]{ 63.0f, 125, 500, 1000, 2000, 4000, 6000, 8000 },
        //     new float[]{ 31.5f, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 },
        //     new float[]{ 25.0f, 31.5f, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000 },
        //     new float[]{ 20.0f, 25, 31.5f, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 },
        // };
        // public static float[] Bandwidths = {
        //     1.414f, // 2^(1/2)
        //     1.260f, // 2^(1/3)
        //     1.414f, // 2^(1/2)
        //     1.414f, // 2^(1/2)
        //     1.122f, // 2^(1/6)
        //     1.122f  // 2^(1/6)
        // };

    }
}