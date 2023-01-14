#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace AI4Animation {
	public class AudioSpectrumModule : Module {
        
        public AudioSpectrum AudioSpectrum = null;
        public AudioSpectrum[] AudioSpectrums = null;
        public bool AdaptiveLowpass = true;
		
        public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            if(AudioSpectrum == null) {
                return null;
            }
			Series instance = new Series(global);
			for(int i=0; i<instance.Samples.Length; i++) {
                float t = timestamp + instance.Samples[i].Timestamp;
                if(AdaptiveLowpass) {
                    instance.Values[i] = AudioSpectrum.GetFiltered(t, global.MaximumFrequency);
                } else {
                    instance.Values[i] = AudioSpectrum.GetSample(t);
                }
			}
			return instance;
        }

		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
            if(AudioSpectrums == null) {
                AudioSpectrums = new AudioSpectrum[0];
            }
		}

		protected override void DerivedUnload(MotionEditor editor) {
            if(AudioSpectrum != null) {
                AudioSpectrum.StopMusic();
            }
            if(AudioSpectrums != null) {
                foreach(AudioSpectrum spectrum in AudioSpectrums) {
                    if(spectrum != null) {
                        spectrum.StopMusic();
                    }
                }
            }
		}

        public override void OnTriggerPlay(MotionEditor editor) {
            if(!editor.IsPlaying()) {
                if(AudioSpectrum != null) {
                    AudioSpectrum.StopMusic();
                }
                if(AudioSpectrums != null) {
                    foreach(AudioSpectrum spectrum in AudioSpectrums) {
                        if(spectrum != null) {
                            spectrum.StopMusic();
                        }
                    }
                }
            }
        }
        
        public AudioSpectrum GetAudioSpectrum(float timestamp) {
            if(AudioSpectrums != null && AudioSpectrums.Length == 0) {
                return AudioSpectrum;
            }
            int index = Asset.GetFrame(timestamp).Index;
            for(int i=0; i<Asset.Sequences.Length; i++) {
                if(Asset.Sequences[i].Contains(index)) {
                    return AudioSpectrums[i];
                }
            }
            return null;
        }

        public float GetAudioSpectrumTimestamp(float timestamp) {
            if(AudioSpectrums != null && AudioSpectrums.Length == 0) {
                return timestamp;
            }
            int index = Asset.GetFrame(timestamp).Index;
            for(int i=0; i<Asset.Sequences.Length; i++) {
                if(Asset.Sequences[i].Contains(index)) {
                    int frames = index - Asset.Sequences[i].Start;
                    return frames / (float)AudioSpectrums[i].Framerate;
                }
            }
            return timestamp;
        }

		protected override void DerivedCallback(MotionEditor editor) {
            AudioSpectrum active = GetAudioSpectrum(editor.GetTimestamp());
            if(editor.IsPlaying()) {
                bool force = Mathf.Abs(editor.GetTimestamp() - active.GetTimestamp()) > 0.1f;
                active.PlayMusic(GetAudioSpectrumTimestamp(editor.GetTimestamp()), force);
            }
            foreach(AudioSpectrum spectrum in AudioSpectrums) {
                if(spectrum != active) {
                    spectrum.StopMusic();
                }
            }
            
            // if(Asset.name.ContainsAny("kth", "@")) {
            //     return;
            // }
            // void CorrectToe(string hip, string ankle, string toe) {
            //     float range = 0.05f;
            //     Transform hipTransform = editor.GetSession().GetActor().FindBone(hip).GetTransform();
            //     Transform ankleTransform = editor.GetSession().GetActor().FindBone(ankle).GetTransform();
            //     Transform toeTransform = editor.GetSession().GetActor().FindBone(toe).GetTransform();
            //     Vector3 footUp = toeTransform.up;
            //     Vector3 groundUp = Utility.GetNormal(toeTransform.position, LayerMask.GetMask("Ground"));
            //     float height = Utility.GetHeight(toeTransform.position, LayerMask.GetMask("Ground"));
            //     float ratio = toeTransform.position.y.Ratio(height, height+range);
            //     toeTransform.rotation = Quaternion.Slerp(toeTransform.rotation, Quaternion.FromToRotation(footUp, groundUp) * toeTransform.rotation, 1f-ratio);

            //     UltimateIK.IK ik = UltimateIK.IK.Create(hipTransform, ankleTransform, toeTransform);
            //     ik.Objectives[0].SolveRotation = false;
            //     ik.Objectives[0].SetTarget(
            //         new Vector3(ankleTransform.position.x, Mathf.Max(ankleTransform.position.y, height + (ankleTransform.position.y - toeTransform.position.y)), ankleTransform.position.z)
            //     );
            //     ik.Objectives[1].SetTarget(
            //         new Vector3(toeTransform.position.x, Mathf.Max(toeTransform.position.y, height), toeTransform.position.z),
            //         toeTransform.rotation
            //     );
            //     ik.Solve();
            // }
            // CorrectToe("m_avg_L_Hip", "m_avg_L_Ankle", "m_avg_L_Foot");
            // CorrectToe("m_avg_R_Hip", "m_avg_R_Ankle", "m_avg_R_Foot");
		}

		protected override void DerivedGUI(MotionEditor editor) {
            
		}

		protected override void DerivedDraw(MotionEditor editor) {
            if(AudioSpectrum == null) {
                return;
            }
            ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror).Draw();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            AudioSpectrum = EditorGUILayout.ObjectField("Audio Spectrum", AudioSpectrum, typeof(AudioSpectrum), true) as AudioSpectrum;
            for(int i=0; i<AudioSpectrums.Length; i++) {
                AudioSpectrums[i] = EditorGUILayout.ObjectField("Audio Spectrum " + (i+1).ToString(), AudioSpectrums[i], typeof(AudioSpectrum), true) as AudioSpectrum;
            }
            if(AudioSpectrum != null) {
                EditorGUILayout.HelpBox("Clip Length: " + AudioSpectrum.Clip.length, MessageType.None);
            }
            AdaptiveLowpass = EditorGUILayout.Toggle("Adaptive Lowpass", AdaptiveLowpass);
		}

		public class Series : TimeSeries.Component {
			
            public static float[] AS, AB, AF, AMFCC, AC, AZC;

            public AudioSpectrum.Sample[] Values;

			public Series(TimeSeries global) : base(global) {
                Values = new AudioSpectrum.Sample[Samples.Length];
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					Values[i] = Values[i+1];
				}
			}

			public override void Interpolate(int start, int end) {

			}

			public override void GUI() {
				if(DrawGUI) {

				}
			}

			public override void Draw() {
				if(DrawGUI) {
					UltiDraw.Begin();

                    {
                        AS = AS.Validate(Values[0].Spectogram.Length);
                        float[][] spectrums = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.Spectogram;
                            spectrums[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AS[j] = Mathf.Max(AS[j], v[j]);
                                spectrums[i][j] = v[j] / AS[j];
                            }
                        }
                        UltiDraw.PlotBars(new Vector2(0.15f, 0.75f), new Vector2(0.25f, 0.4f), spectrums, 0f, 1f);
                    }

                    {
                        AB = AB.Validate(Values[0].Beats.Length);
                        float[][] beats = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.Beats;
                            beats[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AB[j] = Mathf.Max(AB[j], v[j]);
                                beats[i][j] = v[j] / AB[j];
                            }
                        }
                        UltiDraw.PlotFunctions(new Vector2(0.15f, 0.4f), new Vector2(0.25f, 0.2f), beats, UltiDraw.Dimension.Y, 0f, 1f);
                    }

                    {
                        AF = AF.Validate(Values[0].Flux.Length);
                        float[][] flux = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.Flux;
                            flux[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AF[j] = Mathf.Max(AF[j], v[j]);
                                flux[i][j] = v[j] / AF[j];
                            }
                        }
                        UltiDraw.PlotFunction(new Vector2(0.15f, 0.15f), new Vector2(0.25f, 0.2f), flux.Flatten(), 0f, 1f);
                    }

                    {
                        AMFCC = AMFCC.Validate(Values[0].MFCC.Length);
                        float[][] mfcc = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.MFCC;
                            mfcc[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AMFCC[j] = Mathf.Max(AMFCC[j], v[j]);
                                mfcc[i][j] = v[j] / AMFCC[j];
                            }
                        }
                        UltiDraw.PlotBars(new Vector2(0.5f, 0.75f), new Vector2(0.25f, 0.4f), mfcc, 0f, 1f);
                    }

                    {
                        AC = AC.Validate(Values[0].Chroma.Length);
                        float[][] chroma = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.Chroma;
                            chroma[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AC[j] = Mathf.Max(AC[j], v[j]);
                                chroma[i][j] = v[j] / AC[j];
                            }
                        }
                        UltiDraw.PlotBars(new Vector2(0.5f, 0.25f), new Vector2(0.25f, 0.4f), chroma, 0f, 1f);
                    }

                    {
                        AZC = AZC.Validate(Values[0].ZeroCrossing.Length);
                        float[][] zc = new float[KeyCount][];
                        for(int i=0; i<KeyCount; i++) {
                            AudioSpectrum.Sample sample = Values[GetKey(i).Index];
                            float[] v = sample.ZeroCrossing;
                            zc[i] = new float[v.Length];
                            for(int j=0; j<v.Length; j++) {
                                AZC[j] = Mathf.Max(AZC[j], v[j]);
                                zc[i][j] = v[j] / AZC[j];
                            }
                        }
                        UltiDraw.PlotFunction(new Vector2(0.85f, 0.5f), new Vector2(0.25f, 0.4f), zc.Flatten(), 0f, 1f);
                    }

                    //Debug.Log(AS.Length); //80
                    //Debug.Log(AB.Length); //2
                    //Debug.Log(AF.Length); //1
                    //Debug.Log(AMFCC.Length); //20
                    //Debug.Log(AC.Length); //12
                    //Debug.Log(AZC.Length); //1
                    //80+2+1+20+12+1 = 116
                    //116*13=1503

					UltiDraw.End();
				}
			}
		}

	}
}
#endif
