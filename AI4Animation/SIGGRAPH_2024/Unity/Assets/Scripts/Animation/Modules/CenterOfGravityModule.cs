using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class CenterOfGravityModule : Module {
        [NonSerialized] public static bool Smooth = false;

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global);
            int[] bones = null;
            if(parameters.Length > 0) {
                bones = Asset.Source.GetBoneIndices((string[])parameters);
            }
            for(int i=0; i<instance.Samples.Length; i++) {
                instance.Positions[i] = GetCenterOfGravity(timestamp + instance.Samples[i].Timestamp, mirrored, Smooth ? global : null, bones);
                instance.Velocities[i] = GetVelocityOfGravity(timestamp + instance.Samples[i].Timestamp, mirrored, Smooth ? global : null, bones);
            }
			return instance;
		}

		private Matrix4x4 GetBoneTransformation(float timestamp, bool mirrored, int bone) {
			return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
		}

		private Vector3 GetBoneVelocity(float timestamp, bool mirrored, int bone) {
			return Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
		}

        public Vector3 GetCenterOfGravity(float timestamp, bool mirrored, TimeSeries smoothing, int[] bones) {
            float GetBoneLength(int bone) {
                if(Asset.Source.Bones[bone].HasParent()) {
                    return Vector3.Distance(
                        GetBoneTransformation(timestamp, mirrored, Asset.Source.Bones[bone].Parent).GetPosition(),
                        GetBoneTransformation(timestamp, mirrored, bone).GetPosition()
                    );
                } else {
                    return 0f;
                }
            }
            if(bones == null) {
                bones = ArrayExtensions.CreateEnumerated(Asset.Source.Bones.Length);
            }
            if(smoothing != null) {
                Vector3 point = Vector3.zero;
                float sum = 0f;
                for(int i=0; i<smoothing.KeyCount; i++) {
                    float t = timestamp + smoothing.GetKey(i).Timestamp;
                    for(int k=0; k<bones.Length; k++) {
                        float l = GetBoneLength(bones[k]);
                        Vector3 p = GetBoneTransformation(t, mirrored, bones[k]).GetPosition();
                        point += l * p;
                        sum += l;
                    }
                }
                return point/sum;
            } else {
                Vector3 point = Vector3.zero;
                float sum = 0f;
                for(int k=0; k<bones.Length; k++) {
                    float l = GetBoneLength(bones[k]);
                    Vector3 p = GetBoneTransformation(timestamp, mirrored, bones[k]).GetPosition();
                    point += l * p;
                    sum += l;
                }
                return point/sum;
            }
        }

        public Vector3 GetVelocityOfGravity(float timestamp, bool mirrored, TimeSeries smoothing, int[] bones) {
            return (GetCenterOfGravity(timestamp, mirrored, smoothing, bones) - GetCenterOfGravity(timestamp - Asset.GetDeltaTime(), mirrored, smoothing, bones)) / Asset.GetDeltaTime();
        }
#if UNITY_EDITOR
		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
            
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
		
		protected override void DerivedCallback(MotionEditor editor) {

		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
            // ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror, VRModule.LowerBodyNames).Draw();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            Smooth = EditorGUILayout.Toggle("Smooth", Smooth);
		}
#endif

		public class Series : TimeSeries.Component {
            public Vector3[] Positions = new Vector3[0];
            public Vector3[] Velocities = new Vector3[0];

			public Series(TimeSeries global) : base(global) {
                Positions = new Vector3[global.SampleCount];
                Velocities = new Vector3[global.SampleCount];
			}

			public override void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
					Positions[i] = Positions[i+1];
                    Velocities[i] = Velocities[i+1];
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {

				}
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawScene) {
                    Draw(Color.white, 0, KeyCount);
				}
			}

			public void Draw(Color color, int start, int end, float size=0.025f) {
				if(DrawScene) {
                    UltiDraw.Begin();
                    for(int i=start; i<end; i++) {
                        UltiDraw.DrawSphere(Positions[GetKey(i).Index], Quaternion.identity, size, color);
                        UltiDraw.DrawLine(Positions[GetKey(i).Index], Positions[GetKey(i).Index] + Velocities[GetKey(i).Index], 0.5f*size, 0f, color.Opacity(0.25f*color.a));
                    }
                    UltiDraw.End();
				}
			}

            public void Blend(Vector3 source) {
                Vector3 delta = source - Positions[Pivot];
                for(int i=Pivot; i<Samples.Length; i++) {
                    float weight = 1f - i.Ratio(Pivot, Samples.Length-1);
                    Positions[i] = Positions[i] + weight * delta;
                }
            }

            public void Blend(Vector3 source, Vector3 target) {
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					Positions[i] = Utility.Interpolate(source, target, weight);
				}
            }
        }

	}
}