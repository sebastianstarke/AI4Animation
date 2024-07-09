using System;
using UnityEngine;
using UnityEditor;

using SIGGRAPH_2024;

namespace AI4Animation {
	public class AvatarModule : Module {

		public int Head = 0;
		public Vector3 Offset = Vector3.zero;
		public Axis View = Axis.ZPositive;
		public float FOV = 100f;

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global);

			// for(int i=0; i<instance.SampleCount; i++) {
			// 	float t = timestamp + instance.Samples[i].Timestamp;
			// 	instance.HipsTransformations[i] = GetBoneTransformation(t, mirrored, Blueman.HipsIndex, global);
			// 	instance.HipsVelocities[i] = GetBoneVelocity(t, mirrored, Blueman.HipsIndex, global);
			// 	instance.Orientations[i] = GetUpperBodyOrientation(t, mirrored);
			// 	instance.Centers[i] = GetUpperBodyCenter(t, mirrored, global);
			// }

			return instance;
		}

		public Matrix4x4 GetBoneTransformation(float timestamp, bool mirrored, int bone, TimeSeries smoothing=null) {
            if(smoothing == null) {
			    return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
            }
            Matrix4x4 pivot = Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
            Matrix4x4[] values = new Matrix4x4[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
                values[i] = Asset.GetFrame(timestamp + smoothing.GetKey(i).Timestamp).GetBoneTransformation(bone, mirrored).TransformationTo(pivot);
            }
            return values.Gaussian().TransformationFrom(pivot);
		}

		public Vector3 GetBoneVelocity(float timestamp, bool mirrored, int bone, TimeSeries smoothing=null) {
            if(smoothing == null) {
			    return Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
            }
            Vector3 pivot = Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
            Vector3[] values = new Vector3[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
                values[i] = Asset.GetFrame(timestamp + smoothing.GetKey(i).Timestamp).GetBoneVelocity(bone, mirrored) - pivot;
            }
            return pivot + values.Gaussian();
		}

		public Vector3 GetUpperBodyOrientation(float timestamp, bool mirrored) {
			Vector3 value = Vector3.zero;
			Vector3 origin = GetBoneTransformation(timestamp, mirrored, Blueman.HipsIndex).GetPosition();
			for(int i=1; i<Blueman.UpperBodyIndices.Length; i++) {
				Vector3 target = GetBoneTransformation(timestamp, mirrored, Blueman.UpperBodyIndices[i]).GetPosition();
				value += target - origin;
			}
			return value.normalized;
		}

		public Vector3 GetUpperBodyCenter(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			Vector3 GetCenter(float t, bool m) {
				Vector3 value = Vector3.zero;
				for(int i=0; i<Blueman.UpperBodyIndices.Length; i++) {
					value += GetBoneTransformation(t, m, Blueman.UpperBodyIndices[i]).GetPosition();
				}
				return value / Blueman.UpperBodyIndices.Length;
			}
			if(smoothing == null) {
				return GetCenter(timestamp, mirrored);
			}
            Vector3 pivot = GetCenter(timestamp, mirrored);
            Vector3[] values = new Vector3[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
				values[i] = GetCenter(timestamp + smoothing.GetKey(i).Timestamp, mirrored) - pivot;
            }
            return pivot + values.Gaussian();
		}

#if UNITY_EDITOR
		protected override void DerivedInitialize() {

		}
		
		protected override void DerivedLoad(MotionEditor editor) {
            Blueman.RegisterIndices(Asset);
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
		
		protected override void DerivedCallback(MotionEditor editor) {

		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
			UltiDraw.Begin();
			UltiDraw.DrawArrow(
				GetCameraPosition(editor.GetTimestamp(), editor.Mirror), 
				GetCameraPosition(editor.GetTimestamp(), editor.Mirror) + GetViewDirection(editor.GetTimestamp(), editor.Mirror),
				0.8f,
				0.025f,
				0.05f, 
				UltiDraw.Cyan
			);

			foreach(string bone in Blueman.FullBodyNames) {
				UltiDraw.DrawLine(
					GetCameraPosition(editor.GetTimestamp(), editor.Mirror),
					GetBonePosition(bone, editor.GetTimestamp(), editor.Mirror),
					GetVisibility(bone, editor.GetTimestamp(), editor.Mirror) == 1f ? UltiDraw.Green : UltiDraw.Red
				);
			}

			UltiDraw.PlotBars(new Vector2(0.5f, 0.9f), new Vector2(0.5f, 0.1f), GetVisibilities(Blueman.FullBodyNames, editor.GetTimestamp(), editor.Mirror), 0f, 1f);
			UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Head = EditorGUILayout.Popup("Head", Head, Asset.Source.GetBoneNames());
			Offset = EditorGUILayout.Vector3Field("Offset", Offset);
			View = (Axis)EditorGUILayout.EnumPopup("View", View);
			FOV = EditorGUILayout.FloatField("FOV", FOV);
		}
#endif

		public Vector3 GetCameraPosition(float timestamp, bool mirrored) {
			return Offset.PositionFrom(Asset.GetFrame(timestamp).GetBoneTransformation(Head, mirrored));
		}

		public Vector3 GetViewDirection(float timestamp, bool mirrored) {
			return View.GetAxis().DirectionFrom(Asset.GetFrame(timestamp).GetBoneTransformation(Head, mirrored));
		}

		public Vector3 GetBonePosition(string bone, float timestamp, bool mirrored) {
			return Asset.GetFrame(timestamp).GetBonePosition(bone, mirrored);
		}

		public Vector3 GetBonePosition(int bone, float timestamp, bool mirrored) {
			return Asset.GetFrame(timestamp).GetBonePosition(bone, mirrored);
		}

		public float GetVisibility(string bone, float timestamp, bool mirrored) {
			Vector3 from = GetViewDirection(timestamp, mirrored);
			Vector3 to = GetBonePosition(bone, timestamp, mirrored) - GetCameraPosition(timestamp, mirrored);
			return Vector3.Angle(from, to) <= FOV ? 1f : 0f;
		}

		public float GetVisibility(int bone, float timestamp, bool mirrored) {
			Vector3 from = GetViewDirection(timestamp, mirrored);
			Vector3 to = GetBonePosition(bone, timestamp, mirrored) - GetCameraPosition(timestamp, mirrored);
			return Vector3.Angle(from, to) <= FOV ? 1f : 0f;
		}

		public float[] GetVisibilities(string[] bones, float timestamp, bool mirrored) {
			float[] values = new float[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				values[i] = GetVisibility(bones[i], timestamp, mirrored);
			}
			return values;
		}

		public float[] GetVisibilities(int[] bones, float timestamp, bool mirrored) {
			float[] values = new float[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				values[i] = GetVisibility(bones[i], timestamp, mirrored);
			}
			return values;
		}

		public class Series : TimeSeries.Component {

            public Matrix4x4[] HipsTransformations;
            public Vector3[] HipsVelocities;
			public Vector3[] Orientations;
			public Vector3[] Centers;

			public Series(TimeSeries global, Actor actor=null) : base(global) {
                HipsTransformations = new Matrix4x4[SampleCount];
                HipsVelocities = new Vector3[SampleCount];
				Orientations = new Vector3[SampleCount];
				Centers = new Vector3[SampleCount];
				if(actor != null) {
					HipsTransformations.SetAll(actor.GetBoneTransformation(Blueman.HipsName));
					HipsVelocities.SetAll(Vector3.zero);
					Orientations.SetAll(Vector3.up);
					Centers.SetAll(actor.GetBonePosition(Blueman.HipsName));
				}
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					HipsTransformations[i] = HipsTransformations[i+1];
					HipsVelocities[i] = HipsVelocities[i+1];
					Orientations[i] = Orientations[i+1];
					Centers[i] = Centers[i+1];
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {

				}
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawScene) {
					Draw(0, KeyCount);
				}
			}

			public void Draw(int start, int end) {
				UltiDraw.Begin();
				for(int i=start+1; i<end; i++) {
					int previous = GetKey(i-1).Index;
					int next = GetKey(i).Index;
					UltiDraw.DrawLine(HipsTransformations[previous].GetPosition(), HipsTransformations[next].GetPosition(), UltiDraw.Black);
				}
				for(int i=start; i<end; i++) {
					int index = GetKey(i).Index;
					UltiDraw.DrawSphere(HipsTransformations[index].GetPosition(), Quaternion.identity, 0.025f, UltiDraw.Red);
					UltiDraw.DrawLine(HipsTransformations[index].GetPosition(), HipsTransformations[index].GetPosition() + GetTemporalScale(HipsVelocities[index]), 0.0125f, 0f, UltiDraw.Green);
					UltiDraw.DrawArrow(HipsTransformations[index].GetPosition(), HipsTransformations[index].GetPosition() + 0.125f*Orientations[index], 0.8f, 0.0125f, 0.025f, UltiDraw.Orange);
					UltiDraw.DrawSphere(Centers[index], Quaternion.identity, 0.025f, UltiDraw.Magenta);
				}
				UltiDraw.End();
			}
        }

	}
}