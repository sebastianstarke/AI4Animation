using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif

namespace AI4Animation {
	public class RootModule : Module {
		public enum TOPOLOGY {Biped, Quadruped, Bone};
		public bool Primary = false;

		public TOPOLOGY Topology;
		public LayerMask Ground = 0;
		public bool SmoothPositions;
		public bool SmoothRotations;
		public float SmoothingWindow;
		public float LockThreshold = 0.1f;

		public int RootBone;
		public int Hips, Neck, LeftShoulder, RightShoulder, LeftHip, RightHip, Head;

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global);
			for(int i=0; i<instance.Samples.Length; i++) {
                float t = timestamp + instance.Samples[i].Timestamp;
				instance.Transformations[i] = GetRootTransformation(t, mirrored);
				instance.Velocities[i] = GetRootVelocity(t, mirrored, parameters.Length > 0 ? (TimeSeries)parameters[0] : null);
				instance.AngularVelocities[i] = GetAngularVelocity(t, mirrored, parameters.Length > 0 ? (TimeSeries)parameters[0] : null);
				instance.Locks[i] = GetRootLock(t, mirrored, parameters.Length > 0 ? (TimeSeries)parameters[0] : null);
			}
			return instance;
		}

#if UNITY_EDITOR
		protected override void DerivedInitialize() {
			// MotionAsset.Hierarchy.Bone h = Asset.Source.FindBoneContains("Hips");
			// Hips = h == null ? 0 : h.Index;
			// MotionAsset.Hierarchy.Bone n = Asset.Source.FindBoneContains("Neck");
			// Neck = n == null ? 0 : n.Index;
			// MotionAsset.Hierarchy.Bone ls = Asset.Source.FindBoneContains("LeftShoulder");
			// LeftShoulder = ls == null ? 0 : ls.Index;
			// MotionAsset.Hierarchy.Bone rs = Asset.Source.FindBoneContains("RightShoulder");
			// RightShoulder = rs == null ? 0 : rs.Index;
			// MotionAsset.Hierarchy.Bone lh = Asset.Source.FindBoneContainsAny("LeftHip", "LeftUpLeg");
			// LeftHip = lh == null ? 0 : lh.Index;
			// MotionAsset.Hierarchy.Bone rh = Asset.Source.FindBoneContainsAny("RightHip", "RightUpLeg");
			// RightHip = rh == null ? 0 : rh.Index;
			// Ground = 0;
		}

		protected override void DerivedLoad(MotionEditor editor) {
			
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
	
		protected override void DerivedCallback(MotionEditor editor) {
			if(Primary) {
				Actor.Bone[] bones = editor.GetSession().GetActor().GetRootBones();
				Matrix4x4[] transformations = new Matrix4x4[bones.Length];
				for(int i=0; i<bones.Length; i++) {
					transformations[i] = bones[i].GetTransformation();
				}
				editor.GetSession().GetActor().GetRoot().position = GetRootPosition(editor.GetTimestamp(), editor.Mirror);
				editor.GetSession().GetActor().GetRoot().rotation = GetRootRotation(editor.GetTimestamp(), editor.Mirror);
				for(int i=0; i<bones.Length; i++) {
					bones[i].SetTransformation(transformations[i]);
				}
			}
		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
			float[] values = new float[editor.GetTimeSeries().KeyCount];
			UltiDraw.Begin();
			for(int i=0; i<editor.GetTimeSeries().KeyCount; i++) {
				float t = editor.GetTimestamp() + editor.GetTimeSeries().GetKey(i).Timestamp;
				UltiDraw.DrawLine(GetRootPosition(t, editor.Mirror), GetRootPosition(t, editor.Mirror) + GetRootRotation(t, editor.Mirror).GetForward(), 0.1f, 0f, UltiDraw.Cyan);
				float angle = GetRootAngle(t, editor.Mirror, editor.GetTimestamp(), 10);
				values[i] = angle;
				Vector3 forward = Quaternion.AngleAxis(angle, Vector3.up).GetForward().DirectionFrom(GetRootTransformation(editor.GetTimestamp(), editor.Mirror));
				UltiDraw.DrawLine(GetRootPosition(t, editor.Mirror), GetRootPosition(t, editor.Mirror) + forward, 0.1f, 0f, UltiDraw.Magenta);
			}
			UltiDraw.PlotFunction(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), values, -180f, 180f);
			UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Primary = EditorGUILayout.Toggle("Primary", Primary);
			Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
			Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
			SmoothPositions = EditorGUILayout.Toggle("Smooth Positions", SmoothPositions);
			SmoothRotations = EditorGUILayout.Toggle("Smooth Rotations", SmoothRotations);
			SmoothingWindow = EditorGUILayout.FloatField("Smoothing Window", SmoothingWindow);
			LockThreshold = EditorGUILayout.FloatField("Lock Threshold", LockThreshold);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				RootBone = EditorGUILayout.Popup("Root Bone", RootBone, Asset.Source.GetBoneNames());
				Hips = EditorGUILayout.Popup("Hips", Hips, Asset.Source.GetBoneNames());
				Neck = EditorGUILayout.Popup("Neck", Neck, Asset.Source.GetBoneNames());
				LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Asset.Source.GetBoneNames());
				RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Asset.Source.GetBoneNames());
				LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Asset.Source.GetBoneNames());
				RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Asset.Source.GetBoneNames());
			}
		}
#endif
		public Matrix4x4 GetRootTransformation(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			return Matrix4x4.TRS(GetRootPosition(timestamp, mirrored, smoothing), GetRootRotation(timestamp, mirrored, smoothing), Vector3.one);
		}

		public Vector3 GetRootPosition(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			return ComputeRootPosition(timestamp, mirrored, smoothing);
		}

		public Quaternion GetRootRotation(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			return ComputeRootRotation(timestamp, mirrored, smoothing);
		}

		public Vector3 GetRootVelocity(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			Vector3 RootVelocity(float t) {
				return (GetRootPosition(t, mirrored) - GetRootPosition(t - Asset.GetDeltaTime(), mirrored)) / Asset.GetDeltaTime();
			}
			if(smoothing == null) {
				return RootVelocity(timestamp);
			}
            Vector3[] values = new Vector3[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
				values[i] = RootVelocity(timestamp + smoothing.GetKey(i).Timestamp);
            }
            return values.Gaussian();
		}

		public Matrix4x4 GetRootDelta(float from, float to, bool mirrored) {
			Matrix4x4 a = GetRootTransformation(from, mirrored);
			Matrix4x4 b = GetRootTransformation(to, mirrored);
			return b.TransformationTo(a);
			// Vector3 deltaPosition = b.TransformationTo(a).GetPosition();
			// float deltaAngle = GetRootAngle(to, mirrored, from, samples);
			// if(deltaAngle < -180f || deltaAngle > 180f) {
			// 	Debug.Log("DETECTED: " + deltaAngle);
			// }
			// return new Vector3(deltaPosition.x, deltaAngle, deltaPosition.z);
		}

		public Vector3 GetRootDelta(float timestamp, bool mirrored, float deltaTime) {
			Matrix4x4 a = GetRootTransformation(timestamp - deltaTime, mirrored);
			Matrix4x4 b = GetRootTransformation(timestamp, mirrored);
			Matrix4x4 c = b.TransformationTo(a);
			return new Vector3(c.GetPosition().x, Vector3.SignedAngle(Vector3.forward, c.GetForward(), Vector3.up), c.GetPosition().z);
			// Vector3 deltaPosition = b.TransformationTo(a).GetPosition();
			// float deltaAngle = GetRootAngle(to, mirrored, from, samples);
			// if(deltaAngle < -180f || deltaAngle > 180f) {
			// 	Debug.Log("DETECTED: " + deltaAngle);
			// }
			// return new Vector3(deltaPosition.x, deltaAngle, deltaPosition.z);
		}

		// public Matrix4x4 GetRootDelta(float from, float to, bool mirrored, int samples) {
		// 	Matrix4x4 a = GetRootTransformation(from, mirrored);
		// 	Matrix4x4 b = GetRootTransformation(to, mirrored);
		// 	return b.TransformationTo(a);
		// 	// Vector3 deltaPosition = b.TransformationTo(a).GetPosition();
		// 	// float deltaAngle = GetRootAngle(to, mirrored, from, samples);
		// 	// if(deltaAngle < -180f || deltaAngle > 180f) {
		// 	// 	Debug.Log("DETECTED: " + deltaAngle);
		// 	// }
		// 	// return new Vector3(deltaPosition.x, deltaAngle, deltaPosition.z);
		// }

		public float GetRootAngle(float timestamp, bool mirrored, float pivot, int samples) {
			float angle = 0f;
			float step = (timestamp-pivot)/(samples-1);
			for(int i=1; i<samples; i++) {
				float from = pivot + (i-1)*step;
				float to = pivot + i*step;
				angle += Vector3.SignedAngle(GetRootRotation(from, mirrored).GetForward(), GetRootRotation(to, mirrored).GetForward(), Vector3.up);
			}
			return angle;
		}

		public float GetAngularVelocity(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			float AngularVelocity(float t) {
				Vector3 from = GetRootRotation(t - Asset.GetDeltaTime(), mirrored).GetForward();
				Vector3 to = GetRootRotation(t, mirrored).GetForward();
				return Vector3.SignedAngle(from, to, Vector3.up) / 180f / Asset.GetDeltaTime();
			}
			if(smoothing == null) {
				return AngularVelocity(timestamp);
			}
            float[] values = new float[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
				values[i] = AngularVelocity(timestamp + smoothing.GetKey(i).Timestamp);
            }
            return values.Gaussian();
		}

		public float GetRootLock(float timestamp, bool mirrored, TimeSeries smoothing=null) {
			return GetRootVelocity(timestamp, mirrored, smoothing).magnitude < LockThreshold ? 1f : 0f;
		}

		//Calculation Block Start
		private Vector3 ComputeRootPosition(float timestamp, bool mirrored, TimeSeries smoothing) {
			if(smoothing == null)  {
				return RootPosition(timestamp, mirrored);
			}
			Vector3[] values = new Vector3[smoothing.KeyCount];
			Vector3 pivot = RootPosition(timestamp, mirrored);
			for(int i=0; i<values.Length; i++) {
				values[i] = RootPosition(timestamp + smoothing.GetKey(i).Timestamp, mirrored) - pivot;
			}
			return values.Gaussian() + pivot;
		}
		private Quaternion ComputeRootRotation(float timestamp, bool mirrored, TimeSeries smoothing) {
			if(smoothing == null)  {
				return RootRotation(timestamp, mirrored);
			}
			Quaternion[] values = new Quaternion[smoothing.KeyCount];
			for(int i=0; i<values.Length; i++) {
				values[i] = RootRotation(timestamp + smoothing.GetKey(i).Timestamp, mirrored);
			}
			float[] activations = new float[values.Length-1];
			for(int i=0; i<activations.Length; i++) {
				activations[i] = Vector3.SignedAngle(values[i].GetForward(), values[i+1].GetForward(), Vector3.up) / (smoothing.GetKey(i+1).Timestamp - smoothing.GetKey(i).Timestamp);
			}
			float power = Mathf.Deg2Rad*Mathf.Abs(activations.Gaussian());
			return values.Gaussian(power);
		}
		private Vector3 RootPosition(float timestamp, bool mirrored) {
			Frame frame = Asset.GetFrame(timestamp);
			Vector3 position = Vector3.zero;
			if(Topology == TOPOLOGY.Biped) {
				position = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
			}
			if(Topology == TOPOLOGY.Quadruped) {
				position = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
			}
			if(Topology == TOPOLOGY.Bone) {
				position = frame.GetBoneTransformation(RootBone, mirrored).GetPosition();
			}
			if(Ground == 0) {
				position.y = 0f;
			} else {
				position = Utility.ProjectGround(position, Ground);
			}
			return position;
		}
		private Quaternion RootRotation(float timestamp, bool mirrored) {
			Frame frame = Asset.GetFrame(timestamp);
			if(Topology == TOPOLOGY.Biped) {
				Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightHip, mirrored).GetPosition() - frame.GetBoneTransformation(LeftHip, mirrored).GetPosition(), Vector3.up).normalized;
				Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightShoulder, mirrored).GetPosition() - frame.GetBoneTransformation(LeftShoulder, mirrored).GetPosition(), Vector3.up).normalized;
				Vector3 v = (v1+v2).normalized;
				Vector3 forward = Vector3.ProjectOnPlane(Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
				return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
			}
			if(Topology == TOPOLOGY.Quadruped) {
				Vector3 neck = frame.GetBoneTransformation(Neck, mirrored).GetPosition();
				Vector3 hips = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
				Vector3 forward = Vector3.ProjectOnPlane(neck - hips, Vector3.up).normalized;;
				return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward.normalized, Vector3.up);
			}
			if(Topology == TOPOLOGY.Bone) {
				return frame.GetBoneTransformation(RootBone, mirrored).GetRotation();
			}
			return Quaternion.identity;
		}
		//Calculation Block End

		public class Series : TimeSeries.Component {

			public Matrix4x4[] Deltas;
			public Matrix4x4[] Transformations;
			public Vector3[] Velocities;
			public float[] AngularVelocities;
			public float[] Locks;

			public Series(TimeSeries global) : base(global) {
				Deltas = new Matrix4x4[Samples.Length];
				Transformations = new Matrix4x4[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				AngularVelocities = new float[Samples.Length];
				Locks = new float[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					Deltas[i] = Matrix4x4.identity;
					Transformations[i] = Matrix4x4.identity;
					Velocities[i] = Vector3.zero;
					AngularVelocities[i] = 0f;
					Locks[i] = 0f;
				}
			}

			public Series(TimeSeries global, Transform root) : base(global) {
				Deltas = new Matrix4x4[Samples.Length];
				Transformations = new Matrix4x4[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				AngularVelocities = new float[Samples.Length];
				Locks = new float[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					Deltas[i] = Matrix4x4.identity;
					Transformations[i] = Matrix4x4.identity;
					Velocities[i] = Vector3.zero;
					AngularVelocities[i] = 0f;
					Locks[i] = 0f;
				}
				Transformations.SetAll(root.GetWorldMatrix());
			}

			public Series(TimeSeries global, Matrix4x4 root) : base(global) {
				Deltas = new Matrix4x4[Samples.Length];
				Transformations = new Matrix4x4[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				AngularVelocities = new float[Samples.Length];
				Locks = new float[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					Deltas[i] = Matrix4x4.identity;
					Transformations[i] = Matrix4x4.identity;
					Velocities[i] = Vector3.zero;
					AngularVelocities[i] = 0f;
					Locks[i] = 0f;
				}
				Transformations.SetAll(root);
			}

			public void SetTransformation(int index, Matrix4x4 transformation) {
				Transformations[index] = transformation;
			}

			public void SetTransformation(int index, Matrix4x4 transformation, float weight) {
				Transformations[index] = Utility.Interpolate(Transformations[index], transformation, weight);
			}

			public Matrix4x4 GetTransformation(int index) {
				return Transformations[index];
			}

			public void SetPosition(int index, Vector3 value) {
				Matrix4x4Extensions.SetPosition(ref Transformations[index], value);
			}

			public Vector3 GetPosition(int index) {
				return Transformations[index].GetPosition();
			}

			public void SetRotation(int index, Quaternion value) {
				Matrix4x4Extensions.SetRotation(ref Transformations[index], value);
			}

			public Quaternion GetRotation(int index) {
				return Transformations[index].GetRotation();
			}

			public void SetDirection(int index, Vector3 value) {
				Matrix4x4Extensions.SetRotation(ref Transformations[index], Quaternion.LookRotation(value == Vector3.zero ? Vector3.forward : value, Vector3.up));
			}

			public Vector3 GetDirection(int index) {
				return Transformations[index].GetForward();
			}

			public void SetVelocity(int index, Vector3 value) {
				Velocities[index] = value;
			}

			public void SetVelocity(int index, Vector3 value, float weight) {
				Velocities[index] = Vector3.Lerp(Velocities[index], value, weight);
			}

			public Vector3 GetVelocity(int index) {
				return Velocities[index];
			}

			public void TransformFrom(Matrix4x4 space) {
				Transformations.TransformationsFrom(space, true);
				Velocities.DirectionsFrom(space, true);
			}

			public void TransformTo(Matrix4x4 space) {
				Transformations.TransformationsTo(space, true);
				Velocities.DirectionsTo(space, true);
			}

			public void TransformFromTo(Matrix4x4 from, Matrix4x4 to) {
				Transformations.TransformationsFromTo(from, to, true);
				Velocities.DirectionsFromTo(from, to, true);
			}

			public void TransformFromTo(int index, Matrix4x4 from, Matrix4x4 to) {
				Transformations[index] = Transformations[index].TransformationFromTo(from, to);
				Velocities[index] = Velocities[index].DirectionFromTo(from, to);
			}

			public Vector3 GetIntegratedTranslation(int start, int end) {
				Vector3 result = Vector3.zero;
				for(int i=start+1; i<=end; i++) {
					result += GetPosition(i) - GetPosition(i-1);
				}
				return result;
			}
			
			public Vector3 GetIntegratedDirection(int start, int end) {
				Vector3 result = Vector3.zero;
				for(int i=start; i<=end; i++) {
					result += GetDirection(i);
				}
				return result.normalized;
			}

			public float GetAngle(Matrix4x4 pivot, int index) {
				float angle = Vector3.SignedAngle(pivot.GetForward(), GetDirection(Pivot), Vector3.up);
				if(index > Pivot) {
					for(int i=Pivot; i<index; i++) {
						Vector3 from = GetDirection(i);
						Vector3 to = GetDirection(i+1);
						angle += Vector3.SignedAngle(from, to, Vector3.up);
					}
				}
				if(index < Pivot) {
					for(int i=Pivot; i>index; i--) {
						Vector3 from = GetDirection(i);
						Vector3 to = GetDirection(i-1);
						angle += Vector3.SignedAngle(from, to, Vector3.up);
					}
				}
				return angle;
			}

			public void Blend(Matrix4x4 sourceTransformation, float power=1f) {
				Matrix4x4 deltaT = sourceTransformation.TransformationTo(Transformations[Pivot]);
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = 1f - i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(Transformations[i], Transformations[i] * deltaT, weight);
				}
			}

			public void Blend(Matrix4x4 sourceTransformation, Matrix4x4 targetTransformation, float power=1f) {
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(sourceTransformation, targetTransformation, weight);
				}
			}

			public void Blend(Matrix4x4 sourceTransformation, Vector3 sourceVelocity, float power=1f) {
				Matrix4x4 deltaT = sourceTransformation.TransformationTo(Transformations[Pivot]);
				Vector3 deltaV = sourceVelocity - Velocities[Pivot];
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = 1f - i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(Transformations[i], Transformations[i] * deltaT, weight);
					Velocities[i] = Utility.Interpolate(Velocities[i], Velocities[i] + deltaV, weight);
				}
			}

			public void Blend(Matrix4x4 sourceTransformation, Vector3 sourceVelocity, Matrix4x4 targetTransformation, Vector3 targetVelocity, float power=1f) {
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(sourceTransformation, targetTransformation, weight);
					Velocities[i] = Utility.Interpolate(sourceVelocity, targetVelocity, weight);
				}
			}

			public void Generate(Matrix4x4 source, Matrix4x4 target, float arc, float power=1f) {
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(source, target, weight);
				}
				float angle = Vector3.Angle(source.GetForward(), target.GetForward());
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					Vector3 step = angle / 180f * arc * Vector3.Slerp(source.GetForward(), target.GetForward(), weight);
					SetPosition(i, GetPosition(i) + step);
				}
			}

			public void Generate(Matrix4x4 source, Matrix4x4 target, float power=1f) {
				for(int i=Pivot; i<Samples.Length; i++) {
					float weight = i.Ratio(Pivot, Samples.Length-1);
					weight = Mathf.Pow(weight, power);
					Transformations[i] = Utility.Interpolate(source, target, weight);
				}
			}

			public void Interpolate() {
				for(int i=0; i<SampleCount; i++) {
					float weight = (float)(i % Resolution) / (float)Resolution;
					int prevIndex = GetPreviousKey(i).Index;
					int nextIndex = GetNextKey(i).Index;
					if(prevIndex != nextIndex) {
						SetPosition(i, Vector3.Lerp(GetPosition(prevIndex), GetPosition(nextIndex), weight));
						SetDirection(i, Vector3.Lerp(GetDirection(prevIndex), GetDirection(nextIndex), weight).normalized);
						SetVelocity(i, Vector3.Lerp(GetVelocity(prevIndex), GetVelocity(nextIndex), weight));
					}
				}
			}

			private Vector3[] CopyPositions;
			private Quaternion[] CopyRotations;
			private Vector3[] CopyVelocities;
			public void Control(Vector3 move, Vector3 face, float weight, float positionBias=1f, float directionBias=1f, float velocityBias=1f) {
				float ControlWeight(float x, float weight) {
					// return 1f - Mathf.Pow(1f-x, weight);
					return x.SmoothStep(2f, 1f-weight);
					// return x.ActivateCurve(weight, 0f, 1f);
				}
				// Increment(0, Samples.Length-1);
				CopyPositions = new Vector3[Samples.Length];
				CopyRotations = new Quaternion[Samples.Length];
				CopyVelocities = new Vector3[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					CopyPositions[i] = GetPosition(i);
					CopyRotations[i] = GetRotation(i);
					CopyVelocities[i] = GetVelocity(i);
				}
				for(int i=Pivot; i<Samples.Length; i++) {
					float ratio = i.Ratio(Pivot-1, Samples.Length-1);
					//Root Positions
					CopyPositions[i] = CopyPositions[i-1] +
						Vector3.Lerp(
							GetPosition(i) - GetPosition(i-1),
							1f/FutureSamples * move,
							weight * ControlWeight(ratio, positionBias)
						);

					//Root Rotations
					CopyRotations[i] = CopyRotations[i-1] *
						Quaternion.Slerp(
							GetRotation(i).RotationTo(GetRotation(i-1)),
							face != Vector3.zero ? Quaternion.LookRotation(face, Vector3.up).RotationTo(CopyRotations[i-1]) : Quaternion.identity,
							weight * ControlWeight(ratio, directionBias)
						);

					//Root Velocities
					CopyVelocities[i] = CopyVelocities[i-1] +
						Vector3.Lerp(
							GetVelocity(i) - GetVelocity(i-1),
							move-CopyVelocities[i-1],
							weight * ControlWeight(ratio, velocityBias)
						);
				}
				for(int i=0; i<Samples.Length; i++) {
					SetPosition(i, CopyPositions[i]);
					SetRotation(i, CopyRotations[i]);
					SetVelocity(i, CopyVelocities[i]);
				}
			}

			public void ResolveCollisions(float safety, LayerMask mask) {
				for(int i=Pivot; i<Samples.Length; i++) {
					Vector3 previous = GetPosition(i-1);
					Vector3 current = GetPosition(i);
					RaycastHit hit;
					if(Physics.Raycast(previous, (current-previous).normalized, out hit, Vector3.Distance(current, previous), mask, QueryTriggerInteraction.Collide)) {
						//This makes sure no point would ever fall into a geometry volume by projecting point i to i-1
						for(int j=i; j<Samples.Length; j++) {
							SetPosition(j, GetPosition(j-1));
						}
					}
					//This generates a safety-slope around objects over multiple frames in a waterflow-fashion
					Vector3 corrected = SafetyProjection(GetPosition(i));
					SetPosition(i, corrected);
					SetVelocity(i, GetVelocity(i) + (corrected-current) / (Samples[i].Timestamp - Samples[i-1].Timestamp));
				}

				Vector3 SafetyProjection(Vector3 pivot) {
					Vector3 point = Utility.GetClosestPointOverlapSphere(pivot, safety, mask);
					return point + safety * (pivot - point).normalized;
				}
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					Deltas[i] = Deltas[i+1];
					Transformations[i] = Transformations[i+1];
					Velocities[i] = Velocities[i+1];
					AngularVelocities[i] = AngularVelocities[i+1];
					Locks[i] = Locks[i+1];
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawScene) {
					Draw(0, KeyCount);
				}
			}

			public void Draw(int start, int end) {
				Draw(start, end, UltiDraw.Black, UltiDraw.Orange.Opacity(0.75f), UltiDraw.Green.Opacity(0.25f), 2f);
			}
			
			public void Draw(
				int start, 
				int end, 
				Color positionColor, 
				Color directionColor, 
				Color velocityColor, 
				float thickness=1f,
				bool drawConnections=true,
				bool drawPositions=true,
				bool drawDirections=true,
				bool drawVelocities=true,
				bool drawAngularVelocities=true,
				bool drawLocks=true
			) {
				UltiDraw.Begin();

				// start = PivotKey+1;
				// end = KeyCount;

				//Connections
				if(drawConnections) {
					for(int i=start; i<end-1; i++) {
						int current = GetKey(i).Index;
						int next = GetKey(i+1).Index;
						UltiDraw.DrawLine(Transformations[current].GetPosition(), Transformations[next].GetPosition(), Transformations[current].GetUp(), thickness*0.01f, positionColor);
					}
				}

				//Positions
				if(drawPositions) {
					for(int i=start; i<end; i++) {
						int index = GetKey(i).Index;
						UltiDraw.DrawSphere(Transformations[index].GetPosition(), Quaternion.identity, thickness*0.025f, positionColor);
					}
				}

				//Locks
				if(drawLocks) {
					for(int i=start; i<end; i++) {
						int index = GetKey(i).Index;
						UltiDraw.DrawSphere(Transformations[index].GetPosition(), Quaternion.identity, 0.1f, UltiDraw.Red.Opacity(Locks[index]));
					}
				}

				//Directions
				if(drawDirections) {
					for(int i=start; i<end; i++) {
						int index = GetKey(i).Index;
						UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + 0.25f*Transformations[index].GetForward(), Transformations[index].GetUp(), thickness*0.025f, 0f, directionColor);
					}
					// for(int i=start; i<end; i++) {
					// 	int index = GetKey(i).Index;
					// 	Vector3 direction = Quaternion.AngleAxis(GetRelativeAngle(index), Vector3.up).GetForward().DirectionFrom(Transformations[Pivot]);
					// 	UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + 0.5f*direction, Transformations[index].GetUp(), thickness*0.025f, 0f, UltiDraw.Purple);
					// }
				}

				//Velocities
				if(drawVelocities) {
					for(int i=start; i<end; i++) {
						int index = GetKey(i).Index;
						UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + GetTemporalScale(Velocities[index]), Transformations[index].GetUp(), thickness*0.0125f, 0f, velocityColor);
					}
				}

				//Angular Velocities
				if(drawAngularVelocities) {
					for(int i=start; i<end; i++) {
						int index = GetKey(i).Index;
						UltiDraw.DrawLine(GetPosition(index), GetPosition(index) + GetTemporalScale(AngularVelocities[index]) * GetRotation(index).GetRight(), thickness*0.0125f, 0f, UltiDraw.Red);
					}
				}

				// UltiDraw.PlotBars(new Vector2(0.5f, 0.9f), new Vector2(0.5f, 0.1f), Locks, 0f, 1f);

				UltiDraw.End();
			}
		}

	}
}