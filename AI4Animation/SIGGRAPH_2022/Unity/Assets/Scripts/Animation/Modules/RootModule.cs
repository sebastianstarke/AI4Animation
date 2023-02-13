using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif

namespace AI4Animation {
	public class RootModule : Module {

		#if UNITY_EDITOR
		public enum TOPOLOGY {Biped, Quadruped, CenterOfGravity};

		public TOPOLOGY Topology = TOPOLOGY.Biped;
		public int Hips, Neck, LeftShoulder, RightShoulder, LeftHip, RightHip, Head, LeftWrist, RightWrist;
		public LayerMask Ground = 0;
		public bool SmoothRotations = true;
		public float Window = 2f;
		public int History = 24;

		//Precomputed
		private Actor Actor = null;
		private int[] Mapping = null;
		private float[] SmoothingWindow = null;
		// private Vector3[] SmoothingPositions = null;
		private Quaternion[] SmoothingRotations = null;
		private float[] SmoothingAngles = null;
		
		private Precomputable<Matrix4x4> PrecomputedTransformations = null;
		private Precomputable<Vector3> PrecomputedPositions = null;
		private Precomputable<Quaternion> PrecomputedRotations = null;
		private Precomputable<Vector3> PrecomputedVelocities = null;
		private Precomputable<float> PrecomputedRegularLengths = null;

		public override void DerivedResetPrecomputation() {
			SmoothingWindow = null;
			// SmoothingPositions = null;
			SmoothingRotations = null;
			SmoothingAngles = null;
			PrecomputedTransformations = new Precomputable<Matrix4x4>(this);
			PrecomputedPositions = new Precomputable<Vector3>(this);
			PrecomputedRotations = new Precomputable<Quaternion>(this);
			PrecomputedVelocities = new Precomputable<Vector3>(this);
			PrecomputedRegularLengths = new Precomputable<float>(this);
		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global);
			for(int i=0; i<instance.Samples.Length; i++) {
                float t = timestamp + instance.Samples[i].Timestamp;
				instance.Transformations[i] = GetRootTransformation(t, mirrored);
				instance.Velocities[i] = GetRootVelocity(t, mirrored);
			}
			return instance;
		}

		protected override void DerivedInitialize() {
			MotionAsset.Hierarchy.Bone h = Asset.Source.FindBoneContains("Hips");
			Hips = h == null ? 0 : h.Index;
			MotionAsset.Hierarchy.Bone n = Asset.Source.FindBoneContains("Neck");
			Neck = n == null ? 0 : n.Index;
			MotionAsset.Hierarchy.Bone ls = Asset.Source.FindBoneContains("LeftShoulder");
			LeftShoulder = ls == null ? 0 : ls.Index;
			MotionAsset.Hierarchy.Bone rs = Asset.Source.FindBoneContains("RightShoulder");
			RightShoulder = rs == null ? 0 : rs.Index;
			MotionAsset.Hierarchy.Bone lh = Asset.Source.FindBoneContainsAny("LeftHip", "LeftUpLeg");
			LeftHip = lh == null ? 0 : lh.Index;
			MotionAsset.Hierarchy.Bone rh = Asset.Source.FindBoneContainsAny("RightHip", "RightUpLeg");
			RightHip = rh == null ? 0 : rh.Index;
			MotionAsset.Hierarchy.Bone head = Asset.Source.FindBoneContainsAny("Head");
			Head = head == null ? 0 : head.Index;
			MotionAsset.Hierarchy.Bone lw = Asset.Source.FindBoneContainsAny("LeftHand");
			LeftWrist = lw == null ? 0 : lw.Index;
			MotionAsset.Hierarchy.Bone rw = Asset.Source.FindBoneContainsAny("RightHand");
			RightWrist = rw == null ? 0 : rw.Index;
			Ground = LayerMask.GetMask("Ground");
		}

		protected override void DerivedLoad(MotionEditor editor) {

		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
	
		protected override void DerivedCallback(MotionEditor editor) {
			Actor = editor.GetSession().GetActor();
			Mapping = editor.GetSession().GetBoneMapping();

			Frame frame = editor.GetCurrentFrame();

			Actor.Bone[] rootBones = editor.GetSession().GetActor().GetRootBones();
			Matrix4x4[] rootTransformations = new Matrix4x4[rootBones.Length];
			for(int i=0; i<rootBones.Length; i++) {
				rootTransformations[i] = rootBones[i].GetTransformation();
			}
			editor.GetSession().GetActor().transform.position = GetRootPosition(frame.Timestamp, editor.Mirror);
			editor.GetSession().GetActor().transform.rotation = GetRootRotation(frame.Timestamp, editor.Mirror);
			for(int i=0; i<rootBones.Length; i++) {
				rootBones[i].SetTransformation(rootTransformations[i]);
			}
		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {

		}

		protected override void DerivedInspector(MotionEditor editor) {
			Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
			Hips = EditorGUILayout.Popup("Hips", Hips, Asset.Source.GetBoneNames());
			Neck = EditorGUILayout.Popup("Neck", Neck, Asset.Source.GetBoneNames());
			LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Asset.Source.GetBoneNames());
			RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Asset.Source.GetBoneNames());
			LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Asset.Source.GetBoneNames());
			RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Asset.Source.GetBoneNames());
			Head = EditorGUILayout.Popup("Head", Head, Asset.Source.GetBoneNames());
			LeftWrist = EditorGUILayout.Popup("Left Wrist", LeftWrist, Asset.Source.GetBoneNames());
			RightWrist = EditorGUILayout.Popup("Right Wrist", RightWrist, Asset.Source.GetBoneNames());
			Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
			SmoothRotations = EditorGUILayout.Toggle("Smooth Rotations", SmoothRotations);
			Window = EditorGUILayout.FloatField("Window", Window);
			Actor = EditorGUILayout.ObjectField("Actor", Actor, typeof(Actor), true) as Actor;
		}

		public Matrix4x4 GetRootDelta(float timestamp, bool mirrored) {
			return GetRootTransformation(timestamp, mirrored).TransformationTo(GetRootTransformation(timestamp-Asset.GetDeltaTime(), mirrored));
		}

		public Matrix4x4 GetRootTransformation(float timestamp, bool mirrored) {
			return PrecomputedTransformations.Get(timestamp, mirrored, () => Compute());
			Matrix4x4 Compute() {
				return Matrix4x4.TRS(GetRootPosition(timestamp, mirrored), GetRootRotation(timestamp, mirrored), Vector3.one);
			}
		}

		public Vector3 GetRootPosition(float timestamp, bool mirrored) {
			return PrecomputedPositions.Get(timestamp, mirrored, () => Compute());
			Vector3 Compute() {
				return GetRootPosition(timestamp, mirrored, Topology);
			}
		}

		public Vector3 GetRootPosition(float timestamp, bool mirrored, TOPOLOGY topology) {
			return RootPosition(timestamp, mirrored, topology);
		}

		public Quaternion GetRootRotation(float timestamp, bool mirrored) {
			return PrecomputedRotations.Get(timestamp, mirrored, () => Compute());
			Quaternion Compute() {
				return GetRootRotation(timestamp, mirrored, Topology);
			}
		}

		public Quaternion GetRootRotation(float timestamp, bool mirrored, TOPOLOGY topology) {
			if(!SmoothRotations)  {
				return RootRotation(timestamp, mirrored, topology);
			}
			SmoothingWindow = SmoothingWindow == null ? Asset.GetTimeWindow(Window, 1f) : SmoothingWindow;
			SmoothingRotations = SmoothingRotations.Validate(SmoothingWindow.Length);
			SmoothingAngles = SmoothingAngles.Validate(SmoothingRotations.Length-1);
			for(int i=0; i<SmoothingWindow.Length; i++) {
				SmoothingRotations[i] = RootRotation(timestamp + SmoothingWindow[i], mirrored, topology);
			}
			for(int i=0; i<SmoothingAngles.Length; i++) {
				SmoothingAngles[i] = Vector3.SignedAngle(SmoothingRotations[i].GetForward(), SmoothingRotations[i+1].GetForward(), Vector3.up) / (SmoothingWindow[i+1] - SmoothingWindow[i]);
			}
			float power = Mathf.Deg2Rad*Mathf.Abs(SmoothingAngles.Gaussian());
			return SmoothingRotations.Gaussian(power);
		}

		public Vector3 GetRootVelocity(float timestamp, bool mirrored) {
			return PrecomputedVelocities.Get(timestamp, mirrored, () => Compute());
			Vector3 Compute() {
				return (GetRootPosition(timestamp, mirrored) - GetRootPosition(timestamp - Asset.GetDeltaTime(), mirrored)) / Asset.GetDeltaTime();
			}
		}

		private Vector3 RootPosition(float timestamp, bool mirrored, TOPOLOGY topology) {
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
			if(timestamp < start || timestamp > end) {
				float boundary = Mathf.Clamp(timestamp, start, end);
				float pivot = 2f*boundary - timestamp;
				float clamped = Mathf.Clamp(pivot, start, end);
				return 2f*RootPosition(Asset.GetFrame(boundary)) - RootPosition(Asset.GetFrame(clamped));
			} else {
				return RootPosition(Asset.GetFrame(timestamp));
			}

			Vector3 RootPosition(Frame frame) {
				Vector3 position = Vector3.zero;
				if(topology == TOPOLOGY.Biped) {
					position = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
				}
				if(topology == TOPOLOGY.Quadruped) {
					position = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
				}
				if(topology == TOPOLOGY.CenterOfGravity) {
					position = frame.GetBoneTransformations(Mapping, mirrored).GetPositions().Mean(Actor.GetBoneLengths());
				}
				if(Ground == 0) {
					position.y = 0f;
				} else {
					position = Utility.ProjectGround(position, Ground);
				}
				return position;
			}
		}

		private Quaternion RootRotation(float timestamp, bool mirrored, TOPOLOGY topology) {
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
			if(timestamp < start || timestamp > end) {
				float boundary = Mathf.Clamp(timestamp, start, end);
				float pivot = 2f*boundary - timestamp;
				float clamped = Mathf.Clamp(pivot, start, end);
				return RootRotation(Asset.GetFrame(clamped));
			} else {
				return RootRotation(Asset.GetFrame(timestamp));
			}

			Quaternion RootRotation(Frame frame) {
				if(topology == TOPOLOGY.Biped) {
					Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightHip, mirrored).GetPosition() - frame.GetBoneTransformation(LeftHip, mirrored).GetPosition(), Vector3.up).normalized;
					Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightShoulder, mirrored).GetPosition() - frame.GetBoneTransformation(LeftShoulder, mirrored).GetPosition(), Vector3.up).normalized;
					Vector3 v = (v1+v2).normalized;
					Vector3 forward = Vector3.ProjectOnPlane(Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
					return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
				}
				if(topology == TOPOLOGY.Quadruped) {
					Vector3 neck = frame.GetBoneTransformation(Neck, mirrored).GetPosition();
					Vector3 hips = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
					Vector3 forward = Vector3.ProjectOnPlane(neck - hips, Vector3.up).normalized;;
					return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward.normalized, Vector3.up);
				}
				if(topology == TOPOLOGY.CenterOfGravity) {
					Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightHip, mirrored).GetPosition() - frame.GetBoneTransformation(LeftHip, mirrored).GetPosition(), Vector3.up).normalized;
					Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightShoulder, mirrored).GetPosition() - frame.GetBoneTransformation(LeftShoulder, mirrored).GetPosition(), Vector3.up).normalized;
					Vector3 v = (v1+v2).normalized;
					Vector3 forward = Vector3.ProjectOnPlane(Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
					return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
				}
				return Quaternion.identity;
			}
		}

		public float GetRootLength(float timestamp, bool mirrored) {
		return PrecomputedRegularLengths.Get(timestamp, mirrored, () => Compute());

        float Compute() {
            SmoothingWindow = SmoothingWindow == null ? Asset.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SmoothingWindow;
            float value = 0f;
            for(int i=0; i<SmoothingWindow.Length; i++) {
                value += GetRootVelocity(timestamp + SmoothingWindow[i], mirrored).magnitude;
            }
            return value / SmoothingWindow.Length;
        }
    }

		#endif

		public class Series : TimeSeries.Component {

			public Matrix4x4[] Transformations;
			public Vector3[] Velocities;

			public Color PositionColor = UltiDraw.Red;
			public Color DirectionColor = UltiDraw.Orange;
			public Color VelocityColor = UltiDraw.DarkGreen;
			public float Size = 2f;
			public float Opacity = 1f;

			public Series(TimeSeries global) : base(global) {
				Transformations = new Matrix4x4[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					Transformations[i] = Matrix4x4.identity;
					Velocities[i] = Vector3.zero;
				}
			}

			public Series(TimeSeries global, Transform root) : base(global) {
				Transformations = new Matrix4x4[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				Matrix4x4 m = root.GetWorldMatrix();
				for(int i=0; i<Samples.Length; i++) {
					Transformations[i] = m;
					Velocities[i] = Vector3.zero;
				}
			}
			
			public void SetTransformation(int index, Matrix4x4 transformation) {
				Transformations[index] = transformation;
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

			public float ControlWeight(float x, float weight) {
				// return 1f - Mathf.Pow(1f-x, weight);
				return x.SmoothStep(2f, 1f-weight);
				// return x.ActivateCurve(weight, 0f, 1f);
			}

			// public float CorrectionWeight(float x, float weight) {
			// 	// return 1f - Mathf.Pow(1f-x, weight);
			// 	return x.SmoothStep(2f, 1f-weight);
			// 	// return x.ActivateCurve(weight, 0f, 1f);
			// }

			public Vector3 GetIntegratedTranslation(int start, int end) {
				Vector3 result = Vector3.zero;
				for(int i=start+1; i<=end; i++) {
					result += GetPosition(i) - GetPosition(i-1);
				}
				return result;
				// return result.normalized;
				// return result.magnitude < 0.1f ? Vector3.zero : result.normalized;
			}
			
			public Vector3 GetIntegratedDirection(int start, int end) {
				Vector3 result = Vector3.zero;
				for(int i=start; i<=end; i++) {
					result += GetDirection(i);
				}
				return result.normalized;
			}

			// public float GetIntegratedRotation(int start, int end) {
			// 	float result = 0f;
			// 	for(int i=start+1; i<=end; i++) {
			// 		result += Vector3.SignedAngle(GetDirection(i-1), GetDirection(i), Vector3.up);
			// 	}
			// 	return result;
			// }

			private Vector3[] CopyPositions;
			private Quaternion[] CopyRotations;
			private Vector3[] CopyVelocities;
			public void Control(Vector3 move, Vector3 face, float weight, float positionBias=1f, float directionBias=1f, float velocityBias=1f) {
				Increment(0, Samples.Length-1);
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
							1f/FutureSampleCount * move,
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
					Transformations[i] = Transformations[i+1];
					Velocities[i] = Velocities[i+1];
				}
			}

			public override void Interpolate(int start, int end) {
				for(int i=start; i<end; i++) {
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

			public override void GUI() {
				
			}

			public override void Draw() {
				if(DrawScene) {
					UltiDraw.Begin();

					int step = Resolution;

					//Connections
					for(int i=0; i<Transformations.Length-step; i+=step) {
						UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Transformations[i].GetUp(), Size*0.01f, UltiDraw.Black.Opacity(Opacity));
					}

					//Positions
					for(int i=0; i<Transformations.Length; i+=step) {
						UltiDraw.DrawCircle(Transformations[i].GetPosition(), Size*0.025f, i % Resolution == 0 ? UltiDraw.Black : PositionColor.Opacity(0.5f*Opacity));
					}

					//Directions
					for(int i=0; i<Transformations.Length; i+=step) {
						UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + 0.25f*Transformations[i].GetForward(), Transformations[i].GetUp(), Size*0.025f, 0f, DirectionColor.Opacity(0.75f*Opacity));
					}

					//Velocities
					for(int i=0; i<Velocities.Length; i+=step) {
						UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + GetTemporalScale(Velocities[i]), Transformations[i].GetUp(), Size*0.0125f, 0f, VelocityColor.Opacity(0.25f*Opacity));
					}

					UltiDraw.End();
				}
			}
		}

	}
}