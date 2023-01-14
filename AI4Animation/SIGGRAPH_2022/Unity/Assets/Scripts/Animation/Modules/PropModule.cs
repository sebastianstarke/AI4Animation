#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace AI4Animation {
	public class PropModule : Module {

		public int PropBone = -1;
        public int[] PropBones = new int[0];
        public float Size = 1f;
		public float Window = 2f;

		private GameObject Ball = null;
		private RootModule RootModule = null;

		private Precomputable<Vector3> PrecomputedPivots = null;
		private Precomputable<Vector3> PrecomputedMomentums = null;

		public override void DerivedResetPrecomputation() {
			PrecomputedPivots = new Precomputable<Vector3>(this);
			PrecomputedMomentums = new Precomputable<Vector3>(this);
		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global, Asset.GetModule<RootModule>().ExtractSeries(global, timestamp, mirrored) as RootModule.Series);
			for(int i=0; i<instance.Samples.Length; i++) {
				instance.Positions[i] = GetPropPosition(timestamp + instance.Samples[i].Timestamp, mirrored);
				instance.Velocities[i] = GetPropVelocity(timestamp + instance.Samples[i].Timestamp, mirrored);
				instance.Pivots[i] = GetPropPivot(timestamp + instance.Samples[i].Timestamp, mirrored);
				instance.Momentums[i] = GetPropMomentum(timestamp + instance.Samples[i].Timestamp, mirrored);
			}
			return instance;
		}

		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
		}

		protected override void DerivedUnload(MotionEditor editor) {
            if(Ball != null && !PrefabUtility.IsPartOfRegularPrefab(Ball)) {
				Utility.Destroy(Ball);
			}
		}
		

		protected override void DerivedCallback(MotionEditor editor) {
			if(PropBone == -1) {
				return;
			}
			
			if(Ball == null) {
				Transform ball = editor.transform.Find("Ball");
				if(ball != null) {
					Ball = ball.gameObject;
				} else {
					Ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
					Ball.name = "Ball";
					Ball.transform.localScale = Size*Vector3.one;
					Ball.transform.SetParent(editor.transform);
					Ball.layer = LayerMask.NameToLayer("Prop");
					Ball.GetComponent<SphereCollider>().radius = 0.75f;
				}
			}

			Matrix4x4 transformation = editor.GetCurrentFrame().GetBoneTransformation(PropBone, editor.Mirror);
			Ball.transform.position = transformation.GetPosition();
			Ball.transform.rotation = transformation.GetRotation();
		}

		protected override void DerivedGUI(MotionEditor editor) {
            
		}

		protected override void DerivedDraw(MotionEditor editor) {
            UltiDraw.Begin();
			// UltiDraw.SetDepthRendering(true);
			// Matrix4x4 root = Asset.GetModule<RootModule>().GetRootTransformation(editor.GetTimestamp(), editor.Mirror);
			// UltiDraw.DrawCircle(root.GetPosition() + new Vector3(0f, 0.01f, 0f), Quaternion.Euler(90f, 0f, 0f), 2f*InteractionRadius, UltiDraw.Yellow.Opacity(0.25f));
			// UltiDraw.SetDepthRendering(false);
			// if(PropBone != -1) {
			// 	Matrix4x4 transformation = editor.GetCurrentFrame().GetBoneTransformation(PropBone, editor.Mirror);
			// 	UltiDraw.DrawSphere(transformation.GetPosition(), transformation.GetRotation(), Size, UltiDraw.Cyan);
			// 	UltiDraw.DrawWireSphere(transformation.GetPosition(), transformation.GetRotation(), Size, UltiDraw.Magenta.Opacity(0.5f));
			// 	UltiDraw.DrawLine(root.GetPosition(), root.GetPosition() + root.GetRotation()*GetPropPivot(editor.GetTimestamp(), editor.Mirror), 0.05f, 0f, UltiDraw.Magenta.Opacity(0.5f));
			// }
			int index = 0;
			foreach(int bone in PropBones) {
				Matrix4x4 transformation = editor.GetCurrentFrame().GetBoneTransformation(bone, editor.Mirror);
				UltiDraw.DrawWireSphere(transformation.GetPosition(), transformation.GetRotation(), Size, UltiDraw.Magenta.Opacity(0.25f));
				index += 1;// speed.Values[i] = Mathf.Lerp(speed.Values[i], 0f, idle.Values[i]);
			}
            UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
			PropBone = EditorGUILayout.Popup("Prop Bone", PropBone, Asset.Source.GetBoneNames());
			for(int i=0; i<PropBones.Length; i++) {
				PropBones[i] = EditorGUILayout.Popup("Prop " + (i+1), PropBones[i], Asset.Source.GetBoneNames());
			}
            Size = EditorGUILayout.FloatField("Size", Size);
			Window = EditorGUILayout.FloatField("Window", Window);
		}

		// public Matrix4x4 GetPropTransformation(float timestamp, bool mirrored) {
		// 	return Asset.GetFrame(timestamp).GetBoneTransformation(PropBone, mirrored);
		// }

		public Vector3 GetPropPosition(float timestamp, bool mirrored) {
			return Asset.GetFrame(timestamp).GetBoneTransformation(PropBone, mirrored).GetPosition();
		}

		public Vector3 GetPropVelocity(float timestamp, bool mirrored) {
			return Asset.GetFrame(timestamp).GetBoneVelocity(PropBone, mirrored);
		}

		public Vector3 GetPropPivot(float timestamp, bool mirrored) {
			return PrecomputedPivots.Get(timestamp, mirrored, () => Compute());
			Vector3 Compute() {
				float[] window = Asset.GetTimeWindow(Window, 1f);
				Vector3[] vectors = new Vector3[window.Length];
				float[] angles = new float[vectors.Length-1];
				for(int i=0; i<vectors.Length; i++) {
					float t = timestamp + window[i];
					// vectors[i] = GetPropTransformation(t, mirrored).GetPosition().PositionTo(GetRootModule().GetRootTransformation(t, mirrored)).ZeroY().normalized;
					vectors[i] = GetPropPosition(t, mirrored).PositionTo(GetRootModule().GetRootTransformation(t, mirrored)).ZeroY().normalized;
				}
				for(int i=0; i<angles.Length; i++) {
					angles[i] = Vector3.SignedAngle(vectors[i], vectors[i+1], Vector3.up) / (window[i+1] - window[i]);
				}
				float power = Mathf.Deg2Rad*Mathf.Abs(angles.Gaussian());
				Vector3 pivot = vectors.Gaussian(power);
				return pivot;
			}
		}

		public Vector3 GetPropMomentum(float timestamp, bool mirrored) {
			return PrecomputedMomentums.Get(timestamp, mirrored, () => Compute());
			Vector3 Compute() {
				return (GetPropPivot(timestamp, mirrored) - GetPropPivot(timestamp - Asset.GetDeltaTime(), mirrored)) / Asset.GetDeltaTime();
			}
		}

		public float GetPropDistance(float timestamp, bool mirrored) {
			float[] window = Asset.GetTimeWindow(Window, 1f);
			float[] distances = new float[window.Length];
			float[] deltas = new float[distances.Length-1];
			for(int i=0; i<distances.Length; i++) {
				float t = timestamp + window[i];
				// distances[i] = Vector3.Distance(GetRootModule().GetRootTransformation(t, mirrored).GetPosition().ZeroY(), GetPropTransformation(t, mirrored).GetPosition().ZeroY());
				distances[i] = Vector3.Distance(GetRootModule().GetRootTransformation(t, mirrored).GetPosition().ZeroY(), GetPropPosition(t, mirrored).ZeroY());
			}
			for(int i=0; i<deltas.Length; i++) {
				deltas[i] = (distances[i+1] - distances[i]) / (window[i+1] - window[i]);
			}
			float power = Mathf.Abs(deltas.Gaussian());
			float distance = distances.Gaussian(power);
			return distance;
		}

		private RootModule GetRootModule() {
			if(RootModule == null) {
				RootModule = Asset.GetModule<RootModule>();
			}
			return RootModule;
		}

		public class Series : TimeSeries.Component {

			public RootModule.Series RootSeries;
			public Vector3[] Positions;
			public Vector3[] Velocities;
			public Vector3[] Pivots;
			public Vector3[] Momentums;

			private float PropSize = 0.25f;

			private UltiDraw.GUIRect View = new UltiDraw.GUIRect(0.125f, 0.175f, 0.15f, 0.15f);

			public Series(TimeSeries global, RootModule.Series rootSeries) : base(global) {
				RootSeries = rootSeries;
				Positions = new Vector3[Samples.Length];
				Velocities = new Vector3[Samples.Length];
				Pivots = new Vector3[Samples.Length];
				Momentums = new Vector3[Samples.Length];
				for(int i=0; i<Samples.Length; i++) {
					Positions[i] = Vector3.zero;
					Velocities[i] = Vector3.zero;
					Pivots[i] = Vector3.zero;
					Momentums[i] = Vector3.zero;
				}
			}

			public void Update() {
				for(int i=0; i<Pivot; i++) {
					Positions[i] = Positions[i+1];
					Velocities[i] = Velocities[i+1];
				}
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					Pivots[i] = Pivots[i+1];
					Momentums[i] = Momentums[i+1];
				}
			}

			public float GetPropSize() {
				return PropSize;
			}

			public float GetVisualizationArea() {
				return 1f;
			}

			public Matrix4x4 GetCoordinateSystem() {
				return Matrix4x4.TRS(
					Positions[Pivot],
					Quaternion.LookRotation((Positions[Pivot] - RootSeries.GetPosition(Pivot)).ZeroY().normalized, Vector3.up),
					Vector3.one
				);
			}

			public Vector3 InterpolatePivot(Vector3 from, Vector3 to, float vectorWeight, float heightWeight) {
				float magnitude = Mathf.Lerp(from.ZeroY().magnitude, to.ZeroY().magnitude, vectorWeight);
				Vector3 vector = Vector3.Lerp(from.ZeroY(), to.ZeroY(), vectorWeight).normalized;
				float height = Mathf.Lerp(from.y, to.y, heightWeight);
				return (magnitude * vector).SetY(height);
			}

			public Vector3 InterpolateMomentum(Vector3 from, Vector3 to, float vectorWeight, float heightWeight) {
				return Vector3.Lerp(from.ZeroY(), to.ZeroY(), vectorWeight).SetY(Mathf.Lerp(from.y, to.y, heightWeight));
			}

			public override void Interpolate(int start, int end) {
				for(int i=start; i<end; i++) {
					float weight = (float)(i % Resolution) / (float)Resolution;
					int prevIndex = GetPreviousKey(i).Index;
					int nextIndex = GetNextKey(i).Index;
					if(prevIndex != nextIndex) {
						Pivots[i] = InterpolatePivot(Pivots[prevIndex], Pivots[nextIndex], weight, weight); 
						Momentums[i] = InterpolateMomentum(Momentums[prevIndex], Momentums[nextIndex], weight, weight);
						Positions[i] = Vector3.Lerp(Positions[prevIndex], Positions[nextIndex], weight);
						Velocities[i] = Vector3.Lerp(Velocities[prevIndex], Velocities[nextIndex], weight);
					}
				}
			}

			public override void GUI() {
				if(DrawGUI) {

				}
			}

			public override void Draw() {
				if(DrawScene) {
					UltiDraw.Begin();

					Matrix4x4 GetRoot(int index) {
						return RootSeries.Transformations[index];
					}				
					Color GetPivotColor(int index) {
						return UltiDraw.Orange;
					}
					Color GetMomentumColor(int index) {
						return UltiDraw.Purple;
					}

					Color circleColor = UltiDraw.DarkGrey.Opacity(0.5f);
					Color wireColor = UltiDraw.Cyan.Opacity(0.5f);
					Color referenceColor = UltiDraw.White.Opacity(0.5f);

					if(DrawGUI) {
						//Image Space
						UltiDraw.GUICircle(View.GetCenter(), View.W, circleColor);
						UltiDraw.GUICircle(View.GetCenter() + View.ToScreen(new Vector2(0f, 1f)), 0.01f, referenceColor);
						UltiDraw.GUICircle(View.GetCenter() + View.ToScreen(new Vector2(0f, -1f)), 0.01f, referenceColor);
						UltiDraw.GUICircle(View.GetCenter() + View.ToScreen(new Vector2(1f, 0f)), 0.01f, referenceColor);
						UltiDraw.GUICircle(View.GetCenter() + View.ToScreen(new Vector2(-1f, 0f)), 0.01f, referenceColor);
						int step = Resolution;
						for(int i=0; i<Samples.Length; i+=step) {
							Vector3 current = View.GetCenter() + View.ToScreen(new Vector2(Pivots[i].x, Pivots[i].z));
							Vector3 target = View.GetCenter() + View.ToScreen(new Vector2(Pivots[i].x, Pivots[i].z) + GetTemporalScale(new Vector2(Momentums[i].x, Momentums[i].z)));
							if(i < Samples.Length-step) {
								Vector3 next = View.GetCenter() + View.ToScreen(new Vector2(Pivots[i+step].x, Pivots[i+step].z));
								UltiDraw.GUILine(current, next, UltiDraw.Red);
							}
							UltiDraw.GUICircle(current, 0.01f, GetPivotColor(i));
							UltiDraw.GUILine(current, target, GetMomentumColor(i));
						}
					}
					if(DrawScene) {
						Vector3 position = GetRoot(Pivot).GetPosition();
						Quaternion rotation = GetRoot(Pivot).GetRotation();
						//World Space
						UltiDraw.DrawCircle(position, Quaternion.Euler(90f, 0f, 0f), 2f*GetVisualizationArea(), circleColor);
						UltiDraw.DrawWireCircle(position, Quaternion.Euler(90f, 0f, 0f), 2f*GetVisualizationArea(), wireColor);
						UltiDraw.DrawCircle(position + GetVisualizationArea()*rotation.GetForward(), 0.05f, referenceColor);
						UltiDraw.DrawCircle(position + GetVisualizationArea()*rotation.GetRight(), 0.05f, referenceColor);
						UltiDraw.DrawCircle(position - GetVisualizationArea()*rotation.GetForward(), 0.05f, referenceColor);
						UltiDraw.DrawCircle(position - GetVisualizationArea()*rotation.GetRight(), 0.05f, referenceColor);

						//Pivots and Momentums
						for(int i=0; i<Samples.Length; i+=Resolution) {
							float size = Mathf.Sqrt((float)(i+1) / (float)Samples.Length);
							Vector3 location = position + GetVisualizationArea()*(rotation*Pivots[i]);
							Vector3 momentum = rotation*GetTemporalScale(Momentums[i]);
							UltiDraw.DrawSphere(location, Quaternion.identity, size * 0.05f, GetPivotColor(i));
							UltiDraw.DrawArrow(location, location + momentum, 0.8f, 0.0125f, 0.025f, GetMomentumColor(i));
						}
						//Connections
						for(int i=0; i<Samples.Length-1; i++) {
							Vector3 prev = position + GetVisualizationArea()*(rotation*Pivots[i]);
							Vector3 next = position + GetVisualizationArea()*(rotation*Pivots[i+1]);
							UltiDraw.DrawLine(prev, next, UltiDraw.DarkGrey);
						}

						//Prop
						// for(int i=0; i<Pivot; i+=Resolution) {
						for(int i=0; i<Samples.Length; i+=Resolution) {
							UltiDraw.DrawSphere(Positions[i], Quaternion.identity, 0.025f, UltiDraw.White);
							UltiDraw.DrawArrow(Positions[i], Positions[i] + GetTemporalScale(Velocities[i]), 0.8f, 0.005f, 0.025f, UltiDraw.White);
						}
						// for(int i=0; i<Pivot; i++) {
						for(int i=0; i<Samples.Length-1; i++) {
							UltiDraw.DrawLine(Positions[i], Positions[i+1], UltiDraw.White);
						}

						Matrix4x4 root = RootSeries.Transformations[Pivot];
						Vector3 ball = Positions[Pivot];
						Vector3 pivot = Pivots[Pivot];
						UltiDraw.DrawSphere(ball, Quaternion.identity, GetPropSize(), UltiDraw.Cyan);
						UltiDraw.DrawWireSphere(ball, Quaternion.identity, GetPropSize(), UltiDraw.Magenta.Opacity(0.5f));
						UltiDraw.DrawLine(root.GetPosition(), root.GetPosition() + root.GetRotation()*pivot, 0.05f, 0f, UltiDraw.Magenta.Opacity(0.5f));

						UltiDraw.DrawTranslateGizmo(GetCoordinateSystem(), 0.5f);
					}

					UltiDraw.End();
				}
			}
		}

	}
}
#endif
