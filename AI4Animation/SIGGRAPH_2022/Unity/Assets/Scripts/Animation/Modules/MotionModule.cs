using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class MotionModule : Module {

		#if UNITY_EDITOR
		public int[] Bones = new int[0];

		public string Filter = string.Empty;

		public bool ExtrapolateMotion = false;

		//Precomputed
		private Precomputable<Matrix4x4[]> PrecomputedTransformations = null;
		private Precomputable<Vector3[]> PrecomputedVelocities = null;

		public override void DerivedResetPrecomputation() {
			PrecomputedTransformations = new Precomputable<Matrix4x4[]>(this);
			PrecomputedVelocities = new Precomputable<Vector3[]>(this);
		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global, GetNames());
			for(int i=0; i<instance.Samples.Length; i++) {
				instance.Transformations[i] = GetBoneTransformations(timestamp + instance.Samples[i].Timestamp, mirrored).Copy();
				instance.Velocities[i] = GetBoneVelocities(timestamp + instance.Samples[i].Timestamp, mirrored).Copy();
			}
			return instance;
		}

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

		}

		protected override void DerivedInspector(MotionEditor editor) {
			Filter = EditorGUILayout.TextField("Filter", Filter);
			for(int i=0; i<Bones.Length; i++) {
				EditorGUILayout.BeginHorizontal();
				if(Filter != string.Empty) {
					string[] names = Asset.Source.GetBoneNames().Filter(Filter);
					int original = names.FindIndex(Asset.Source.Bones[Bones[i]].Name);
					int modified = EditorGUILayout.Popup(original, names);
					if(modified != original) {
						Bones[i] = Asset.Source.FindBone(names[modified]).Index;
					}
				} else {
					Bones[i] = EditorGUILayout.Popup(Bones[i], Asset.Source.GetBoneNames());
				}
				if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f)) {
					RemoveBone(i);
					GUIUtility.ExitGUI();
				}
				EditorGUILayout.EndHorizontal();
			}
			if(Utility.GUIButton("Add Bone", UltiDraw.DarkGrey, UltiDraw.White)) {
				AddBone(editor.GetSession().GetActor().Bones.First().GetName());
			}
			ExtrapolateMotion = EditorGUILayout.Toggle("Extrapolate Motion", ExtrapolateMotion);
		}

		public Matrix4x4[] GetBoneTransformations(float timestamp, bool mirrored) {
			return PrecomputedTransformations.Get(timestamp, mirrored, () => Compute());
			Matrix4x4[] Compute() {
				if(ExtrapolateMotion) {
					Matrix4x4[] transformations = new Matrix4x4[GetBones().Length];
					for(int i=0; i<GetBones().Length; i++) {
						transformations[i] = BoneTransformation(GetBones()[i], timestamp, mirrored);
					}
					return transformations;
				} else {
					return Asset.GetFrame(timestamp).GetBoneTransformations(GetBones(), mirrored);
				}
			}
		}

		public Vector3[] GetBoneVelocities(float timestamp, bool mirrored) {
			return PrecomputedVelocities.Get(timestamp, mirrored, () => Compute());
			Vector3[] Compute() {
				if(ExtrapolateMotion) {
					Vector3[] velocities = new Vector3[GetBones().Length];
					for(int i=0; i<GetBones().Length; i++) {
						velocities[i] = BoneVelocity(GetBones()[i], timestamp, mirrored);
					}
					return velocities;
				} else {
					return Asset.GetFrame(timestamp).GetBoneVelocities(GetBones(), mirrored);
				}
			}
		}

		private Matrix4x4 BoneTransformation(int bone, float timestamp, bool mirrored) {
			float start = Asset.Frames.First().Timestamp;
			float end = Asset.Frames.Last().Timestamp;
			if(timestamp < start || timestamp > end) {
				float boundary = Mathf.Clamp(timestamp, start, end);
				float pivot = 2f*boundary - timestamp;
				float clamped = Mathf.Clamp(pivot, start, end);
				Matrix4x4 tBoundary = Asset.GetFrame(boundary).GetBoneTransformation(bone, mirrored);
				Matrix4x4 tClamped = Asset.GetFrame(clamped).GetBoneTransformation(bone, mirrored);
				return Matrix4x4.TRS(
					2f*tBoundary.GetPosition() - tClamped.GetPosition(),
					tClamped.GetRotation(),
					Vector3.one
				);
			} else {
				return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
			}
		}

		private Vector3 BoneVelocity(int bone, float timestamp, bool mirrored) {
			float start = Asset.Frames.First().Timestamp;
			float end = Asset.Frames.Last().Timestamp;
			if(timestamp < start || timestamp > end) {
				return (BoneTransformation(bone, timestamp, mirrored).GetPosition() - BoneTransformation(bone, timestamp - Asset.GetDeltaTime(), mirrored).GetPosition()) / Asset.GetDeltaTime();
			} else {
				return Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
			}
		}

		public int[] GetBones() {
			return Bones;
		}

		public string[] GetNames() {
			return Asset.Source.GetBoneNames(GetBones());
		}

		public void Clear() {
			ArrayExtensions.Clear(ref Bones);
		}

		public void SetBones(params string[] names) {
			Clear();
			foreach(string n in names) {
				AddBone(n);
			}
		}

		public void AddBone(string name) {
			ArrayExtensions.Append(ref Bones, Asset.Source.FindBone(name).Index);
		}

		public void AddBone(int index) {
			ArrayExtensions.Append(ref Bones, index);
		}

		public void RemoveBone(int index) {
			ArrayExtensions.RemoveAt(ref Bones, index);
		}
		#endif

		public class Series : TimeSeries.Component {
			public Actor Actor;
			public string[] Bones;

			public Matrix4x4[][] Transformations;
			public Vector3[][] Velocities;
			public float[][] FilterCurves;
			
			public Series(TimeSeries global, params string[] bones) : base(global) {
				Bones = bones;

				Transformations = new Matrix4x4[Samples.Length][];
				Velocities = new Vector3[Samples.Length][];
				FilterCurves = new float[Samples.Length][];
				for(int i=0; i<Samples.Length; i++) {
					Transformations[i] = new Matrix4x4[Bones.Length];
					Velocities[i] = new Vector3[Bones.Length];
					FilterCurves[i] = new float[Bones.Length];
					for(int j=0; j<Bones.Length; j++) {
						Transformations[i][j] = Matrix4x4.identity;
						Velocities[i][j] = Vector3.zero;
						FilterCurves[i][j] = 0f;
					}
				}
			}

			public void TransformFrom(RootModule.Series space) {
				for(int i=0; i<Transformations.Length; i++) {
					Transformations[i] = Transformations[i].TransformationsFrom(space.Transformations[i], true);
					Velocities[i] = Velocities[i].DirectionsFrom(space.Transformations[i], true);
				}
			}

			public void TransformTo(RootModule.Series space) {
				for(int i=0; i<Transformations.Length; i++) {
					Transformations[i] = Transformations[i].TransformationsTo(space.Transformations[i], true);
					Velocities[i] = Velocities[i].DirectionsTo(space.Transformations[i], true);
				}
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					for(int j=0; j<Bones.Length; j++) {
						Transformations[i][j] = Transformations[i+1][j];
						Velocities[i][j] = Velocities[i+1][j];
					}
				}
			}

			public override void Interpolate(int start, int end) {
				for(int i=start; i<end; i++) {
					for(int j=0; j<Bones.Length; j++) {
						float weight = (float)(i % Resolution) / (float)Resolution;
						int prevIndex = GetPreviousKey(i).Index;
						int nextIndex = GetNextKey(i).Index;
						if(prevIndex != nextIndex) {
							Transformations[i][j] = Utility.Interpolate(Transformations[prevIndex][j], Transformations[nextIndex][j], weight, weight);
							Velocities[i][j] = Vector3.Lerp(Velocities[prevIndex][j], Velocities[nextIndex][j], weight);
						}
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

					float size = 1f;
					int step = Resolution;

					//Connections
					for(int i=0; i<Transformations.Length-step; i+=step) {
						for(int j=0; j<Bones.Length; j++) {
							UltiDraw.DrawLine(Transformations[i][j].GetPosition(), Transformations[i+step][j].GetPosition(), size*0.01f, UltiDraw.Grey.Opacity(0.5f));
						}
					}

					//Positions
					for(int i=0; i<Transformations.Length; i+=step) {
						for(int j=0; j<Bones.Length; j++) {
							UltiDraw.DrawCircle(Transformations[i][j].GetPosition(), size*0.025f, UltiDraw.Black);
						}
					}

					//Velocities
					for(int i=0; i<Transformations.Length; i+=step) {
						for(int j=0; j<Bones.Length; j++) {
							UltiDraw.DrawLine(Transformations[i][j].GetPosition(), Transformations[i][j].GetPosition() + GetTemporalScale(Velocities[i][j]), size*0.0125f, 0f, UltiDraw.DarkGreen.Opacity(0.25f));
						}
					}

					//Filter Curves
					// UltiDraw.PlotFunctions(new Vector2(0.2f, 0.1f), new Vector2(0.4f, 0.2f), FilterCurves, UltiDraw.Dimension.Y, 0f, 10f);

					UltiDraw.End();
				}
			}
		}

	}
}