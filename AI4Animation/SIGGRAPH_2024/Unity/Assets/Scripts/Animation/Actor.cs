using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	[ExecuteInEditMode]
	public class Actor : MonoBehaviour {

		public enum DRAW {Skeleton, Sketch};

		public bool Inspect = false;
		
		public bool DrawRoot = false;
		public bool DrawSkeleton = true;
		public bool DrawTransforms = false;
		public bool DrawVelocities = false;
		public bool DrawAccelerations = false;
		public bool DrawAlignment = false;
		public bool DrawOffset = false;
		public bool DrawHistory = false;
		public bool PlotHistory = false;

		public float BoneSize = 0.025f;
		public Color BoneColor = UltiDraw.Black;
		public Color JointColor = UltiDraw.White;
		public bool DepthRendering = false;

		public Bone[] Bones = new Bone[0];

		private string[] BoneNames = null;
		
		private List<Pose> History = new List<Pose>();
		private class Pose {
			public Matrix4x4[] Transformations;
			public Vector3[] Velocities;
			public Pose(Matrix4x4[] transformations, Vector3[] velocities) {
				Transformations = transformations;
				Velocities = velocities;
			}
		}

		void Reset() {
			Create(transform.GetAllChilds());
		}

		void FixedUpdate() {
			if(DrawHistory || PlotHistory) {
				int maxLength = Mathf.RoundToInt(1f/Time.fixedDeltaTime);
				while(History.Count >= maxLength) {
					History.RemoveAt(0);
				}
				History.Add(new Pose(GetBoneTransformations(), GetBoneVelocities()));
			} else {
				History.Clear();
			}
		}

		public void Initialize() {
			Reset();
		}

		public void CopySetup(Actor reference) {
			Create(reference.GetBoneNames());
		}

		public void CopyBones(Actor reference) {
			GetRoot().position = reference.GetRoot().position;
			GetRoot().rotation = reference.GetRoot().rotation;
			foreach(Actor.Bone bone in Bones) {
				bone.SetTransformation(reference.FindBone(bone.GetName()).GetTransformation());
			}
		}

		public void CopyTransforms(Actor reference) {
			GetRoot().position = reference.GetRoot().position;
			GetRoot().rotation = reference.GetRoot().rotation;
			foreach(Transform other in reference.transform.GetAllChilds()) {
				Transform self = FindTransform(other.name);
				if(self != null) {
					self.gameObject.SetActive(other.gameObject.activeSelf);
					self.SetTransformation(other.GetWorldMatrix());
				}
			}
		}

		public void CopyLocalBoneRotations(Actor reference) {
			for(int i=0; i<Bones.Length; i++) {
				Bones[i].GetTransform().localRotation = reference.Bones[i].GetTransform().localRotation;
			}
		}

		public void RenameBones(string from, string to) {
			void Recursion(Transform t) {
				t.name = t.name.Replace(from, to);
				for(int i=0; i<t.childCount; i++) {
					Recursion(t.GetChild(i));
				}
			}
			Recursion(transform);
			BoneNames = new string[0];
		}

		public void SwitchNames(string a, string b) {
			void Recursion(Transform t) {
				string name = t.name;
				if(name.Contains(a)) {
					t.name = t.name.Replace(a, b);
				}
				if(name.Contains(b)) {
					t.name = t.name.Replace(b, a);
				}
				for(int i=0; i<t.childCount; i++) {
					Recursion(t.GetChild(i));
				}
			}
			Recursion(transform);
			BoneNames = new string[0];
		}

		public Transform GetRoot() {
			return transform;
		}

		public Bone[] GetRootBones() {
			List<Bone> bones = new List<Bone>();
			for(int i=0; i<Bones.Length; i++) {
				if(Bones[i].GetParent() == null) {
					bones.Add(Bones[i]);
				}
			}
			return bones.ToArray();
		}

		public void SetBoneLengths(float[] values) {
			for(int i=0; i<Bones.Length; i++) {
				Bones[i].SetLength(values[i]);
			}
		}

		public void SetBoneLengths(string[] names, float[] values) {
			for(int i=0; i<names.Length; i++) {
				FindBone(names[i]).SetLength(values[i]);
			}
		}

		public float[] GetBoneLengths() {
			float[] lengths = new float[Bones.Length];
			for(int i=0; i<Bones.Length; i++) {
				lengths[i] = Bones[i].GetLength();
			}
			return lengths;
		}

		public float[] GetBoneLengths(params string[] names) {
			float[] lengths = new float[names.Length];
			for(int i=0; i<names.Length; i++) {
				lengths[i] = FindBone(names[i]).GetLength();
			}
			return lengths;
		}

		public Transform[] FindTransforms(params string[] names) {
			Transform[] transforms = new Transform[names.Length];
			for(int i=0; i<transforms.Length; i++) {
				transforms[i] = FindTransform(names[i]);
			}
			return transforms;
		}

		public Transform FindTransform(string name) {
			Transform element = null;
			Action<Transform> recursion = null;
			recursion = new Action<Transform>((transform) => {
				if(transform.name == name) {
					element = transform;
					return;
				}
				for(int i=0; i<transform.childCount; i++) {
					recursion(transform.GetChild(i));
				}
			});
			recursion(GetRoot());
			if(element == null) {
				Debug.Log("Could not find transform of name " + name + ".");
			}
			return element;
		}

		public Bone[] FindBones(params Transform[] transforms) {
			Bone[] bones = new Bone[transforms.Length];
			for(int i=0; i<bones.Length; i++) {
				bones[i] = FindBone(transforms[i]);
			}
			return bones;
		}

		public Bone[] FindBones(params string[] names) {
			Bone[] bones = new Bone[names.Length];
			for(int i=0; i<bones.Length; i++) {
				bones[i] = FindBone(names[i]);
			}
			return bones;
		}

		public Bone FindBone(Transform transform) {
			return Array.Find(Bones, x => x.GetTransform() == transform);
		}

		public Bone FindBone(string name) {
			return Array.Find(Bones, x => x.GetName() == name);
		}

		public string[] GetBoneNames() {
			if(BoneNames == null || BoneNames.Length != Bones.Length) {
				BoneNames = new string[Bones.Length];
				for(int i=0; i<BoneNames.Length; i++) {
					BoneNames[i] = Bones[i].GetName();
				}
			}
			return BoneNames;
		}

		public string[] GetBoneNames(Bone[] bones) {
			string[] names = new string[bones.Length];
			for(int i=0; i<names.Length; i++) {
				names[i] = bones[i].GetName();
			}
			return names;
		}

		public int GetBoneIndex(string name) {
			Bone bone = FindBone(name);
			return bone == null ? -1 : bone.GetIndex();
		}

		public int[] GetBoneIndices(params string[] names) {
			int[] indices = new int[names.Length];
			for(int i=0; i<indices.Length; i++) {
				Bone bone = FindBone(names[i]);
				indices[i] = bone == null ? -1 : bone.GetIndex();
			}
			return indices;
		}

		public Transform[] GetBoneTransforms(params string[] names) {
			Transform[] transforms = new Transform[names.Length];
			for(int i=0; i<names.Length; i++) {
				transforms[i] = FindTransform(names[i]);
			}
			return transforms;
		}

		public void Create() {
			Create(null as Transform);
		}

		public void Create(params Transform[] bones) {
			ArrayExtensions.Clear(ref Bones);
			Action<Transform, Bone> recursion = null;
			recursion = new Action<Transform, Bone>((transform, parent) => {
				if(bones == null || System.Array.Find(bones, x => x == transform)) {
					Bone bone = new Bone(this, transform, Bones.Length, parent);
					ArrayExtensions.Append(ref Bones, bone);
					parent = bone;
				}
				for(int i=0; i<transform.childCount; i++) {
					recursion(transform.GetChild(i), parent);
				}
			});
			recursion(GetRoot(), null);
			BoneNames = new string[0];
		}

		public void Create(params string[] bones) {
			Create(FindTransforms(bones));
		}

		public void CreateSimplifiedSkeleton() {
			Transform actor = new GameObject(name + "_Simplified").transform;
			for(int i=0; i<Bones.Length; i++) {
				Transform bone = new GameObject(Bones[i].GetName()).transform;
				bone.SetParent(Bones[i].GetParent() != null ? actor.FindRecursive(Bones[i].GetParent().GetName()) : actor);
				bone.transform.position = Bones[i].GetPosition();
				bone.transform.rotation = Bones[i].GetRotation();
			}
		}

		public void SetBoneTransformations(Matrix4x4[] values) {
			if(values.Length != Bones.Length) {
				return;
			}
			for(int i=0; i<Bones.Length; i++) {
				Bones[i].SetTransformation(values[i]);
			}
		}

		public void SetBoneTransformations(Matrix4x4[] values, params string[] bones) {
			for(int i=0; i<bones.Length; i++) {
				SetBoneTransformation(values[i], bones[i]);
			}
		}

		public void SetBoneTransformation(Matrix4x4 value, string bone) {
			Bone b = FindBone(bone);
			if(b != null) {
				b.SetTransformation(value);
			}
		}

		public Matrix4x4[] GetBoneTransformations() {
			Matrix4x4[] transformations = new Matrix4x4[Bones.Length];
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = Bones[i].GetTransformation();
			}
			return transformations;
		}

		public Matrix4x4[] GetBoneTransformations(params int[] bones) {
			Matrix4x4[] transformations = new Matrix4x4[bones.Length];
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = Bones[bones[i]].GetTransformation();
			}
			return transformations;
		}

		public Matrix4x4[] GetBoneTransformations(params string[] bones) {
			Matrix4x4[] transformations = new Matrix4x4[bones.Length];
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = GetBoneTransformation(bones[i]);
			}
			return transformations;
		}

		public Matrix4x4 GetBoneTransformation(string bone) {
			return FindBone(bone).GetTransformation();
		}

		public void SetBoneVelocities(Vector3[] values) {
			if(values.Length != Bones.Length) {
				return;
			}
			for(int i=0; i<Bones.Length; i++) {
				Bones[i].SetVelocity(values[i]);
			}
		}

		public void SetBoneVelocities(Vector3[] values, params string[] bones) {
			for(int i=0; i<bones.Length; i++) {
				SetBoneVelocity(values[i], bones[i]);
			}
		}

		public void SetBoneVelocity(Vector3 value, string bone) {
			Bone b = FindBone(bone);
			if(b != null) {
				b.SetVelocity(value);
			}
		}

		public Vector3[] GetBoneVelocities() {
			Vector3[] velocities = new Vector3[Bones.Length];
			for(int i=0; i<velocities.Length; i++) {
				velocities[i] = Bones[i].GetVelocity();
			}
			return velocities;
		}

		public Vector3[] GetBoneVelocities(params int[] bones) {
			Vector3[] velocities = new Vector3[bones.Length];
			for(int i=0; i<velocities.Length; i++) {
				velocities[i] = Bones[bones[i]].GetVelocity();
			}
			return velocities;
		}

		public Vector3[] GetBoneVelocities(params string[] bones) {
			Vector3[] velocities = new Vector3[bones.Length];
			for(int i=0; i<velocities.Length; i++) {
				velocities[i] = GetBoneVelocity(bones[i]);
			}
			return velocities;
		}

		public Vector3 GetBoneVelocity(string bone) {
			return FindBone(bone).GetVelocity();
		}


		public void SetBoneAccelerations(Vector3[] values) {
			if(values.Length != Bones.Length) {
				return;
			}
			for(int i=0; i<Bones.Length; i++) {
				Bones[i].SetAcceleration(values[i]);
			}
		}

		public void SetBoneAccelerations(Vector3[] values, params string[] bones) {
			for(int i=0; i<bones.Length; i++) {
				SetBoneAcceleration(values[i], bones[i]);
			}
		}

		public void SetBoneAcceleration(Vector3 value, string bone) {
			Bone b = FindBone(bone);
			if(b != null) {
				b.SetAcceleration(value);
			}
		}

		public Vector3[] GetBoneAccelerations() {
			Vector3[] values = new Vector3[Bones.Length];
			for(int i=0; i<values.Length; i++) {
				values[i] = Bones[i].GetAcceleration();
			}
			return values;
		}

		public Vector3[] GetBoneAccelerations(params int[] bones) {
			Vector3[] values = new Vector3[bones.Length];
			for(int i=0; i<values.Length; i++) {
				values[i] = Bones[bones[i]].GetAcceleration();
			}
			return values;
		}

		public Vector3[] GetBoneAccelerations(params string[] bones) {
			Vector3[] values = new Vector3[bones.Length];
			for(int i=0; i<values.Length; i++) {
				values[i] = GetBoneAcceleration(bones[i]);
			}
			return values;
		}

		public Vector3 GetBoneAcceleration(string bone) {
			return FindBone(bone).GetAcceleration();
		}

		public Vector3[] GetBonePositions() {
			Vector3[] positions = new Vector3[Bones.Length];
			for(int i=0; i<positions.Length; i++) {
				positions[i] = Bones[i].GetPosition();
			}
			return positions;
		}

		public Vector3[] GetBonePositions(params int[] bones) {
			Vector3[] positions = new Vector3[bones.Length];
			for(int i=0; i<positions.Length; i++) {
				positions[i] = Bones[bones[i]].GetPosition();
			}
			return positions;
		}

		public Vector3[] GetBonePositions(params string[] bones) {
			Vector3[] positions = new Vector3[bones.Length];
			for(int i=0; i<positions.Length; i++) {
				positions[i] = GetBonePosition(bones[i]);
			}
			return positions;
		}

		public Vector3[] GetBonePositions(params Bone[] bones) {
			Vector3[] positions = new Vector3[bones.Length];
			for(int i=0; i<positions.Length; i++) {
				positions[i] = bones[i].GetPosition();
			}
			return positions;
		}

		public Vector3 GetBonePosition(string bone) {
			return FindBone(bone).GetPosition();
		}

		public Quaternion[] GetBoneRotations() {
			Quaternion[] rotation = new Quaternion[Bones.Length];
			for(int i=0; i<rotation.Length; i++) {
				rotation[i] = Bones[i].GetRotation();
			}
			return rotation;
		}

		public Quaternion[] GetBoneRotations(params string[] bones) {
			Quaternion[] rotation = new Quaternion[bones.Length];
			for(int i=0; i<rotation.Length; i++) {
				rotation[i] = GetBoneRotation(bones[i]);
			}
			return rotation;
		}

		public Quaternion[] GetBoneRotations(params int[] bones) {
			Quaternion[] rotations = new Quaternion[bones.Length];
			for(int i=0; i<rotations.Length; i++) {
				rotations[i] = Bones[bones[i]].GetRotation();
			}
			return rotations;
		}

		public Quaternion[] GetBoneRotations(params Bone[] bones) {
			Quaternion[] rotations = new Quaternion[bones.Length];
			for(int i=0; i<rotations.Length; i++) {
				rotations[i] = bones[i].GetRotation();
			}
			return rotations;
		}


		public Quaternion GetBoneRotation(string bone) {
			return FindBone(bone).GetRotation();
		}

		public Vector3[] GetBoneAngles() {
			Vector3[] angles = new Vector3[Bones.Length];
			for(int i=0; i<angles.Length; i++) {
				angles[i] = Bones[i].GetTransform().localEulerAngles;
			}
			return angles;
		}

		public Vector3 GetCenterOfGravity(params string[] bones) {
			Vector3 point = Vector3.zero;
			float sum = 0f;
			foreach(Bone bone in bones.Length == 0 ? Bones : FindBones(bones)) {
				float length = bone.GetLength();
				point += length * bone.GetPosition();
				sum += length;
			}
			return point/sum;
		}

		public void RestoreAlignment() {
			RestoreAlignment(Bones);
		}

		public void RestoreAlignment(Bone[] bones) {
			foreach(Bone bone in bones) {
				bone.RestoreAlignment();
			}
		}
		
		public void Draw(Matrix4x4[] transformations, Color boneColor, Color jointColor, DRAW mode) {
			if(transformations.Length != Bones.Length) {
				Debug.Log("Number of given transformations does not match number of bones.");
				return;
			}
			UltiDraw.Begin();
			if(mode == DRAW.Skeleton) {
				void Recursion(Bone bone) {
					Matrix4x4 current = transformations[bone.GetIndex()];
					if(bone.GetParent() != null) {
						Matrix4x4 parent = transformations[bone.GetParent().GetIndex()];
						UltiDraw.DrawBone(
							parent.GetPosition(),
							Quaternion.FromToRotation(parent.GetForward(), current.GetPosition() - parent.GetPosition()) * parent.GetRotation(),
							12.5f*BoneSize*bone.GetLength(), bone.GetLength(),
							boneColor
						);
					}
					UltiDraw.DrawSphere(
						current.GetPosition(),
						Quaternion.identity,
						5f/8f * BoneSize,
						jointColor
					);
					for(int i=0; i<bone.GetChildCount(); i++) {
						Recursion(bone.GetChild(i));
					}
				}
				foreach(Bone bone in GetRootBones()) {
					Recursion(bone);
				}
			}
			if(mode == DRAW.Sketch) {
				void Recursion(Bone bone) {
					Matrix4x4 current = transformations[bone.GetIndex()];
					if(bone.GetParent() != null) {
						Matrix4x4 parent = transformations[bone.GetParent().GetIndex()];
						UltiDraw.DrawLine(parent.GetPosition(), current.GetPosition(), boneColor);
					}
					UltiDraw.DrawCube(current.GetPosition(), current.GetRotation(), 0.02f, jointColor);
					for(int i=0; i<bone.GetChildCount(); i++) {
						Recursion(bone.GetChild(i));
					}
				}
				foreach(Bone bone in GetRootBones()) {
					Recursion(bone);
				}
			}
			UltiDraw.End();
		}


		public void Draw(Matrix4x4[] transformations, string[] bones, Color boneColor, Color jointColor, DRAW mode) {
			if(transformations.Length != bones.Length) {
				Debug.Log("Number of given transformations does not match number of given bones.");
				return;
			}
			UltiDraw.Begin();
			if(mode == DRAW.Skeleton) {
				void Recursion(Bone bone, int parent) {
					int index = bones.FindIndex(bone.GetName());
					if(index >= 0) {
						Matrix4x4 boneMatrix = transformations[index];
						if(parent >= 0) {
							Matrix4x4 parentMatrix = transformations[parent];
							float length = Vector3.Distance(parentMatrix.GetPosition(), boneMatrix.GetPosition());
							UltiDraw.DrawBone(
								parentMatrix.GetPosition(),
								Quaternion.FromToRotation(parentMatrix.GetForward(), boneMatrix.GetPosition() - parentMatrix.GetPosition()) * parentMatrix.GetRotation(),
								12.5f*BoneSize*length, length,
								boneColor
							);
						}
						UltiDraw.DrawSphere(
							boneMatrix.GetPosition(),
							Quaternion.identity,
							5f/8f * BoneSize,
							jointColor
						);
						parent = index;
					}
					for(int i=0; i<bone.GetChildCount(); i++) {
						Recursion(bone.GetChild(i), parent);
					}
				}
				foreach(Bone bone in GetRootBones()) {
					Recursion(bone, -1);
				}
			}
			if(mode == DRAW.Sketch) {
				void Recursion(Bone bone, int parent) {
					int index = bones.FindIndex(bone.GetName());
					if(index >= 0) {
						Matrix4x4 boneMatrix = transformations[index];
						if(parent >= 0) {
							Matrix4x4 parentMatrix = transformations[parent];
							UltiDraw.DrawLine(parentMatrix.GetPosition(), boneMatrix.GetPosition(), boneColor);
						}
						UltiDraw.DrawCube(boneMatrix.GetPosition(), boneMatrix.GetRotation(), 0.02f, jointColor);
						parent = index;
					}
					for(int i=0; i<bone.GetChildCount(); i++) {
						Recursion(bone.GetChild(i), parent);
					}
				}
				foreach(Bone bone in GetRootBones()) {
					Recursion(bone, -1);
				}
			}
			UltiDraw.End();
		}

		public void DrawRootTransformation(Matrix4x4 root) {
			UltiDraw.Begin();
			UltiDraw.DrawWireCube(root.GetPosition(), root.GetRotation(), 0.1f, UltiDraw.Magenta);
			UltiDraw.DrawCube(root.GetPosition(), root.GetRotation(), 0.075f, UltiDraw.Cyan.Opacity(0.75f));
			UltiDraw.DrawTranslateGizmo(root.GetPosition(), root.GetRotation(), 0.1f);
			UltiDraw.End();
		}

		public void DrawCharacterIcon(Color color) {
			UltiDraw.Begin();
			UltiDraw.DrawPyramid(transform.position.SetY(GetBonePositions().Max().y+0.6f), transform.rotation, 0.3f, -0.3f, color);
			UltiDraw.End();
		}

		public Transform CreateVisualInstance() {
			Transform instance = Instantiate(gameObject).transform;
			foreach(Component c in instance.GetAllChilds()) {
				if(c is SkinnedMeshRenderer || c is Renderer) {
					
				} else {
					Utility.Destroy(c);
				}
			}
			return instance;
		}

		void OnRenderObject() {
			UltiDraw.SetDepthRendering(DepthRendering);

			if(DrawSkeleton) {
				Draw(GetBoneTransformations(), BoneColor, JointColor, DRAW.Skeleton);
			}

			if(DrawRoot) {
				DrawRootTransformation(GetRoot().GetWorldMatrix());
			}

			UltiDraw.Begin();
			if(DrawVelocities) {
				for(int i=0; i<Bones.Length; i++) {
					UltiDraw.DrawArrow(
						Bones[i].GetPosition(),
						Bones[i].GetPosition() + Bones[i].GetVelocity(),
						0.75f,
						0.0075f,
						0.05f,
						UltiDraw.DarkGreen.Opacity(0.5f)
					);
				}
			}

			if(DrawAccelerations) {
				for(int i=0; i<Bones.Length; i++) {
					UltiDraw.DrawArrow(
						Bones[i].GetPosition(),
						Bones[i].GetPosition() + Bones[i].GetAcceleration(),
						0.75f,
						0.0075f,
						0.05f,
						UltiDraw.DarkBlue.Opacity(0.5f)
					);
				}
			}

			if(DrawTransforms) {
				foreach(Bone bone in Bones) {
					UltiDraw.DrawTranslateGizmo(bone.GetPosition(), bone.GetRotation(), 0.05f);
				}
			}

			if(DrawAlignment) {
				foreach(Bone bone in Bones) {
					UltiDraw.DrawLine(bone.GetPosition(), bone.GetPosition() + bone.GetRotation() * bone.GetAlignment(), 0.05f, 0f, UltiDraw.Magenta);
				}
			}

			if(DrawOffset) {
				foreach(Bone bone in Bones) {
					if(bone.GetParent() != null) {
						UltiDraw.DrawLine(bone.GetParent().GetPosition(), bone.GetParent().GetPosition() + bone.GetParent().GetRotation() * bone.GetOffset(), 0.05f, 0f, UltiDraw.Green);
					}
				}
			}
			UltiDraw.End();

			if(DrawHistory) {
				foreach(Pose pose in History) {
					Draw(pose.Transformations, UltiDraw.Black, UltiDraw.Black, DRAW.Sketch);
				}
			}

			if(PlotHistory) {
				UltiDraw.Begin();
				float[][] values = new float[10][];
				for(int i=0; i<10; i++) {
					values[i] = new float[100];
					for(int j=0; j<100; j++) {
						values[i][j] = UnityEngine.Random.value;
					}
				}
				UltiDraw.PlotFunctions(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.25f), values, UltiDraw.Dimension.X, yMin:0f, yMax:1f, backgroundColor:UltiDraw.Black, lineColors:UltiDraw.GetRainbowColors(10));
				UltiDraw.End();
			}

			UltiDraw.SetDepthRendering(false);
		}

		public class State {
			public Matrix4x4[] Transformations;
			public Vector3[] Velocities;
		}

		[Serializable]
		public class Bone {
			[SerializeField] private Actor Actor;
			[SerializeField] private Transform Transform;
			[SerializeField] private Vector3 Velocity;
			[SerializeField] private Vector3 Acceleration;
			[SerializeField] private Quaternion AngularVelocity;
			[SerializeField] private int Index;
			[SerializeField] private int Parent;
			[SerializeField] private int[] Childs;
			[SerializeField] private Vector3 Offset; //The default local position of the joint in the parent space.
			[SerializeField] private Vector3 Alignment; //The axis pointing from the joint's child to this joint along which the rotation needs to be aligned;

			public Bone(Actor actor, Transform transform, int index, Bone parent) {
				Actor = actor;
				Transform = transform;
				Velocity = Vector3.zero;
				Acceleration = Vector3.zero;
				AngularVelocity = Quaternion.identity;
				Index = index;
				Childs = new int[0];
				if(parent != null) {
					Parent = parent.Index;
					ArrayExtensions.Append(ref parent.Childs, Index);
				} else {
					Parent = -1;
				}
			}

			public Actor GetActor() {
				return Actor;
			}

			public Transform GetTransform() {
				return Transform;
			}

			public string GetName() {
				return Transform.name;
			}

			public int GetIndex() {
				return Index;
			}

			public Bone GetParent() {
				return Parent == -1 ? null : Actor.Bones[Parent];
			}

			public Bone GetChild(int index) {
				return index >= Childs.Length ? null : Actor.Bones[Childs[index]];
			}

			public Bone[] GetChilds() {
				Bone[] childs = new Bone[Childs.Length];
				for(int i=0; i<childs.Length; i++) {
					childs[i] = GetChild(i);
				}
				return childs;
			}

			public int GetChildCount() {
				return Childs.Length;
			}

			public void SetLength(float value) {
				if(GetParent() == null) {
					return;
				}
				Transform.position = GetParent().Transform.position + value * (Transform.position - GetParent().Transform.position).normalized;
			}

			public float GetLength() {
				return GetParent() == null ? 0f : Vector3.Distance(GetParent().Transform.position, Transform.position);
			}

			public void SetTransformation(Matrix4x4 matrix) {
				Transform.position = matrix.GetPosition();
				Transform.rotation = matrix.GetRotation();
			}

			public Matrix4x4 GetTransformation() {
				return Transform.GetWorldMatrix();
			}

			public void SetPosition(Vector3 position) {
				Transform.position = position;
			}

			public Vector3 GetPosition() {
				return Transform.position;
			}

			public void SetRotation(Quaternion rotation) {
				Transform.rotation = rotation;
			}

			public Quaternion GetRotation() {
				return Transform.rotation;
			}

			public void SetVelocity(Vector3 velocity) {
				Velocity = velocity;
			}

			public Vector3 GetVelocity() {
				return Velocity;
			}

			public void SetAcceleration(Vector3 acceleration) {
				Acceleration = acceleration;
			}

			public Vector3 GetAcceleration() {
				return Acceleration;
			}

			public void SetAngularVelocity(Quaternion value) {
				AngularVelocity = value;
			}

			public Quaternion GetAngularVelocity() {
				return AngularVelocity;
			}

			public bool HasAlignment() {
				return Alignment != Vector3.zero;
			}

			public Vector3 GetAlignment() {
				return Alignment;
			}

			public Vector3 GetOffset() {
				return Offset;
			}

			public void ComputeOffset() {
				Offset = GetParent() == null ? Vector3.zero : Transform.localPosition;
			}

			public void ComputeAlignment() {
				if(Childs.Length != 1) {
					Alignment = Vector3.zero;
				} else {
					Alignment = (GetChild(0).GetPosition() - GetPosition()).DirectionTo(GetTransformation());
				}
			}

			public void RestoreOffset() {
				if(GetParent() != null) {
					Transform.localPosition = Offset;
				}
			}

			public void RestoreAlignment() {
				if(!HasAlignment()) {
					return;
				}
				Vector3 position = GetPosition();
				Quaternion rotation = GetRotation();
				Vector3 childPosition = GetChild(0).GetPosition();
				Quaternion childRotation = GetChild(0).GetRotation();
				Vector3 target = childPosition-position;
				Vector3 aligned = rotation * Alignment;
				// float[] angles = new float[] {
				// 	Vector3.Angle(rotation.GetRight(), target),
				// 	Vector3.Angle(rotation.GetUp(), target),
				// 	Vector3.Angle(rotation.GetForward(), target),
				// 	Vector3.Angle(-rotation.GetRight(), target),
				// 	Vector3.Angle(-rotation.GetUp(), target),
				// 	Vector3.Angle(-rotation.GetForward(), target)
				// };
				// Vector3 vector = Vector3.zero;
				// float min = angles.Min();
				// if(min == angles[0]) {
				// 	vector = rotation.GetRight();
				// }
				// if(min == angles[1]) {
				// 	vector = rotation.GetUp();
				// }
				// if(min == angles[2]) {
				// 	vector = rotation.GetForward();
				// }
				// if(min == angles[3]) {
				// 	vector = -rotation.GetRight();
				// }
				// if(min == angles[4]) {
				// 	vector = -rotation.GetUp();
				// }
				// if(min == angles[5]) {
				// 	vector = -rotation.GetForward();
				// }
				// SetRotation(Quaternion.FromToRotation(vector, target) * rotation);
				SetRotation(Quaternion.FromToRotation(aligned, target) * rotation);
				// GetChild(0).SetPosition(position + Alignment.magnitude * target.normalized);
				GetChild(0).SetPosition(childPosition);
				GetChild(0).SetRotation(childRotation);
			}
		}

		#if UNITY_EDITOR
		[CustomEditor(typeof(Actor))]
		public class Actor_Editor : Editor {

			public Actor Target;

			private string AName = string.Empty;
			private string BName = string.Empty;

			void Awake() {
				Target = (Actor)target;
			}

			public override void OnInspectorGUI() {
				Undo.RecordObject(Target, Target.name);

				Target.DrawRoot = EditorGUILayout.Toggle("Draw Root", Target.DrawRoot);
				Target.DrawSkeleton = EditorGUILayout.Toggle("Draw Skeleton", Target.DrawSkeleton);
				Target.DrawTransforms = EditorGUILayout.Toggle("Draw Transforms", Target.DrawTransforms);
				Target.DrawVelocities = EditorGUILayout.Toggle("Draw Velocities", Target.DrawVelocities);
				Target.DrawAccelerations = EditorGUILayout.Toggle("Draw Acceleration", Target.DrawAccelerations);
				Target.DrawAlignment = EditorGUILayout.Toggle("Draw Alignment", Target.DrawAlignment);
				Target.DrawOffset = EditorGUILayout.Toggle("Draw Offset", Target.DrawOffset);
				Target.DrawHistory = EditorGUILayout.Toggle("Draw History", Target.DrawHistory);
				Target.PlotHistory = EditorGUILayout.Toggle("Plot History", Target.PlotHistory);

				Utility.SetGUIColor(Color.grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Utility.GUIButton("Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Inspect = !Target.Inspect;
					}
					if(Target.Inspect) {
						{
							Actor reference = (Actor)EditorGUILayout.ObjectField("Copy Setup", null, typeof(Actor), true);
							if(reference != null) {
								Target.CopySetup(reference);
							}
						}

						{
							Actor reference = (Actor)EditorGUILayout.ObjectField("Copy Bones", null, typeof(Actor), true);
							if(reference != null) {
								Target.CopyBones(reference);
							}
						}

						{
							Actor reference = (Actor)EditorGUILayout.ObjectField("Copy Transforms", null, typeof(Actor), true);
							if(reference != null) {
								Target.CopyTransforms(reference);
							}
						}

						{
							Actor reference = (Actor)EditorGUILayout.ObjectField("Copy Local Bone Rotations", null, typeof(Actor), true);
							if(reference != null) {
								Target.CopyLocalBoneRotations(reference);
							}
						}

						EditorGUILayout.BeginHorizontal();
						AName = EditorGUILayout.TextField(AName, GUILayout.Width(175f));
						EditorGUILayout.LabelField(">", GUILayout.Width(10f));
						BName = EditorGUILayout.TextField(BName, GUILayout.Width(175f));
						if(Utility.GUIButton("Rename", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.RenameBones(AName, BName);
						}
						if(Utility.GUIButton("Switch", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.SwitchNames(AName, BName);
						}
						EditorGUILayout.EndHorizontal();

						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField("Bones: " + Target.Bones.Length);
						if(Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.Create(new Transform[0]);
						}
						EditorGUILayout.EndHorizontal();
						if(Utility.GUIButton("Compute Alignment", UltiDraw.DarkGrey, UltiDraw.White)) {
							foreach(Bone bone in Target.Bones) {
								bone.ComputeAlignment();
							}
						}
						if(Utility.GUIButton("Compute Offset", UltiDraw.DarkGrey, UltiDraw.White)) {
							foreach(Bone bone in Target.Bones) {
								bone.ComputeOffset();
							}
						}
						if(Utility.GUIButton("Create Simplified Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
							Target.CreateSimplifiedSkeleton();
						}
						Target.BoneSize = EditorGUILayout.FloatField("Bone Size", Target.BoneSize);
						Target.BoneColor = EditorGUILayout.ColorField("Bone Color", Target.BoneColor);
						Target.JointColor = EditorGUILayout.ColorField("Joint Color", Target.JointColor);
						Target.DepthRendering = EditorGUILayout.Toggle("Depth Rendering", Target.DepthRendering);
						InspectSkeleton(Target.GetRoot(), 0);
					}
				}

				if(GUI.changed) {
					EditorUtility.SetDirty(Target);
				}
			}

			private void InspectSkeleton(Transform transform, int indent) {
				float indentSpace = 10f;
				Bone bone = Target.FindBone(transform);
				Utility.SetGUIColor(bone == null ? UltiDraw.LightGrey : UltiDraw.Mustard);
				using(new EditorGUILayout.HorizontalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Bone", bone == null ? UltiDraw.White : UltiDraw.DarkGrey, bone == null ? UltiDraw.DarkGrey : UltiDraw.White)) {
						Transform[] bones = new Transform[Target.Bones.Length];
						for(int i=0; i<bones.Length; i++) {
							bones[i] = Target.Bones[i].GetTransform();
						}
						if(bone == null) {
							ArrayExtensions.Append(ref bones, transform);
							Target.Create(bones);
						} else {
							ArrayExtensions.Remove(ref bones, transform);
							Target.Create(bones);
						}
					}
					for(int i=0; i<indent; i++) {
						EditorGUILayout.LabelField("|", GUILayout.Width(indentSpace));
					}
					EditorGUILayout.LabelField("-", GUILayout.Width(indentSpace));
					EditorGUILayout.LabelField(transform.name, GUILayout.Width(400f - indent*indentSpace));
					GUILayout.FlexibleSpace();
					if(bone != null) {
						// EditorGUILayout.LabelField("Parent: " + (bone.GetParent() == null ? "None" : bone.GetParent().GetName()));
						EditorGUILayout.LabelField("Index: " + bone.GetIndex().ToString(), GUILayout.Width(60f));
						EditorGUILayout.LabelField("Length: " + bone.GetLength().ToString("F3"), GUILayout.Width(90f));
						EditorGUILayout.LabelField("Offset: " + bone.GetOffset().ToString(), GUILayout.Width(150f));
						EditorGUILayout.LabelField("Alignment: " + bone.GetAlignment().ToString(), GUILayout.Width(150f));
					}
					EditorGUILayout.EndHorizontal();
				}
				for(int i=0; i<transform.childCount; i++) {
					InspectSkeleton(transform.GetChild(i), indent+1);
				}
			}

		}
		#endif

	}
}