#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Actor))]
[RequireComponent(typeof(Animator))]
public class AnimatorImporter : MonoBehaviour {

	public Object[] Animations = new Object[0];
	public Actor Retarget = null;
	public float Speed = 1f;
	public int Framerate = 60;
	public string Destination = string.Empty;

	public Refinement[] Refinements = new Refinement[0];
	public Quaternion[] Mapping = new Quaternion[0];
	public Vector3[] Corrections = new Vector3[0];
	
	private Actor Actor = null;
	private Animator Animator = null;
	private bool Baking = false;
	private List<Sample> Samples = new List<Sample>();

	private Object Current = null;
	
	public Actor GetActor() {
		if(Actor == null) {
			Actor = GetComponent<Actor>();
		}
		return Actor;
	}

	public Animator GetAnimator() {
		if(Animator == null) {
			Animator = GetComponent<Animator>();
		}
		return Animator;
	}

	private void PostProcess() {
		//Mapping
		Matrix4x4[] posture = GetActor().GetPosture();
		for(int i=0; i<posture.Length; i++) {
			posture[i] *= Matrix4x4.TRS(Vector3.zero, Mapping[i], Vector3.one);
		}
		for(int i=0; i<Retarget.Bones.Length; i++) {
			Retarget.Bones[i].Transform.position = posture[i].GetPosition();
			Retarget.Bones[i].Transform.rotation = posture[i].GetRotation();
		}
		for(int i=0; i<Retarget.Bones.Length; i++) {
			Retarget.Bones[i].Transform.rotation *= Quaternion.Euler(Corrections[i]);
		}

		//Refinements
		Quaternion[] refined = new Quaternion[Refinements.Length];
		for(int i=0; i<Refinements.Length; i++) {
			Quaternion[] rotations = new Quaternion[Refinements[i].ReferenceBones.Length];
			for(int j=0; j<Refinements[i].ReferenceBones.Length; j++) {
				rotations[j] = Retarget.Bones[Refinements[i].ReferenceBones[j]].Transform.rotation;
			}
			refined[i] = Utility.QuaternionAverage(rotations);
		}
		for(int i=0; i<Refinements.Length; i++) {
			Retarget.Bones[Refinements[i].TargetBone].Transform.OverrideRotation(refined[i]);
		}
	}

	public IEnumerator Play() {
		yield return new WaitForEndOfFrame();
	}

	public IEnumerator Bake() {
		if(Application.isPlaying) {
			Current = null;
			Baking = true;
			string destination = Destination;
			if(!AssetDatabase.IsValidFolder(destination)) {
				Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
			} else {
				for(int k=0; k<Animations.Length; k++) {
					string name = Animations[k].name;
					if(AssetDatabase.LoadAssetAtPath(destination+"/"+name+".asset", typeof(MotionData)) == null) {
						Current = Animations[k];

						GetActor().transform.position = Vector3.zero;
						GetActor().transform.rotation = Quaternion.identity;
						Retarget.transform.position = Vector3.zero;
						Retarget.transform.rotation = Quaternion.identity;

						//Initialise
						AnimatorOverrideController aoc = new AnimatorOverrideController(GetAnimator().runtimeAnimatorController);
						var anims = new List<KeyValuePair<AnimationClip, AnimationClip>>();
						foreach (var a in aoc.animationClips)
							anims.Add(new KeyValuePair<AnimationClip, AnimationClip>(a, (AnimationClip)AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(Animations[k]), typeof(AnimationClip))));
						aoc.ApplyOverrides(anims);
						GetAnimator().runtimeAnimatorController = aoc;

						//Start Bake
						GetAnimator().speed = Speed;
						GetAnimator().Play("Animation", 0, 0f);
						yield return new WaitForEndOfFrame();

						Samples = new List<Sample>();
						while(GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime < 1f) {
							GetAnimator().speed = Speed;
							PostProcess();
							Samples.Add(new Sample(GetTimestamp(), Retarget.GetPosture(), Retarget.GetPosture()));
							yield return new WaitForEndOfFrame();
						}
						GetAnimator().speed = Speed;
						PostProcess();
						Samples.Add(new Sample(GetTimestamp(),Retarget.GetPosture(), Retarget.GetPosture()));

						//Save Bake
						MotionData data = ScriptableObject.CreateInstance<MotionData>();

						//Assign Name
						data.name = name;

						AssetDatabase.CreateAsset(data , destination+"/"+data.name+".asset");

						//Create Source Data
						data.Source = new MotionData.Hierarchy();
						for(int i=0; i<Retarget.Bones.Length; i++) {
							data.Source.AddBone(Retarget.Bones[i].GetName(), Retarget.Bones[i].GetParent() == null ? "None" : Retarget.Bones[i].GetParent().GetName());
						}

						//Set Frames
						ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt((float)Framerate * GetRecordedTime()));

						//Set Framerate
						data.Framerate = (float)Framerate;

						//Compute Frames
						List<Sample> frames = Resample();
						for(int i=0; i<frames.Count; i++) {
							data.Frames[i] = new Frame(data, i+1, frames[i].Timestamp);
							for(int j=0; j<Retarget.Bones.Length; j++) {
								data.Frames[i].Local[j] = frames[i].LocalPosture[j];
								data.Frames[i].World[j] = frames[i].WorldPosture[j];
							}
						}

						//Finalise
						data.DetectSymmetry();
						data.AddSequence();

						EditorUtility.SetDirty(data);

						//Stop Bake
						GetAnimator().speed = 0f;
						yield return new WaitForEndOfFrame();
					} else {
						Debug.Log("File with name " + name + " already exists.");
					}
				}
				yield return new WaitForEndOfFrame();
				AssetDatabase.SaveAssets();
				AssetDatabase.Refresh();
			}
			Current = null;
			Baking = false;
		}
	}

	public void Abort() {
		if(Baking) {
			StopAllCoroutines();
			GetAnimator().speed = 0f;
			Current = null;
			Baking = false;
		}
	}

	public float GetNormalizedTimestamp() {
		if(!Application.isPlaying || Animator == null || Actor == null) {
			return 0f;
		}
		return GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime;
	}

	public float GetTimestamp() {
		if(!Application.isPlaying || Animator == null || Actor == null) {
			return 0f;
		}
		float timestamp = GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime * GetAnimator().GetCurrentAnimatorStateInfo(0).length * Speed;
		return float.IsNaN(timestamp) ? 0f : timestamp;
	}

	public float GetRecordedTime() {
		return Samples.Count == 0 ? 0f : Samples[Samples.Count-1].Timestamp;
	}

	public float GetRecordingFPS() {
		int samples = 0;
		float fps = 0f;
		for(int i=Samples.Count-1; i>Mathf.Max(Samples.Count-Framerate-1, 0); i--) {
			if(i > 0) {
				samples += 1;
				fps += 1f / (Samples[i].Timestamp - Samples[i-1].Timestamp);
			}
		}
		return samples == 0 ? 0f : (fps / samples);
	}

	public void Setup() {
		if(Retarget != null) {
			if(!Application.isPlaying && transform.hasChanged) {
				Mapping = new Quaternion[GetActor().Bones.Length];
				for(int i=0; i<GetActor().Bones.Length; i++) {
					Mapping[i] = Retarget.Bones[i].Transform.rotation.GetRelativeRotationTo(GetActor().Bones[i].Transform.GetWorldMatrix());
				}
				ArrayExtensions.Resize(ref Corrections, GetActor().Bones.Length);
			}
		} else {
			ArrayExtensions.Resize(ref Mapping, 0);
			ArrayExtensions.Resize(ref Corrections, 0);
			ArrayExtensions.Resize(ref Refinements, 0);
		}
	}

	public List<Sample> Resample() {
		List<Sample> samples = new List<Sample>();
		float[] timestamps = new float[Mathf.RoundToInt((float)Framerate * GetRecordedTime())];
		int index = 0;
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = (float)i/(float)Framerate;
			while(!(Samples[Mathf.Clamp(index-1, 0, samples.Count)].Timestamp <= timestamps[i] && Samples[index].Timestamp >= timestamps[i])) {
				index += 1;
			}
			Sample a = Samples[Mathf.Clamp(index-1, 0, samples.Count)];
			Sample b = Samples[index];
			float weight = a.Timestamp == b.Timestamp ? 0f : (timestamps[i] - a.Timestamp) / (b.Timestamp - a.Timestamp);
			Matrix4x4[] local = new Matrix4x4[Retarget.Bones.Length];
			Matrix4x4[] world = new Matrix4x4[Retarget.Bones.Length];
			for(int j=0; j<Retarget.Bones.Length; j++) {
				local[j] = Utility.Interpolate(a.LocalPosture[j], b.LocalPosture[j], weight);
				world[j] = Utility.Interpolate(a.WorldPosture[j], b.WorldPosture[j], weight);
			}
			samples.Add(new Sample(timestamps[i], world, local));
		}
		return samples;
	}

	[System.Serializable]	
	public class Refinement {
		public int TargetBone = 0;
		public int[] ReferenceBones = new int[0];
	}

	public class Sample {
		public float Timestamp;
		public Matrix4x4[] WorldPosture;
		public Matrix4x4[] LocalPosture;
		public Sample(float timestamp, Matrix4x4[] world, Matrix4x4[] local) {
			Timestamp = timestamp;
			WorldPosture = world;
			LocalPosture = local;
		}
	}

	[CustomEditor(typeof(AnimatorImporter))]
	public class AnimatorImporter_Editor : Editor {

		public AnimatorImporter Target;

		void Awake() {
			Target = (AnimatorImporter)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);
			
			Target.Setup();

			EditorGUI.BeginDisabledGroup(Target.Baking);

			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("Add Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
				ArrayExtensions.Expand(ref Target.Animations);
			}
			if(Utility.GUIButton("Remove Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
				ArrayExtensions.Shrink(ref Target.Animations);
			}
			EditorGUILayout.EndHorizontal();
			for(int i=0; i<Target.Animations.Length; i++) {
				Object o = (Object)EditorGUILayout.ObjectField("Animation " + (i+1), Target.Animations[i], typeof(Object), true);
				if(Target.Animations[i] != o) {
					if(AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(o), typeof(AnimationClip)) != null) {
						Target.Animations[i] = o;
					} else { 
						Target.Animations[i] = null;
					}
				}
			}

			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("Add Refinement", UltiDraw.DarkGrey, UltiDraw.White)) {
				ArrayExtensions.Add(ref Target.Refinements, new Refinement());
			}
			if(Utility.GUIButton("Remove Refinement", UltiDraw.DarkGrey, UltiDraw.White)) {
				ArrayExtensions.Shrink(ref Target.Refinements);
			}
			EditorGUILayout.EndHorizontal();
			for(int i=0; i<Target.Refinements.Length; i++) {
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					Target.Refinements[i].TargetBone = Mathf.Clamp(EditorGUILayout.IntField("Refinement " + (i+1) + " Target Bone", Target.Refinements[i].TargetBone), 0, Target.GetActor().Bones.Length-1);
					EditorGUILayout.LabelField(Target.Retarget.Bones[Target.Refinements[i].TargetBone].GetName());
					EditorGUILayout.EndHorizontal();
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Add Reference", UltiDraw.DarkGrey, UltiDraw.White)) {
						ArrayExtensions.Expand(ref Target.Refinements[i].ReferenceBones);
					}
					if(Utility.GUIButton("Remove Referece", UltiDraw.DarkGrey, UltiDraw.White)) {
						ArrayExtensions.Shrink(ref Target.Refinements[i].ReferenceBones);
					}
					EditorGUILayout.EndHorizontal();
					for(int j=0; j<Target.Refinements[i].ReferenceBones.Length; j++) {
						EditorGUILayout.BeginHorizontal();
						Target.Refinements[i].ReferenceBones[j] = Mathf.Clamp(EditorGUILayout.IntField("Reference " + (j+1), Target.Refinements[i].ReferenceBones[j]), 0, Target.GetActor().Bones.Length-1);
						EditorGUILayout.LabelField(Target.Retarget.Bones[Target.Refinements[i].ReferenceBones[j]].GetName());
						EditorGUILayout.EndHorizontal();
					}
				}
			}

			EditorGUI.EndDisabledGroup();

			Target.Retarget = (Actor)EditorGUILayout.ObjectField("Retarget", Target.Retarget, typeof(Actor), true);
			Target.Destination = EditorGUILayout.TextField("Destination", Target.Destination);
			Target.Speed = EditorGUILayout.FloatField("Speed", Target.Speed);
			Target.Framerate = EditorGUILayout.IntField("Framerate", Target.Framerate);

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				for(int i=0; i<Target.Mapping.Length; i++) {
					Utility.SetGUIColor(Target.Corrections[i].magnitude != 0f ? UltiDraw.Cyan : UltiDraw.Grey);
					Target.Corrections[i] = EditorGUILayout.Vector3Field(Target.GetActor().Bones[i].GetName() + " <-> " + Target.Retarget.Bones[i].GetName(), Target.Corrections[i]);
					Utility.ResetGUIColor();
				}
			}

			if(!Target.Baking) {
				if(Utility.GUIButton("Bake", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.StartCoroutine(Target.Bake());
				}
			} else {
				EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Target.GetNormalizedTimestamp() * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Cyan.Transparent(0.75f));
				EditorGUILayout.LabelField("Animation: " + Target.Current.name);
				EditorGUILayout.LabelField("Recorded Samples: " + Target.Samples.Count);
				EditorGUILayout.LabelField("Recorded Time: " + Target.GetRecordedTime());
				EditorGUILayout.LabelField("Recording FPS: " + Target.GetRecordingFPS());
				if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.Abort();
				}
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}

}
#endif