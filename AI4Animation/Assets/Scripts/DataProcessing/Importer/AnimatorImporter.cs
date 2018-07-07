#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Actor))]
[RequireComponent(typeof(Animator))]
public class AnimatorImporter : MonoBehaviour {

	public AnimationClip[] Animations = new AnimationClip[0];
	public Actor Retarget = null;
	public float Speed = 1f;
	public int Framerate = 60;
	public string Destination = string.Empty;

	public Refinement[] Refinements = new Refinement[0];
	public Matrix4x4[] Mapping = new Matrix4x4[0];
	
	private Actor Actor = null;
	private Animator Animator = null;
	private bool Baking = false;
	private List<Sample> Samples = new List<Sample>();
	
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

	void LateUpdate() {
		for(int i=0; i<Refinements.Length; i++) {
			Quaternion[] rotations = new Quaternion[Refinements[i].ReferenceBones.Length];
			for(int j=0; j<Refinements[i].ReferenceBones.Length; j++) {
				rotations[j] = GetActor().Bones[Refinements[i].ReferenceBones[j]].Transform.rotation;
			}
			GetActor().Bones[Refinements[i].TargetBone].Transform.OverrideRotation(Utility.QuaternionAverage(rotations));
		}
	}

	public IEnumerator Bake() {
		if(Application.isPlaying) {
			Baking = true;
			string destination = "Assets/" + Destination;
			if(!AssetDatabase.IsValidFolder(destination)) {
				Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
			} else {
				for(int k=0; k<Animations.Length; k++) {
					string name = Animations[k].name.Substring(Animations[k].name.IndexOf("|")+1);
					if(AssetDatabase.LoadAssetAtPath(destination+"/"+name+".asset", typeof(MotionData)) == null) {
						//Initialise
						AnimatorOverrideController aoc = new AnimatorOverrideController(GetAnimator().runtimeAnimatorController);
						var anims = new List<KeyValuePair<AnimationClip, AnimationClip>>();
						foreach (var a in aoc.animationClips)
							anims.Add(new KeyValuePair<AnimationClip, AnimationClip>(a, Animations[k]));
						aoc.ApplyOverrides(anims);
						GetAnimator().runtimeAnimatorController = aoc;

						//Start Bake
						transform.position = Vector3.zero;
						transform.rotation = Quaternion.identity;
						GetAnimator().speed = Speed;
						GetAnimator().Play("Animation", 0, 0f);
						yield return new WaitForEndOfFrame();

						Samples = new List<Sample>();
						while(GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime < 1f) {
							Samples.Add(new Sample(GetTimestamp(), GetActor().GetWorldPosture(), GetActor().GetLocalPosture()));
							yield return new WaitForEndOfFrame();
						}
						Samples.Add(new Sample(GetTimestamp(), GetActor().GetWorldPosture(), GetActor().GetLocalPosture()));

						//Save Bake
						MotionData data = ScriptableObject.CreateInstance<MotionData>();

						//Assign Name
						data.Name = name;

						AssetDatabase.CreateAsset(data , destination+"/"+data.Name+".asset");

						//Create Source Data
						data.Source = new MotionData.Hierarchy();
						for(int i=0; i<GetActor().Bones.Length; i++) {
							data.Source.AddBone(GetActor().Bones[i].GetName(), GetActor().Bones[i].GetParent() == null ? "None" : GetActor().Bones[i].GetParent().GetName());
						}

						//Set Frames
						ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt((float)Framerate * GetRecordedTime()));

						//Set Framerate
						data.Framerate = (float)Framerate;

						//Compute Frames
						List<Sample> frames = Resample();
						for(int i=0; i<frames.Count; i++) {
							data.Frames[i] = new Frame(data, i+1, frames[i].Timestamp);
							for(int j=0; j<GetActor().Bones.Length; j++) {
								data.Frames[i].Local[j] = frames[i].LocalPosture[j] * Mapping[j];
								data.Frames[i].World[j] = frames[i].WorldPosture[j] * Mapping[j];
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
				AssetDatabase.SaveAssets();
				AssetDatabase.Refresh();
			}
			Baking = false;
		}
	}

	public void Abort() {
		if(Baking) {
			StopAllCoroutines();
			GetAnimator().speed = 0f;
			Baking = false;
		}
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
		return Samples.Count == 0 ? 0f : Samples.Count / GetRecordedTime();
	}

	public void ComputeMapping() {
		Mapping = new Matrix4x4[GetActor().Bones.Length];
		for(int i=0; i<GetActor().Bones.Length; i++) {
			Mapping[i] = Retarget.Bones[i].Transform.GetWorldMatrix().GetRelativeTransformationTo(GetActor().Bones[i].Transform.GetWorldMatrix());
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
			Matrix4x4[] local = new Matrix4x4[GetActor().Bones.Length];
			Matrix4x4[] world = new Matrix4x4[GetActor().Bones.Length];
			for(int j=0; j<GetActor().Bones.Length; j++) {
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
				Target.Animations[i] = (AnimationClip)AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(o), typeof(AnimationClip));
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
					EditorGUILayout.LabelField(Target.GetActor().Bones[Target.Refinements[i].TargetBone].GetName());
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
						EditorGUILayout.LabelField(Target.GetActor().Bones[Target.Refinements[i].ReferenceBones[j]].GetName());
						EditorGUILayout.EndHorizontal();
					}
				}
			}

			if(Utility.GUIButton("Compute Mapping", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.ComputeMapping();
			}

			Target.Retarget = (Actor)EditorGUILayout.ObjectField("Retarget", Target.Retarget, typeof(Actor), true);
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Target.Destination = EditorGUILayout.TextField(Target.Destination);
			EditorGUILayout.EndHorizontal();
			Target.Speed = EditorGUILayout.FloatField("Speed", Target.Speed);
			Target.Framerate = EditorGUILayout.IntField("Framerate", Target.Framerate);

			EditorGUI.EndDisabledGroup();

			EditorGUILayout.LabelField("Recorded Samples: " + Target.Samples.Count);
			EditorGUILayout.LabelField("Recorded Time: " + Target.GetRecordedTime());
			EditorGUILayout.LabelField("Recording FPS: " + Target.GetRecordingFPS());

			if(!Target.Baking) {
				if(Utility.GUIButton("Bake", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.StartCoroutine(Target.Bake());
				}
			} else {
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