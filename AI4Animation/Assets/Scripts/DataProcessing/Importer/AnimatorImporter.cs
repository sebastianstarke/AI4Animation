#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Actor))]
[RequireComponent(typeof(Animator))]
public class AnimatorImporter : MonoBehaviour {

	public Actor Retarget;
	public AnimationClip Animation;
	public float Speed = 1f;
	public int Framerate = 60;
	public string Destination = string.Empty;

	public Matrix4x4[] Mapping = new Matrix4x4[0];
	
	private Actor Actor;
	private Animator Animator;
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

	public IEnumerator Bake() {
		if(Application.isPlaying) {
			string destination = "Assets/" + Destination;
			string name = Animation.name.Substring(Animation.name.IndexOf("|")+1);
			if(!AssetDatabase.IsValidFolder(destination)) {
				Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
			} else if(AssetDatabase.LoadAssetAtPath(destination+"/"+name+".asset", typeof(MotionData)) == null) {
				//Start Bake
				Baking = true;
				AnimatorOverrideController aoc = new AnimatorOverrideController(GetAnimator().runtimeAnimatorController);
				var anims = new List<KeyValuePair<AnimationClip, AnimationClip>>();
				foreach (var a in aoc.animationClips)
					anims.Add(new KeyValuePair<AnimationClip, AnimationClip>(a, Animation));
				aoc.ApplyOverrides(anims);
				GetAnimator().runtimeAnimatorController = aoc;
				GetAnimator().speed = Speed;

				//Bake
				transform.position = Vector3.zero;
				transform.rotation = Quaternion.identity;
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
				AssetDatabase.SaveAssets();
				AssetDatabase.Refresh();

				//Stop Bake
				GetAnimator().speed = 0f;
				Baking = false;
			} else {
				Debug.Log("File with name " + name + " already exists.");
			}
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
		if(!Application.isPlaying || Animator == null || Animation == null || Actor == null) {
			return 0f;
		}
		float timestamp = GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime * GetAnimator().GetCurrentAnimatorStateInfo(0).length * Speed;
		return float.IsNaN(timestamp) ? 0f : timestamp;
	}

	public float GetRecordedTime() {
		if(Samples.Count == 0) {
			return 0f;
		} else {
			return Samples[Samples.Count-1].Timestamp;
		}
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
			Target.Retarget = (Actor)EditorGUILayout.ObjectField("Retarget", Target.Retarget, typeof(Actor), true);
			Target.Animation = (AnimationClip)EditorGUILayout.ObjectField("Animation", Target.Animation, typeof(AnimationClip), true);
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

			if(Utility.GUIButton("Compute Mapping", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.ComputeMapping();
			}

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