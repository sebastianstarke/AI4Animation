#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AnimatorImporter : MonoBehaviour {

	public Actor Actor;
	public Actor Retarget;
	public Animator Animator;
	public AnimationClip Animation;
	public float Speed = 1f;
	public int Framerate = 60;
	public string Name = string.Empty;
	public string Destination = string.Empty;

	public Quaternion[] Mapping = new Quaternion[0];
	
	private bool Baking = false;
	private List<Sample> Samples = new List<Sample>();
	
	public IEnumerator Bake() {
		if(Application.isPlaying) {
			StartBake();
			yield return new WaitForEndOfFrame();

			Samples = new List<Sample>();
			while(Animator.GetCurrentAnimatorStateInfo(0).normalizedTime < 1f) {
				Samples.Add(new Sample(GetTimestamp(), Actor.GetWorldPosture(), Actor.GetLocalPosture()));
				yield return new WaitForEndOfFrame();
			}
			Samples.Add(new Sample(GetTimestamp(), Actor.GetWorldPosture(), Actor.GetLocalPosture()));

			StopBake();
			yield return new WaitForEndOfFrame();
		}
	}

	public float GetTimestamp() {
		if(!Application.isPlaying || Animator == null || Animation == null || Actor == null) {
			return 0f;
		}
		float timestamp = Animator.GetCurrentAnimatorStateInfo(0).normalizedTime * Animator.GetCurrentAnimatorStateInfo(0).length * Animator.GetCurrentAnimatorStateInfo(0).speedMultiplier;
		return float.IsNaN(timestamp) ? 0f : timestamp;
	}

	public float GetRecordedTime() {
		if(Samples.Count == 0) {
			return 0f;
		} else {
			return Samples[Samples.Count-1].Timestamp;
		}
	}

	public void StartBake() {
		if(Application.isPlaying) {
			Baking = true;
			AnimatorOverrideController aoc = new AnimatorOverrideController(Animator.runtimeAnimatorController);
			var anims = new List<KeyValuePair<AnimationClip, AnimationClip>>();
			foreach (var a in aoc.animationClips)
				anims.Add(new KeyValuePair<AnimationClip, AnimationClip>(a, Animation));
			aoc.ApplyOverrides(anims);
			Animator.runtimeAnimatorController = aoc;
			transform.position = Vector3.zero;
			transform.rotation = Quaternion.identity;
			Animator.SetFloat("Speed", Speed);
			Animator.Play("Animation", 0, 0f);
		}
	}

	public void StopBake() {
		if(Application.isPlaying) {
			StopAllCoroutines();
			Animator.SetFloat("Speed", 0f);
			Baking = false;
		}
	}

	public void SaveBake() {
		if(Application.isPlaying) {
			string destination = "Assets/" + Destination;
			if(!AssetDatabase.IsValidFolder(destination)) {
				Debug.Log("Folder " + "'" + destination + "'" + " is not valid.");
			} else {
				if(AssetDatabase.LoadAssetAtPath(destination+"/"+Name+".asset", typeof(MotionData)) == null) {
					MotionData data = ScriptableObject.CreateInstance<MotionData>();

					//Assign Name
					data.Name = Name;

					AssetDatabase.CreateAsset(data , destination+"/"+data.Name+".asset");

					//Create Source Data
					data.Source = new MotionData.Hierarchy();
					for(int i=0; i<Actor.Bones.Length; i++) {
						data.Source.AddBone(Actor.Bones[i].GetName(), Actor.Bones[i].GetParent() == null ? "None" : Actor.Bones[i].GetParent().GetName());
					}

					//Set Frames
					ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt((float)Framerate * GetRecordedTime()));

					//Set Framerate
					data.Framerate = (float)Framerate;

					//Compute Frames
					List<Sample> frames = Resample();
					for(int i=0; i<frames.Count; i++) {
						data.Frames[i] = new Frame(data, i+1, frames[i].Timestamp);
						for(int j=0; j<Actor.Bones.Length; j++) {
							data.Frames[i].Local[j] = Matrix4x4.TRS(frames[i].LocalPosture[j].GetPosition(), frames[i].LocalPosture[j].GetRotation() * Mapping[j], Vector3.one);
							data.Frames[i].World[j] = Matrix4x4.TRS(frames[i].WorldPosture[j].GetPosition(), frames[i].WorldPosture[j].GetRotation() * Mapping[j], Vector3.one);
						}
					}

					//Finalise
					data.DetectSymmetry();
					data.AddSequence();

					EditorUtility.SetDirty(data);
					AssetDatabase.SaveAssets();
					AssetDatabase.Refresh();
				} else {
					Debug.Log("File with name " + Animation.name + " already exists.");
				}
			}
		}
	}

	public void ComputeMapping() {
		Mapping = new Quaternion[Actor.Bones.Length];
		for(int i=0; i<Actor.Bones.Length; i++) {
			Mapping[i] = Retarget.Bones[i].Transform.rotation.GetRelativeRotationTo(Actor.Bones[i].Transform.GetWorldMatrix());
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
			Matrix4x4[] local = new Matrix4x4[Actor.Bones.Length];
			Matrix4x4[] world = new Matrix4x4[Actor.Bones.Length];
			for(int j=0; j<Actor.Bones.Length; j++) {
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
			Target.Actor = (Actor)EditorGUILayout.ObjectField("Actor", Target.Actor, typeof(Actor), true);
			Target.Retarget = (Actor)EditorGUILayout.ObjectField("Retarget", Target.Retarget, typeof(Actor), true);
			Target.Animator = (Animator)EditorGUILayout.ObjectField("Animator", Target.Animator, typeof(Animator), true);
			Target.Animation = (AnimationClip)EditorGUILayout.ObjectField("Animation", Target.Animation, typeof(AnimationClip), true);
			Target.Name = EditorGUILayout.TextField("Name", Target.Name);
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Target.Destination = EditorGUILayout.TextField(Target.Destination);
			EditorGUILayout.EndHorizontal();
			Target.Speed = EditorGUILayout.FloatField("Speed", Target.Speed);
			Target.Framerate = EditorGUILayout.IntField("Framerate", Target.Framerate);
			EditorGUI.EndDisabledGroup();

			EditorGUILayout.LabelField("Timestamp: " + Target.GetTimestamp());
			EditorGUILayout.LabelField("Recorded Samples: " + Target.Samples.Count);
			EditorGUILayout.LabelField("Recorded Time: " + Target.GetRecordedTime());

			if(!Target.Baking) {
				if(Utility.GUIButton("Bake", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.StartCoroutine(Target.Bake());
				}
			} else {
				if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.StopBake();
				}
			}
			if(Utility.GUIButton("Compute Mapping", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.ComputeMapping();
			}
			if(Utility.GUIButton("Save", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.SaveBake();
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}

}
#endif