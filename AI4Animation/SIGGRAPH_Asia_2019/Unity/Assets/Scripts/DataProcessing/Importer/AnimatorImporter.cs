#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[RequireComponent(typeof(Animator))]
public class AnimatorImporter : MonoBehaviour {

	public AnimationClip[] Animations = new AnimationClip[0];
	public bool[] Import = new bool[0];
	public Actor Retarget = null;
	public Actor Skeleton = null;
	public float Speed = 1f;
	public int Framerate = 60;
	public int Smoothing = 0;
	public string Source = string.Empty;
	public string Destination = string.Empty;
	public bool Demo = false;

	public int[] Mapping = new int[0];
	public Vector3[] Delta = new Vector3[0];
	public Vector3[] Corrections = new Vector3[0];
	
	private Animator Animator = null;
	private bool Baking = false;
	private List<Sample> Samples = new List<Sample>();

	private AnimationClip Current = null;

	public Animator GetAnimator() {
		if(Animator == null) {
			Animator = GetComponent<Animator>();
		}
		return Animator;
	}

	public void LoadAnimations() {
		Animations = new AnimationClip[0];
		Import = new bool[0];
		foreach(string file in System.IO.Directory.GetFiles(Source)) {
			AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(file, typeof(AnimationClip));
			if(clip != null) {
				ArrayExtensions.Add(ref Animations, clip);
				ArrayExtensions.Add(ref Import, true);
			}
		}
	}

	private void PostProcess() {
		Matrix4x4[] posture = Skeleton.GetBoneTransformations();
		for(int i=0; i<posture.Length; i++) {
			posture[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Delta[i]), Vector3.one);
			posture[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Corrections[i]), Vector3.one);
		}
		for(int i=0; i<Retarget.Bones.Length; i++) {
			Retarget.Bones[Mapping[i]].Transform.position = posture[i].GetPosition();
			Retarget.Bones[Mapping[i]].Transform.rotation = posture[i].GetRotation();
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
					if(Import[k]) {
						string name = Animations[k].name + k;
						if(!Directory.Exists(destination+"/"+name) ) {
							Current = Animations[k];

							Skeleton.transform.position = Vector3.zero;
							Skeleton.transform.rotation = Quaternion.identity;
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

							Samples.Clear();
							while(GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime < 1f) {
								PostProcess();
								StoreSample();
								yield return new WaitForEndOfFrame();
							}
							PostProcess();
							StoreSample();
							yield return new WaitForEndOfFrame();

							if(!Demo) {
								//Save Bake
								AssetDatabase.CreateFolder(destination, name.Substring(name.IndexOf("|")+1));
								MotionData data = ScriptableObject.CreateInstance<MotionData>();
								data.name = "Data";
								AssetDatabase.CreateAsset(data, destination+"/"+name+"/"+data.name+".asset");

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
										data.Frames[i].World[j] = frames[i].WorldPosture[j];
									}
								}

								//Detect Symmetry
								data.DetectSymmetry();

								//Add Scene
								data.CreateScene();
								data.AddSequence();

								//Smooth Motion
								if(Smoothing > 0) {
									Debug.Log("Smoothing ignored.");
									//Debug.Log("Applying smoothing to " + data.name + ".");
									//data.SmoothMotion(Smoothing);
								}

								EditorUtility.SetDirty(data);
							}

							//Stop Bake
							GetAnimator().speed = 0f;
							yield return new WaitForEndOfFrame();
						} else {
							Debug.Log("File with name " + name + " already exists.");
						}
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

	public void StoreSample() {
		if(!Demo) {
			Samples.Add(new Sample(GetTimestamp(), Retarget.GetBoneTransformations()));
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
		if(!Application.isPlaying || Animator == null || Retarget == null) {
			return 0f;
		}
		return GetAnimator().GetCurrentAnimatorStateInfo(0).normalizedTime;
	}

	public float GetTimestamp() {
		if(!Application.isPlaying || Animator == null || Retarget == null) {
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
				if(Samples[i-1].Timestamp == Samples[i].Timestamp) {
					Debug.Log("Generated samples with identical timestamps.");
				} else {
					samples += 1;
					fps += 1f / (Samples[i].Timestamp - Samples[i-1].Timestamp);
				}
			}
		}
		return samples == 0 ? 0f : (fps / samples);
	}

	public void DetectMapping() {
		if(Retarget != null & Skeleton != null) {
			if(!Application.isPlaying && transform.hasChanged) {
				Mapping = new int[Retarget.Bones.Length];
				for(int i=0; i<Retarget.Bones.Length; i++) {
					Mapping[i] = i;
				}
			}
		}
	}

	public void ComputeDelta() {
		if(Retarget != null & Skeleton != null) {
			if(!Application.isPlaying && transform.hasChanged) {
				Delta = new Vector3[Retarget.Bones.Length];
				for(int i=0; i<Retarget.Bones.Length; i++) {
					Delta[i] = Retarget.Bones[Mapping[i]].Transform.rotation.GetRelativeRotationTo(Skeleton.Bones[i].Transform.GetWorldMatrix()).eulerAngles;
				}
			}
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
			Matrix4x4[] world = new Matrix4x4[Retarget.Bones.Length];
			for(int j=0; j<Retarget.Bones.Length; j++) {
				world[j] = Utility.Interpolate(a.WorldPosture[j], b.WorldPosture[j], weight);
			}
			samples.Add(new Sample(timestamps[i], world));
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
		public Sample(float timestamp, Matrix4x4[] world) {
			Timestamp = timestamp;
			WorldPosture = world;
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

			if(Utility.GUIButton("Load Animations", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.LoadAnimations();
			}

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.IntField("Animations", Target.Animations.Length);
				for(int i=0; i<Target.Animations.Length; i++) {
					Target.Import[i] = EditorGUILayout.Toggle(Target.Animations[i].name, Target.Import[i]);
				}
			}
			EditorGUI.EndDisabledGroup();
			Target.Skeleton = (Actor)EditorGUILayout.ObjectField("Skeleton", Target.Skeleton, typeof(Actor), true);
			Target.Retarget = (Actor)EditorGUILayout.ObjectField("Retarget", Target.Retarget, typeof(Actor), true);
			Target.Source = EditorGUILayout.TextField("Source", Target.Source);
			Target.Destination = EditorGUILayout.TextField("Destination", Target.Destination);
			Target.Speed = EditorGUILayout.FloatField("Speed", Target.Speed);
			Target.Framerate = EditorGUILayout.IntField("Framerate", Target.Framerate);
			Target.Smoothing = EditorGUILayout.IntField("Smoothing", Target.Smoothing);
			Target.Demo = EditorGUILayout.Toggle("Demo", Target.Demo);

			EditorGUILayout.BeginHorizontal();
			if(Target.Corrections.Length != Target.Skeleton.Bones.Length) {
				Target.Corrections = new Vector3[Target.Skeleton.Bones.Length];
			}
			if(Target.Mapping.Length != Target.Skeleton.Bones.Length) {
				Target.Mapping = new int[Target.Skeleton.Bones.Length];
			}
			if(Target.Delta.Length != Target.Skeleton.Bones.Length) {
				Target.Delta = new Vector3[Target.Skeleton.Bones.Length];
			}
			if(Utility.GUIButton("Detect Mapping", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.DetectMapping();
			}
			if(Utility.GUIButton("Compute Delta", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.ComputeDelta();
			}
			EditorGUILayout.EndHorizontal();

			string[] skeletonNames = Target.Skeleton.GetBoneNames();
			string[] retargetNames = Target.Retarget.GetBoneNames();
			for(int i=0; i<Target.Skeleton.Bones.Length; i++) {
				Utility.SetGUIColor(Target.Corrections[i] != Vector3.zero ? UltiDraw.Cyan : UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.TextField(skeletonNames[i]);
					EditorGUI.EndDisabledGroup();
					Target.Mapping[i] = EditorGUILayout.Popup(Target.Mapping[i], retargetNames);
					EditorGUILayout.EndHorizontal();
					Target.Delta[i] = EditorGUILayout.Vector3Field("Delta", Target.Delta[i]);
					Target.Corrections[i] = EditorGUILayout.Vector3Field("Correction", Target.Corrections[i]);
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