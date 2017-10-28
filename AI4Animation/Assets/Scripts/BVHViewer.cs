using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class BVHViewer : MonoBehaviour {

	public float UnitScale = 10f;
	public string Path = string.Empty;

	private Transform[] Bones = new Transform[0];
	private Keyframe[] Keyframes = new Keyframe[0];
	private Keyframe CurrentKeyframe = null;

	private int TotalFrames = 0;
	private float Timescale = 1f;
	private float TotalTime = 0f;
	private float FrameTime = 0f;
	private float PlayTime = 0f;
	private bool Loaded = false;
	private bool Playing = false;

	private bool ShowCapture = false;

	public BVHViewer() {
		EditorApplication.update += EditorUpdate;
	}

	public void EditorUpdate() {
		if(EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling) {
			Unload();
			return;
		}
		if(Playing) {
			//Debug.Log(Time.deltaTime);
			PlayTime += Timescale*Time.deltaTime;
			if(PlayTime > TotalTime) {
				PlayTime -= TotalTime;
			}
			ShowKeyframe(PlayTime);
		}
	}

	public void Load(string path) {
		if(Application.isPlaying) {
			return;
		}
		Unload();

		string[] lines = File.ReadAllLines(Path);
		char[] whitespace = new char[] {' '};
		int index = 0;

		//Build Hierarchy
		List<Transform> bones = new List<Transform>();
		List<int[]> channels = new List<int[]>();
		List<Transformation> zero = new List<Transformation>();
		Transform current = transform;
		for(index = 0; index<lines.Length; index++) {
			if(lines[index] == "MOTION") {
				break;
			}
			string[] entries = lines[index].Split(whitespace);
			for(int entry=0; entry<entries.Length; entry++) {
				//Debug.Log("Line " + index + " Entry " + entry + ": " + entries[entry]);
				if(entries[entry].Contains("ROOT")) {
					current.name = entries[entry+1];
					bones.Add(transform);
					break;
				} else if(entries[entry].Contains("JOINT")) {
					Transform t = new GameObject(entries[entry+1]).transform;
					t.SetParent(current, false);
					current = t;
					bones.Add(t);
					break;
				} else if(entries[entry].Contains("End")) {
					index += 3;
					break;
				} else if(entries[entry].Contains("OFFSET")) {
					zero.Add(new Transformation(
						new Vector3(ReadFloat(entries[entry+1]), ReadFloat(entries[entry+2]), ReadFloat(entries[entry+3])), 
						Quaternion.identity
						));
					break;
				} else if(entries[entry].Contains("CHANNELS")) {
					int dimensions = ReadInt(entries[entry+1]);
					int[] channel = new int[dimensions+1];
					channel[0] = dimensions;
					for(int i=0; i<dimensions; i++) {
						if(entries[entry+2+i] == "Xposition") {
							channel[1+i] = 1;
						} else if(entries[entry+2+i] == "Yposition") {
							channel[1+i] = 2;
						} else if(entries[entry+2+i] == "Zposition") {
							channel[1+i] = 3;
						} else if(entries[entry+2+i] == "Xrotation") {
							channel[1+i] = 4;
						} else if(entries[entry+2+i] == "Yrotation") {
							channel[1+i] = 5;
						} else if(entries[entry+2+i] == "Zrotation") {
							channel[1+i] = 6;
						}
					}
					channels.Add(channel);
					break;
				} else if(entries[entry].Contains("}")) {
					if(current != transform) {
						current = current.parent;
					}
					break;
				}
			}
		}
		Bones = bones.ToArray();

		//Skip frame count
		index += 1;
		TotalFrames = ReadInt(lines[index].Substring(8));

		//Read frame time
		index += 1;
		FrameTime = ReadFloat(lines[index].Substring(12));

		//Compute total time
		TotalTime = TotalFrames * FrameTime;

		//Read motions
		index += 1;
		float[][] motions = new float[TotalFrames][];
		for(int i=index; i<lines.Length; i++) {
			motions[i-index] = ReadArray(lines[i]);
		}

		//Build Keyframes
		System.Array.Resize(ref Keyframes, TotalFrames);
		for(int i=0; i<TotalFrames; i++) {
			Keyframes[i] = new Keyframe(Bones, zero, channels, motions[i], i+1, i*FrameTime, UnitScale);
		}

		//Load First Keyframe
		ShowKeyframe(1);

		Loaded = true;
	}

	public void Unload() {
		if(!Loaded) {
			return;
		}

		transform.name = "Root";
		while(transform.childCount > 0) {
			DestroyImmediate(transform.GetChild(0).gameObject);
		}
		System.Array.Resize(ref Bones, 0);
		System.Array.Resize(ref Keyframes, 0);
		CurrentKeyframe = null;

		TotalFrames = 0;
		Timescale = 1f;
		TotalTime = 0f;
		FrameTime = 0f;
		PlayTime = 0f;

		Loaded = false;
		Playing = false;
		ShowCapture = false;
	}

	public bool IsLoaded() {
		return Loaded;
	}

	public bool IsPlaying() {
		return Playing;
	}

	public Transform[] GetBones() {
		return Bones;
	}

	public Keyframe[] GetKeyframes() {
		return Keyframes;
	}

	public Keyframe GetCurrentKeyframe() {
		return CurrentKeyframe;
	}

	public int GetTotalFrames() {
		return TotalFrames;
	}

	public void SetTimescale(float value) {
		Timescale = value;
	}

	public float GetTimescale() {
		return Timescale;
	}

	public float GetTotalTime() {
		return TotalTime;
	}

	public float GetFrameTime() {
		return FrameTime;
	}

	public void SetPlayTime(float time) {
		PlayTime = time;
	}

	public float GetPlayTime() {
		return PlayTime;
	}

	public void DrawCapture(bool value) {
		ShowCapture = value;
	}

	public bool IsDrawingCapture() {
		return ShowCapture;
	}

	public void Play() {
		Playing = true;
	}

	public void Stop() {
		Playing = false;
	}

	public void ShowKeyframe(Keyframe keyframe) {
		if(keyframe == null) {
			return;
		}
		if(CurrentKeyframe == keyframe) {
			return;
		}
		CurrentKeyframe = keyframe;
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].position = CurrentKeyframe.Positions[i];
			Bones[i].rotation = CurrentKeyframe.Rotations[i];
		}
	}

	public void ShowKeyframe(int index) {
		if(CurrentKeyframe != null) {
			if(CurrentKeyframe.Index == index) {
				return;
			}
		}
		CurrentKeyframe = GetKeyframe(index);
		if(CurrentKeyframe == null) {
			return;
		}
		//PlayTime = CurrentKeyframe.Timestamp;
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].position = CurrentKeyframe.Positions[i];
			Bones[i].rotation = CurrentKeyframe.Rotations[i];
		}
	}

	public void ShowKeyframe(float time) {
		if(CurrentKeyframe != null) {
			if(CurrentKeyframe.Timestamp == time) {
				return;
			}
		}
		CurrentKeyframe = GetKeyframe(time);
		if(CurrentKeyframe == null) {
			return;
		}
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].position = CurrentKeyframe.Positions[i];
			Bones[i].rotation = CurrentKeyframe.Rotations[i];
		}
	}

	public Keyframe GetKeyframe(int index) {
		if(index < 1 || index > TotalFrames) {
			Debug.Log("Please specify an index between 1 and " + TotalFrames + ".");
			return null;
		}
		return Keyframes[index-1];
	}

	public Keyframe GetKeyframe(float time) {
		if(time < 0f || time > TotalTime) {
			Debug.Log("Please specify a time between 0 and " + TotalTime + ".");
			return null;
		}
		return GetKeyframe(Mathf.Min(Mathf.RoundToInt(time / FrameTime) + 1, TotalFrames));
	}

	public static int ReadInt(string value) {
		value = FilterValueField(value);
		return ParseInt(value);
	}

	public static float ReadFloat(string value) {
		value = FilterValueField(value);
		return ParseFloat(value);
	}

	public static float[] ReadArray(string value) {
		value = FilterValueField(value);
		if(value.StartsWith(" ")) {
			value = value.Substring(1);
		}
		if(value.EndsWith(" ")) {
			value = value.Substring(0, value.Length-1);
		}
		string[] values = value.Split(' ');
		float[] array = new float[values.Length];
		for(int i=0; i<array.Length; i++) {
			array[i] = ParseFloat(values[i]);
		}
		return array;
	}

	public static Vector3 ReadVector3(string value) {
		value = FilterValueField(value);
		string[] values = value.Split(' ');
		float x = ParseFloat(values[0]);
		float y = ParseFloat(values[1]);
		float z = ParseFloat(values[2]);
		return new Vector3(x,y,z);
	}

	public static Vector4 ReadColor(string value) {
		value = FilterValueField(value);
		string[] values = value.Split(' ');
		float r = ParseFloat(values[0]);
		float g = ParseFloat(values[1]);
		float b = ParseFloat(values[2]);
		float a = ParseFloat(values[3]);
		return new Color(r,g,b,a);
	}

	public static string FilterValueField(string value) {
		while(value.Contains("  ")) {
			value = value.Replace("  "," ");
		}
		while(value.Contains("< ")) {
			value = value.Replace("< ","<");
		}
		while(value.Contains(" >")) {
			value = value.Replace(" >",">");
		}
		while(value.Contains(" .")) {
			value = value.Replace(" ."," 0.");
		}
		while(value.Contains(". ")) {
			value = value.Replace(". ",".0");
		}
		while(value.Contains("<.")) {
			value = value.Replace("<.","<0.");
		}
		return value;
	}

	public static int ParseInt(string value) {
		int parsed = 0;
		if(int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + "!");
			return 0;
		}
	}

	public static float ParseFloat(string value) {
		float parsed = 0f;
		if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + "!");
			return 0f;
		}
	}

	public class Keyframe {
		public int Index;
		public float Timestamp;
		public Vector3[] Positions;
		public Quaternion[] Rotations;

		public Keyframe(Transform[] bones, List<Transformation> zero, List<int[]> channels, float[] motion, int index, float timestamp, float unitScale) {
			Index = index;
			Timestamp = timestamp;
			Positions = new Vector3[bones.Length];
			Rotations = new Quaternion[bones.Length];
			//Forward Kinematics
			int channel = 0;
			for(int i=0; i<bones.Length; i++) {
				if(i == 0) {
					Vector3 position = new Vector3(motion[channel+0], motion[channel+1], motion[channel+2]);
					Quaternion rotation =
						GetAngleAxis(motion[channel+3], channels[i][4]) *
						GetAngleAxis(motion[channel+4], channels[i][5]) *
						GetAngleAxis(motion[channel+5], channels[i][6]);
					bones[i].localPosition = position / unitScale;
					bones[i].localRotation = rotation;
					channel += 6;
				} else {
					Quaternion rotation =
						GetAngleAxis(motion[channel+0], channels[i][1]) *
						GetAngleAxis(motion[channel+1], channels[i][2]) *
						GetAngleAxis(motion[channel+2], channels[i][3]);
					bones[i].localPosition = zero[i].Position / unitScale;
					bones[i].localRotation = zero[i].Rotation * rotation;
					channel += 3;
				}
			}
			for(int i=0; i<bones.Length; i++) {
				Positions[i] = bones[i].position;
				Rotations[i] = bones[i].rotation;
			}
		}

		private Quaternion GetAngleAxis(float angle, int axis) {
			if(axis == 4) {
				return Quaternion.AngleAxis(angle, Vector3.right);
			}
			if(axis == 5) {
				return Quaternion.AngleAxis(angle, Vector3.up);
			}
			if(axis == 6) {
				return Quaternion.AngleAxis(angle, Vector3.forward);
			}
			return Quaternion.identity;
		}
	}

	void OnDrawGizmos() {
		if(Loaded) {
			if(ShowCapture) {
				DrawCapture();
			}
			DrawSkeleton(transform, Color.cyan, Color.black);
		}
	}

	private void DrawCapture() {
		Keyframe frame = CurrentKeyframe;
		float step = 2f;
		for(float i = 0f; i<=TotalTime; i+=step) {
			ShowKeyframe(i);
			Color boneColor = Color.Lerp(Color.red, Color.green, i/TotalTime);
			boneColor.a = 0.25f;
			Color jointColor = Color.black;
			jointColor.a = 0.25f;
			DrawSkeleton(transform, boneColor, jointColor);
		}
		ShowKeyframe(frame);
	}

	private void DrawSkeleton(Transform t, Color boneColor, Color jointColor) {
		Gizmos.color = boneColor;
		for(int i=0; i<t.childCount; i++) {
			Gizmos.DrawLine(t.position, t.GetChild(i).position);
		}
		Gizmos.color = jointColor;
		Gizmos.DrawSphere(t.position, 0.025f);
		for(int i=0; i<t.childCount; i++) {
			DrawSkeleton(t.GetChild(i), boneColor, jointColor);
		}
	}

}
