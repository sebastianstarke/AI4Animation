/*
#if UNITY_EDITOR
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MotionEditor))]
public class StateEditor : MonoBehaviour {

	public List<State> States = new List<State>();

	private MotionEditor Editor;

	private MotionEditor GetEditor() {
		if(Editor == null) {
			Editor = GetComponent<MotionEditor>();
		}
		return Editor;
	}
	
	public void AddState() {
		States.Add(new State(GetEditor().Data.GetFrame(Editor.GetState().Index)));
	}

	public void ClearStates() {
		States.Clear();
	}

	public float[] GetStateVector(MotionData.Frame frame) {
		float[] losses = new float[States.Count];
		float min = float.MaxValue;
		float max = float.MinValue;
		float[] feature = GetFeature(frame);
		for(int i=0; i<losses.Length; i++) {
			losses[i] = GetDistance(GetFeature(States[i].Frame), feature);
			min = Mathf.Min(losses[i], min);
			max = Mathf.Max(losses[i], max);
		}
		float[] scores = new float[States.Count];
		for(int i=0; i<scores.Length; i++) {
			scores[i] = Utility.Normalise(losses[i], min, max, 1f, 0f);
		}
		Utility.SoftMax(ref scores);
		return scores;
	}

		public float[] GetFeature(MotionData.Frame frame) {
			int dim = 6*frame.Data.Source.Bones.Length;
			float[] feature = new float[dim];
			int pivot = 0;
			Matrix4x4 root = frame.GetRootTransformation(false);
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneTransformation(i, false).GetRelativeTransformationTo(root).GetPosition().x;
			}
			pivot += frame.Data.Source.Bones.Length;
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneTransformation(i, false).GetRelativeTransformationTo(root).GetPosition().x;
			}
			pivot += frame.Data.Source.Bones.Length;
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneTransformation(i, false).GetRelativeTransformationTo(root).GetPosition().x;
			}
			pivot += frame.Data.Source.Bones.Length;
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneVelocity(i, false).GetRelativeDirectionTo(root).x;
			}
			pivot += frame.Data.Source.Bones.Length;
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneVelocity(i, false).GetRelativeDirectionTo(root).y;
			}
			pivot += frame.Data.Source.Bones.Length;
			for(int i=0; i<frame.Data.Source.Bones.Length; i++) {
				feature[pivot + i] = frame.GetBoneVelocity(i, false).GetRelativeDirectionTo(root).z;
			}
			pivot += frame.Data.Source.Bones.Length;
			return feature;
		}

		public float GetDistance(float[] from, float[] to) {
			float distance = 0f;
			for(int i=0; i<from.Length; i++) {
				distance += Mathf.Pow(from[i] - to[i], 2f);
			}
			distance = Mathf.Sqrt(distance);
			return distance;
		}

	void Draw() {
		UltiDraw.Begin();

		float window = 0.25f;
		MotionData.Frame[] frames = GetEditor().Data.GetFrames(Mathf.Clamp(GetEditor().GetState().Timestamp - window/2f, 0f, GetEditor().Data.GetTotalTime()), Mathf.Clamp(GetEditor().GetState().Timestamp + window/2f, 0f, GetEditor().Data.GetTotalTime()));

		List<float[]> values = new List<float[]>();
		for(int i=0; i<States.Count; i++) {
			values.Add(new float[frames.Length]);
		}
		for(int i=0; i<frames.Length; i++) {
			float[] stateVector = GetStateVector(frames[i]);
			for(int j=0; j<States.Count; j++) {
				values[j][i] = stateVector[j];
			}
		}

		UltiDraw.DrawGUIFunctions(0.5f*Vector2.one, Vector2.one, values, 0f, 1f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(States.Count));

		UltiDraw.End();
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	[System.Serializable]
	public class State {
		public MotionData.Frame Frame;

		public State(MotionData.Frame frame) {
			Frame = frame;
		}
	}

	[CustomEditor(typeof(StateEditor))]
	public class StateEditor_Editor : Editor {

		public StateEditor Target;

		void Awake() {
			Target = (StateEditor)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			EditorGUILayout.IntField("States", Target.States.Count);

			if(Utility.GUIButton("Add State", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.AddState();
			}

			if(Utility.GUIButton("Clear States", UltiDraw.DarkGrey, UltiDraw.White)) {
				Target.ClearStates();
			}

			float[] state = Target.GetStateVector(Target.GetEditor().Data.GetFrame(Target.GetEditor().GetState().Index));
			for(int i=1; i<=state.Length; i++) {
				EditorGUILayout.FloatField("State " + i, state[i-1]);
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
	}

}
#endif
*/