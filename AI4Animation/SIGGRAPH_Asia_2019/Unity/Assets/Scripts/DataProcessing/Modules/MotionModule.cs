#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class MotionModule : Module {

	public bool[] Bones = new bool[0];
	public float[] Velocities = new float[0];
	//public float DistanceThreshold = 0.05f;
	public float VelocityThreshold = 0.1f;
	//public LayerMask Mask = -1;

	public override ID GetID() {
		return ID.Motion;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Bones = new bool[data.Source.Bones.Length];
		Compute();
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		
	}

	public void ToggleBone(int index) {
		Bones[index] = !Bones[index];
		Compute();
	}

	public bool[] GetState(Frame frame, bool mirrored) {
		List<bool> state = new List<bool>();
		for(int i=0; i<Bones.Length; i++) {
			if(Bones[i]) {
				state.Add(
					//(DistanceThreshold == 0f || Physics.CheckSphere(frame.GetBoneTransformation(i, mirrored).GetPosition(), DistanceThreshold, Mask))
					//&& 
					(frame.GetBoneVelocity(i, mirrored, 1f/Data.Framerate).magnitude <= VelocityThreshold)
				);
			}
		}
		return state.ToArray();
	}

	private void FindKey(MotionEditor editor, int increment) {
		Frame seed = editor.GetCurrentFrame();
		for(int i=seed.Index+increment; increment==1 ? i<=Data.GetTotalFrames() : i>=1; i+=increment) {
			Frame current = Data.GetFrame(i);
			Frame previous = Data.GetFrame(i-1);
			Frame next = Data.GetFrame(i+1);
			if(previous != null && next != null) {
				bool[] currentState = GetState(current, editor.Mirror);
				bool[] previousState = GetState(previous, editor.Mirror);
				bool[] nextState = GetState(next, editor.Mirror);
				if(AllTrue(currentState)) {
					if(AllTrue(previousState) && !AllTrue(nextState)) {
						editor.LoadFrame(i);
						break;
					}
					if(!AllTrue(previousState) && AllTrue(nextState)) {
						editor.LoadFrame(i);
						break;
					}
				}
				if(AllFalse(currentState)) {
					if(AllFalse(previousState) && !AllFalse(nextState)) {
						editor.LoadFrame(i);
						break;
					}
					if(!AllFalse(previousState) && AllFalse(nextState)) {
						editor.LoadFrame(i);
						break;
					}
				}
			}
		}
	}

	private void Compute() {
		Velocities = new float[Data.GetTotalFrames()];
		float min = float.MaxValue;
		float max = 0f;
		for(int i=0; i<Velocities.Length; i++) {
			float velocity = 0f;
			for(int j=0; j<Bones.Length; j++) {
				if(Bones[j]) {
					velocity = Mathf.Max(velocity, Data.Frames[i].GetBoneVelocity(j, false, 1f/Data.Framerate).magnitude);
				}
			}
			min = Mathf.Min(min, velocity);
			max = Mathf.Max(max, velocity);
			Velocities[i] = velocity;
		}
		for(int i=0; i<Velocities.Length; i++) {
			Velocities[i] = Utility.Normalise(Velocities[i], min, max, 0f, 1f);
		}
	}

	private bool AllTrue(bool[] values) {
		for(int i=0; i<values.Length; i++) {
			if(values[i] == false) {
				return false;
			}
		}
		return true;
	}

	private bool AllFalse(bool[] values) {
		for(int i=0; i<values.Length; i++) {
			if(values[i] == true) {
				return false;
			}
		}
		return true;
	}

	protected override void DerivedDraw(MotionEditor editor) {
		UltiDraw.Begin();
		bool[] state = GetState(editor.GetCurrentFrame(), editor.Mirror);
		int index = 0;
		for(int i=0; i<Bones.Length; i++) {
			if(Bones[i]) {
				UltiDraw.DrawSphere(editor.GetCurrentFrame().GetBoneTransformation(i, editor.Mirror).GetPosition(), Quaternion.identity, 0.05f, state[index] ? UltiDraw.Red.Transparent(0.5f) : UltiDraw.White.Transparent(0.25f));
				index += 1;
			}
		}
		UltiDraw.End();
	}

	public void Fix() {
		if(Velocities == null || Velocities.Length != Data.GetTotalFrames()) {
			Compute();
		}
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Fix();

		//DistanceThreshold = EditorGUILayout.FloatField("Distance Threshold", DistanceThreshold);
		VelocityThreshold = EditorGUILayout.FloatField("Velocity Threshold", VelocityThreshold);
		//Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
		EditorGUILayout.BeginHorizontal();
		if(Utility.GUIButton("Active All", UltiDraw.DarkGrey, UltiDraw.White)) {
			for(int i=0; i<Bones.Length; i++) {
				Bones[i] = true;
			}
			Compute();
		}
		if(Utility.GUIButton("Active None", UltiDraw.DarkGrey, UltiDraw.White)) {
			for(int i=0; i<Bones.Length; i++) {
				Bones[i] = false;
			}
			Compute();
		}
		if(Utility.GUIButton("Active Setup", UltiDraw.DarkGrey, UltiDraw.White)) {
			for(int i=0; i<Bones.Length; i++) {
				Bones[i] = editor.GetActor().FindBone(Data.Source.Bones[i].Name) != null;
			}
			Compute();
		}
		EditorGUILayout.EndHorizontal();
		EditorGUILayout.BeginHorizontal();
		GUILayout.FlexibleSpace();
		for(int i=0; i<Bones.Length; i++) {
			if(Utility.GUIButton(Data.Source.Bones[i].Name, Bones[i] ? UltiDraw.Cyan : UltiDraw.Grey, Bones[i] ? UltiDraw.DarkGrey : UltiDraw.LightGrey, EditorGUIUtility.currentViewWidth*0.075f, 20f)) {
				ToggleBone(i);
			}
			if(i%10 == 9) {
				GUILayout.FlexibleSpace();
				EditorGUILayout.EndHorizontal();
				EditorGUILayout.BeginHorizontal();
				GUILayout.FlexibleSpace();
			}
		}
		GUILayout.FlexibleSpace();
		EditorGUILayout.EndHorizontal();

		EditorGUILayout.BeginHorizontal();
		UltiDraw.Begin();
		if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			FindKey(editor, -1);
		}

		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		Vector3Int view = editor.GetView();

		Vector3 prevPos = Vector3.zero;
		Vector3 newPos = Vector3.zero;
		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

		for(int i=1; i<view.z; i++) {
			prevPos.x = rect.xMin + (float)(i-1)/(view.z-1) * rect.width;
			prevPos.y = rect.yMax - Velocities[i+view.x-1] * rect.height;
			newPos.x = rect.xMin + (float)(i)/(view.z-1) * rect.width;
			newPos.y = rect.yMax - Velocities[i+view.x] * rect.height;
			UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Gold);
		}

		/*
		//Seconds
		float timestamp = startTime;
		while(timestamp <= endTime) {
			float floor = Mathf.FloorToInt(timestamp);
			if(floor >= startTime && floor <= endTime) {
				top.x = rect.xMin + (float)(Data.GetFrame(floor).Index-start)/elements * rect.width;
				UltiDraw.DrawWireCircle(top, 5f, UltiDraw.White);
			}
			timestamp += 1f;
		}
		//
		*/
		UltiDraw.End();

		editor.DrawPivot(rect);

		Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
		EditorGUILayout.EndVertical();

		if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			FindKey(editor, 1);
		}
		EditorGUILayout.EndHorizontal();

	}

}
#endif
