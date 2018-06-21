#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class StyleModule : Module {

	public float Transition = 0.5f;
	public StyleFunction[] Functions = new StyleFunction[0];

	public override TYPE Type() {
		return TYPE.Style;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		Transition = 0.5f;
		Functions = new StyleFunction[0];
		return this;
	}

	public void SetTransition(float value) {
		if(Transition != value) {
			Transition = value;
			for(int i=0; i<Functions.Length; i++) {
				for(int j=1; j<=Data.GetTotalFrames(); j++) {
					if(Functions[i].IsStyleKey(Data.GetFrame(j))) {
						Functions[i].ApplyStyleKeyValues(Data.GetFrame(j));;
					}
				}
			}
		}
	}

	public void AddStyle(string name) {
		if(System.Array.Exists(Functions, x => x.Name == name)) {
			Debug.Log("Style with name " + name + " already exists.");
			return;
		}
		ArrayExtensions.Add(ref Functions, new StyleFunction(this, name));
	}

	public void RemoveStyle() {
		ArrayExtensions.Shrink(ref Functions);
	}

	public void RemoveStyle(string name) {
		int index = System.Array.FindIndex(Functions, x => x.Name == name);
		if(index >= 0) {
			ArrayExtensions.RemoveAt(ref Functions, index);
		} else {
			Debug.Log("Style with name " + name + " does not exist.");
		}
	}

	public float[] GetStyle(Frame frame) {
		float[] style = new float[Functions.Length];
		for(int i=0; i<style.Length; i++) {
			style[i] = Functions[i].GetValue(frame);
		}
		return style;
	}

	public Frame GetAnyNextStyleKey(Frame frame) {
		while(frame.Index < Data.GetTotalFrames()) {
			frame = frame.GetNextFrame();
			if(IsAnyStyleKey(frame)) {
				return frame;
			}
		}
		return null;
	}

	public Frame GetAnyPreviousStyleKey(Frame frame) {
		while(frame.Index > 1) {
			frame = frame.GetPreviousFrame();
			if(IsAnyStyleKey(frame)) {
				return frame;
			}
		}
		return null;
	}

	public bool IsAnyStyleKey(Frame frame) {
		for(int i=0; i<Functions.Length; i++) {
			if(Functions[i].IsStyleKey(frame)) {
				return true;
			}
		}
		return false;
	}

	public bool IsAnyTransition(Frame frame) {
		for(int i=0; i<Functions.Length; i++) {
			if(Functions[i].IsTransition(frame)) {
				return true;
			}
		}
		return false;
	}

	public override void Draw(MotionEditor editor) {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		Frame frame = Data.GetFrame(editor.GetState().Index);

		EditorGUILayout.BeginHorizontal();
		SetTransition(EditorGUILayout.Slider("Transition", Transition, 0.1f, 1f));
		if(Utility.GUIButton("Add Style", UltiDraw.DarkGrey, UltiDraw.White)) {
			AddStyle("Style");
		}
		if(Utility.GUIButton("Remove Style", UltiDraw.DarkGrey, UltiDraw.White)) {
			RemoveStyle();
		}
		EditorGUILayout.EndHorizontal();

		Color[] colors = UltiDraw.GetRainbowColors(Functions.Length);
		for(int i=0; i<Functions.Length; i++) {
			float height = 25f;
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton(!Functions[i].GetFlag(frame) ? "Off" : "On", !Functions[i].GetFlag(frame) ? colors[i].Transparent(0.25f) : colors[i], UltiDraw.White, 200f, height)) {
				Functions[i].ToggleStyle(frame);
			}
			Rect c = EditorGUILayout.GetControlRect();
			Rect r = new Rect(c.x, c.y, Functions[i].GetValue(frame) * c.width, height);
			EditorGUI.DrawRect(r, colors[i].Transparent(0.75f));
			EditorGUILayout.FloatField(Functions[i].GetValue(frame), GUILayout.Width(50f));
			Functions[i].Name = EditorGUILayout.TextField(Functions[i].Name);
			EditorGUILayout.EndHorizontal();
		}
		EditorGUILayout.BeginHorizontal();
		if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			Frame previous = GetAnyPreviousStyleKey(frame);
			editor.LoadFrame(previous == null ? 0f : previous.Timestamp);
		}
		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		UltiDraw.Begin();

		float startTime = frame.Timestamp-editor.GetWindow()/2f;
		float endTime = frame.Timestamp+editor.GetWindow()/2f;
		if(startTime < 0f) {
			endTime -= startTime;
			startTime = 0f;
		}
		if(endTime > Data.GetTotalTime()) {
			startTime -= endTime-Data.GetTotalTime();
			endTime = Data.GetTotalTime();
		}
		startTime = Mathf.Max(0f, startTime);
		endTime = Mathf.Min(Data.GetTotalTime(), endTime);
		int start = Data.GetFrame(startTime).Index;
		int end = Data.GetFrame(endTime).Index;
		int elements = end-start;

		Vector3 prevPos = Vector3.zero;
		Vector3 newPos = Vector3.zero;
		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

		//Sequences
		for(int i=0; i<Data.Sequences.Length; i++) {
			float _start = (float)(Mathf.Clamp(Data.Sequences[i].Start, start, end)-start) / (float)elements;
			float _end = (float)(Mathf.Clamp(Data.Sequences[i].End, start, end)-start) / (float)elements;
			float left = rect.x + _start * rect.width;
			float right = rect.x + _end * rect.width;
			Vector3 a = new Vector3(left, rect.y, 0f);
			Vector3 b = new Vector3(right, rect.y, 0f);
			Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
			Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
			UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
			UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
		}

		//Styles
		for(int i=0; i<Functions.Length; i++) {
			int x = start;
			for(int j=start; j<end; j++) {
				float val = Functions[i].Values[j];
				if(
					Functions[i].Values[x]<1f && val==1f ||
					Functions[i].Values[x]>0f && val==0f
					) {
					float _start = (float)(Mathf.Clamp(x-1, start, end)-1-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(j, start, end)-1-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					float yStart = rect.y + (1f - Functions[i].Values[Mathf.Max(x-1, 0)]) * rect.height;
					float yEnd = rect.y + (1f - Functions[i].Values[j]) * rect.height;
					UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
					x = j;
				}
				if(
					Functions[i].Values[x]==0f && val>0f || 
					Functions[i].Values[x]==1f && val<1f
					) {
					float _start = (float)(Mathf.Clamp(x, start, end)-1-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(j-1, start, end)-1-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					float yStart = rect.y + (1f - Functions[i].Values[x]) * rect.height;
					float yEnd = rect.y + (1f - Functions[i].Values[j-1]) * rect.height;
					UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
					x = j;
				}
				if(j==Data.GetTotalFrames()-1) {
					float _start = (float)(Mathf.Clamp(x, start, end)-1-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(j-1, start, end)-1-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					float yStart = rect.y + (1f - Functions[i].Values[x]) * rect.height;
					float yEnd = rect.y + (1f - Functions[i].Values[j-1]) * rect.height;
					UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
					x = j;
				}
			}
		}

		//Current Pivot
		top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
		UltiDraw.DrawCircle(top, 3f, UltiDraw.Green);
		UltiDraw.DrawCircle(bottom, 3f, UltiDraw.Green);

		UltiDraw.End();
		EditorGUILayout.EndVertical();
		if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
			Frame next = GetAnyNextStyleKey(frame);
			editor.LoadFrame(next == null ? Data.GetTotalTime() : next.Timestamp);
		}
		EditorGUILayout.EndHorizontal();
	}

	[System.Serializable]
	public class StyleFunction {
		public StyleModule Module;
		public string Name;
		public float[] Values;
		public bool[] Flags;

		public StyleFunction(StyleModule module, string name) {
			Module = module;
			Name = name;
			Values = new float[Module.Data.GetTotalFrames()];
			Flags = new bool[Module.Data.GetTotalFrames()]; 
		}

		public float GetValue(Frame frame) {
			return Values[frame.Index-1];
		}

		public bool GetFlag(Frame frame) {
			return Flags[frame.Index-1];
		}

		public void ToggleStyle(Frame frame) {
			Frame next = GetNextStyleKey(frame);
			Frame start = frame;
			Frame end = next == null ? Module.Data.GetFrame(Module.Data.GetTotalFrames()) : next.GetPreviousFrame();
			bool value = !GetFlag(frame);
			for(int i=start.Index; i<=end.Index; i++) {
				Flags[i-1] = value;
			}
			ApplyStyleKeyValues(frame);
		}

		public void ApplyStyleKeyValues(Frame frame) {
			//Previous Frames
			Frame previousKey = GetPreviousStyleKey(frame);
			previousKey = previousKey == null ? Module.Data.GetFrame(1) : previousKey;
			Frame pivot = Module.Data.GetFrame(Mathf.Max(previousKey.Index, Module.Data.GetFrame(Mathf.Max(frame.Timestamp - Module.Transition, previousKey.Timestamp)).Index));
			if(pivot == frame) {
				Values[frame.Index-1] = GetFlag(frame) ? 1f : 0f;
			} else {
				for(int i=previousKey.Index; i<=pivot.Index; i++) {
					Values[i-1] = GetFlag(previousKey) ? 1f : 0f;
				}
				float valA = GetFlag(pivot) ? 1f : 0f;
				float valB = GetFlag(frame) ? 1f : 0f;
				for(int i=pivot.Index; i<=frame.Index; i++) {
					float weight = (float)(i-pivot.Index) / (float)(frame.Index - pivot.Index);
					Values[i-1] = (1f-weight) * valA + weight * valB;
				}
			}
			//Next Frames
			Frame nextKey = GetNextStyleKey(frame);
			nextKey = nextKey == null ? Module.Data.GetFrame(Module.Data.GetTotalFrames()) : nextKey;
			for(int i=frame.Index; i<=nextKey.Index; i++) {
				Values[i-1] = GetFlag(frame) ? 1f : 0f;
			}
			previousKey = GetFlag(frame) ? frame : previousKey;
			pivot = Module.Data.GetFrame(Mathf.Max(previousKey.Index, Module.Data.GetFrame(Mathf.Max(nextKey.Timestamp - Module.Transition, frame.Timestamp)).Index));
			if(pivot == nextKey) {
				Values[frame.Index-1] = GetFlag(frame) ? 1f : 0f;
			} else {
				float valA = GetFlag(pivot) ? 1f : 0f;
				float valB = GetFlag(frame) ? 1f : 0f;
				for(int i=pivot.Index; i<=nextKey.Index; i++) {
					float weight = (float)(i-pivot.Index) / (float)(nextKey.Index - pivot.Index);
					Values[i-1] = (1f-weight) * valA + weight * valB;
				}
			}
		}

		public Frame GetNextStyleKey(Frame frame) {
			while(frame.Index < Module.Data.GetTotalFrames()) {
				frame = frame.GetNextFrame();
				if(IsStyleKey(frame)) {
					return frame;
				}
			}
			return null;
		}

		public Frame GetPreviousStyleKey(Frame frame) {
			while(frame.Index > 1) {
				frame = frame.GetPreviousFrame();
				if(IsStyleKey(frame)) {
					return frame;
				}
			}
			return null;
		}

		public bool IsStyleKey(Frame frame) {
			Frame previous = frame.GetPreviousFrame();;
			if(!GetFlag(frame) && GetFlag(previous)) {
				return true;
			}
			if(GetFlag(frame) && !GetFlag(previous)) {
				return true;
			}
			return false;
		}

		public bool IsTransition(Frame frame) {
			return Values[frame.Index-1] > 0f & Values[frame.Index-1] < 1f;
		}
	}

}
#endif