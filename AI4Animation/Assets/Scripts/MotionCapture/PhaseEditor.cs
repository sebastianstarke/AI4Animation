#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[RequireComponent(typeof(MotionEditor))]
public class PhaseEditor : MonoBehaviour {

	public float Pivot = 0f;
	public float TimeWindow = 0f;

	public float MaximumVelocity = 5f;
	public float VelocityThreshold = 0.1f;

	public float[] RegularPhase;
	public float[] InversePhase;

	public bool[] RegularVariables;
	public bool[] InverseVariables;
	public float[] RegularVelocities;
	public float[] InverseVelocities;
	public float[] RegularNormalisedVelocities;
	public float[] InverseNormalisedVelocities;

	private PhaseFunction RegularPhaseFunction;
	private PhaseFunction InversePhaseFunction;

	private MotionEditor Editor;

	void Reset() {
		Pivot = 0f;
		TimeWindow = GetEditor().Data.GetTotalTime();
		RegularPhase = new float[GetEditor().Data.GetTotalFrames()];
		InversePhase = new float[GetEditor().Data.GetTotalFrames()];
		RegularVariables = new bool[GetEditor().Data.Source.Bones.Length];
		InverseVariables = new bool[GetEditor().Data.Source.Bones.Length];
		RegularVelocities = new float[GetEditor().Data.GetTotalFrames()];
		RegularNormalisedVelocities = new float[GetEditor().Data.GetTotalFrames()];
		InverseVelocities = new float[GetEditor().Data.GetTotalFrames()];
		InverseNormalisedVelocities = new float[GetEditor().Data.GetTotalFrames()];
	}

	private MotionEditor GetEditor() {
		if(Editor == null) {
			Editor = GetComponent<MotionEditor>();
		}
		return Editor;
	}

	public void SetPivot(float value) {
		if(Pivot != value) {
			Pivot = value;
			GetEditor().LoadFrame(Pivot * GetEditor().Data.GetTotalTime());
		}
	}

	public void SetMaximumVelocity(float value) {
		value = Mathf.Max(1f, value);
		if(MaximumVelocity != value) {
			MaximumVelocity = value;
			ComputeValues();
		}
	}

	public void SetVelocityThreshold(float value) {
		value = Mathf.Max(0f, value);
		if(VelocityThreshold != value) {
			VelocityThreshold = value;
			ComputeValues();
		}
	}

	public void ToggleVariable(int index) {
		RegularVariables[index] = !RegularVariables[index];
		InverseVariables[GetEditor().Data.Symmetry[index]] = RegularVariables[index];
		ComputeValues();
	}

	private void ComputeValues() {
		float min, max;

		RegularVelocities = new float[GetEditor().Data.GetTotalFrames()];
		RegularNormalisedVelocities = new float[GetEditor().Data.GetTotalFrames()];
		min = float.MaxValue;
		max = float.MinValue;
		for(int i=0; i<RegularVelocities.Length; i++) {
			for(int j=0; j<RegularVariables.Length; j++) {
				if(RegularVariables[j]) {
					float boneVelocity = Mathf.Min(GetEditor().Data.Frames[i].GetBoneVelocity(j, false).magnitude, MaximumVelocity);
					RegularVelocities[i] += boneVelocity;
				}
			}
			if(RegularVelocities[i] < VelocityThreshold) {
				RegularVelocities[i] = 0f;
			}
			if(RegularVelocities[i] < min) {
				min = RegularVelocities[i];
			}
			if(RegularVelocities[i] > max) {
				max = RegularVelocities[i];
			}
		}
		for(int i=0; i<RegularVelocities.Length; i++) {
			RegularNormalisedVelocities[i] = Utility.Normalise(RegularVelocities[i], min, max, 0f, 1f);
		}

		InverseVelocities = new float[GetEditor().Data.GetTotalFrames()];
		InverseNormalisedVelocities = new float[GetEditor().Data.GetTotalFrames()];
		min = float.MaxValue;
		max = float.MinValue;
		for(int i=0; i<InverseVelocities.Length; i++) {
			for(int j=0; j<InverseVariables.Length; j++) {
				if(InverseVariables[j]) {
					float boneVelocity = Mathf.Min(GetEditor().Data.Frames[i].GetBoneVelocity(j, false).magnitude, MaximumVelocity);
					InverseVelocities[i] += boneVelocity;
				}
			}
			if(InverseVelocities[i] < VelocityThreshold) {
				InverseVelocities[i] = 0f;
			}
			if(InverseVelocities[i] < min) {
				min = InverseVelocities[i];
			}
			if(InverseVelocities[i] > max) {
				max = InverseVelocities[i];
			}
		}
		for(int i=0; i<InverseVelocities.Length; i++) {
			InverseNormalisedVelocities[i] = Utility.Normalise(InverseVelocities[i], min, max, 0f, 1f);
		}
	}

	public class PhaseFunction {
		public PhaseEditor Editor;

		public float[] Phase;
		public bool[] Keys;

		public bool ShowCycle;
		public float[] Cycle;
		public float[] NormalisedCycle;

		public PhaseEvolution Optimiser;
		public bool Optimising;

		public PhaseFunction(PhaseEditor editor, float[] values) {
			Editor = editor;

			int frames = editor.GetEditor().Data.GetTotalFrames();
			int bones = editor.GetEditor().Data.Source.Bones.Length;

			Phase = values.Length != frames ? new float[frames] : values;
			Keys = new bool[frames];
			Cycle = new float[frames];
			NormalisedCycle = new float[frames];

			Optimiser = new PhaseEvolution(Editor, this);

			for(int i=0; i<Phase.Length; i++) {
				Keys[i] = Phase[i] == 1f;
			}
		}

		public void Save() {
			if(this == Editor.RegularPhaseFunction) {
				Editor.RegularPhase = (float[])Phase.Clone();
			} else {
				Editor.InversePhase = (float[])Phase.Clone();
			}
		}

		public void SetKey(MotionData.Frame frame, bool value) {
			if(value) {
				if(IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = true;
				Phase[frame.Index-1] = 1f;
				Interpolate(frame);
			} else {
				if(!IsKey(frame)) {
					return;
				}
				Keys[frame.Index-1] = false;
				Phase[frame.Index-1] = 0f;
				Interpolate(frame);
			}
		}

		public bool IsKey(MotionData.Frame frame) {
			return Keys[frame.Index-1];
		}

		public void SetPhase(MotionData.Frame frame, float value) {
			if(Phase[frame.Index-1] != value) {
				Phase[frame.Index-1] = value;
				Interpolate(frame);
			}
		}

		public float GetPhase(MotionData.Frame frame) {
			return Phase[frame.Index-1];
		}

		public MotionData.Frame GetPreviousKey(MotionData.Frame frame) {
			if(frame != null) {
				for(int i=frame.Index-1; i>=1; i--) {
					if(Keys[i-1]) {
						return Editor.GetEditor().Data.Frames[i-1];
					}
				}
			}
			return Editor.GetEditor().Data.Frames[0];
		}

		public MotionData.Frame GetNextKey(MotionData.Frame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Editor.GetEditor().Data.GetTotalFrames(); i++) {
					if(Keys[i-1]) {
						return Editor.GetEditor().Data.Frames[i-1];
					}
				}
			}
			return Editor.GetEditor().Data.Frames[Editor.GetEditor().Data.GetTotalFrames()-1];
		}

		public void Recompute() {
			for(int i=0; i<Editor.GetEditor().Data.Frames.Length; i++) {
				if(IsKey(Editor.GetEditor().Data.Frames[i])) {
					Phase[i] = 1f;
				}
			}
			MotionData.Frame A = Editor.GetEditor().Data.Frames[0];
			MotionData.Frame B = GetNextKey(A);
			while(A != B) {
				Interpolate(A, B);
				A = B;
				B = GetNextKey(A);
			}
		}

		private void Interpolate(MotionData.Frame frame) {
			if(IsKey(frame)) {
				Interpolate(GetPreviousKey(frame), frame);
				Interpolate(frame, GetNextKey(frame));
			} else {
				Interpolate(GetPreviousKey(frame), GetNextKey(frame));
			}
		}

		private void Interpolate(MotionData.Frame a, MotionData.Frame b) {
			if(a == null || b == null) {
				Debug.Log("A given frame was null.");
				return;
			}
			int dist = b.Index - a.Index;
			if(dist >= 2) {
				for(int i=a.Index+1; i<b.Index; i++) {
					float rateA = (float)((float)i-(float)a.Index)/(float)dist;
					float rateB = (float)((float)b.Index-(float)i)/(float)dist;
					Phase[i-1] = rateB*Mathf.Repeat(Phase[a.Index-1], 1f) + rateA*Phase[b.Index-1];
				}
			}

			if(a.Index == 1) {
				MotionData.Frame first = Editor.GetEditor().Data.Frames[0];
				MotionData.Frame next1 = GetNextKey(first);
				MotionData.Frame next2 = GetNextKey(next1);
				Keys[0] = true;
				float xFirst = next1.Timestamp - first.Timestamp;
				float mFirst = next2.Timestamp - next1.Timestamp;
				SetPhase(first, Mathf.Clamp(1f - xFirst / mFirst, 0f, 1f));
			}
			if(b.Index == Editor.GetEditor().Data.GetTotalFrames()) {
				MotionData.Frame last = Editor.GetEditor().Data.Frames[Editor.GetEditor().Data.GetTotalFrames()-1];
				MotionData.Frame previous1 = GetPreviousKey(last);
				MotionData.Frame previous2 = GetPreviousKey(previous1);
				Keys[Editor.GetEditor().Data.GetTotalFrames()-1] = true;
				float xLast = last.Timestamp - previous1.Timestamp;
				float mLast = previous1.Timestamp - previous2.Timestamp;
				SetPhase(last, Mathf.Clamp(xLast / mLast, 0f, 1f));
			}
		}

		public void EditorUpdate() {
			if(Optimising) {
				Optimiser.Optimise();
			}
		}

		public void Inspector() {
			UltiDraw.Begin();

			Utility.SetGUIColor(UltiDraw.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField(this == Editor.RegularPhaseFunction ? "Regular" : "Inverse");
				}

				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Optimising) {
						if(Utility.GUIButton("Stop Optimisation", UltiDraw.LightGrey, UltiDraw.Black)) {
							Optimising = !Optimising;
							Save();
						}
					} else {
						if(Utility.GUIButton("Start Optimisation", UltiDraw.DarkGrey, UltiDraw.White)) {
							Optimising = !Optimising;
						}
					}
					if(Utility.GUIButton("Restart", UltiDraw.Brown, UltiDraw.White)) {
						Optimiser.Initialise();
					}
					if(Utility.GUIButton("Clear", UltiDraw.Brown, UltiDraw.White)) {
						Optimiser.Clear();
					}
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Fitness: " + Optimiser.GetFitness(), GUILayout.Width(150f));
					float[] configuration = Optimiser.GetPeakConfiguration();
					EditorGUILayout.LabelField("Peak: " + configuration[0] + " | " + configuration[1] + " | " + configuration[2] + " | " + configuration[3] + " | " + configuration[4]);
					EditorGUILayout.EndHorizontal();
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Exploration", GUILayout.Width(100f));
					GUILayout.FlexibleSpace();
					Optimiser.Behaviour = EditorGUILayout.Slider(Optimiser.Behaviour, 0f, 1f);
					GUILayout.FlexibleSpace();
					EditorGUILayout.LabelField("Exploitation", GUILayout.Width(100f));
					EditorGUILayout.EndHorizontal();
					Optimiser.SetAmplitude(EditorGUILayout.Slider("Amplitude", Optimiser.Amplitude, 0, PhaseEvolution.AMPLITUDE));
					Optimiser.SetFrequency(EditorGUILayout.Slider("Frequency", Optimiser.Frequency, 0f, PhaseEvolution.FREQUENCY));
					Optimiser.SetShift(EditorGUILayout.Slider("Shift", Optimiser.Shift, 0, PhaseEvolution.SHIFT));
					Optimiser.SetOffset(EditorGUILayout.Slider("Offset", Optimiser.Offset, 0, PhaseEvolution.OFFSET));
					Optimiser.SetSlope(EditorGUILayout.Slider("Slope", Optimiser.Slope, 0, PhaseEvolution.SLOPE));
					Optimiser.SetWindow(EditorGUILayout.Slider("Window", Optimiser.Window, 0.1f, PhaseEvolution.WINDOW));
					Optimiser.Blending = EditorGUILayout.Slider("Blending", Optimiser.Blending, 0f, 1f);
				}

				MotionData.Frame frame = Editor.GetEditor().Data.GetFrame(Editor.GetEditor().GetState().Index);

				if(IsKey(frame)) {
					SetPhase(frame, EditorGUILayout.Slider("Phase", GetPhase(frame), 0f, 1f));
				} else {
					EditorGUI.BeginDisabledGroup(true);
					SetPhase(frame, EditorGUILayout.Slider("Phase", GetPhase(frame), 0f, 1f));
					EditorGUI.EndDisabledGroup();
				}

				ShowCycle = EditorGUILayout.Toggle("Show Cycle", ShowCycle);

				if(IsKey(frame)) {
					if(Utility.GUIButton("Unset Key", UltiDraw.Grey, UltiDraw.White)) {
						SetKey(frame, false);
						Save();
					}
				} else {
					if(Utility.GUIButton("Set Key", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetKey(frame, true);
						Save();
					}
				}
				
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Editor.GetEditor().LoadFrame((GetPreviousKey(frame).Timestamp));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = frame.Timestamp-Editor.TimeWindow/2f;
				float endTime = frame.Timestamp+Editor.TimeWindow/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Editor.GetEditor().Data.GetTotalTime()) {
					startTime -= endTime-Editor.GetEditor().Data.GetTotalTime();
					endTime = Editor.GetEditor().Data.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Editor.GetEditor().Data.GetTotalTime(), endTime);
				int start = Editor.GetEditor().Data.GetFrame(startTime).Index;
				int end = Editor.GetEditor().Data.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				//Regular Velocities
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Editor.RegularNormalisedVelocities[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Editor.RegularNormalisedVelocities[i+start] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, this == Editor.RegularPhaseFunction ? UltiDraw.Green : UltiDraw.Red);
				}

				//Inverse Velocities
				for(int i=1; i<elements; i++) {
					prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
					prevPos.y = rect.yMax - Editor.InverseNormalisedVelocities[i+start-1] * rect.height;
					newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
					newPos.y = rect.yMax - Editor.InverseNormalisedVelocities[i+start] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, this == Editor.RegularPhaseFunction ? UltiDraw.Red : UltiDraw.Green);
				}
				
				//Cycle
				if(ShowCycle) {
					for(int i=1; i<elements; i++) {
						prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
						prevPos.y = rect.yMax - NormalisedCycle[i+start-1] * rect.height;
						newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
						newPos.y = rect.yMax - NormalisedCycle[i+start] * rect.height;
						UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Yellow);
					}
				}

				//Phase
				/*
				for(int i=1; i<Editor.GetEditor().Data.Frames.Length; i++) {
					MotionData.Frame A = Editor.GetEditor().Data.Frames[i-1];
					MotionData.Frame B = Editor.GetEditor().Data.Frames[i];
					prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
					prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
					newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
				}
				*/
				
				MotionData.Frame A = Editor.GetEditor().Data.GetFrame(start);
				if(A.Index == 1) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta.Transparent(0.5f));
				}
				MotionData.Frame B = GetNextKey(A);
				while(A != B) {
					prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
					prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
					newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
					UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
					bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta.Transparent(0.5f));
					A = B;
					B = GetNextKey(A);
					if(B.Index > end) {
						break;
					}
				}

				//Seconds
				float timestamp = startTime;
				while(timestamp <= endTime) {
					float floor = Mathf.FloorToInt(timestamp);
					if(floor >= startTime && floor <= endTime) {
						top.x = rect.xMin + (float)(Editor.GetEditor().Data.GetFrame(floor).Index-start)/elements * rect.width;
						UltiDraw.DrawCircle(top, 5f, UltiDraw.White);
					}
					timestamp += 1f;
				}
				//

				//Current Pivot
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
				UltiDraw.DrawCircle(top, 3f, UltiDraw.Green);
				UltiDraw.DrawCircle(bottom, 3f, UltiDraw.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					Editor.GetEditor().LoadFrame(GetNextKey(frame).Timestamp);
				}
				EditorGUILayout.EndHorizontal();
			}

			UltiDraw.End();
		}
	}


	public class PhaseEvolution {
		public static float AMPLITUDE = 10f;
		public static float FREQUENCY = 2.5f;
		public static float SHIFT = Mathf.PI;
		public static float OFFSET = 10f;
		public static float SLOPE = 5f;
		public static float WINDOW = 5f;
		
		public PhaseEditor Editor;
		public PhaseFunction Function;

		public Population[] Populations;

		public float[] LowerBounds;
		public float[] UpperBounds;

		public float Amplitude = AMPLITUDE;
		public float Frequency = FREQUENCY;
		public float Shift = SHIFT;
		public float Offset = OFFSET;
		public float Slope = SLOPE;

		public float Behaviour = 1f;

		public float Window = 1f;
		public float Blending = 1f;

		public PhaseEvolution(PhaseEditor editor, PhaseFunction function) {
			Editor = editor;
			Function = function;

			LowerBounds = new float[5];
			UpperBounds = new float[5];

			SetAmplitude(Amplitude);
			SetFrequency(Frequency);
			SetShift(Shift);
			SetOffset(Offset);
			SetSlope(Slope);

			Initialise();
		}

		public void SetAmplitude(float value) {
			Amplitude = value;
			LowerBounds[0] = -value;
			UpperBounds[0] = value;
		}

		public void SetFrequency(float value) {
			Frequency = value;
			LowerBounds[1] = 0f;
			UpperBounds[1] = value;
		}

		public void SetShift(float value) {
			Shift = value;
			LowerBounds[2] = -value;
			UpperBounds[2] = value;
		}

		public void SetOffset(float value) {
			Offset = value;
			LowerBounds[3] = -value;
			UpperBounds[3] = value;
		}

		public void SetSlope(float value) {
			Slope = value;
			LowerBounds[4] = -value;
			UpperBounds[4] = value;
		}

		public void SetWindow(float value) {
			if(Window != value) {
				Window = value;
				Initialise();
			}
		}

		public void Initialise() {
			Interval[] intervals = new Interval[Mathf.FloorToInt(Editor.GetEditor().Data.GetTotalTime() / Window) + 1];
			for(int i=0; i<intervals.Length; i++) {
				int start = Editor.GetEditor().Data.GetFrame(i*Window).Index-1;
				int end = Editor.GetEditor().Data.GetFrame(Mathf.Min(Editor.GetEditor().Data.GetTotalTime(), (i+1)*Window)).Index-2;
				if(end == Editor.GetEditor().Data.GetTotalFrames()-2) {
					end += 1;
				}
				intervals[i] = new Interval(start, end);
			}
			Populations = new Population[intervals.Length];
			for(int i=0; i<Populations.Length; i++) {
				Populations[i] = new Population(this, 50, 5, intervals[i]);
			}
		}

		public void Optimise() {
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Active = IsActive(i);
			}
			for(int i=0; i<Populations.Length; i++) {
				Populations[i].Evolve(GetPreviousPopulation(i), GetNextPopulation(i), GetPreviousPivotPopulation(i), GetNextPivotPopulation(i));
			}
			Assign();
		}

		public void Clear() {
			for(int i=0; i<Function.Phase.Length; i++) {
				Function.Phase[i] = 0f;
				Function.Keys[i] = false;
			}
		}

		
		public void Assign() {
			for(int i=0; i<Editor.GetEditor().Data.GetTotalFrames(); i++) {
				Function.Keys[i] = false;
				Function.Phase[i] = 0f;
				Function.Cycle[i] = 0f;
				Function.NormalisedCycle[i] = 0f;
			}

			//Compute cycle
			float min = float.MaxValue;
			float max = float.MinValue;
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					Function.Cycle[j] = Interpolate(i, j);
					min = Mathf.Min(min, Function.Cycle[j]);
					max = Mathf.Max(max, Function.Cycle[j]);
				}
			}
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					Function.NormalisedCycle[j] = Utility.Normalise(Function.Cycle[j], min, max, 0f, 1f);
				}
			}

			//Fill with frequency negative turning points
			for(int i=0; i<Populations.Length; i++) {
				for(int j=Populations[i].Interval.Start; j<=Populations[i].Interval.End; j++) {
					if(InterpolateD2(i, j) <= 0f && InterpolateD2(i, j+1) >= 0f) {
						Function.Keys[j] = true;
					}
				}
			}

			//Compute phase
			for(int i=0; i<Function.Keys.Length; i++) {
				if(Function.Keys[i]) {
					Function.SetPhase(Editor.GetEditor().Data.Frames[i], i == 0 ? 0f : 1f);
				}
			}
		}

		public Population GetPreviousPopulation(int current) {
			return Populations[Mathf.Max(0, current-1)];
		}

		public Population GetPreviousPivotPopulation(int current) {
			for(int i=current-1; i>=0; i--) {
				if(Populations[i].Active) {
					return Populations[i];
				}
			}
			return Populations[0];
		}

		public Population GetNextPopulation(int current) {
			return Populations[Mathf.Min(Populations.Length-1, current+1)];
		}

		public Population GetNextPivotPopulation(int current) {
			for(int i=current+1; i<Populations.Length; i++) {
				if(Populations[i].Active) {
					return Populations[i];
				}
			}
			return Populations[Populations.Length-1];
		}

		public bool IsActive(int interval) {
			float velocity = 0f;
			for(int i=Populations[interval].Interval.Start; i<=Populations[interval].Interval.End; i++) {
				velocity += Editor.RegularVelocities[i] + Editor.InverseVelocities[i];
			}
			return velocity / Populations[interval].Interval.Length > 0f;
		}

		public float Interpolate(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD1(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype1(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype1(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype1(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD2(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype2(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype2(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype2(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float InterpolateD3(int interval, int frame) {
			interval = Mathf.Clamp(interval, 0, Populations.Length-1);
			Population current = Populations[interval];
			float value = current.Phenotype3(current.GetWinner().Genes, frame);
			float pivot = (float)(frame-current.Interval.Start) / (float)(current.Interval.Length-1) - 0.5f;
			float threshold = 0.5f * (1f - Blending);
			if(pivot < -threshold) {
				Population previous = GetPreviousPopulation(interval);
				float blend = 0.5f * (pivot + threshold) / (-0.5f + threshold);
				float prevValue = previous.Phenotype3(previous.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * prevValue;
			}
			if(pivot > threshold) {
				Population next = GetNextPopulation(interval);
				float blend = 0.5f * (pivot - threshold) / (0.5f - threshold);
				float nextValue = next.Phenotype3(next.GetWinner().Genes, frame);
				value = (1f-blend) * value + blend * nextValue;
			}
			return value;
		}

		public float GetFitness() {
			float fitness = 0f;
			for(int i=0; i<Populations.Length; i++) {
				fitness += Populations[i].GetFitness();
			}
			return fitness / Populations.Length;
		}

		public float[] GetPeakConfiguration() {
			float[] configuration = new float[5];
			for(int i=0; i<5; i++) {
				configuration[i] = float.MinValue;
			}
			for(int i=0; i<Populations.Length; i++) {
				for(int j=0; j<5; j++) {
					configuration[j] = Mathf.Max(configuration[j], Mathf.Abs(Populations[i].GetWinner().Genes[j]));
				}
			}
			return configuration;
		}

		public class Population {
			public PhaseEvolution Evolution;
			public int Size;
			public int Dimensionality;
			public Interval Interval;

			public bool Active;

			public Individual[] Individuals;
			public Individual[] Offspring;
			public float[] RankProbabilities;
			public float RankProbabilitySum;

			public Population(PhaseEvolution evolution, int size, int dimensionality, Interval interval) {
				Evolution = evolution;
				Size = size;
				Dimensionality = dimensionality;
				Interval = interval;

				//Create individuals
				Individuals = new Individual[Size];
				Offspring = new Individual[Size];
				for(int i=0; i<Size; i++) {
					Individuals[i] = new Individual(Dimensionality);
					Offspring[i] = new Individual(Dimensionality);
				}

				//Compute rank probabilities
				RankProbabilities = new float[Size];
				float rankSum = (float)(Size*(Size+1)) / 2f;
				for(int i=0; i<Size; i++) {
					RankProbabilities[i] = (float)(Size-i)/(float)rankSum;
				}
				for(int i=0; i<Size; i++) {
					RankProbabilitySum += RankProbabilities[i];
				}

				//Initialise randomly
				for(int i=0; i<Size; i++) {
					Reroll(Individuals[i]);
				}

				//Evaluate fitness
				for(int i=0; i<Size; i++) {
					Individuals[i].Fitness = ComputeFitness(Individuals[i].Genes);
				}

				//Sort
				SortByFitness(Individuals);

				//Evaluate extinctions
				AssignExtinctions(Individuals);
			}

			public void Evolve(Population previous, Population next, Population previousPivot, Population nextPivot) {
				if(Active) {
					//Copy elite
					Copy(Individuals[0], Offspring[0]);

					//Memetic exploitation
					Exploit(Offspring[0]);

					//Remaining individuals
					for(int o=1; o<Size; o++) {
						Individual offspring = Offspring[o];
						if(Random.value <= Evolution.Behaviour) {
							Individual parentA = Select(Individuals);
							Individual parentB = Select(Individuals);
							while(parentB == parentA) {
								parentB = Select(Individuals);
							}
							Individual prototype = Select(Individuals);
							while(prototype == parentA || prototype == parentB) {
								prototype = Select(Individuals);
							}

							float mutationRate = GetMutationProbability(parentA, parentB);
							float mutationStrength = GetMutationStrength(parentA, parentB);

							for(int i=0; i<Dimensionality; i++) {
								float weight;

								//Recombination
								weight = Random.value;
								float momentum = Random.value * parentA.Momentum[i] + Random.value * parentB.Momentum[i];
								if(Random.value < 0.5f) {
									offspring.Genes[i] = parentA.Genes[i] + momentum;
								} else {
									offspring.Genes[i] = parentB.Genes[i] + momentum;
								}

								//Store
								float gene = offspring.Genes[i];

								//Mutation
								if(Random.value <= mutationRate) {
									float span = Evolution.UpperBounds[i] - Evolution.LowerBounds[i];
									offspring.Genes[i] += Random.Range(-mutationStrength*span, mutationStrength*span);
								}
								
								//Adoption
								weight = Random.value;
								offspring.Genes[i] += 
									weight * Random.value * (0.5f * (parentA.Genes[i] + parentB.Genes[i]) - offspring.Genes[i])
									+ (1f-weight) * Random.value * (prototype.Genes[i] - offspring.Genes[i]);

								//Constrain
								offspring.Genes[i] = Mathf.Clamp(offspring.Genes[i], Evolution.LowerBounds[i], Evolution.UpperBounds[i]);

								//Momentum
								offspring.Momentum[i] = Random.value * momentum + (offspring.Genes[i] - gene);
							}
						} else {
							Reroll(offspring);
						}
					}

					//Evaluate fitness
					for(int i=0; i<Size; i++) {
						Offspring[i].Fitness = ComputeFitness(Offspring[i].Genes);
					}

					//Sort
					SortByFitness(Offspring);

					//Evaluate extinctions
					AssignExtinctions(Offspring);

					//Form new population
					for(int i=0; i<Size; i++) {
						Copy(Offspring[i], Individuals[i]);
					}
				} else {
					//Postprocess
					for(int i=0; i<Size; i++) {
						Individuals[i].Genes[0] = 1f;
						Individuals[i].Genes[1] = 1f;
						Individuals[i].Genes[2] = 0.5f * (previousPivot.GetWinner().Genes[2] + nextPivot.GetWinner().Genes[2]);
						Individuals[i].Genes[3] = 0.5f * (previousPivot.GetWinner().Genes[3] + nextPivot.GetWinner().Genes[3]);
						Individuals[i].Genes[4] = 0f;
						for(int j=0; j<5; j++) {
							Individuals[i].Momentum[j] = 0f;
						}
						Individuals[i].Fitness = 0f;
						Individuals[i].Extinction = 0f;
					}
				}
			}

			//Returns the mutation probability from two parents
			private float GetMutationProbability(Individual parentA, Individual parentB) {
				float extinction = 0.5f * (parentA.Extinction + parentB.Extinction);
				float inverse = 1f/(float)Dimensionality;
				return extinction * (1f-inverse) + inverse;
			}

			//Returns the mutation strength from two parents
			private float GetMutationStrength(Individual parentA, Individual parentB) {
				return 0.5f * (parentA.Extinction + parentB.Extinction);
			}

			public Individual GetWinner() {
				return Individuals[0];
			}

			public float GetFitness() {
				return GetWinner().Fitness;
			}

			private void Copy(Individual from, Individual to) {
				for(int i=0; i<Dimensionality; i++) {
					to.Genes[i] = Mathf.Clamp(from.Genes[i], Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					to.Momentum[i] = from.Momentum[i];
				}
				to.Extinction = from.Extinction;
				to.Fitness = from.Fitness;
			}

			private void Reroll(Individual individual) {
				for(int i=0; i<Dimensionality; i++) {
					individual.Genes[i] = Random.Range(Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
				}
			}

			private void Exploit(Individual individual) {
				individual.Fitness = ComputeFitness(individual.Genes);
				for(int i=0; i<Dimensionality; i++) {
					float gene = individual.Genes[i];

					float span = Evolution.UpperBounds[i] - Evolution.LowerBounds[i];

					float incGene = Mathf.Clamp(gene + Random.value*individual.Fitness*span, Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					individual.Genes[i] = incGene;
					float incFitness = ComputeFitness(individual.Genes);

					float decGene = Mathf.Clamp(gene - Random.value*individual.Fitness*span, Evolution.LowerBounds[i], Evolution.UpperBounds[i]);
					individual.Genes[i] = decGene;
					float decFitness = ComputeFitness(individual.Genes);

					individual.Genes[i] = gene;

					if(incFitness < individual.Fitness) {
						individual.Genes[i] = incGene;
						individual.Momentum[i] = incGene - gene;
						individual.Fitness = incFitness;
					}

					if(decFitness < individual.Fitness) {
						individual.Genes[i] = decGene;
						individual.Momentum[i] = decGene - gene;
						individual.Fitness = decFitness;
					}
				}
			}

			//Rank-based selection of an individual
			private Individual Select(Individual[] pool) {
				double rVal = Random.value * RankProbabilitySum;
				for(int i=0; i<Size; i++) {
					rVal -= RankProbabilities[i];
					if(rVal <= 0.0) {
						return pool[i];
					}
				}
				return pool[Size-1];
			}

			//Sorts all individuals starting with best (lowest) fitness
			private void SortByFitness(Individual[] individuals) {
				System.Array.Sort(individuals,
					delegate(Individual a, Individual b) {
						return a.Fitness.CompareTo(b.Fitness);
					}
				);
			}

			//Multi-Objective RMSE
			private float ComputeFitness(float[] genes) {
				float fitness = 0f;
				for(int i=Interval.Start; i<=Interval.End; i++) {
					float y1 = Evolution.Function == Evolution.Editor.RegularPhaseFunction ? Evolution.Editor.RegularVelocities[i] : Evolution.Editor.InverseVelocities[i];
					float y2 = Evolution.Function == Evolution.Editor.RegularPhaseFunction ? Evolution.Editor.InverseVelocities[i] : Evolution.Editor.RegularVelocities[i];
					float x = Phenotype(genes, i);
					float error = (y1-x)*(y1-x) + (-y2-x)*(-y2-x);
					float sqrError = error*error;
					fitness += sqrError;
				}
				fitness /= Interval.Length;
				fitness = Mathf.Sqrt(fitness);
				return fitness;
			}
			
			public float Phenotype(float[] genes, int frame) {
				return Utility.LinSin(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Editor.GetEditor().Data.Framerate, 
					genes[4], 
					frame/Evolution.Editor.GetEditor().Data.Framerate
					);
			}

			public float Phenotype1(float[] genes, int frame) {
				return Utility.LinSin1(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Editor.GetEditor().Data.Framerate,
					genes[4], 
					frame/Evolution.Editor.GetEditor().Data.Framerate
					);
			}

			public float Phenotype2(float[] genes, int frame) {
				return Utility.LinSin2(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Editor.GetEditor().Data.Framerate,
					genes[4], 
					frame/Evolution.Editor.GetEditor().Data.Framerate
					);
			}

			public float Phenotype3(float[] genes, int frame) {
				return Utility.LinSin3(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Editor.GetEditor().Data.Framerate,
					genes[4], 
					frame/Evolution.Editor.GetEditor().Data.Framerate
					);
			}

			//Compute extinction values
			private void AssignExtinctions(Individual[] individuals) {
				float min = individuals[0].Fitness;
				float max = individuals[Size-1].Fitness;
				for(int i=0; i<Size; i++) {
					float grading = (float)i/((float)Size-1);
					individuals[i].Extinction = (individuals[i].Fitness + min*(grading-1f)) / max;
				}
			}
		}

		public class Individual {
			public float[] Genes;
			public float[] Momentum;
			public float Extinction;
			public float Fitness;
			public Individual(int dimensionality) {
				Genes = new float[dimensionality];
				Momentum = new float[dimensionality];
			}
		}

		public class Interval {
			public int Start;
			public int End;
			public int Length;
			public Interval(int start, int end) {
				Start = start;
				End = end;
				Length = end-start+1;
			}
		}

	}

	[CustomEditor(typeof(PhaseEditor))]
	public class PhaseEditor_Editor : Editor {

		public PhaseEditor Target;

		private float RefreshRate = 30f;
		private System.DateTime Timestamp;

		void Awake() {
			Target = (PhaseEditor)target;
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy() {
			EditorApplication.update -= EditorUpdate;
		}

		public void EditorUpdate() {
			if(Utility.GetElapsedTime(Timestamp) >= 1f/RefreshRate) {
				Repaint();
				Timestamp = Utility.GetTimestamp();
			}
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);
			Inspector();
			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		public void Inspector() {
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.DarkGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();

						EditorGUILayout.ObjectField("Data", Target.GetEditor().Data, typeof(MotionData), true);
					}

					Target.SetPivot(EditorGUILayout.Slider("Pivot", Target.Pivot, 0f, 1f));
					Target.TimeWindow = EditorGUILayout.Slider("Time Window", Target.TimeWindow, 0f, Target.GetEditor().Data.GetTotalTime());
					Target.SetMaximumVelocity(EditorGUILayout.FloatField("Maximum Velocity", Target.MaximumVelocity));
					Target.SetVelocityThreshold(EditorGUILayout.FloatField("Velocity Threshold", Target.VelocityThreshold));
					for(int i=0; i<Target.GetEditor().Data.Source.Bones.Length; i++) {
						if(Target.RegularVariables[i]) {
							if(Utility.GUIButton(Target.GetEditor().Data.Source.Bones[i].Name, UltiDraw.Mustard, UltiDraw.White)) {
								Target.ToggleVariable(i);
							}
						} else if(Target.InverseVariables[i]) {
							if(Utility.GUIButton(Target.GetEditor().Data.Source.Bones[i].Name, UltiDraw.Cyan, UltiDraw.White)) {
								Target.ToggleVariable(Target.GetEditor().Data.Symmetry[i]);
							}
						} else {
							if(Utility.GUIButton(Target.GetEditor().Data.Source.Bones[i].Name, UltiDraw.DarkRed, UltiDraw.White)) {
								Target.ToggleVariable(i);
							}
						}
					}

					if(Target.RegularPhaseFunction == null) {
						Target.RegularPhaseFunction = new PhaseFunction(Target, Target.RegularPhase);
					}
					if(Target.InversePhaseFunction == null) {
						Target.InversePhaseFunction = new PhaseFunction(Target, Target.InversePhase);
					}
					Target.RegularPhaseFunction.Inspector();
					Target.RegularPhaseFunction.EditorUpdate();
					Target.InversePhaseFunction.Inspector();
					Target.InversePhaseFunction.EditorUpdate();
				}
			}

		}
	}

}
#endif