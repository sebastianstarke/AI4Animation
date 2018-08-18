#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class PhaseModule : Module {

	public float MaximumVelocity = 10f;
	public float VelocityThreshold = 0.1f;

	public PhaseFunction RegularPhaseFunction = null;
	public PhaseFunction InversePhaseFunction = null;
	public bool[] Variables = new bool[0];

	public bool ShowVelocities = true;
	public bool ShowCycle = true;

	private bool Optimising = false;	

	public override TYPE Type() {
		return TYPE.Phase;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		RegularPhaseFunction = new PhaseFunction(this);
		InversePhaseFunction = new PhaseFunction(this);
		Variables = new bool[Data.Source.Bones.Length];
		return this;
	}

	public void SetMaximumVelocity(float value) {
		value = Mathf.Max(1f, value);
		if(MaximumVelocity != value) {
			MaximumVelocity = value;
			RegularPhaseFunction.ComputeVelocities();
			InversePhaseFunction.ComputeVelocities();
		}
	}

	public void SetVelocityThreshold(float value) {
		value = Mathf.Max(0f, value);
		if(VelocityThreshold != value) {
			VelocityThreshold = value;
			RegularPhaseFunction.ComputeVelocities();
			InversePhaseFunction.ComputeVelocities();
		}
	}

	public void ToggleVariable(int index) {
		Variables[index] = !Variables[index];
		RegularPhaseFunction.ComputeVelocities();
		InversePhaseFunction.ComputeVelocities();
	}

	public float GetPhase(Frame frame, bool mirrored) {
		return mirrored ? InversePhaseFunction.GetPhase(frame) : RegularPhaseFunction.GetPhase(frame);
	}

	public override void Draw(MotionEditor editor) {
		UltiDraw.Begin();
		for(int i=0; i<Variables.Length; i++) {
			if(Variables[i]) {
				UltiDraw.DrawSphere(editor.GetCurrentFrame().GetBoneTransformation(i, editor.Mirror).GetPosition(), Quaternion.identity, 0.05f, UltiDraw.Red);
			}
		}
		UltiDraw.End();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		ShowVelocities = EditorGUILayout.Toggle("Show Velocities", ShowVelocities);
		ShowCycle = EditorGUILayout.Toggle("Show Cycle", ShowCycle);
		SetMaximumVelocity(EditorGUILayout.FloatField("Maximum Velocity", MaximumVelocity));
		SetVelocityThreshold(EditorGUILayout.FloatField("Velocity Threshold", VelocityThreshold));
		int index = EditorGUILayout.Popup("Phase Detector", 0, ArrayExtensions.Concat("Select...", Data.Source.GetNames()));
		if(index > 0) {
			ToggleVariable(index-1);
		}
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			if(Variables[i]) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField(Data.Source.Bones[i].Name);
					EditorGUILayout.LabelField(Data.Source.Bones[Data.Symmetry[i]].Name);
					EditorGUILayout.EndHorizontal();
				}
			}
		}
		Utility.SetGUIColor(UltiDraw.Grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Optimising) {
				if(Utility.GUIButton("Stop Optimisation", UltiDraw.LightGrey, UltiDraw.Black)) {
					EditorCoroutines.StopCoroutine(RegularPhaseFunction.Optimise(), this);
					EditorCoroutines.StopCoroutine(InversePhaseFunction.Optimise(), this);
					Optimising = false;
				}
			} else {
				if(Utility.GUIButton("Start Optimisation", UltiDraw.DarkGrey, UltiDraw.White)) {
					EditorCoroutines.StartCoroutine(RegularPhaseFunction.Optimise(), this);
					EditorCoroutines.StartCoroutine(InversePhaseFunction.Optimise(), this);
					Optimising = true;
				}
			}
			if(Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White)) {
				RegularPhaseFunction.Clear();
				InversePhaseFunction.Clear();
			}
		}
		RegularPhaseFunction.Inspector(editor);
		InversePhaseFunction.Inspector(editor);
	}

	[System.Serializable]
	public class PhaseFunction {
		public PhaseModule Module;

		public float[] Phase;
		public bool[] Keys;
		
		public float[] Cycle;
		public float[] NormalisedCycle;
		public float[] Velocities;
		public float[] NVelocities;

		public PhaseFunction(PhaseModule module) {
			Module = module;
			Phase = new float[module.Data.GetTotalFrames()];
			Keys = new bool[module.Data.GetTotalFrames()];
			Cycle = new float[module.Data.GetTotalFrames()];
			NormalisedCycle = new float[module.Data.GetTotalFrames()];
			ComputeVelocities();
		}
		
		public IEnumerator Optimise() {
			PhaseEvolution optimiser = new PhaseEvolution(this);
			while(true) {
				optimiser.Optimise();
				yield return new WaitForSeconds(0f);
			}
		}

		public void Clear() {
			Phase = new float[Module.Data.GetTotalFrames()];
			Keys = new bool[Module.Data.GetTotalFrames()];
			Cycle = new float[Module.Data.GetTotalFrames()];
			NormalisedCycle = new float[Module.Data.GetTotalFrames()];
		}

		public void ComputeVelocities() {
			float min, max;
			Velocities = new float[Module.Data.GetTotalFrames()];
			NVelocities = new float[Module.Data.GetTotalFrames()];
			min = float.MaxValue;
			max = float.MinValue;
			for(int i=0; i<Velocities.Length; i++) {
				for(int j=0; j<Module.Variables.Length; j++) {
					if(Module.Variables[j]) {
						float boneVelocity = Mathf.Min(Module.Data.Frames[i].GetBoneVelocity(j,  (this == Module.RegularPhaseFunction ? false : true)).magnitude, Module.MaximumVelocity);
						Velocities[i] += boneVelocity;
					}
				}
				if(Velocities[i] < Module.VelocityThreshold) {
					Velocities[i] = 0f;
				}
				if(Velocities[i] < min) {
					min = Velocities[i];
				}
				if(Velocities[i] > max) {
					max = Velocities[i];
				}
			}
			for(int i=0; i<Velocities.Length; i++) {
				NVelocities[i] = Utility.Normalise(Velocities[i], min, max, 0f, 1f);
			}
		}

		public void SetKey(Frame frame, bool value) {
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

		public bool IsKey(Frame frame) {
			return Keys[frame.Index-1];
		}

		public void SetPhase(Frame frame, float value) {
			if(Phase[frame.Index-1] != value) {
				Phase[frame.Index-1] = value;
				Interpolate(frame);
			}
		}

		public float GetPhase(Frame frame) {
			return Phase[frame.Index-1];
		}

		public Frame GetPreviousKey(Frame frame) {
			if(frame != null) {
				for(int i=frame.Index-1; i>=1; i--) {
					if(Keys[i-1]) {
						return Module.Data.Frames[i-1];
					}
				}
			}
			return Module.Data.Frames[0];
		}

		public Frame GetNextKey(Frame frame) {
			if(frame != null) {
				for(int i=frame.Index+1; i<=Module.Data.GetTotalFrames(); i++) {
					if(Keys[i-1]) {
						return Module.Data.Frames[i-1];
					}
				}
			}
			return Module.Data.Frames[Module.Data.GetTotalFrames()-1];
		}

		public void Recompute() {
			for(int i=0; i<Module.Data.Frames.Length; i++) {
				if(IsKey(Module.Data.Frames[i])) {
					Phase[i] = 1f;
				}
			}
			Frame A = Module.Data.Frames[0];
			Frame B = GetNextKey(A);
			while(A != B) {
				Interpolate(A, B);
				A = B;
				B = GetNextKey(A);
			}
		}

		private void Interpolate(Frame frame) {
			if(IsKey(frame)) {
				Interpolate(GetPreviousKey(frame), frame);
				Interpolate(frame, GetNextKey(frame));
			} else {
				Interpolate(GetPreviousKey(frame), GetNextKey(frame));
			}
		}

		private void Interpolate(Frame a, Frame b) {
			if(a == null || b == null) {
				Debug.Log("A given frame was null.");
				return;
			}
			if(a == b) {
				return;
			}
			if(a == Module.Data.Frames[0] && b == Module.Data.Frames[Module.Data.Frames.Length-1]) {
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
				Frame first = Module.Data.Frames[0];
				Frame next1 = GetNextKey(first);
				Frame next2 = GetNextKey(next1);
				if(next2 == Module.Data.Frames[Module.Data.Frames.Length-1]) {
					float ratio = 1f - next1.Timestamp / Module.Data.GetTotalTime();
					SetPhase(first, ratio);
					SetPhase(next2, ratio);
				} else {
					float xFirst = next1.Timestamp - first.Timestamp;
					float mFirst = next2.Timestamp - next1.Timestamp;
					SetPhase(first, Mathf.Clamp(1f - xFirst / mFirst, 0f, 1f));
				}
			}
			if(b.Index == Module.Data.GetTotalFrames()) {
				Frame last = Module.Data.Frames[Module.Data.GetTotalFrames()-1];
				Frame previous1 = GetPreviousKey(last);
				Frame previous2 = GetPreviousKey(previous1);
				if(previous2 == Module.Data.Frames[0]) {
					float ratio = 1f - previous1.Timestamp / Module.Data.GetTotalTime();
					SetPhase(last, ratio);
					SetPhase(previous2, ratio);
				} else {
					float xLast = last.Timestamp - previous1.Timestamp;
					float mLast = previous1.Timestamp - previous2.Timestamp;
					SetPhase(last, Mathf.Clamp(xLast / mLast, 0f, 1f));
				}
			}
		}

		public void Inspector(MotionEditor editor) {
			UltiDraw.Begin();
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Frame frame = editor.GetCurrentFrame();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField(this == Module.RegularPhaseFunction ? "Regular" : "Inverse");
				}

				if(IsKey(frame)) {
					SetPhase(frame, EditorGUILayout.Slider("Phase", GetPhase(frame), 0f, 1f));
				} else {
					EditorGUI.BeginDisabledGroup(true);
					SetPhase(frame, EditorGUILayout.Slider("Phase", GetPhase(frame), 0f, 1f));
					EditorGUI.EndDisabledGroup();
				}

				if(IsKey(frame)) {
					if(Utility.GUIButton("Unset Key", UltiDraw.Grey, UltiDraw.White)) {
						SetKey(frame, false);
					}
				} else {
					if(Utility.GUIButton("Set Key", UltiDraw.DarkGrey, UltiDraw.White)) {
						SetKey(frame, true);
					}
				}
				
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					editor.LoadFrame((GetPreviousKey(frame).Timestamp));
				}

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = frame.Timestamp-editor.GetWindow()/2f;
				float endTime = frame.Timestamp+editor.GetWindow()/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > Module.Data.GetTotalTime()) {
					startTime -= endTime-Module.Data.GetTotalTime();
					endTime = Module.Data.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(Module.Data.GetTotalTime(), endTime);
				int start = Module.Data.GetFrame(startTime).Index;
				int end = Module.Data.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 prevPos = Vector3.zero;
				Vector3 newPos = Vector3.zero;
				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				//Sequences
				for(int i=0; i<Module.Data.Sequences.Length; i++) {
					float _start = (float)(Mathf.Clamp(Module.Data.Sequences[i].Start, start, end)-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(Module.Data.Sequences[i].End, start, end)-start) / (float)elements;
					float left = rect.x + _start * rect.width;
					float right = rect.x + _end * rect.width;
					Vector3 a = new Vector3(left, rect.y, 0f);
					Vector3 b = new Vector3(right, rect.y, 0f);
					Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
					Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
					UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
					UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
				}

				if(Module.ShowVelocities) {
					//Regular Velocities
					for(int i=1; i<elements; i++) {
						prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
						prevPos.y = rect.yMax - Module.RegularPhaseFunction.NVelocities[i+start-1] * rect.height;
						newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
						newPos.y = rect.yMax - Module.RegularPhaseFunction.NVelocities[i+start] * rect.height;
						UltiDraw.DrawLine(prevPos, newPos, this == Module.RegularPhaseFunction ? UltiDraw.Green : UltiDraw.Red);
					}

					//Inverse Velocities
					for(int i=1; i<elements; i++) {
						prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
						prevPos.y = rect.yMax - Module.InversePhaseFunction.NVelocities[i+start-1] * rect.height;
						newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
						newPos.y = rect.yMax - Module.InversePhaseFunction.NVelocities[i+start] * rect.height;
						UltiDraw.DrawLine(prevPos, newPos, this == Module.RegularPhaseFunction ? UltiDraw.Red : UltiDraw.Green);
					}
				}
				
				if(Module.ShowCycle) {
					//Cycle
					for(int i=1; i<elements; i++) {
						prevPos.x = rect.xMin + (float)(i-1)/(elements-1) * rect.width;
						prevPos.y = rect.yMax - NormalisedCycle[i+start-1] * rect.height;
						newPos.x = rect.xMin + (float)(i)/(elements-1) * rect.width;
						newPos.y = rect.yMax - NormalisedCycle[i+start] * rect.height;
						UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Yellow);
					}
				}

				//Phase
				//for(int i=1; i<Module.Data.Frames.Length; i++) {
				//	Frame A = Module.Data.Frames[i-1];
				//	Frame B = Module.Data.Frames[i];
				//	prevPos.x = rect.xMin + (float)(A.Index-start)/elements * rect.width;
				//	prevPos.y = rect.yMax - Mathf.Repeat(Phase[A.Index-1], 1f) * rect.height;
				//	newPos.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
				//	newPos.y = rect.yMax - Phase[B.Index-1] * rect.height;
				//	UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
				//	bottom.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
				//	top.x = rect.xMin + (float)(B.Index-start)/elements * rect.width;
				//}
				
				Frame A = Module.Data.GetFrame(start);
				if(A.Index == 1) {
					bottom.x = rect.xMin;
					top.x = rect.xMin;
					UltiDraw.DrawLine(bottom, top, UltiDraw.Magenta.Transparent(0.5f));
				}
				Frame B = GetNextKey(A);
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
						top.x = rect.xMin + (float)(Module.Data.GetFrame(floor).Index-start)/elements * rect.width;
						UltiDraw.DrawCircle(top, 5f, UltiDraw.White);
					}
					timestamp += 1f;
				}
				//

				//Current Pivot
				float pStart = (float)(Module.Data.GetFrame(Mathf.Clamp(frame.Timestamp-1f, 0f, Module.Data.GetTotalTime())).Index-start) / (float)elements;
				float pEnd = (float)(Module.Data.GetFrame(Mathf.Clamp(frame.Timestamp+1f, 0f, Module.Data.GetTotalTime())).Index-start) / (float)elements;
				float pLeft = rect.x + pStart * rect.width;
				float pRight = rect.x + pEnd * rect.width;
				Vector3 pA = new Vector3(pLeft, rect.y, 0f);
				Vector3 pB = new Vector3(pRight, rect.y, 0f);
				Vector3 pC = new Vector3(pLeft, rect.y+rect.height, 0f);
				Vector3 pD = new Vector3(pRight, rect.y+rect.height, 0f);
				UltiDraw.DrawTriangle(pA, pC, pB, UltiDraw.White.Transparent(0.1f));
				UltiDraw.DrawTriangle(pB, pC, pD, UltiDraw.White.Transparent(0.1f));
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...
				EditorGUILayout.EndVertical();

				if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
					editor.LoadFrame(GetNextKey(frame).Timestamp);
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

		public PhaseEvolution(PhaseFunction function) {
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
			Interval[] intervals = new Interval[Mathf.FloorToInt(Function.Module.Data.GetTotalTime() / Window) + 1];
			for(int i=0; i<intervals.Length; i++) {
				int start = Function.Module.Data.GetFrame(i*Window).Index-1;
				int end = Function.Module.Data.GetFrame(Mathf.Min(Function.Module.Data.GetTotalTime(), (i+1)*Window)).Index-2;
				if(end == Function.Module.Data.GetTotalFrames()-2) {
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

		public void Assign() {
			for(int i=0; i<Function.Module.Data.GetTotalFrames(); i++) {
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
					Function.SetPhase(Function.Module.Data.Frames[i], i == 0 ? 0f : 1f);
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
				velocity += Function.Module.RegularPhaseFunction.Velocities[i] + Function.Module.InversePhaseFunction.Velocities[i];
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
					float y1 = Evolution.Function == Evolution.Function.Module.RegularPhaseFunction ? Evolution.Function.Module.RegularPhaseFunction.Velocities[i] : Evolution.Function.Module.InversePhaseFunction.Velocities[i];
					float y2 = Evolution.Function == Evolution.Function.Module.RegularPhaseFunction ? Evolution.Function.Module.InversePhaseFunction.Velocities[i] : Evolution.Function.Module.RegularPhaseFunction.Velocities[i];
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
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Function.Module.Data.Framerate, 
					genes[4], 
					frame/Evolution.Function.Module.Data.Framerate
					);
			}

			public float Phenotype1(float[] genes, int frame) {
				return Utility.LinSin1(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Function.Module.Data.Framerate,
					genes[4], 
					frame/Evolution.Function.Module.Data.Framerate
					);
			}

			public float Phenotype2(float[] genes, int frame) {
				return Utility.LinSin2(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Function.Module.Data.Framerate,
					genes[4], 
					frame/Evolution.Function.Module.Data.Framerate
					);
			}

			public float Phenotype3(float[] genes, int frame) {
				return Utility.LinSin3(
					genes[0], 
					genes[1], 
					genes[2], 
					genes[3] - (float)(frame-Interval.Start)*genes[4]/Evolution.Function.Module.Data.Framerate,
					genes[4], 
					frame/Evolution.Function.Module.Data.Framerate
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

}
#endif