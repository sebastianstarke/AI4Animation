using System;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class StyleModule : Module {
		public enum STATE {Passive, Active, Inactive}

		public int Window = 30;

		public Key[] Keys = new Key[0];
		public Function[] Functions = new Function[0];

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global, GetNames());
			for(int i=0; i<instance.Samples.Length; i++) {
				instance.Values[i] = GetValues(timestamp + instance.Samples[i].Timestamp, mirrored);
				// for(int j=0; j<Functions.Length; j++) {
				// 	if(Functions[j].Type == Series.TYPE.Continuous) {
				// 		instance.Values[i][j] *= Asset.Scale;
				// 	}
				// }
			}
			for(int i=0; i<instance.Styles.Length; i++) {
				instance.DisplayMin[i] = Functions[i].DisplayMin;
				instance.DisplayMax[i] = Functions[i].DisplayMax;
			}
			return instance;
		}
#if UNITY_EDITOR
		protected override void DerivedInitialize() {
			Clear();
		}

		protected override void DerivedLoad(MotionEditor editor) {
			foreach(Key key in Keys) {
				key.States = key.States.Validate(Functions.Length);
			}
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
	
		protected override void DerivedCallback(MotionEditor editor) {
			
		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
			// UltiDraw.Begin();
			// VRModule module = Asset.GetModule<VRModule>();
			// UltiDraw.DrawSphere(ExtractFootLocation(editor.GetTimestamp(), editor.Mirror, module.LeftAnkle, module.RightAnkle), Quaternion.identity, 0.25f, Color.magenta);
			// UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Frame frame = editor.GetCurrentFrame();
			
			Color GetColorForState(Color color, STATE state) {
				if(state == STATE.Passive) {
					return UltiDraw.Grey;
				}
				if(state == STATE.Active) {
					return color.Opacity(1f);
				}
				if(state == STATE.Inactive) {
					return color.Opacity(0.5f);
				}
				return UltiDraw.Transparent;
			}

			Color[] colors = UltiDraw.GetRainbowColors(Functions.Length);
			for(int i=0; i<Functions.Length; i++) {
				float height = 25f;
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton(Functions[i].Name, GetColorForState(colors[i], GetKey(frame).States[i]), UltiDraw.White, 150f, height)) {
					Functions[i].Toggle(frame);
				}
				Rect c = EditorGUILayout.GetControlRect();
				Rect r = new Rect(c.x, c.y, Functions[i].GetValue(frame, editor.Mirror) * c.width, height);
				EditorGUI.DrawRect(r, colors[i].Opacity(0.75f));
				EditorGUILayout.FloatField(Functions[i].GetValue(frame, editor.Mirror), GUILayout.Width(50f));
				Functions[i].Name = EditorGUILayout.TextField(Functions[i].Name);
				Functions[i].DisplayMin = EditorGUILayout.FloatField(Functions[i].DisplayMin);
				Functions[i].DisplayMax = EditorGUILayout.FloatField(Functions[i].DisplayMax);
				Functions[i].Type = (Series.TYPE)EditorGUILayout.EnumPopup(Functions[i].Type);
				if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f)) {
					RemoveFunction(Functions[i].Name);
				}
				EditorGUILayout.EndHorizontal();
			}

			Window = EditorGUILayout.IntField("Window", Window);

			if(Utility.GUIButton("Add Function", UltiDraw.DarkGrey, UltiDraw.White)) {
				AddFunction("Function " + (Functions.Length+1));
				EditorGUIUtility.ExitGUI();
			}
			EditorGUILayout.BeginHorizontal();
			if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
				editor.LoadFrame(GetPreviousKey(frame).Timestamp);
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
			if(endTime > Asset.GetTotalTime()) {
				startTime -= endTime-Asset.GetTotalTime();
				endTime = Asset.GetTotalTime();
			}
			startTime = Mathf.Max(0f, startTime);
			endTime = Mathf.Min(Asset.GetTotalTime(), endTime);
			int start = Asset.GetFrame(startTime).Index;
			int end = Asset.GetFrame(endTime).Index;
			int elements = end-start;

			//Styles
			for(int i=0; i<Functions.Length; i++) {
				for(int j=start; j<end; j++) {
					int prev = j;
					int next = j+1;
					float prevValue = GetValue(Asset.GetFrame(Mathf.Clamp(prev, start, end)).Timestamp, editor.Mirror, i);
					float nextValue = GetValue(Asset.GetFrame(Mathf.Clamp(next, start, end)).Timestamp, editor.Mirror, i);

					prevValue = prevValue.Normalize(Functions[i].DisplayMin, Functions[i].DisplayMax, 0f, 1f);
					nextValue = nextValue.Normalize(Functions[i].DisplayMin, Functions[i].DisplayMax, 0f, 1f);

					float _start = (float)(Mathf.Clamp(prev, start, end)-start) / (float)elements;
					float _end = (float)(Mathf.Clamp(next, start, end)-start) / (float)elements;
					float xStart = rect.x + _start * rect.width;
					float xEnd = rect.x + _end * rect.width;
					float yStart = rect.y + (1f - prevValue) * rect.height;
					float yEnd = rect.y + (1f - nextValue) * rect.height;
					UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[i]);
				}
			}

			//Keys
			Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
			Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
			for(int i=start-1; i<end-1; i++) {
				if(Keys[i].IsActive()) {
					top.x = rect.xMin + (float)(i+1-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(i+1-start)/elements * rect.width;
					UltiDraw.DrawLine(top, bottom, UltiDraw.White);
				}
			}

			UltiDraw.End();

			editor.DrawPivot(rect);
			
			EditorGUILayout.EndVertical();
			if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f)) {
				editor.LoadFrame(GetNextKey(frame).Timestamp);
			}
			EditorGUILayout.EndHorizontal();
		}
#endif

		public void Clear() {
			Keys = new Key[Asset.Frames.Length];
			for(int i=0; i<Keys.Length; i++) {
				Keys[i] = new Key(this);
			}
			for(int i=0; i<Functions.Length; i++) {
				Functions[i].StandardValues.SetAll(0f);
				Functions[i].MirroredValues.SetAll(0f);
			}
			Functions = new Function[0];
		}

		public bool HasFunction(string name) {
			return Array.Exists(Functions, x => x.Name == name);
		}

		public Function AddFunction(string name) {
			if(HasFunction(name)) {
				Debug.Log("Function with name " + name + " already exists in asset: " + Asset.name);
				return GetFunction(name);
			} else {
				Function function = ArrayExtensions.Append(ref Functions, new Function(this, name));
				for(int i=0; i<Keys.Length; i++) {
					ArrayExtensions.Append(ref Keys[i].States, STATE.Passive);
				}
				return function;
			}
		}

		public void AddFunctions(params string[] names) {
			for(int i=0; i<names.Length; i++) {
				AddFunction(names[i]);
			}
		}

		public void RemoveFunction(string name) {
			int index = Array.FindIndex(Functions, x => x.Name == name);
			if(index >= 0) {
				ArrayExtensions.RemoveAt(ref Functions, index);
				for(int i=0; i<Keys.Length; i++) {
					ArrayExtensions.RemoveAt(ref Keys[i].States, index);
				}
			} else {
				Debug.Log("Function with name " + name + " does not exist.");
			}
		}

		public Function GetFunction(string name) {
			return Array.Find(Functions, x => x.Name == name);
		}

		public float[] GetValues(float timestamp, bool mirrored) {
			float[] values = new float[Functions.Length];
			for(int i=0; i<Functions.Length; i++) {
				values[i] = GetValue(timestamp, mirrored, i);
			}
			return values;
		}

		private float GetValue(float timestamp, bool mirrored, int index) {
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
			if(timestamp < start || timestamp > end) {
				float boundary = Mathf.Clamp(timestamp, start, end);
				float pivot = 2f*boundary - timestamp;
				float clamped = Mathf.Clamp(pivot, start, end);
				return Functions[index].GetValue(Asset.GetFrame(clamped), mirrored);
			} else {
				return Functions[index].GetValue(Asset.GetFrame(timestamp), mirrored);
			}
		}

		public string[] GetNames() {
			string[] names = new string[Functions.Length];
			for(int i=0; i<Functions.Length; i++) {
				names[i] = Functions[i].Name;
			}
			return names;
		}

		public bool IsKey(Frame frame) {
			return Keys[frame.Index-1].IsActive();
		}

		public Key GetKey(Frame frame) {
			return Keys[frame.Index-1];
		}

		public Frame GetPreviousKey(Frame frame) {
			while(frame.Index > 1) {
				frame = Asset.GetFrame(frame.Index-1);
				if(IsKey(frame)) {
					return frame;
				}
			}
			return Asset.Frames.First();
		}

		public Frame GetPreviousStateKey(Frame frame, int index) {
			while(frame.Index > 1) {
				frame = Asset.GetFrame(frame.Index-1);
				if(IsKey(frame) && (GetKey(frame).States[index] == STATE.Active || GetKey(frame).States[index] == STATE.Inactive)) {
					return frame;
				}
			}
			return Asset.Frames.First();
		}

		public Frame GetNextKey(Frame frame) {
			while(frame.Index < Asset.Frames.Length) {
				frame = Asset.GetFrame(frame.Index+1);
				if(IsKey(frame)) {
					return frame;
				}
			}
			return Asset.Frames.Last();
		}

		public Frame GetNextStateKey(Frame frame, int index) {
			while(frame.Index < Asset.Frames.Length) {
				frame = Asset.GetFrame(frame.Index+1);
				if(IsKey(frame) && (GetKey(frame).States[index] == STATE.Active || GetKey(frame).States[index] == STATE.Inactive)) {
					return frame;
				}
			}
			return Asset.Frames.Last();
		}

		public bool EstimateStepping(float timestamp, bool mirrored, params ContactModule.Sensor[] sensors) {
			foreach(ContactModule.Sensor sensor in sensors) {
				if(sensor.GetContact(timestamp, mirrored) == 0f) {
					return true;
				}
			}
			return false;
		}

		public float EstimateSpeed(RootModule.Series rootSeries) {
			float speed = 0f;
			for(int i=0; i<rootSeries.Samples.Length; i++) {
				speed += rootSeries.Velocities[i].magnitude;
			}
			return speed /= rootSeries.Samples.Length;
		}

		public bool EstimateJumping(
			float timestamp, 
			bool mirrored, 
			int head,
			int hips,
			int leftFoot, 
			int rightFoot, 
			float headHeightThreshold,
			float hipsHeightThreshold,
			float feetHeightThresold
			// CenterOfGravityModule cogModule,
			// int[] cogBones, 
			// float cogHeightThreshld, 
			// ContactModule.Sensor[] sensors
		) {
			// bool isInAir = cogModule.GetCenterOfGravity(timestamp, mirrored, null, cogBones).y > cogHeightThreshld;
			float GetHeight(int bone) {
				return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored).GetPosition().y;
			}
			bool headInAir = GetHeight(head) > headHeightThreshold;
			bool hipsInAir = GetHeight(hips) > hipsHeightThreshold;
			bool leftFootInAir = GetHeight(leftFoot) > feetHeightThresold;
			bool righFootInAir = GetHeight(rightFoot) > feetHeightThresold;
			// bool isContacting = false;
			// foreach(ContactModule.Sensor sensor in sensors) {
			// 	isContacting = isContacting || sensor.GetContact(timestamp, mirrored) == 1f;
			// }
			return headInAir && hipsInAir && leftFootInAir && righFootInAir;
		}

        // public Vector3 ExtractFootLocation(float timestamp, bool mirrored, int leftAnkle, int rightAnkle) {
        //     float[] timestamps = Asset.SimulateTimestamps(2f);
        //     Vector3[] values = new Vector3[timestamps.Length];
        //     for(int i=0; i<timestamps.Length; i++) {
        //         Frame frame = Asset.GetFrame(timestamp);
        //         Vector3 leftFoot = frame.GetBoneTransformation(leftAnkle, mirrored).GetPosition();
        //         Vector3 rightFoot = frame.GetBoneTransformation(rightAnkle, mirrored).GetPosition();
        //         Vector3 pivot = 0.5f * (leftFoot + rightFoot);
        //         values[i] = pivot.ZeroY();
        //     }
        //     return values.Gaussian();
        // }

        // public float ExtractHorizontalFeetPivot(float timestamp, bool mirrored, RootModule rootModule, int hips, int leftAnkle, int rightAnkle) {
        //     float[] timestamps = Asset.SimulateTimestamps(timestamp-1f, timestamp+1f);
        //     float[] values = new float[timestamps.Length];
        //     for(int i=0; i<timestamps.Length; i++) {
        //         Frame frame = Asset.GetFrame(timestamp);
        //         Vector3 position = frame.GetBoneTransformation(hips, mirrored).GetPosition();
        //         Quaternion rotation = rootModule.GetRootRotation(timestamp, mirrored);
        //         Matrix4x4 reference = Matrix4x4.TRS(position, rotation, Vector3.one);
        //         Vector3 leftFoot = frame.GetBoneTransformation(leftAnkle, mirrored).GetPosition();
        //         Vector3 rightFoot = frame.GetBoneTransformation(rightAnkle, mirrored).GetPosition();
        //         leftFoot = leftFoot.PositionTo(reference);
        //         rightFoot = rightFoot.PositionTo(reference);
        //         float pivot = 0.5f * (leftFoot.x + rightFoot.x);
        //         values[i] = pivot;
        //     }
        //     return values.Gaussian();
        // }

		[Serializable]
		public class Key {
			public STATE[] States;
			public Key(StyleModule module) {
				States = new STATE[module.Functions.Length];
			}
			public bool IsActive() {
				return !States.All(STATE.Passive);
			}
		}

		[Serializable]
		public class Function {
			public StyleModule Module;
			public string Name;
			public Series.TYPE Type = Series.TYPE.Binary;
			public float[] StandardValues;
			public float[] MirroredValues;
			public float DisplayMin = 0f;
			public float DisplayMax = 1f;
			
			public Function(StyleModule module, string name) {
				Module = module;
				Name = name;
				StandardValues = new float[Module.Asset.Frames.Length];
				MirroredValues = new float[Module.Asset.Frames.Length];
			}

			public float GetValue(Frame frame, bool mirrored) {
				return mirrored ? MirroredValues[frame.Index-1] : StandardValues[frame.Index-1];
			}

			public int GetIndex() {
				return Module.Functions.FindIndex(this);
			}

			public void Toggle(Frame frame) {
				SetState(frame, (STATE)(((int)Module.GetKey(frame).States[GetIndex()] + 1) % 3), Module.Window);
			}

			public void SetState(Frame frame, STATE state, int window) {
				if(window == 0 || Module.IsKey(frame) || frame == Module.Asset.Frames.First() || frame == Module.Asset.Frames.Last()) {
					Module.GetKey(frame).States[GetIndex()] = state;
					Compute(frame);
				} else {
					Frame start = Module.Asset.GetFrame(Mathf.Clamp(frame.Index - window/2, 1, Module.Asset.Frames.Length));
					Frame end = Module.Asset.GetFrame(Mathf.Clamp(frame.Index + window/2, 1, Module.Asset.Frames.Length));
					int index = GetIndex();
					if(Module.GetKey(start).States[index] == STATE.Passive) {
						SetState(start, STATE.Inactive, 0);
						SetState(end, STATE.Active, 0);
					} else if(Module.GetKey(start).States[index] == STATE.Inactive) {
						SetState(start, STATE.Active, 0);
						SetState(end, STATE.Inactive, 0);
					} else if(Module.GetKey(start).States[index] == STATE.Active) {
						SetState(start, STATE.Passive, 0);
						SetState(end, STATE.Passive, 0);
					}
				}
			}

			public void Compute(Frame frame) {

				float GetTarget(Frame pivot) {
					if(Module.GetKey(pivot).States[GetIndex()] == STATE.Active) {
						return 1f;
					}
					if(Module.GetKey(pivot).States[GetIndex()] == STATE.Inactive) {
						return 0f;
					}
					// Debug.Log("Querying target for passive state. This should not have happened!");
					return 0f;
				}

				int index = Module.Functions.FindIndex(this);
				Frame current = frame;
				Frame previous = Module.GetPreviousStateKey(current, index);
				Frame next = Module.GetNextStateKey(current, index);

				if(Module.IsKey(frame) && Module.GetKey(frame).States[index] != STATE.Passive) {
					//Current Frame
					StandardValues[frame.Index-1] = GetTarget(frame);
					MirroredValues[frame.Index-1] = StandardValues[frame.Index-1];
					//Previous Frames
					{
						float a = GetTarget(previous);
						float b = GetTarget(current);
						for(int i=previous.Index; i<current.Index; i++) {
							float weight = i.Ratio(previous.Index, current.Index);
							weight = weight.SmoothStep(2f, 0.5f);
							StandardValues[i-1] = (1f-weight) * a + weight * b;
							MirroredValues[i-1] = StandardValues[i-1];
						}
					}
					//Next Frames
					{
						float a = GetTarget(current);
						float b = GetTarget(next);
						for(int i=current.Index+1; i<=next.Index; i++) {
							float weight = i.Ratio(current.Index, next.Index);
							weight = weight.SmoothStep(2f, 0.5f);
							StandardValues[i-1] = (1f-weight) * a + weight * b;
							MirroredValues[i-1] = StandardValues[i-1];
						}
					}
				} else {
					float A = GetTarget(previous);
					float B = GetTarget(next);
					for(int i=previous.Index; i<=next.Index; i++) {
						float weight = (float)(i-previous.Index) / (float)(next.Index-previous.Index);
						weight = weight.SmoothStep(2f, 0.5f);
						StandardValues[i-1] = (1f-weight)*A + weight*B;
						MirroredValues[i-1] = StandardValues[i-1];
					}
				}
			}
		}

		public class Series : TimeSeries.Component {
			public enum TYPE {Binary, Continuous}
			public string[] Styles;
			public float[][] Values;
			public TYPE[] Types;
			public float[] DisplayMin;
			public float[] DisplayMax;

			private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.8f, 0.8f, 0.3f, 0.15f);

			public Series(TimeSeries global, params string[] styles) : base(global) {
				Styles = styles;
				Values = new float[Samples.Length][];
				for(int i=0; i<Values.Length; i++) {
					Values[i] = new float[Styles.Length];
				}
				Types = new TYPE[styles.Length];
				Types.SetAll(TYPE.Binary);
				DisplayMin = new float[Styles.Length];
				DisplayMax = new float[Styles.Length];
				DisplayMin.SetAll(0f);
				DisplayMax.SetAll(1f);
			}

			public Series(TimeSeries global, string[] styles, float[] seed) : base(global) {
				Styles = styles;
				Values = new float[Samples.Length][];
				for(int i=0; i<Values.Length; i++) {
					Values[i] = new float[Styles.Length];
				}
				if(styles.Length != seed.Length) {
					Debug.Log("Given number of styles and seed do not match.");
					return;
				}
				for(int i=0; i<Values.Length; i++) {
					for(int j=0; j<Styles.Length; j++) {
						Values[i][j] = seed[j];
					}
				}
				DisplayMin = new float[Styles.Length];
				DisplayMax = new float[Styles.Length];
				DisplayMin.SetAll(0f);
				DisplayMax.SetAll(1f);
			}
			
			public void SetRange(int dim, float min, float max) {
				DisplayMin[dim] = min;
				DisplayMax[dim] = max;
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					for(int j=0; j<Styles.Length; j++) {
						Values[i][j] = Values[i+1][j];
					}
				}
			}

			public void Interpolate(int start, int end) {
				for(int i=start; i<end; i++) {
					float weight = (float)(i % Resolution) / (float)Resolution;
					int prevIndex = GetPreviousKey(i).Index;
					int nextIndex = GetNextKey(i).Index;
					for(int j=0; j<Styles.Length; j++) {
						Values[i][j] = Mathf.Lerp(Values[prevIndex][j], Values[nextIndex][j], weight);
					}
				}
			}

			public void Control(float[] actions, float weight) {
				Increment(0, Samples.Length-1);
				for(int i=Pivot; i<Samples.Length; i++) {
					for(int j=0; j<Styles.Length; j++) {
						Values[i][j] = Mathf.Lerp(
							Values[i][j], 
							actions[j],
							weight
						);
					}
				}
			}

			public void Control(float[] actions, float[] weights) {
				Increment(0, Samples.Length-1);
				for(int i=Pivot; i<Samples.Length; i++) {
					for(int j=0; j<Styles.Length; j++) {
						Values[i][j] = Mathf.Lerp(
							Values[i][j], 
							actions[j],
							weights[j]
						);
					}
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.GUIRect area = rect == null ? Rect : rect;
					UltiDraw.Begin();
					UltiDraw.OnGUILabel(area.GetCenter() + new Vector2(0f, 0.1f), area.GetSize(), 0.0175f, "Actions", UltiDraw.White);
					Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
					for(int i=0; i<Styles.Length; i++) {
						float value = Values[Pivot][i];
						UltiDraw.OnGUILabel(new Vector2(area.X, value.Normalize(0f, 1f, area.Y-area.H/2f, area.Y+area.H/2f)), area.GetSize(), 0.0175f, Styles[i], colors[i]);
					}
					UltiDraw.End();
				}
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.GUIRect area = rect == null ? Rect : rect;
					UltiDraw.Begin();
					Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
					float[][] values = Values.GetTranspose();
					UltiDraw.GUIRectangle(area.GetCenter(), area.GetSize(), UltiDraw.White.Opacity(0.75f));
					for(int i=0; i<Styles.Length; i++) {
						UltiDraw.PlotFunction(area.GetCenter(), area.GetSize(), values[i], yMin: DisplayMin[i], yMax: DisplayMax[i], thickness: 0.0025f, backgroundColor:UltiDraw.Transparent, lineColor:colors[i]);
					}
					UltiDraw.End();
				}
			}

			public void SetValue(int index, string name, float value) {
				int idx = ArrayExtensions.FindIndex(ref Styles, name);
				if(idx == -1) {
					// Debug.Log("Style " + name + " could not be found.");
					return;
				}
				Values[index][idx] = value;
			}

			public float GetValue(int index, string name) {
				int idx = ArrayExtensions.FindIndex(ref Styles, name);
				if(idx == -1) {
					// Debug.Log("Style " + name + " could not be found.");
					return 0f;
				}
				return Values[index][idx];
			}

			public float[] GetValues(int index, params string[] names) {
				if(names.Length == 0) {
					return Values[index];
				}
				float[] values = new float[names.Length];
				for(int i=0; i<names.Length; i++) {
					values[i] = GetValue(index, names[i]);
				}
				return values;
			}
		}

	}
}