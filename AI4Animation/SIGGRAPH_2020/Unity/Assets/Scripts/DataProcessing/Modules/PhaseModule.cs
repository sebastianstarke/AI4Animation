#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Threading.Tasks;

public class PhaseModule : Module {

    public enum Optimizer {Evolution, Cobyla, NelderMead, ConjugateGradient, BFGS};

    public enum Rescaling {None, Window, Adaptive, Snap};

    public bool ShowNormalized = true;
    public bool ShowHighlighted = true;
    public bool ShowValues = true;
    public bool ShowFitting = true;
    public bool ShowZero = true;
    public bool ShowPhase = true;
    public bool ShowWindow = true;
    public bool DisplayValues = true;

    public Optimizer Technique = Optimizer.Evolution;
    public int MaxIterations = 10;
    public int Individuals = 50;
    public int Elites = 5;
    public float MaxFrequency = 4f;
    public float Exploration = 0.2f;
    public float Memetism = 0.1f;

    public bool ApplyButterworth = true;

    public bool LocalVelocities = false;

    public Rescaling RescalingMethod = Rescaling.Window;

    [NonSerialized] private bool ShowParameters = false;

    public Function[] Functions = new Function[0];

    private string[] Names = null;

    private bool Token = false;
    private bool[] Threads = null;
    private int[] Iterations = null;
    private float[] Progress = null;

    private Precomputable<float[]>[] PrecomputedRegularPhases = null;
    private Precomputable<float[]>[] PrecomputedInversePhases = null;
    private Precomputable<float[]>[] PrecomputedRegularAmplitudes = null;
    private Precomputable<float[]>[] PrecomputedInverseAmplitudes = null;

	public override ID GetID() {
		return ID.Phase;
	}

    public override void DerivedResetPrecomputation() {
        PrecomputedRegularPhases = Data.ResetPrecomputable(PrecomputedRegularPhases);
        PrecomputedInversePhases = Data.ResetPrecomputable(PrecomputedInversePhases);
        PrecomputedRegularAmplitudes = Data.ResetPrecomputable(PrecomputedRegularAmplitudes);
        PrecomputedInverseAmplitudes = Data.ResetPrecomputable(PrecomputedInverseAmplitudes);
    }

    public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
        PhaseSeries instance = new PhaseSeries(global, GetNames());
        for(int i=0; i<instance.Samples.Length; i++) {
            instance.Phases[i] = GetPhases(timestamp + instance.Samples[i].Timestamp, mirrored);
            instance.Amplitudes[i] = GetAmplitudes(timestamp + instance.Samples[i].Timestamp, mirrored);
        }
        return instance;
    }

	protected override void DerivedInitialize() {
        ContactModule module = Data.GetModule<ContactModule>();
        if(module != null) {
            SetFunctions(module.GetNames());
        } else {
            Functions = new Function[0];
        }
	}

	protected override void DerivedLoad(MotionEditor editor) {
        Validate();
    }

	protected override void DerivedCallback(MotionEditor editor) {

    }


	protected override void DerivedInspector(MotionEditor editor) {
        Validate();

        Frame frame = editor.GetCurrentFrame();
        Vector3Int view = editor.GetView();

        ShowNormalized = EditorGUILayout.Toggle("Show Normalized", ShowNormalized);
        ShowHighlighted = EditorGUILayout.Toggle("Show Highlighted", ShowHighlighted);
        ShowValues = EditorGUILayout.Toggle("Show Values", ShowValues);
        ShowFitting = EditorGUILayout.Toggle("Show Fitting", ShowFitting);
        ShowZero = EditorGUILayout.Toggle("Show Zero", ShowZero);
        ShowPhase = EditorGUILayout.Toggle("Show Phase", ShowPhase);
        ShowWindow = EditorGUILayout.Toggle("Show Window", ShowWindow);
        DisplayValues = EditorGUILayout.Toggle("Display Values", DisplayValues);

        Technique = (Optimizer)EditorGUILayout.EnumPopup("Optimizer", Technique);
        MaxIterations = EditorGUILayout.IntField("Max Iterations", MaxIterations);
        Individuals = EditorGUILayout.IntField("Individuals", Individuals);
        Elites = EditorGUILayout.IntField("Elites", Elites);
        MaxFrequency = EditorGUILayout.FloatField("Max Frequency", MaxFrequency);
        Exploration = EditorGUILayout.Slider("Exploration", Exploration, 0f, 1f);
        Memetism = EditorGUILayout.Slider("Memetism", Memetism, 0f, 1f);

        ApplyButterworth = EditorGUILayout.Toggle("Apply Butterworth", ApplyButterworth);
        LocalVelocities = EditorGUILayout.Toggle("Local Velocities", LocalVelocities);

        RescalingMethod = (Rescaling)EditorGUILayout.EnumPopup("Rescaling Method", RescalingMethod);

        ShowParameters = EditorGUILayout.Toggle("Show Parameters", ShowParameters);

        if(Utility.GUIButton("Compute", UltiDraw.DarkGrey, UltiDraw.White)) {
            foreach(Function f in Functions) {
                f.Compute(true);
            }
        }

        bool fitting = IsFitting();
        EditorGUI.BeginDisabledGroup(Token);
        if(Utility.GUIButton(fitting ? "Stop" : "Optimise", fitting ? UltiDraw.DarkRed : UltiDraw.DarkGrey, UltiDraw.White)) {
            if(!fitting) {
                StartFitting();
            } else {
                StopFitting();
            }
        }
        EditorGUI.EndDisabledGroup();
        
        float max = 0f;
        if(ShowHighlighted) {
            foreach(Function f in Functions) {
                max = Mathf.Max(max, (float)f.Amplitudes.Max());
            }
        }

        float height = 50f;

        for(int i=0; i<Functions.Length; i++) {
            Function f = Functions[i];
            EditorGUILayout.BeginHorizontal();

            Utility.GUIButton(f.GetName(), UltiDraw.GetRainbowColor(i, Functions.Length).Opacity(0.5f), UltiDraw.Black, 150f, height);

            EditorGUILayout.BeginVertical(GUILayout.Height(height));
            Rect ctrl = EditorGUILayout.GetControlRect();
            Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
            EditorGUI.DrawRect(rect, UltiDraw.Black);

            UltiDraw.Begin();

            Vector3 prevPos = Vector3.zero;
            Vector3 newPos = Vector3.zero;
            Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
            Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

            //Zero
            if(ShowZero) {
                prevPos.x = rect.xMin;
                prevPos.y = rect.yMax - (float)(ShowNormalized ? 0.0.Normalize(f.MinValue, f.MaxValue, 0f, 1f) : 0f) * rect.height;
                newPos.x = rect.xMin + rect.width;
                newPos.y = rect.yMax - (float)(ShowNormalized ? 0.0.Normalize(f.MinValue, f.MaxValue, 0f, 1f) : 0f) * rect.height;
                UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(0.5f));
            }

            //Values
            if(ShowValues) {
                for(int j=1; j<view.z; j++) {
                    prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                    prevPos.y = rect.yMax - (float)f.GetValue(view.x+j-1-1, ShowNormalized) * rect.height;
                    newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    newPos.y = rect.yMax - (float)f.GetValue(view.x+j-1, ShowNormalized) * rect.height;
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
                }
            }

            //Fitting
            if(ShowFitting) {
                for(int j=1; j<view.z; j++) {
                    prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                    prevPos.y = rect.yMax - (float)f.GetFit(view.x+j-1-1, ShowNormalized) * rect.height;
                    newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    newPos.y = rect.yMax - (float)f.GetFit(view.x+j-1, ShowNormalized) * rect.height;
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Green);
                }
            }

            //Phase
            if(ShowPhase) {
                for(int j=0; j<view.z; j++) {
                    prevPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    prevPos.y = rect.yMax;
                    newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    newPos.y = rect.yMax - (float)f.GetPhase(view.x+j-1) * rect.height;
                    float weight = ShowHighlighted ? (float)f.GetAmplitude(view.x+j-1).Normalize(0f, max, 0f, 1f) : 1f;
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Cyan.Opacity(weight));
                }
            }
            UltiDraw.End();

            //Phase Window
            if(ShowWindow) {
               int padding = f.GetPhaseWindow(frame.Index-1) / 2;
               editor.DrawRect(Data.GetFrame(frame.Index - padding), Data.GetFrame(frame.Index + padding), 1f, UltiDraw.Gold.Opacity(0.25f), rect);
            }

            //Window
            if(ShowWindow) {
                int padding = f.Windows[frame.Index-1] / 2;
                editor.DrawRect(Data.GetFrame(frame.Index - padding), Data.GetFrame(frame.Index + padding), 1f, UltiDraw.IndianRed.Opacity(0.5f), rect);
            }

            editor.DrawPivot(rect);
            
            EditorGUILayout.EndVertical();

            //Progress Bar
            if(fitting && Threads != null && Iterations != null && Progress != null && Threads.Length > 0 && Iterations.Length > 0 && Progress.Length > 0) {
                float ratio = (float)Iterations[i] / (float)MaxIterations;
                // EditorGUILayout.LabelField(Mathf.RoundToInt(100f * ratio) + "%", GUILayout.Width(40f));

                EditorGUI.DrawRect(new Rect(ctrl.x, ctrl.y, ratio * ctrl.width, height), UltiDraw.Lerp(UltiDraw.Red, UltiDraw.Green, ratio).Opacity(0.5f));
                
                // if(Progress[i] > 0f && Progress[i] < 1f) {
                //     EditorGUI.DrawRect(new Rect(ctrl.x, ctrl.y, Progress[i] * ctrl.width, height), UltiDraw.Lerp(UltiDraw.Red, UltiDraw.Green, Progress[i]).Opacity(0.5f));
                // }
            }

            EditorGUILayout.EndHorizontal();

            if(DisplayValues) {
                EditorGUILayout.BeginHorizontal();
                EditorGUI.BeginDisabledGroup(true);

                // float value = (float)Functions[i].GetValue(editor.GetCurrentFrame().Index-1, false);
                // value = Mathf.Round(value * 100f) / 100f;
                // EditorGUILayout.FloatField(value, GUILayout.Width(35f));

                GUILayout.FlexibleSpace();

                float amplitude = (float)GetAmplitude(i, editor.GetCurrentFrame().Timestamp, false);
                amplitude = Mathf.Round(amplitude * 100f) / 100f;
                EditorGUILayout.LabelField("Amplitude", GUILayout.Width(100f));
                EditorGUILayout.FloatField(amplitude, GUILayout.Width(50f));

                float phase = (float)GetPhase(i, editor.GetCurrentFrame().Timestamp, false);
                phase = Mathf.Round(phase * 100f) / 100f;
                EditorGUILayout.LabelField("Phase", GUILayout.Width(100f));
                EditorGUILayout.FloatField(phase, GUILayout.Width(50f));
                
                GUILayout.FlexibleSpace();

                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();
            }
        }
        {
            EditorGUILayout.BeginHorizontal();

            Utility.GUIButton("Amplitudes", UltiDraw.White, UltiDraw.Black, 150f, height);

            EditorGUILayout.BeginVertical(GUILayout.Height(height));
            Rect ctrl = EditorGUILayout.GetControlRect();
            Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
            EditorGUI.DrawRect(rect, UltiDraw.Black);

            UltiDraw.Begin();

            Vector3 prevPos = Vector3.zero;
            Vector3 newPos = Vector3.zero;
            Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
            Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

            for(int i=0; i<Functions.Length; i++) {
                Function f = Functions[i];
                for(int j=1; j<view.z; j++) {
                    prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                    prevPos.y = rect.yMax - (float)f.GetAmplitude(view.x+j-1-1).Normalize(0f, max, 0f, 1f) * rect.height;
                    newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    newPos.y = rect.yMax - (float)f.GetAmplitude(view.x+j-1).Normalize(0f, max, 0f, 1f) * rect.height;
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.GetRainbowColor(i, Functions.Length));
                }
            }

            UltiDraw.End();

            editor.DrawPivot(rect);
            
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndHorizontal();
        }
        EditorGUILayout.HelpBox("Active Threads: " + (Threads == null ? 0 : Threads.Count(true)), MessageType.None);
    }

    protected override void DerivedGUI(MotionEditor editor) {

    }

	protected override void DerivedDraw(MotionEditor editor) {
        UltiDraw.Begin();
        if(ShowParameters) {
            float[] timestamps = Data.GetTimestamps(editor.GetCurrentFrame().Timestamp - editor.PastWindow, editor.GetCurrentFrame().Timestamp + editor.FutureWindow);
            float[][] amplitudes = new float[timestamps.Length][];
            float[][] frequencies = new float[timestamps.Length][];
            float[][] shifts = new float[timestamps.Length][];
            float[][] offsets = new float[timestamps.Length][];
            for(int i=0; i<timestamps.Length; i++) {
                amplitudes[i] = GetAmplitudes(timestamps[i], editor.Mirror);
                frequencies[i] = GetFrequencies(timestamps[i], editor.Mirror);
                shifts[i] = GetShifts(timestamps[i], editor.Mirror);
                offsets[i] = GetOffsets(timestamps[i], editor.Mirror);
            }
            UltiDraw.PlotFunctions(new Vector2(0.5f, 0.6f), new Vector2(0.5f, 0.1f), amplitudes, UltiDraw.Dimension.Y, -1f, 1f);
            UltiDraw.PlotFunctions(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.1f), frequencies, UltiDraw.Dimension.Y, 0f, MaxFrequency);
            UltiDraw.PlotFunctions(new Vector2(0.5f, 0.4f), new Vector2(0.5f, 0.1f), shifts, UltiDraw.Dimension.Y, -1f, 1f);
            UltiDraw.PlotFunctions(new Vector2(0.5f, 0.3f), new Vector2(0.5f, 0.1f), offsets, UltiDraw.Dimension.Y, -1f, 1f);
        }
        UltiDraw.End();
	}

    public string[] GetNames() {
        if(!Names.Verify(Functions.Length)) {
            Names = new string[Functions.Length];
            for(int i=0; i<Functions.Length; i++) {
                Names[i] = Functions[i].GetName();
            }
        }
        return Names;
    }

    public void SetFunctions(params string[] bones) {
        Functions = new Function[bones.Length];
        for(int i=0; i<Functions.Length; i++) {
            Functions[i] = new Function(this, Data.Source.FindBone(bones[i]).Index);
        }
        Names = null;
    }

    public float[] GetPhases(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInversePhases[index] == null) {
				PrecomputedInversePhases[index] = new Precomputable<float[]>(Compute());
			}
			if(!mirrored && PrecomputedRegularPhases[index] == null) {
				PrecomputedRegularPhases[index] = new Precomputable<float[]>(Compute());
			}
			return mirrored ? PrecomputedInversePhases[index].Value : PrecomputedRegularPhases[index].Value;
		}

        return Compute();
        float[] Compute() {
            float[] values = new float[Functions.Length];
            for(int i=0; i<Functions.Length; i++) {
                values[i] = GetPhase(i, timestamp, mirrored);
            }
            return values;
        }
    }

    public float GetPhase(int function, float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float repeated = Mathf.Repeat(pivot-start, end-start) + start;
            return
            Mathf.Repeat(
                (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Data.GetFrame(boundary).Index-1) -
                Utility.SignedPhaseUpdate(
                    (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Data.GetFrame(boundary).Index-1),
                    (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Data.GetFrame(repeated).Index-1)
                ), 1f
            );
        } else {
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Data.GetFrame(timestamp).Index-1);
        }
    }

    public float[] GetAmplitudes(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseAmplitudes[index] == null) {
				PrecomputedInverseAmplitudes[index] = new Precomputable<float[]>(Compute());
			}
			if(!mirrored && PrecomputedRegularAmplitudes[index] == null) {
				PrecomputedRegularAmplitudes[index] = new Precomputable<float[]>(Compute());
			}
			return mirrored ? PrecomputedInverseAmplitudes[index].Value : PrecomputedRegularAmplitudes[index].Value;
		}

        return Compute();
        float[] Compute() {
            float[] values = new float[Functions.Length];
            for(int i=0; i<Functions.Length; i++) {
                values[i] = GetAmplitude(i, timestamp, mirrored);
            }
            return values;
        }
    }

    public float GetAmplitude(int function, float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float repeated = Mathf.Repeat(pivot-start, end-start) + start;
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetAmplitude(Data.GetFrame(repeated).Index-1);
        } else {
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetAmplitude(Data.GetFrame(timestamp).Index-1);
        }
    }

    public float[] GetFrequencies(float timestamp, bool mirrored) {
        float[] values = new float[Functions.Length];
        for(int i=0; i<Functions.Length; i++) {
            values[i] = GetFrequency(i, timestamp, mirrored);
        }
        return values;
    }

    public float GetFrequency(int function, float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float repeated = Mathf.Repeat(pivot-start, end-start) + start;
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetFrequency(Data.GetFrame(repeated).Index-1);
        } else {
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetFrequency(Data.GetFrame(timestamp).Index-1);
        }
    }

    public float[] GetShifts(float timestamp, bool mirrored) {
        float[] values = new float[Functions.Length];
        for(int i=0; i<Functions.Length; i++) {
            values[i] = GetShift(i, timestamp, mirrored);
        }
        return values;
    }

    public float GetShift(int function, float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float repeated = Mathf.Repeat(pivot-start, end-start) + start;
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetShift(Data.GetFrame(repeated).Index-1);
        } else {
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetShift(Data.GetFrame(timestamp).Index-1);
        }
    }

    public float[] GetOffsets(float timestamp, bool mirrored) {
        float[] values = new float[Functions.Length];
        for(int i=0; i<Functions.Length; i++) {
            values[i] = GetOffset(i, timestamp, mirrored);
        }
        return values;
    }

    public float GetOffset(int function, float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float repeated = Mathf.Repeat(pivot-start, end-start) + start;
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetOffset(Data.GetFrame(repeated).Index-1);
        } else {
            return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetOffset(Data.GetFrame(timestamp).Index-1);
        }
    }

    public float[] GetUpdateRates(float from, float to, bool mirrored) {
        float[] rates = new float[Functions.Length];
        for(int i=0; i<Functions.Length; i++) {
            float delta = 0f;
            for(float t=from; t<to-Data.GetDeltaTime(); t+=Data.GetDeltaTime()) {
                delta += Utility.SignedPhaseUpdate(GetPhase(i, t, mirrored), GetPhase(i, t+Data.GetDeltaTime(), mirrored));
            }
            rates[i] = delta / (to-from);
        }
        return rates;
    }

    public void StartFitting() {
        foreach(Function f in Functions) {
            f.Preprocess();
        }
        Token = false;
        Threads = new bool[Functions.Length];
        Iterations = new int[Functions.Length];
        Progress = new float[Functions.Length];
        for(int i=0; i<Functions.Length; i++) {
            int thread = i;
            Threads[thread] = true;
            Task.Factory.StartNew(() => {
                Functions[thread].Optimise(ref Token, ref Threads, ref Iterations, ref Progress, thread);
            });
        }
    }

    public void StopFitting() {
        Token = true;
        Task.Factory.StartNew(() => {
            while(IsFitting()) {
                System.Threading.Thread.Sleep(1);
            }
            Token = false;
        });
    }

    public bool IsFitting() {
        if(Threads != null && Threads.Any(true)) {
            return true;
        } else {
            Token = false;
            Threads = null;
            Iterations = null;
            return false;
        }
    }

    private void Validate() {
		if(Functions.Length == Data.Source.Bones.Length) {
            ContactModule module = Data.GetModule<ContactModule>();
            if(Functions.Length != module.Sensors.Length) {
                Debug.LogWarning("Auto-removing unused alignment functions in asset " + Data.GetName() + ".");
                int[] indices = Data.Source.GetBoneIndices(module.GetNames());
                Functions = Functions.GatherByIndices(indices);
            }
        }
        foreach(Function f in Functions) {
            f.Values = f.Values.Validate(Data.Frames.Length);
            f.Fit = f.Fit.Validate(Data.Frames.Length);
            f.Phases = f.Phases.Validate(Data.Frames.Length);
            f.Amplitudes = f.Amplitudes.Validate(Data.Frames.Length);
            f.Windows = f.Windows.Validate(Data.Frames.Length);
        }
    }

    [System.Serializable]
    public class Solution {
        public double[] Values;
        public Solution() {
            Values = new double[Function.Dimensionality];
        }
    }

	[System.Serializable]
    public class Function {
        public PhaseModule Module;
        public int Bone;

        public double MinValue;
        public double MaxValue;

        public double[] Values;
        public double[] Fit;
        public double[] Phases;
        public double[] Amplitudes;

        public int[] Windows;

        public Solution[] Solutions;

        public const int Dimensionality = 4;

        private Function Symmetric = null;

        public Function(PhaseModule module, int bone) {
            Module = module;
            Bone = bone;
            Values = new double[Module.Data.GetTotalFrames()];
            Fit = new double[Module.Data.GetTotalFrames()];
            Phases = new double[Module.Data.GetTotalFrames()];
            Amplitudes = new double[Module.Data.GetTotalFrames()];
            Windows = new int[Module.Data.GetTotalFrames()];
        }

        public Function GetSymmetricFunction() {
            if(Symmetric == null) {
                int selfIndex = Module.Data.Source.FindBone(GetName()).Index;
                int otherIndex = Module.Data.Symmetry[selfIndex];
                string otherName = Module.Data.Source.Bones[otherIndex].Name;
                Symmetric = System.Array.Find(Module.Functions, x => x.GetName() == otherName);
            }
            return Symmetric == null ? this : Symmetric;
        }

        public string GetName() {
            return Module.Data.Source.Bones[Bone].Name;
        }

        public double GetValue(int index, bool normalized) {
            Values = Values.Validate(Module.Data.Frames.Length);
            return normalized ? Values[index].Normalize(MinValue, MaxValue, 0.0, 1.0) : Values[index];
        }

        public double GetFit(int index, bool normalized) {
            Fit = Fit.Validate(Module.Data.Frames.Length);
            return normalized ? Fit[index].Normalize(MinValue, MaxValue, 0.0, 1.0) : Fit[index];
        }

        public double GetPhase(int index) {
            Phases = Phases.Validate(Module.Data.Frames.Length);
            return Phases[index];
        }

        public double GetAmplitude(int index) {
            Amplitudes = Amplitudes.Validate(Module.Data.Frames.Length);
            return System.Math.Max(Amplitudes[index], 0.0);
        }

        public double GetFrequency(int index) {
            return Solutions[index].Values[1];
        }

        public double GetShift(int index) {
            return Solutions[index].Values[2];
        }

        public double GetOffset(int index) {
            return Solutions[index].Values[3];
        }

        public void Preprocess() {
            ContactModule contact = Module.Data.GetModule<ContactModule>();
            ContactModule.Sensor sensor = contact.GetSensor(GetName());

            //Check Active
            if(sensor.Contacts.All(0f) || sensor.Contacts.All(1f)) {
                Values.SetAll(0.0);
                Windows.SetAll(0);
                return;
            }

            //Start Processing
            for(int i=0; i<Values.Length; i++) {
                Values[i] = GetValue(Module.Data.Frames[i]);
                float GetValue(Frame frame) {
                    if(sensor == null) {
                        return 0f;
                    }
                    if(Module.RescalingMethod == Rescaling.None) {
                        return sensor.GetContact(frame, false);
                    }
                    if(Module.RescalingMethod == Rescaling.Window) {
                        float[] timestamps = Module.Data.SimulateTimestamps(frame, 1f/2f, 1f/2f); //HARDCODED Window = 1s
                        float[] contacts = new float[timestamps.Length];
                        for(int t=0; t<timestamps.Length; t++) {
                            contacts[t] = sensor.GetContact(timestamps[t], false);
                        }
                        float c = sensor.GetContact(frame, false);
                        float mean = contacts.Mean();
                        float std = contacts.Sigma();
                        std = std == 0f ? 1f : std;
                        return (c-mean) / std;
                    }
                    if(Module.RescalingMethod == Rescaling.Adaptive) {
                        float[] timestamps = Module.Data.SimulateTimestamps(frame, sensor.GetStateWindow(Module.Data.Frames[i], false) * Module.Data.GetDeltaTime());
                        float sum = 0f;
                        for(int t=0; t<timestamps.Length; t++) {
                            sum += sensor.GetContact(timestamps[t], false);
                        }
                        float ratio = sum / timestamps.Length;
                        float lower = -ratio;
                        float upper = 1f-ratio;
                        return sensor.GetContact(frame, false).Normalize(0f, 1f, lower, upper);
                    }
                    if(Module.RescalingMethod == Rescaling.Snap) {
                        Frame start = Module.Data.Frames.First();
                        Frame end = Module.Data.Frames.Last();
                        if(sensor.GetContact(frame, false) == 1f) {
                            Frame s = contact.GetPreviousContactEnd(frame, false, sensor);
                            Frame e = contact.GetNextContactStart(frame, false, sensor);
                            start = s != null ? Module.Data.GetFrame(s.Index+1) : start;
                            end = e != null ? Module.Data.GetFrame(e.Index+1) : end;
                        } else {
                            Frame s = contact.GetPreviousContactStart(frame, false, sensor);
                            Frame e = contact.GetNextContactEnd(frame, false, sensor);
                            start = s != null ? s : start;
                            end = e != null ? e : end;
                        }
                        Frame[] frames = Module.Data.GetFrames(start.Index, end.Index);
                        float sum = 0f;
                        for(int f=0; f<frames.Length; f++) {
                            sum += sensor.GetContact(frames[f], false);
                        }
                        float ratio = sum / frames.Length; 
                        float lower = -ratio;
                        float upper = 1f-ratio;
                        return sensor.GetContact(frame, false).Normalize(0f, 1f, lower, upper);
                    }
                    return 0f;
                }
            }

            if(Module.ApplyButterworth) {
                Values = Utility.Butterworth(Values, (double)Module.Data.GetDeltaTime(), (double)Module.MaxFrequency);
            }

            for(int i=0; i<Values.Length; i++) {
                if(System.Math.Abs(Values[i]) < 1e-3) {
                    Values[i] = 0.0;
                }
            }

            //Compute Windows
            for(int i=0; i<Values.Length; i++) {
                int window = sensor.GetStateWindow(Module.Data.Frames[i], false);
                float active = 1f - Values.GatherByOverflowWindow(i, window / 2).Ratio(0.0);
                Windows[i] = Mathf.RoundToInt(active * window);
            }
            int[] tmp = new int[Values.Length];
            for(int i=0; i<Values.Length; i++) {
                int[] windows = Windows.GatherByOverflowWindow(i, 30);      //HARDCODED Window = 1s
                double[] values = Values.GatherByOverflowWindow(i, 30);     //HARDCODED Window = 1s
                bool[] mask = new bool[values.Length];
                for(int j=0; j<values.Length; j++) {
                    mask[j] = values[j] != 0f;
                }
                tmp[i] = Mathf.RoundToInt(windows.Gaussian(mask));
            }
            Windows = tmp;

            MinValue = Values.Min();
            MaxValue = Values.Max();
        }

        public void Optimise(ref bool token, ref bool[] threads, ref int[] iterations, ref float[] progress, int thread) {
            Solutions = new Solution[Values.Length];
            for(int i=0; i<Solutions.Length; i++) {
                Solutions[i] = new Solution();
            }
            Population[] populations = new Population[Values.Length];
            for(int i=0; i<Values.Length; i++) {
                int padding = Windows[i] / 2;
                Frame[] frames = Module.Data.Frames.GatherByOverflowWindow(i, padding);
                double[] values = Values.GatherByOverflowWindow(i, padding);
                if(values.AbsSum() < 1e-3) {
                    continue;
                }
                double min = values.Min();
                double max = values.Max();
                double[] lowerBounds = new double[Dimensionality]{0.0, 1f/Module.MaxFrequency, -0.5, min};
                double[] upperBounds = new double[Dimensionality]{max-min, Module.MaxFrequency, 0.5, max};
                double[] seed = new double[Dimensionality];
                for(int j=0; j<Dimensionality; j++) {
                    seed[j] = 0.5 * (lowerBounds[j] + upperBounds[j]);
                }
                double[] _g = new double[Dimensionality];
                double[] _tmp = new double[Dimensionality];
                Sequence _seq = new Sequence(frames.First().Index, frames.Last().Index);
                System.Func<double[], double> func = x => Loss(_seq, x);
                System.Func<double[], double[]> grad = g => Grad(_seq, g, 0.1, _g, _tmp);
                populations[i] = new Population(this, Module.Individuals, Module.Elites, Module.Exploration, Module.Memetism, lowerBounds, upperBounds, seed, func, grad);
            }

            iterations[thread] = 0;
            while(!token) {
                //Iterate
                for(int i=0; i<populations.Length && !token; i++) {
                    if(populations[i] != null) {
                        populations[i].Evolve();
                        Solutions[i].Values = populations[i].GetSolution();
                    }
                    progress[thread] = (float)(i+1) / (float)populations.Length;
                }
                Compute(false);
                iterations[thread] += 1;
                if(Module.MaxIterations > 0 && iterations[thread] >= Module.MaxIterations) {
                    break;
                }
            }
            Compute(true);
            threads[thread] = false;
        }

        public void Compute(bool postprocess) {
            if(Solutions == null || Solutions.Length != Values.Length || Solutions.Any(null)) {
                Debug.Log("Computing failed because no solutions are available.");
                return;
            }
            for(int i=0; i<Values.Length; i++) {
                // Windows[i] = GetPhaseWindow(i);
                Fit[i] = ComputeFit(i);
                Phases[i] = ComputePhase(i);
                Amplitudes[i] = ComputeAmplitude(i);
            }
            if(postprocess) {
                double[] px = new double[Phases.Length];
                double[] py = new double[Phases.Length];
                for(int i=0; i<Phases.Length; i++) {
                    Vector2 v = Utility.PhaseVector((float)Phases[i]);
                    px[i] = v.x;
                    py[i] = v.y;
                }
                px = Utility.Butterworth(px, (double)Module.Data.GetDeltaTime(), (double)Module.MaxFrequency);
                py = Utility.Butterworth(py, (double)Module.Data.GetDeltaTime(), (double)Module.MaxFrequency);
                for(int i=0; i<Phases.Length; i++) {
                    Phases[i] = Utility.PhaseValue(new Vector2((float)px[i], (float)py[i]).normalized);
                }
                Amplitudes = Utility.Butterworth(Amplitudes, (double)Module.Data.GetDeltaTime(), (double)Module.MaxFrequency);
            }

            double ComputeFit(int index) {
                return Trigonometric(index, Solutions[index].Values);
            }

            double ComputePhase(int index) {
                double[] x = Solutions[index].Values;
                double t = Module.Data.Frames[index].Timestamp;
                double F = x[1];
                double S = x[2];
                return Mathf.Repeat((float)(F * t - S), 1f);
            }
            
            double ComputeAmplitude(int index) {
                Frame[] window = Module.Data.Frames.GatherByOverflowWindow(index, GetPhaseWindow(index) / 2);

                // return GetSmoothedAmplitude(window);

                // return Solutions[index].Values[0] * Module.Data.Frames[index].GetBoneVelocity(Bone, false).magnitude;

                return GetSmoothedAmplitude(window) * GetMaxVelocityMagnitude(window);
            }
        }

        public int GetPhaseWindow(int index) {
            if(Solutions == null || Solutions.Length != Values.Length || Solutions.Any(null)) {
                return 0;
            }
            float f = (float)Solutions[index].Values[1];
            return f == 0f ? 0 : Mathf.RoundToInt(Module.Data.Framerate / f);
        }

        private double GetSmoothedAmplitude(Frame[] frames) {
			double[] values = new double[frames.Length];
			for(int i=0; i<frames.Length; i++) {
				values[i] = Solutions[frames[i].Index-1].Values[0];
			}
            return values.Gaussian();
        }

        private float GetMaxVelocityMagnitude(Frame[] frames) {
            if(!Module.LocalVelocities) {
                float magnitude = 0f;
                for(int i=0; i<frames.Length; i++) {
                    magnitude = Mathf.Max(frames[i].GetBoneVelocity(Bone, false).magnitude, magnitude);
                }
                return magnitude;
            } else {
                RootModule m = Module.Data.GetModule<RootModule>();
                float magnitude = 0f;
                for(int i=0; i<frames.Length; i++) {
                    magnitude = Mathf.Max((frames[i].GetBoneVelocity(Bone, false) - m.GetRootVelocity(frames[i].Timestamp, false)).magnitude, magnitude);
                }
                return magnitude;
            }
        }

        private double Trigonometric(int index, double[] x) {
            double t = Module.Data.Frames[index].Timestamp;
            double A = x[0];
            double F = x[1];
            double S = x[2];
            double B = x[3];
            return A * System.Math.Sin(2.0*System.Math.PI * (F * t - S)) + B;
        }

        private double Loss(Sequence seq, double[] x) {
            double loss = 0.0;
            double count = 0.0;
            for(int i=seq.Start; i<=seq.End; i++) {
                Accumulate(i);
            }
            int padding = seq.GetLength() / 2;
            for(int i=1; i<=padding; i++) {
                double w = 1.0 - (double)(i)/(double)(padding); //Mean
                // double w = Mathf.Exp(-Mathf.Pow((float)i - (float)padding, 2f) / Mathf.Pow(0.5f * (float)padding, 2f)); //Gaussian
                w *= w;
                Connect(seq.Start - i, w);
                Connect(seq.End + i, w);
            }

            loss /= count;
            loss = System.Math.Sqrt(loss);

            return loss;

            void Accumulate(int frame) {
                if(frame >= 1 && frame <= Values.Length) {
                    double error = Values[frame-1] - Trigonometric(frame-1, x);
                    error *= error;
                    loss += error;
                    count += 1.0;
                }
            }

            void Connect(int frame, double weight) {
                if(frame >= 1 && frame <= Values.Length) {
                    // double weight = System.Math.Abs(Values[frame-1]);
                    double error = Fit[frame-1] - Trigonometric(frame-1, x);
                    error *= error;
                    loss += weight * error;
                    count += weight;
                }
            }
        }

        private double[] Grad(Sequence seq, double[] x, double delta, double[] grad, double[] tmp) {
            for(int i=0; i<x.Length; i++) {
                tmp[i] = x[i];
            }
            double loss = Loss(seq, tmp);
            for(int i=0; i<tmp.Length; i++) {
                tmp[i] += delta;
                grad[i] = (Loss(seq, tmp) - loss) / delta;
                tmp[i] -= delta;
            }
            return grad;
        }

        private class Population {

            private Function Function;
            private System.Random RNG;

            private int Size;
            private int Elites;
            private int Dimensionality;
            private double Exploration;
            private double Memetism;
            public double[] LowerBounds;
            public double[] UpperBounds;

            private double[] RankProbabilities;
            private double RankProbabilitySum;

            private System.Func<double[], double> Func;
            private System.Func<double[], double[]> Grad;

            private Individual[] Individuals;
            private Individual[] Offspring;
            
            private Accord.Math.Optimization.Cobyla Memetic;

            private Accord.Math.Optimization.Cobyla Cobyla;
            private Accord.Math.Optimization.NelderMead NelderMead;
            private Accord.Math.Optimization.ConjugateGradient ConjugateGradient;
            private Accord.Math.Optimization.BroydenFletcherGoldfarbShanno BFGS;

            public Population(Function function, int size, int elites, float exploration, float memetism, double[] lowerBounds, double[] upperBounds, double[] seed, System.Func<double[], double> func, System.Func<double[], double[]> grad) {
                Function = function;
                RNG = new System.Random();

                Size = size;
                Elites = elites;
                Dimensionality = seed.Length;
                Exploration = exploration;
                Memetism = memetism;

                LowerBounds = lowerBounds;
                UpperBounds = upperBounds;

                Func = func;
                Grad = grad;

                //Setup Alternatives
                Cobyla = new Accord.Math.Optimization.Cobyla(Dimensionality, Func);
                NelderMead = new Accord.Math.Optimization.NelderMead(Dimensionality, Func);
                ConjugateGradient = new Accord.Math.Optimization.ConjugateGradient(Dimensionality, Func, Grad);
                BFGS = new Accord.Math.Optimization.BroydenFletcherGoldfarbShanno(Dimensionality, Func, Grad);

                //Setup Memetic
                Memetic = new Accord.Math.Optimization.Cobyla(Dimensionality, Func);
                Memetic.MaxIterations = 10;

                //Compute rank probabilities
                double rankSum = (double)(Size*(Size+1)) / 2.0;
                RankProbabilities = new double[Size];
                for(int i=0; i<Size; i++) {
                    RankProbabilities[i] = (double)(Size-i)/rankSum;
                    RankProbabilitySum += RankProbabilities[i];
                }

                //Create population
                Individuals = new Individual[Size];
                Offspring = new Individual[Size];
                for(int i=0; i<size; i++) {
                    Individuals[i] = new Individual(Dimensionality);
                    Offspring[i] = new Individual(Dimensionality);
                }

                //Initialise randomly
                Individuals[0].Genes = (double[])seed.Clone();
                for(int i=1; i<size; i++) {
                    Reroll(Individuals[i]);
                }

                //Finalise
                EvaluateFitness(Offspring);
                SortByFitness(Offspring);
                AssignExtinctions(Offspring);
            }

            public double[] GetSolution() {
                return Individuals[0].Genes;
            }
            
            public void Evolve() {
                //Alternatives
                if(Function.Module.Technique == Optimizer.Cobyla) {
                    Cobyla.MaxIterations = Function.Module.MaxIterations;
                    Cobyla.Minimize(Individuals.First().Genes);
                    for(int j=0; j<Cobyla.Solution.Length; j++) {
                        Individuals.First().Genes[j] = Cobyla.Solution[j];
                    }
                    return;
                }
                if(Function.Module.Technique == Optimizer.NelderMead) {
                    NelderMead.Minimize(Individuals.First().Genes);
                    for(int j=0; j<NelderMead.Solution.Length; j++) {
                        Individuals.First().Genes[j] = NelderMead.Solution[j];
                    }
                    return;
                }
                if(Function.Module.Technique == Optimizer.ConjugateGradient) {
                    ConjugateGradient.MaxIterations = Function.Module.MaxIterations;
                    ConjugateGradient.Minimize(Individuals.First().Genes);
                    for(int j=0; j<ConjugateGradient.Solution.Length; j++) {
                        Individuals.First().Genes[j] = ConjugateGradient.Solution[j];
                    }
                    return;
                }
                if(Function.Module.Technique == Optimizer.BFGS) {
                    BFGS.MaxIterations = Function.Module.MaxIterations;
                    BFGS.Minimize(Individuals.First().Genes);
                    for(int j=0; j<BFGS.Solution.Length; j++) {
                        Individuals.First().Genes[j] = BFGS.Solution[j];
                    }
                    return;
                }

                //Copy elite
                for(int i=0; i<Elites; i++) {
                    Copy(Individuals[i], Offspring[i]);                    
                }

                //Remaining individuals
                for(int o=Elites; o<Size; o++) {
                    Individual child = Offspring[o];
                    if(GetRandom() <= 1.0-Exploration) {
                        Individual parentA = Select(Individuals);
                        Individual parentB = Select(Individuals);
                        while(parentB == parentA) {
                            parentB = Select(Individuals);
                        }
                        Individual prototype = Select(Individuals);
                        while(prototype == parentA || prototype == parentB) {
                            prototype = Select(Individuals);
                        }

                        double mutationRate = GetMutationProbability(parentA, parentB);
                        double mutationStrength = GetMutationStrength(parentA, parentB);

                        for(int i=0; i<Dimensionality; i++) {
                            double weight;

                            //Recombination
                            weight = GetRandom();
                            double momentum = GetRandom() * parentA.Momentum[i] + GetRandom() * parentB.Momentum[i];
                            if(GetRandom() < 0.5) {
                                child.Genes[i] = parentA.Genes[i] + momentum;
                            } else {
                                child.Genes[i] = parentB.Genes[i] + momentum;
                            }

                            //Store
                            double gene = child.Genes[i];

                            //Mutation
                            if(GetRandom() <= mutationRate) {
                                double span = UpperBounds[i] - LowerBounds[i];
                                child.Genes[i] += GetRandom(-mutationStrength*span, mutationStrength*span);
                            }
                            
                            //Adoption
                            weight = GetRandom();
                            child.Genes[i] += 
                                weight * GetRandom() * (0.5f * (parentA.Genes[i] + parentB.Genes[i]) - child.Genes[i])
                                + (1.0-weight) * GetRandom() * (prototype.Genes[i] - child.Genes[i]);

                            //Clamp
                            if(child.Genes[i] < LowerBounds[i]) {
                                child.Genes[i] = LowerBounds[i];
                            }
                            if(child.Genes[i] > UpperBounds[i]) {
                                child.Genes[i] = UpperBounds[i];
                            }

                            //Momentum
                            child.Momentum[i] = GetRandom() * momentum + (child.Genes[i] - gene);
                        }
                    } else {
                        Reroll(child);
                    }
                }

                //Memetic Local Search
                for(int i=0; i<Offspring.Length; i++) {
                    if(GetRandom() <= Memetism) {
                        Memetic.Minimize(Offspring[i].Genes);
                        for(int j=0; j<Memetic.Solution.Length; j++) {
                            if(Memetic.Solution[j] < LowerBounds[j]) {
                                Memetic.Solution[j] = LowerBounds[j];
                            }
                            if(Memetic.Solution[j] > UpperBounds[j]) {
                                Memetic.Solution[j] = UpperBounds[j];
                            }
                            Offspring[i].Momentum[j] = Memetic.Solution[j] - Offspring[i].Genes[j];
                            Offspring[i].Genes[j] = Memetic.Solution[j];
                        }
                    }
                }

                //Finalise
                EvaluateFitness(Offspring);
                SortByFitness(Offspring);
                AssignExtinctions(Offspring);

                //Update
                Utility.Swap(ref Individuals, ref Offspring);
            }

            private double GetRandom(double min=0.0, double max=1.0) { 
                return RNG.NextDouble() * (max - min) + min;
            }

            //Copies an individual from from to to
			private void Copy(Individual from, Individual to) {
				for(int i=0; i<Dimensionality; i++) {
					to.Genes[i] = from.Genes[i];
					to.Momentum[i] = from.Momentum[i];
				}
				to.Extinction = from.Extinction;
				to.Fitness = from.Fitness;
			}

            //Rerolls an individual
			private void Reroll(Individual individual) {
				for(int i=0; i<Dimensionality; i++) {
					individual.Genes[i] = GetRandom(LowerBounds[i], UpperBounds[i]);
                    individual.Momentum[i] = 0.0;
				}
			}

            //Rank-based selection of an individual
            private Individual Select(Individual[] entities) {
                double rVal = GetRandom() * RankProbabilitySum;
                for(int i=0; i<Size; i++) {
                    rVal -= RankProbabilities[i];
                    if(rVal <= 0.0) {
                        return entities[i];
                    }
                }
                return entities[Size-1];
            }

			//Returns the mutation probability from two parents
			private double GetMutationProbability(Individual parentA, Individual parentB) {
				double extinction = 0.5 * (parentA.Extinction + parentB.Extinction);
				double inverse = 1.0/(double)Dimensionality;
				return extinction * (1.0-inverse) + inverse;
			}

			//Returns the mutation strength from two parents
			private double GetMutationStrength(Individual parentA, Individual parentB) {
				return 0.5 * (parentA.Extinction + parentB.Extinction);
			}

            private void EvaluateFitness(Individual[] entities) {
                for(int i=0; i<entities.Length; i++) {
                    entities[i].Fitness = Func(entities[i].Genes);
                }
            }

            //Sorts all individuals starting with best (lowest) fitness
            private void SortByFitness(Individual[] entities) {
                System.Array.Sort(entities,
                    delegate(Individual a, Individual b) {
                        return a.Fitness.CompareTo(b.Fitness);
                    }
                );
            }

            //Compute extinction values
            private void AssignExtinctions(Individual[] entities) {
                double min = entities[0].Fitness;
                double max = entities[Size-1].Fitness;
                for(int i=0; i<entities.Length; i++) {
                    double grading = (double)i/((double)entities.Length-1.0);
                    entities[i].Extinction = (entities[i].Fitness + min*(grading-1.0)) / max;
                }
            }

            private class Individual {
                public double Fitness;
                public double[] Genes;
                public double[] Momentum;
                public double Extinction;

                public Individual(int dimensionality) {
                    Genes = new double[dimensionality];
                    Momentum = new double[dimensionality];
                }
            }

        }

    }
}
#endif




                // Vector3 Velocity(int bone, int frame) {
                //     Frame previous = Module.Data.Frames[Mathf.Clamp(frame-1, 0, Module.Data.Frames.Length-1)];
                //     Frame current = Module.Data.Frames[Mathf.Clamp(frame, 0, Module.Data.Frames.Length-1)];

                //     RootModule root = (RootModule)Module.Data.GetModule(AlignmentModule.ID.Root);
                //     Matrix4x4 previousRoot = root.GetRootTransformation(previous.Timestamp, false);
                //     Matrix4x4 currentRoot = root.GetRootTransformation(current.Timestamp, false);

                //     float delta = 1f/Module.Data.Framerate;
                //     Vector3 velocity = current.GetBoneTransformation(bone, false).GetRelativeTransformationTo(currentRoot).GetPosition() - previous.GetBoneTransformation(bone, false).GetRelativeTransformationTo(previousRoot).GetPosition();
                //     return velocity /= delta;
                // }

                // Vector3 Acceleration(int bone, int frame) {
                //     Frame previous2 = Module.Data.Frames[Mathf.Clamp(frame-2, 0, Module.Data.Frames.Length-1)];
                //     Frame previous1 = Module.Data.Frames[Mathf.Clamp(frame-1, 0, Module.Data.Frames.Length-1)];
                //     Frame current = Module.Data.Frames[Mathf.Clamp(frame, 0, Module.Data.Frames.Length-1)];

                //     RootModule root = (RootModule)Module.Data.GetModule(AlignmentModule.ID.Root);
                //     Matrix4x4 previous2Root = root.GetRootTransformation(previous2.Timestamp, false);
                //     Matrix4x4 previous1Root = root.GetRootTransformation(previous1.Timestamp, false);
                //     Matrix4x4 currentRoot = root.GetRootTransformation(current.Timestamp, false);

                //     float delta = 1f/Module.Data.Framerate;
                //     Vector3 previousVelocity = previous1.GetBoneTransformation(bone, false).GetRelativeTransformationTo(previous1Root).GetPosition() - previous2.GetBoneTransformation(bone, false).GetRelativeTransformationTo(previous2Root).GetPosition();
                //     Vector3 currentVelocity = current.GetBoneTransformation(bone, false).GetRelativeTransformationTo(currentRoot).GetPosition() - previous1.GetBoneTransformation(bone, false).GetRelativeTransformationTo(previous1Root).GetPosition();
                //     previousVelocity /= delta;
                //     currentVelocity /= delta;
                //     return (currentVelocity - previousVelocity) / delta;
                // }

                
            // } else {
            //     Frame previous = Module.Data.Frames[i-1 == -1 ? 0 : i-1];
            //     Frame current = Module.Data.Frames[i];
            //     float delta = 1f/Module.Data.Framerate;
            //     if(Bone == Module.Data.Symmetry[Bone]) {
            //         RootModule rootModule = (RootModule)Module.Data.GetModule(AlignmentModule.ID.Root);
            //         Vector3 a = previous.GetBoneTransformation(Bone, false).GetPosition().GetRelativePositionTo(rootModule.GetRootTransformation(previous.Timestamp, false));
            //         Vector3 b = current.GetBoneTransformation(Bone, false).GetPosition().GetRelativePositionTo(rootModule.GetRootTransformation(current.Timestamp, false));
            //         Values[i] = (b.magnitude - a.magnitude) / delta;
            //     } else {
            //         Vector3 a = current.GetBoneVelocity(Bone, false, delta);
            //         Vector3 b = current.GetBoneVelocity(Module.Data.Symmetry[Bone], false, delta);
            //         Values[i] = b.magnitude - a.magnitude;
            //     }
            // }

                    // double magnitude = 0.0;
                    // double count = 0.0;
                    // int padding = Module.Window / 2;
                    // for(int i=0; i<=padding; i++) {
                    //     if(i==0) {
                    //         Accumulate(index, 1.0);
                    //     } else {
                    //         // double w = 1.0 - (double)(i)/(double)(window); //Mean
                    //         float w = Mathf.Exp(-Mathf.Pow((float)i - (float)padding, 2f) / Mathf.Pow(0.5f * (float)padding, 2f)); //Gaussian
                    //         Accumulate(index - i, w);
                    //         Accumulate(index + i, w);
                    //     }
                    //     void Accumulate(int frame, double weight) {
                    //         if(frame >= 0 && frame < Values.Length) {
                    //             if(populations[frame] != null) {
                    //                 magnitude += weight * populations[frame].GetSolution()[0];
                    //                 count += weight;
                    //             }
                    //         }
                    //     }
                    // }
                    // return magnitude / count;