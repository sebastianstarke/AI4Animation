using UnityEngine;
using System;
using System.Threading.Tasks;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    public class AlignmentModule : Module {

        #if UNITY_EDITOR
        public enum SOURCE {Contact, Velocity};

        public SOURCE Source = SOURCE.Contact;

        public bool ShowNormalized = true;
        public bool ShowHighlighted = true;
        public bool ShowSource = true;
        public bool ShowValues = true;
        public bool ShowFitting = true;
        public bool ShowZero = true;
        public bool ShowPhase = true;
        public bool ShowWindow = true;
        public bool DisplayValues = true;

        public int MaxIterations = 10;
        public int Individuals = 50;
        public int Elites = 5;
        public float MaxFrequency = 4f;
        public float Exploration = 0.2f;
        public float Memetism = 0.1f;

        public bool ApplyNormalization = true;
        public bool ApplyButterworth = true;
        
        public bool LocalVelocities = false;

        [NonSerialized] private bool ShowParameters = false;

        public Function[] Functions = new Function[0];

        private string[] Identifiers = null;

        private bool Token = false;
        private bool[] Threads = null;
        private int[] Iterations = null;
        private float[] Progress = null;

        private Precomputable<float[]> PrecomputedPhases = null;
        private Precomputable<float[]> PrecomputedAmplitudes = null;

        public override void DerivedResetPrecomputation() {
            PrecomputedPhases = new Precomputable<float[]>(this);
            PrecomputedAmplitudes = new Precomputable<float[]>(this);
        }

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            Series instance = new Series(global, GetIdentifiers());
            for(int i=0; i<instance.Samples.Length; i++) {
                instance.Phases[i] = GetPhases(timestamp + instance.Samples[i].Timestamp, mirrored);
                instance.Amplitudes[i] = GetAmplitudes(timestamp + instance.Samples[i].Timestamp, mirrored);
            }
            return instance;
        }

        protected override void DerivedInitialize() {

        }

        protected override void DerivedLoad(MotionEditor editor) {
            Validate();
        }

        protected override void DerivedUnload(MotionEditor editor) {

        }

        protected override void DerivedCallback(MotionEditor editor) {

        }

        protected override void DerivedInspector(MotionEditor editor) {
            Validate();

            Frame frame = editor.GetCurrentFrame();
            Vector3Int view = editor.GetView();

            Source = (SOURCE)EditorGUILayout.EnumPopup("Source", Source);

            ShowNormalized = EditorGUILayout.Toggle("Show Normalized", ShowNormalized);
            ShowHighlighted = EditorGUILayout.Toggle("Show Highlighted", ShowHighlighted);
            ShowSource = EditorGUILayout.Toggle("Show Source", ShowSource);
            ShowValues = EditorGUILayout.Toggle("Show Values", ShowValues);
            ShowFitting = EditorGUILayout.Toggle("Show Fitting", ShowFitting);
            ShowZero = EditorGUILayout.Toggle("Show Zero", ShowZero);
            ShowPhase = EditorGUILayout.Toggle("Show Phase", ShowPhase);
            ShowWindow = EditorGUILayout.Toggle("Show Window", ShowWindow);
            DisplayValues = EditorGUILayout.Toggle("Display Values", DisplayValues);

            MaxIterations = EditorGUILayout.IntField("Max Iterations", MaxIterations);
            Individuals = EditorGUILayout.IntField("Individuals", Individuals);
            Elites = EditorGUILayout.IntField("Elites", Elites);
            MaxFrequency = EditorGUILayout.FloatField("Max Frequency", MaxFrequency);
            Exploration = EditorGUILayout.Slider("Exploration", Exploration, 0f, 1f);
            Memetism = EditorGUILayout.Slider("Memetism", Memetism, 0f, 1f);

            ApplyNormalization = EditorGUILayout.Toggle("Apply Normalization", ApplyNormalization);
            ApplyButterworth = EditorGUILayout.Toggle("Apply Butterworth", ApplyButterworth);
            LocalVelocities = EditorGUILayout.Toggle("Local Velocities", LocalVelocities);

            ShowParameters = EditorGUILayout.Toggle("Show Parameters", ShowParameters);

            if(Utility.GUIButton("Compute", UltiDraw.DarkGrey, UltiDraw.White)) {
                foreach(Function f in Functions) {
                    f.Compute(true);
                }
            }

            bool fitting = IsFitting();
            EditorGUI.BeginDisabledGroup(Token);
            if(Utility.GUIButton(fitting ? "Stop" : "Optimize", fitting ? UltiDraw.DarkRed : UltiDraw.DarkGrey, UltiDraw.White)) {
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
                    prevPos.y = rect.yMax - (float)(ShowNormalized ? (0.0).Normalize(f.MinValue, f.MaxValue, 0f, 1f) : 0f) * rect.height;
                    newPos.x = rect.xMin + rect.width;
                    newPos.y = rect.yMax - (float)(ShowNormalized ? (0.0).Normalize(f.MinValue, f.MaxValue, 0f, 1f) : 0f) * rect.height;
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(0.5f));
                }

                //Source
                if(ShowSource) {
                    for(int j=1; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax - (float)f.GetSource(view.x+j-1-1, ShowNormalized) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)f.GetSource(view.x+j-1, ShowNormalized) * rect.height;
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White);
                    }
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
                    editor.DrawRect(Asset.GetFrame(frame.Index - padding), Asset.GetFrame(frame.Index + padding), 1f, UltiDraw.Gold.Opacity(0.25f), rect);
                }

                //Window
                if(ShowWindow) {
                    int padding = f.Windows[frame.Index-1] / 2;
                    editor.DrawRect(Asset.GetFrame(frame.Index - padding), Asset.GetFrame(frame.Index + padding), 1f, UltiDraw.IndianRed.Opacity(0.5f), rect);
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

                    float amplitude = (float)GetAmplitude(i, editor.GetTimestamp(), false);
                    amplitude = Mathf.Round(amplitude * 100f) / 100f;
                    EditorGUILayout.LabelField("Amplitude", GUILayout.Width(100f));
                    EditorGUILayout.FloatField(amplitude, GUILayout.Width(50f));

                    float phase = (float)GetPhase(i, editor.GetTimestamp(), false);
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
                float[] timestamps = Asset.GetTimestamps(editor.GetTimestamp() - editor.PastWindow, editor.GetTimestamp() + editor.FutureWindow);
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

        public string[] GetIdentifiers() {
            if(!Identifiers.Verify(Functions.Length)) {
                Identifiers = new string[Functions.Length];
                for(int i=0; i<Functions.Length; i++) {
                    Identifiers[i] = Functions[i].GetName();
                }
            }
            return Identifiers;
        }

        public float[] GetPhases(float timestamp, bool mirrored) {
            return PrecomputedPhases.Get(timestamp, mirrored, () => Compute());
            float[] Compute() {
                float[] values = new float[Functions.Length];
                for(int i=0; i<Functions.Length; i++) {
                    values[i] = GetPhase(i, timestamp, mirrored);
                }
                return values;
            }
        }

        public float GetPhase(int function, float timestamp, bool mirrored) {
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < 0f || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                return
                Mathf.Repeat(
                    (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Asset.GetFrame(boundary).Index-1) -
                    Utility.SignedPhaseUpdate(
                        (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Asset.GetFrame(boundary).Index-1),
                        (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Asset.GetFrame(repeated).Index-1)
                    ), 1f
                );
            } else {
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetPhase(Asset.GetFrame(timestamp).Index-1);
            }
        }

        public float[] GetAmplitudes(float timestamp, bool mirrored) {
            return PrecomputedAmplitudes.Get(timestamp, mirrored, () => Compute());
            float[] Compute() {
                float[] values = new float[Functions.Length];
                for(int i=0; i<Functions.Length; i++) {
                    values[i] = GetAmplitude(i, timestamp, mirrored);
                }
                return values;
            }
        }

        public float GetAmplitude(int function, float timestamp, bool mirrored) {
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < start || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetAmplitude(Asset.GetFrame(repeated).Index-1);
            } else {
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetAmplitude(Asset.GetFrame(timestamp).Index-1);
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
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < start || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetFrequency(Asset.GetFrame(repeated).Index-1);
            } else {
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetFrequency(Asset.GetFrame(timestamp).Index-1);
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
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < start || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetShift(Asset.GetFrame(repeated).Index-1);
            } else {
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetShift(Asset.GetFrame(timestamp).Index-1);
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
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < start || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetOffset(Asset.GetFrame(repeated).Index-1);
            } else {
                return (float)(mirrored ? Functions[function].GetSymmetricFunction() : Functions[function]).GetOffset(Asset.GetFrame(timestamp).Index-1);
            }
        }

        public float[] GetUpdateRates(float from, float to, bool mirrored) {
            float[] rates = new float[Functions.Length];
            for(int i=0; i<Functions.Length; i++) {
                float delta = 0f;
                for(float t=from; t<to-Asset.GetDeltaTime(); t+=Asset.GetDeltaTime()) {
                    delta += Utility.SignedPhaseUpdate(GetPhase(i, t, mirrored), GetPhase(i, t+Asset.GetDeltaTime(), mirrored));
                }
                rates[i] = delta / (to-from);
            }
            return rates;
        }

        public void StartFitting() {
            //Create Functions
            switch(Source) {
                case SOURCE.Contact:
                {
                    ContactModule source = Asset.GetModule<ContactModule>();
                    if(source == null) {
                        return;
                    } else {
                        Functions = new Function[source.Sensors.Length];
                        for(int i=0; i<Functions.Length; i++) {
                            Functions[i] = new Function(this, source.Sensors[i].Bone);
                        }
                    }
                }
                break;

                case SOURCE.Velocity:
                {
                    MotionModule source = Asset.GetModule<MotionModule>();
                    if(source == null) {
                        return;
                    } else {
                        Functions = new Function[source.Bones.Length];
                        for(int i=0; i<Functions.Length; i++) {
                            Functions[i] = new Function(this, source.Bones[i]);
                        }
                    }
                }
                break;
            }

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
                    Functions[thread].Optimize(ref Token, ref Threads, ref Iterations, ref Progress, thread);
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
            foreach(Function f in Functions) {
                f.Values = f.Values.Validate(Asset.Frames.Length);
                f.Fit = f.Fit.Validate(Asset.Frames.Length);
                f.Phases = f.Phases.Validate(Asset.Frames.Length);
                f.Amplitudes = f.Amplitudes.Validate(Asset.Frames.Length);
                f.Windows = f.Windows.Validate(Asset.Frames.Length);
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
            public AlignmentModule Module;
            public int Bone;

            public double MinSource;
            public double MaxSource;
            public double MinValue;
            public double MaxValue;

            public double[] Source;
            public double[] Values;
            public double[] Fit;
            public double[] Phases;
            public double[] Amplitudes;

            public int[] Windows;

            public Solution[] Solutions;

            public const int Dimensionality = 4;

            private Function Symmetric = null;

            public Function(AlignmentModule module, int bone) {
                Module = module;
                Bone = bone;
                Source = new double[Module.Asset.GetTotalFrames()];
                Values = new double[Module.Asset.GetTotalFrames()];
                Fit = new double[Module.Asset.GetTotalFrames()];
                Phases = new double[Module.Asset.GetTotalFrames()];
                Amplitudes = new double[Module.Asset.GetTotalFrames()];
                Windows = new int[Module.Asset.GetTotalFrames()];
            }

            public Function GetSymmetricFunction() {
                if(Symmetric == null) {
                    int selfIndex = Module.Asset.Source.FindBone(GetName()).Index;
                    int otherIndex = Module.Asset.Symmetry[selfIndex];
                    string otherName = Module.Asset.Source.Bones[otherIndex].Name;
                    Symmetric = System.Array.Find(Module.Functions, x => x.GetName() == otherName);
                }
                return Symmetric == null ? this : Symmetric;
            }

            public string GetName() {
                return Module.Asset.Source.Bones[Bone].Name;
            }

            public double GetSource(int index, bool normalized) {
                Source = Source.Validate(Module.Asset.Frames.Length);
                return normalized ? Source[index].Normalize(MinSource, MaxSource, 0.0, 1.0) : Source[index];
            }

            public double GetValue(int index, bool normalized) {
                Values = Values.Validate(Module.Asset.Frames.Length);
                return normalized ? Values[index].Normalize(MinValue, MaxValue, 0.0, 1.0) : Values[index];
            }

            public double GetFit(int index, bool normalized) {
                Fit = Fit.Validate(Module.Asset.Frames.Length);
                return normalized ? Fit[index].Normalize(MinValue, MaxValue, 0.0, 1.0) : Fit[index];
            }

            public double GetPhase(int index) {
                Phases = Phases.Validate(Module.Asset.Frames.Length);
                return Phases[index];
            }

            public double GetAmplitude(int index) {
                Amplitudes = Amplitudes.Validate(Module.Asset.Frames.Length);
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
                //Check Early Termination
                switch(Module.Source) {
                    case SOURCE.Contact:
                    ContactModule.Sensor sensor = Module.Asset.GetModule<ContactModule>().GetSensor(GetName());
                    if(sensor.Contacts.All(0f) || sensor.Contacts.All(1f)) {
                        Source.SetAll(0.0);
                        Values.SetAll(0.0);
                        Fit.SetAll(0.0);
                        Phases.SetAll(0.0);
                        Amplitudes.SetAll(0.0);
                        Windows.SetAll(0);
                        return;
                    }
                    break;
                }

                //Start Processing
                for(int i=0; i<Source.Length; i++) {
                    Source[i] = GetSource(Module.Asset.Frames[i]);
                    float GetSignal(Frame frame) {
                        switch(Module.Source) {
                            case SOURCE.Contact:
                            ContactModule.Sensor sensor = Module.Asset.GetModule<ContactModule>().GetSensor(GetName());
                            return Module.Asset.GetModule<ContactModule>().GetSensor(GetName()).GetContact(frame, false);

                            case SOURCE.Velocity:
                            return frame.GetBoneVelocity(Bone, false).magnitude;

                            default:
                            return 0f;
                        }
                    }
                    float GetSource(Frame frame) {
                        if(Module.ApplyNormalization) {
                            float[] timestamps = Module.Asset.SimulateTimestamps(frame, 1f/2f, 1f/2f); //HARDCODED Window = 1s
                            float[] signals = new float[timestamps.Length];
                            for(int t=0; t<timestamps.Length; t++) {
                                signals[t] = GetSignal(Module.Asset.GetFrame(timestamps[t]));
                            }
                            float c = GetSignal(frame);
                            float mean = signals.Mean();
                            float std = signals.Sigma();
                            std = std == 0f ? 1f : std;
                            return (c-mean) / std;
                        } else {
                            return GetSignal(frame);
                        }
                    }
                }

                if(Module.ApplyButterworth) {
                    Values = Utility.Butterworth(Source, (double)Module.Asset.GetDeltaTime(), (double)Module.MaxFrequency);
                }

                for(int i=0; i<Values.Length; i++) {
                    if(System.Math.Abs(Values[i]) < 1e-3) {
                        Values[i] = 0.0;
                    }
                }

                //Refine Windows
                switch(Module.Source) {
                    case SOURCE.Contact:
                    {
                        for(int i=0; i<Values.Length; i++) {
                            int window = Module.Asset.GetModule<ContactModule>().GetSensor(GetName()).GetStateWindow(Module.Asset.Frames[i], false);
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
                            tmp[i] = Mathf.RoundToInt(windows.Gaussian(mask:mask));
                        }
                        Windows = tmp;
                    }
                    break;

                    case SOURCE.Velocity:
                    Windows.SetAll(30);
                    break;
                }

                MinSource = Source.Min();
                MaxSource = Source.Max();
                MinValue = Values.Min();
                MaxValue = Values.Max();
            }

            public void Optimize(ref bool token, ref bool[] threads, ref int[] iterations, ref float[] progress, int thread) {
                Solutions = new Solution[Values.Length];
                for(int i=0; i<Solutions.Length; i++) {
                    Solutions[i] = new Solution();
                }
                Population[] populations = new Population[Values.Length];
                for(int i=0; i<Values.Length; i++) {
                    int padding = Windows[i] / 2;
                    Frame[] frames = Module.Asset.Frames.GatherByOverflowWindow(i, padding);
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
                    Interval _interval = new Interval(frames.First().Index, frames.Last().Index);
                    System.Func<double[], double> func = x => Loss(_interval, x);
                    System.Func<double[], double[]> grad = g => Grad(_interval, g, 0.1, _g, _tmp);
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
                    if(Module.ApplyButterworth) {
                        px = Utility.Butterworth(px, (double)Module.Asset.GetDeltaTime(), (double)Module.MaxFrequency);
                        py = Utility.Butterworth(py, (double)Module.Asset.GetDeltaTime(), (double)Module.MaxFrequency);
                    }
                    for(int i=0; i<Phases.Length; i++) {
                        Phases[i] = Utility.PhaseValue(new Vector2((float)px[i], (float)py[i]).normalized);
                    }
                    if(Module.ApplyButterworth) {
                        Amplitudes = Utility.Butterworth(Amplitudes, (double)Module.Asset.GetDeltaTime(), (double)Module.MaxFrequency);
                    }
                }

                double ComputeFit(int index) {
                    return Trigonometric(index, Solutions[index].Values);
                }

                double ComputePhase(int index) {
                    double[] x = Solutions[index].Values;
                    double t = Module.Asset.Frames[index].Timestamp;
                    double F = x[1];
                    double S = x[2];
                    return Mathf.Repeat((float)(F * t - S), 1f);
                }
                
                double ComputeAmplitude(int index) {
                    switch(Module.Source) {
                        case SOURCE.Contact:
                        Frame[] window = Module.Asset.Frames.GatherByOverflowWindow(index, GetPhaseWindow(index) / 2);
                        return GetSmoothedAmplitude(window) * GetMaxVelocityMagnitude(window);

                        case SOURCE.Velocity:
                        return Solutions[index].Values[0];

                        default:
                        return 0f;
                    }
                }
            }

            public int GetPhaseWindow(int index) {
                if(Solutions == null || Solutions.Length != Values.Length || Solutions.Any(null)) {
                    return 0;
                }
                float f = (float)Solutions[index].Values[1];
                return f == 0f ? 0 : Mathf.RoundToInt(Module.Asset.Framerate / f);
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
                    RootModule m = Module.Asset.GetModule<RootModule>();
                    float magnitude = 0f;
                    for(int i=0; i<frames.Length; i++) {
                        magnitude = Mathf.Max((frames[i].GetBoneVelocity(Bone, false) - m.GetRootVelocity(frames[i].Timestamp, false)).magnitude, magnitude);
                    }
                    return magnitude;
                }
            }

            private double Trigonometric(int index, double[] x) {
                double t = Module.Asset.Frames[index].Timestamp;
                double A = x[0];
                double F = x[1];
                double S = x[2];
                double B = x[3];
                return A * System.Math.Sin(2.0*System.Math.PI * (F * t - S)) + B;
            }

            private double Loss(Interval interval, double[] x) {
                double loss = 0.0;
                double count = 0.0;
                for(int i=interval.Start; i<=interval.End; i++) {
                    Accumulate(i);
                }
                int padding = interval.GetLength() / 2;
                for(int i=1; i<=padding; i++) {
                    double w = 1.0 - (double)(i)/(double)(padding); //Mean
                    // double w = Mathf.Exp(-Mathf.Pow((float)i - (float)padding, 2f) / Mathf.Pow(0.5f * (float)padding, 2f)); //Gaussian
                    w *= w;
                    Connect(interval.Start - i, w);
                    Connect(interval.End + i, w);
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

            private double[] Grad(Interval interval, double[] x, double delta, double[] grad, double[] tmp) {
                for(int i=0; i<x.Length; i++) {
                    tmp[i] = x[i];
                }
                double loss = Loss(interval, tmp);
                for(int i=0; i<tmp.Length; i++) {
                    tmp[i] += delta;
                    grad[i] = (Loss(interval, tmp) - loss) / delta;
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
        #endif

        public class Series : TimeSeries.Component {
            public string[] Bones;
            public float[][] Amplitudes;
            public float[][] Phases;
            public bool[] Active;

            private float Amplitude = 0f;

            private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.15f, 0.2f, 0.15f);
            // private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.1f, 0.075f, 0.8f, 0.125f);

            public Series(TimeSeries global, params string[] bones) : base(global) {
                Bones = bones;
                Amplitudes = new float[Samples.Length][];
                Phases = new float[Samples.Length][];
                for(int i=0; i<Samples.Length; i++) {
                    Amplitudes[i] = new float[bones.Length];
                    Phases[i] = new float[bones.Length];
                }
                Active = new bool[bones.Length];
                Active.SetAll(true);
            }

            public float[] GetAlignment() {
                int pivot = 0;
                float[] alignment = new float[Bones.Length * KeyCount * 2];
                for(int k=0; k<KeyCount; k++) {
                    int index = GetKey(k).Index;
                    for(int b=0; b<Bones.Length; b++) {
                        Vector2 phase = Active[b] ? (Amplitudes[index][b] * Utility.PhaseVector(Phases[index][b])) : Vector2.zero;
                        alignment[pivot] = phase.x; pivot += 1;
                        alignment[pivot] = phase.y; pivot += 1;
                    }
                }
                return alignment;
            }

            public bool IsActive(params string[] bones) {
                for(int i=0; i<bones.Length; i++) {
                    if(!Active[System.Array.FindIndex(Bones, x => x == bones[i])]) {
                        return false;
                    }
                }
                return true;
            }

            public float GetPhase(int index, int bone) {
                return Active[bone] ? Phases[index][bone] : 0f;
            }

            public float GetAmplitude(int index, int bone) {
                return Active[bone] ? Amplitudes[index][bone] : 0f;
            }

            public override void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
                    for(int j=0; j<Bones.Length; j++) {
                        Phases[i][j] = Phases[i+1][j];
                        Amplitudes[i][j] = Amplitudes[i+1][j];
                    }
                }
            }

            public override void Interpolate(int start, int end) {
                for(int i=start; i<end; i++) {
                    float weight = (float)(i % Resolution) / (float)Resolution;
                    int prevIndex = GetPreviousKey(i).Index;
                    int nextIndex = GetNextKey(i).Index;
                    for(int j=0; j<Bones.Length; j++) {
                        Phases[i][j] = Utility.PhaseValue(Vector2.Lerp(Utility.PhaseVector(Phases[prevIndex][j]), Utility.PhaseVector(Phases[nextIndex][j]), weight).normalized);
                        Amplitudes[i][j] = Mathf.Lerp(Amplitudes[prevIndex][j], Amplitudes[nextIndex][j], weight);
                    }
                }
            }

            public override void GUI() {
                if(DrawGUI) {
                    UltiDraw.Begin();
                    UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.5f*Rect.H + 0.025f), Rect.GetSize(), 0.0175f, "Local Motion Phases", UltiDraw.Black);

                    float xMin = Rect.X;
                    float xMax = Rect.X + Rect.W;
                    float yMin = Rect.Y;
                    float yMax = Rect.Y + Rect.H;
                    for(int b=0; b<Bones.Length; b++) {
                        float w = (float)b/(float)(Bones.Length-1);
                        float vertical = w.Normalize(0f, 1f, yMax, yMin);
                        float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
                        float border = 0.025f*(yMax-yMin)/(Bones.Length-1);
                        if(!Active[b]) {
                            UltiDraw.OnGUILabel(new Vector2(0.5f*(xMin+xMax), vertical), new Vector2(xMax-xMin, height), 0.015f, "Disabled", UltiDraw.White, UltiDraw.Transparent);
                        }
                    }
                    UltiDraw.End();
                }
            }

            public override void Draw() {
                if(DrawGUI) {
                    UltiDraw.Begin();

                    //Vector-Space
                    // {
                    //     float xMin = Rect.X;
                    //     float xMax = Rect.X + Rect.W;
                    //     float yMin = Rect.Y;
                    //     float yMax = Rect.Y + Rect.H;
                    //     float amp = Amplitudes.Flatten().Max();
                    //     for(int b=0; b<Bones.Length; b++) {
                    //         float w = (float)b/(float)(Bones.Length-1);
                    //         float vertical = w.Normalize(0f, 1f, yMax, yMin);
                    //         float border = 0.025f*(yMax-yMin)/(Bones.Length-1);
                    //         Color phaseColor = UltiDraw.White;
                    //         for(int i=0; i<KeyCount; i++) {
                    //             float ratio = (float)(i) / (float)(KeyCount-1);
                    //             Vector2 center = new Vector2(xMin + ratio * (xMax - xMin), vertical);
                    //             float size = 0.95f*(xMax-xMin)/(KeyCount-1);
                    //             if(i < PivotKey) {
                    //                 float phase = Phases[GetKey(i).Index][b];
                    //                 float amplitude = Amplitudes[GetKey(i).Index][b];
                    //                 Color color = phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f));
                    //                 UltiDraw.PlotCircularPivot(center, size, 360f * phase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.DarkGrey, pivotColor: color);
                    //             }
                    //             if(i == PivotKey) {
                    //                 float phase = Phases[GetKey(i).Index][b];
                    //                 float amplitude = Amplitudes[GetKey(i).Index][b];
                    //                 Color color = phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f));
                    //                 UltiDraw.PlotCircularPivot(center, size, 360f * phase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.DarkGrey, pivotColor: color);

                    //                 Vector2[] vectors = GetPhaseStabilization(Pivot, b);
                    //                 float[] phases = new float[vectors.Length];
                    //                 float[] amplitudes = new float[vectors.Length];
                    //                 for(int v=0; v<vectors.Length; v++) {
                    //                     phases[v] = 360f * Utility.PhaseValue(vectors[v]);
                    //                     amplitudes[v] = vectors[v].magnitude.Normalize(0f, amp, 0f, 1f);
                    //                 }
                    //                 UltiDraw.PlotCircularPivots(center, size, phases, amplitudes, backgroundColor: UltiDraw.None, pivotColors: UltiDraw.GetRainbowColors(vectors.Length));
                                    
                    //                 // float stabilizedPhase = GetStabilizedPhase(Pivot, b, vectors, 0.5f);
                    //                 // UltiDraw.PlotCircularPivot(center, size, 360f * stabilizedPhase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.None, pivotColor: UltiDraw.Magenta);
                    //             }
                    //             if(i > PivotKey) {
                    //                 float[] phases = new float[2]{
                    //                     360f * Phases[GetKey(i).Index][b], 
                    //                     360f * GetUpdateRate(b, Pivot, GetKey(i).Index, 1f/30f)
                    //                 };
                    //                 float[] amplitudes = new float[2]{
                    //                     Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f), 
                    //                     Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f)
                    //                 };
                    //                 Color[] colors = new Color[2]{
                    //                     phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f)),
                    //                     UltiDraw.Red.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f))
                    //                 };
                    //                 UltiDraw.PlotCircularPivots(center, size, phases, amplitudes, backgroundColor: UltiDraw.DarkGrey, pivotColors: colors);
                    //             }
                    //         }
                    //     }
                    // }

                    //Phase-Space
                    {
                        Color inactive = UltiDraw.Red.Opacity(0.5f);
                        float xMin = Rect.X;
                        float xMax = Rect.X + Rect.W;
                        float yMin = Rect.Y;
                        float yMax = Rect.Y + Rect.H;
                        float[][] amplitudes;
                        if(Active.All(true)) {
                            amplitudes = Amplitudes;
                        } else {
                            amplitudes = (float[][])Amplitudes.Clone();
                            for(int i=0; i<Bones.Length; i++) {
                                if(!Active[i]) {
                                    for(int j=0; j<SampleCount; j++) {
                                        amplitudes[j][i] = 0f;
                                    }
                                }
                            }
                        }
                        Amplitude = Mathf.Max(Amplitude, Amplitudes.Flatten().Max());
                        for(int b=0; b<=Bones.Length; b++) {
                            float ratio = b.Ratio(0, Bones.Length);
                            float itemSize = Rect.H / (Bones.Length+1);
                            if(b < Bones.Length) {
                                float[] values = new float[Samples.Length];
                                Color[] colors = new Color[Samples.Length];
                                for(int i=0; i<Samples.Length; i++) {
                                    values[i] = GetPhase(i,b);
                                    colors[i] = UltiDraw.Black.Opacity(GetAmplitude(i,b).Normalize(0f, Amplitude, 0f, 1f));
                                }
                                UltiDraw.PlotBars(new Vector2(Rect.X, ratio.Normalize(0f, 1f, Rect.Y + Rect.H/2f - itemSize/2f, Rect.Y - Rect.H/2f + itemSize/2f)), new Vector2(Rect.W, itemSize), values, yMin: 0f, yMax: 1f, barColors: colors, backgroundColor: Active[b] ? UltiDraw.White : inactive);
                                // UltiDraw.PlotCircularPivot(new Vector2(xMax + 0.8f * height/2f, vertical), 0.8f * height/2f, 360f*GetPhase(Pivot,b), GetAmplitude(Pivot,b).Normalize(0f, amp, 0f, 1f), backgroundColor: Active[b] ? UltiDraw.DarkGrey : inactive, pivotColor: colors[Pivot].Invert());
                            } else {
                                UltiDraw.PlotFunctions(new Vector2(Rect.X, ratio.Normalize(0f, 1f, Rect.Y + Rect.H/2f - itemSize/2f, Rect.Y - Rect.H/2f + itemSize/2f)), new Vector2(Rect.W, itemSize), amplitudes, UltiDraw.Dimension.Y, yMin: 0f, yMax: Amplitude);
                            }
                        }
                    }

                    UltiDraw.End();
                }
            }
        }
        
    }
}