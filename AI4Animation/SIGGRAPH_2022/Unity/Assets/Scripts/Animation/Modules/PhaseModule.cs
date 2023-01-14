#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Threading.Tasks;

namespace AI4Animation {
    public class PhaseModule : Module {

        public int MaxIterations = 10;
        public int Individuals = 50;
        public int Elites = 5;
        public float MaxFrequency = 4f;
        public float FilterWindow = 0.5f;
        public float Exploration = 0.2f;

        public bool Plugin = false;

        public Function[] Functions = new Function[0];

        private string[] Identifiers = null;

        private Task[] Threads = null;

        public override void DerivedResetPrecomputation() {

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
            ContactModule contacts = Asset.GetModule<ContactModule>();
            if(contacts != null) {
                SetFunctions(contacts.GetNames());
            } else {
                Functions = new Function[0];
            }
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

            MaxIterations = EditorGUILayout.IntField("Max Iterations", MaxIterations);
            Individuals = EditorGUILayout.IntField("Individuals", Individuals);
            Elites = EditorGUILayout.IntField("Elites", Elites);
            MaxFrequency = EditorGUILayout.FloatField("Max Frequency", MaxFrequency);
            FilterWindow = EditorGUILayout.FloatField("Filter Window", FilterWindow);
            Exploration = EditorGUILayout.Slider("Exploration", Exploration, 0f, 1f);

            Plugin = EditorGUILayout.Toggle("Plugin", Plugin);

            EditorGUI.BeginDisabledGroup(Threads != null);
            if(Utility.GUIButton("Optimize", UltiDraw.DarkGrey, UltiDraw.White)) {
                StartOptimization();
            }
            EditorGUI.EndDisabledGroup();
            
            float max = 0f;
            foreach(Function f in Functions) {
                max = Mathf.Max(max, (float)f.PhaseVectors.Magnitudes().Max());
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
                for(int j=0; j<view.z; j++) {
                    prevPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    prevPos.y = rect.yMax;
                    newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                    newPos.y = rect.yMax - (float)f.GetPhase(view.x+j-1) * rect.height;
                    float weight = (float)f.GetAmplitude(view.x+j-1).Normalize(0f, max, 0f, 1f);
                    UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Cyan.Opacity(weight));
                }
                UltiDraw.End();
                editor.DrawPivot(rect);
                EditorGUILayout.EndVertical();

                EditorGUILayout.EndHorizontal();
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
            EditorGUILayout.HelpBox("Active Threads: " + ActiveThreads(), MessageType.None);
        }

        protected override void DerivedGUI(MotionEditor editor) {

        }

        protected override void DerivedDraw(MotionEditor editor) {
            // UltiDraw.Begin();
            // UltiDraw.PlotCircularPivot(new Vector2(0.5f, 0.5f), 1f/3f, 360f * GetPhase(0, editor.GetTimestamp(), editor.Mirror), 0.9f, Color.white, Color.black);
            // UltiDraw.End();
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

        public void SetFunctions(params string[] bones) {
            Functions = new Function[bones.Length];
            for(int i=0; i<Functions.Length; i++) {
                Functions[i] = new Function(this, Asset.Source.FindBone(bones[i]).Index);
            }
            Identifiers = null;
        }

        public float[] GetPhases(float timestamp, bool mirrored) {
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
            float start = Asset.Frames.First().Timestamp;
            float end = Asset.Frames.Last().Timestamp;
            if(timestamp < start || timestamp > end) {
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

        public void StartOptimization() {
            Threads = new Task[Functions.Length];
            for(int i=0; i<Functions.Length; i++) {
                int index = i;
                Threads[index] = Task.Factory.StartNew(() => {
                    Functions[index].Optimise();
                });
            }
            Task.Factory.StartNew(() => {
                while(ActiveThreads() > 0) {
                    System.Threading.Thread.Sleep(1);
                }
                Threads = null;
            });
        }

        public bool IsOptimizing() {
            return ActiveThreads() > 0;
        }

        private int ActiveThreads() {
            if(Threads == null) {
                return 0;
            }
            int active = 0;
            foreach(Task t in Threads) {
                if(!t.IsCompleted) {
                    active += 1;
                }
            }
            return active;
        }

        private void Validate() {
            if(Functions.Length == Asset.Source.Bones.Length) {
                ContactModule module = Asset.GetModule<ContactModule>();
                if(Functions.Length != module.Sensors.Length) {
                    Debug.LogWarning("Auto-removing unused local phase functions in asset " + Asset.name + ".");
                    int[] indices = Asset.Source.GetBoneIndices(module.GetNames());
                    Functions = Functions.GatherByIndices(indices);
                }
            }
            foreach(Function f in Functions) {
                f.PhaseVectors = f.PhaseVectors.Validate(Asset.Frames.Length);
            }
        }

        [System.Serializable]
        public class Function {
            public PhaseModule Module;
            public int Bone;
            
            public Vector2[] PhaseVectors;

            private Function Symmetric = null;

            public Function(PhaseModule module, int bone) {
                Module = module;
                Bone = bone;
                PhaseVectors = new Vector2[Module.Asset.GetTotalFrames()];
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
                return Module.Asset.Source.Bones[Bone].GetName();
            }

            public double GetPhase(int index) {
                PhaseVectors = PhaseVectors.Validate(Module.Asset.Frames.Length);
                return Utility.PhaseValue(PhaseVectors[index]);
            }

            public double GetAmplitude(int index) {
                PhaseVectors = PhaseVectors.Validate(Module.Asset.Frames.Length);
                return PhaseVectors[index].magnitude;
            }

            public void Optimise() {
                ContactModule contact = Module.Asset.GetModule<ContactModule>();
                ContactModule.Sensor sensor = contact.GetSensor(GetName());
                float[] CONTACTS = new float[Module.Asset.Frames.Length];
                float[] VELOCITIES = new float[Module.Asset.Frames.Length];
                for(int i=0; i<Module.Asset.Frames.Length; i++) {
                    CONTACTS[i] = sensor.Contacts[i];
                    VELOCITIES[i] = Module.Asset.Frames[i].GetBoneVelocity(Bone, false).magnitude;
                }
                Module.ExtractPhase(
                    Module.Asset.Frames.Length, 
                    Module.Asset.Framerate, 
                    Module.MaxFrequency, 
                    Module.FilterWindow,
                    Module.Individuals, 
                    Module.Elites, 
                    Module.Exploration, 
                    Module.MaxIterations, 
                    CONTACTS, 
                    VELOCITIES,
                    PhaseVectors
                );
            }
        }

        public void ExtractPhase(
            int frameCount,
            float frameRate,
            float maxFrequency,
            float filterWindow,
            int individuals,
            int elites,
            float exploration,
            int iterations,

            float[] contacts,
            float[] velocities,
            Vector2[] phaseVectors
        ) {
            const int dimensionality = 4;

            if(Plugin) {
                Debug.Log("Phase plugin not supported anymore.");
                // IntPtr pluginContacts = Eigen.Create(1, Asset.Frames.Length);
                // IntPtr pluginVelocities = Eigen.Create(1, Asset.Frames.Length);
                // IntPtr pluginResult = Eigen.Create(2, Asset.Frames.Length);
                // for(int i=0; i<Asset.Frames.Length; i++) {
                //     Eigen.SetValue(pluginContacts, 0, i, contacts[i]);
                //     Eigen.SetValue(pluginVelocities, 0, i, velocities[i]);
                // }
                // PhaseExtractor.Compute(frameCount, frameRate, maxFrequency, filterWindow, individuals, elites, exploration, iterations, pluginContacts, pluginVelocities, pluginResult);
                // for(int i=0; i<Asset.Frames.Length; i++) {
                //     phaseVectors[i].x = Eigen.GetValue(pluginResult, 0, i);
                //     phaseVectors[i].y = Eigen.GetValue(pluginResult, 1, i);
                // }
                // Eigen.Delete(pluginContacts);
                // Eigen.Delete(pluginVelocities);
                // Eigen.Delete(pluginResult);
            } else {
                //Check Invalid
                if(contacts.Length != frameCount || velocities.Length != frameCount) {
                    return;
                }

                //Check Active
                if(contacts.All(0f) || contacts.All(1f)) {
                return;
                }

                //Compute DeltaTime
                double deltaTime = 1.0/frameRate;

                //Compute Indices
                int[] indices = new int[frameCount];
                for(int i=0; i<frameCount; i++) {
                    indices[i] = i;
                }

                //Compute Timestamps
                double[] timestamps = new double[frameCount];
                for(int i=0; i<frameCount; i++) {
                    timestamps[i] = i * deltaTime;
                }

                //Window-based Normalization
                double[] values = new double[frameCount];
                for(int i=0; i<frameCount; i++) {
                    double[] sampledTimestamps = SampleTimestamps(timestamps[i], filterWindow, filterWindow);
                    double[] sampledContacts = new double[sampledTimestamps.Length];
                    for(int t=0; t<sampledTimestamps.Length; t++) {
                        sampledContacts[t] = GetContact(sampledTimestamps[t]);
                    }
                    double mean = sampledContacts.Mean();
                    double std = sampledContacts.Sigma();
                    std = std == 0.0 ? 1.0 : std;
                    values[i] = (contacts[i]-mean) / std;
                }

                //Butterworth Filter
                values = Utility.Butterworth(values, deltaTime, maxFrequency);

                //Mask-Out Zeros
                for(int i=0; i<frameCount; i++) {
                    if(System.Math.Abs(values[i]) < 1e-3) {
                        values[i] = 0.0;
                    }
                }

                //Compute Windows
                int[] windows = new int[frameCount];
                for(int i=0; i<frameCount; i++) {
                    int window = GetContactWindow(i);
                    double active = 1.0 - values.GatherByOverflowWindow(i, window / 2).Ratio(0.0);
                    windows[i] = Mathf.RoundToInt((float)active * window);
                }
                int[] _windows = new int[frameCount];
                for(int i=0; i<frameCount; i++) {
                    int[] sampledWindows = windows.GatherByOverflowWindow(i, Mathf.RoundToInt(filterWindow * frameRate));
                    double[] sampledValues = values.GatherByOverflowWindow(i, Mathf.RoundToInt(filterWindow * frameRate));
                    bool[] mask = new bool[sampledValues.Length];
                    for(int j=0; j<sampledValues.Length; j++) {
                        mask[j] = sampledValues[j] != 0f;
                    }
                    _windows[i] = Mathf.RoundToInt(sampledWindows.Gaussian(mask:mask));
                }
                windows = _windows;

                //Start Optimization
                Population[] populations = new Population[values.Length];
                for(int i=0; i<values.Length; i++) {
                    int padding = windows[i] / 2;
                    int[] _indices = indices.GatherByOverflowWindow(i, padding);
                    double[] _values = values.GatherByOverflowWindow(i, padding);
                    // if(_values.AbsSum() < 1e-3) {
                    //     continue;
                    // }
                    double min = _values.Min();
                    double max = _values.Max();
                    double[] lowerBounds = new double[dimensionality]{0.0, 1f/maxFrequency, -0.5, min};
                    double[] upperBounds = new double[dimensionality]{max-min, maxFrequency, 0.5, max};
                    double[] seed = new double[dimensionality];
                    for(int j=0; j<dimensionality; j++) {
                        seed[j] = 0.5 * (lowerBounds[j] + upperBounds[j]);
                    }
                    Interval _seq = new Interval(_indices.First(), _indices.Last());
                    System.Func<double[], double> func = x => Loss(_seq, x);
                    populations[i] = new Population(individuals, elites, exploration, lowerBounds, upperBounds, seed, func);
                }

                //Iterate
                for(int k=0; k<iterations; k++) {
                    for(int i=0; i<populations.Length; i++) {
                        populations[i].Evolve();
                    }
                    for(int i=0; i<populations.Length; i++) {
                        double[] x = populations[i].GetSolution();
                        double t = timestamps[i];
                        double F = x[1];
                        double S = x[2];
                        
                        float phase = Mathf.Repeat((float)(F * t - S), 1f);

                        int phaseWindow = (float)F == 0f ? 0 : Mathf.RoundToInt((float)frameRate / (float)F);
                        int[] window = indices.GatherByOverflowWindow(i, phaseWindow / 2);
                        float amplitude = (float)(GetSmoothedAmplitude(window) * GetMaxVelocityMagnitude(window));

                        phaseVectors[i] = amplitude * Utility.PhaseVector(phase);
                    }
                    double[] px = new double[phaseVectors.Length];
                    double[] py = new double[phaseVectors.Length];
                    for(int i=0; i<phaseVectors.Length; i++) {
                        px[i] = phaseVectors[i].x;
                        py[i] = phaseVectors[i].y;
                    }
                    px = Utility.Butterworth(px, deltaTime, maxFrequency);
                    py = Utility.Butterworth(py, deltaTime, maxFrequency);
                    for(int i=0; i<phaseVectors.Length; i++) {
                        phaseVectors[i] = new Vector2((float)px[i], (float)py[i]);
                    }
                }

                double GetSmoothedAmplitude(int[] window) {
                    double[] amplitudes = new double[window.Length];
                    for(int i=0; i<window.Length; i++) {
                        amplitudes[i] = populations[window[i]].GetSolution()[0];
                    }
                    return amplitudes.Gaussian();
                }

                float GetMaxVelocityMagnitude(int[] window) {
                    float magnitude = 0f;
                    for(int i=0; i<window.Length; i++) {
                        magnitude = Mathf.Max((float)velocities[window[i]], magnitude);
                    }
                    return magnitude;
                }

                double GetContact(double timestamp) {
                    double _start = timestamps.First();
                    double _end = timestamps.Last();
                    if(timestamp < _start || timestamp > _end) {
                        float _boundary = Mathf.Clamp((float)timestamp, (float)_start, (float)_end);
                        float _pivot = 2f*_boundary - (float)timestamp;
                        float _clamped = Mathf.Clamp(_pivot, (float)_start, (float)_end);
                        return contacts[Mathf.RoundToInt(_clamped * (float)frameRate)];
                    } else {
                        return contacts[Mathf.RoundToInt((float)timestamp * (float)frameRate)];
                    }
                }

                double[] SampleTimestamps(double timestamp, double pastPadding, double futurePadding) {
                    double _start = timestamp - pastPadding;
                    double[] _timestamps = new double[Mathf.RoundToInt((float)((pastPadding+futurePadding)/deltaTime))+1];
                    for(int i=0; i<_timestamps.Length; i++) {
                        _timestamps[i] = _start + i*deltaTime;
                    }
                    return _timestamps;
                }

                int GetContactWindow(int index) {
                    double state = contacts[index];
                    int window = 1;
                    for(int i=index-1; i>=0; i--) {
                        if(contacts[i] != state) {
                            break;
                        }
                        window += 1;
                    }
                    for(int i=index+1; i<frameCount; i++) {
                        if(contacts[i] != state) {
                            break;
                        }
                        window += 1;
                    }
                    return window;
                }

                double Trigonometric(int index, double[] x) {
                    double t = timestamps[index];
                    double A = x[0];
                    double F = x[1];
                    double S = x[2];
                    double B = x[3];
                    return A * System.Math.Sin(2.0*System.Math.PI * (F * t - S)) + B;
                }

                double Loss(Interval interval, double[] x) {
                    double loss = 0.0;
                    double count = 0.0;
                    for(int i=interval.Start; i<=interval.End; i++) {
                        Accumulate(i, 1f);
                    }
                    int padding = interval.GetLength() / 2;
                    for(int i=1; i<=padding; i++) {
                        double w = 1.0 - (double)(i)/(double)(padding); //Mean
                        w *= w;
                        Accumulate(interval.Start - i, w);
                        Accumulate(interval.End + i, w);
                    }

                    loss /= count;
                    loss = System.Math.Sqrt(loss);

                    return loss;

                    void Accumulate(int index, double weight) {
                        if(index >= 0 && index < values.Length) {
                            double error = values[index] - Trigonometric(index, x);
                            error *= error;
                            loss += error;
                            count += 1.0;
                        }
                    }
                }
            }
        }

        private class Population {
            private System.Random RNG;

            public int Size;
            public int Elites;
            public int Dimensionality;
            public double Exploration;
            public double[] LowerBounds;
            public double[] UpperBounds;

            public double[] RankProbabilities;

            public System.Func<double[], double> Func;

            public Individual[] Individuals;
            public Individual[] Offspring;

            public Population(int size, int elites, float exploration, double[] lowerBounds, double[] upperBounds, double[] seed, System.Func<double[], double> func) {
                RNG = new System.Random();

                Size = size;
                Elites = elites;
                Dimensionality = seed.Length;
                Exploration = exploration;

                LowerBounds = lowerBounds;
                UpperBounds = upperBounds;

                Func = func;

                //Compute rank probabilities
                RankProbabilities = new double[Size];
                for(int i=0; i<Size; i++) {
                    double sum = (double)(Size*(Size+1)) / 2.0;
                    RankProbabilities[i] = (double)(Size-i)/sum;
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
                EvaluateFitness(Individuals);
                SortByFitness(Individuals);
                AssignExtinctions(Individuals);
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
                double rVal = GetRandom();
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

            public class Individual {
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
#endif