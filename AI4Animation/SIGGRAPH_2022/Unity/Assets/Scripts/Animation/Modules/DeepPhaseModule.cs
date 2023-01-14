#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace AI4Animation {
	public class DeepPhaseModule : Module {

        public Channel[] Channels = new Channel[0];
        [NonSerialized] private bool UseOffsets = false;
        [NonSerialized] private bool ShowParameters = false;
        [NonSerialized] private bool ShowNormalized = false;
        [NonSerialized] private bool DrawWindowPoses = false;
        [NonSerialized] private bool DrawPhaseSpace = true;
        [NonSerialized] private bool DrawPivot = true;

        public static DistanceSeries Distances;
        public class DistanceSeries {
            public MotionAsset Asset;
            public Actor Actor;
            public TimeSeries TimeSeries;
            public DeepPhaseModule Module;

            public Map GlobalChannel;
            public Map[] LocalChannels;

            public class Map {
                public float[][] Values;
                public float Min;
                public float Max;
                public Map(int size) {
                    Values = new float[size][];
                    for(int i=0; i<Values.Length; i++) {
                        Values[i] = new float[size];
                    }
                    Min = float.MaxValue;
                    Max = float.MinValue;
                }
            }

            public DistanceSeries(MotionAsset asset, Actor actor, TimeSeries timeSeries, DeepPhaseModule module) {
                Asset = asset;
                Actor = actor;
                TimeSeries = timeSeries;
                Module = module;
                Compute();
            }

            public float[] GetFeatureVector(float timestamp, bool mirrored) {
                return Module.GetManifold(timestamp, mirrored).ToArray().Flatten();
            }

            public float[] GetFeatureVector(float timestamp, bool mirrored, int channel) {
                return Module.Channels[channel].GetManifoldVector(timestamp, mirrored).ToArray();
            }

            public void Compute() {
                GlobalChannel = new Map(Asset.Frames.Length);
                for(int y=0; y<Asset.Frames.Length; y++) {
                    float[] Y = GetFeatureVector(Asset.Frames[y].Timestamp, false);
                    for(int x=0; x<Asset.Frames.Length; x++) {
                        float[] X = GetFeatureVector(Asset.Frames[x].Timestamp, false);
                        float distance = ArrayExtensions.MSE(Y,X);
                        GlobalChannel.Values[y][x] = distance;
                        GlobalChannel.Min = Mathf.Min(GlobalChannel.Min, distance);
                        GlobalChannel.Max = Mathf.Max(GlobalChannel.Max, distance);
                    }
                }

                LocalChannels = new Map[Module.Channels.Length];
                for(int i=0; i<LocalChannels.Length; i++) {
                    LocalChannels[i] = new Map(Asset.Frames.Length);
                    for(int y=0; y<Asset.Frames.Length; y++) {
                        float[] Y = GetFeatureVector(Asset.Frames[y].Timestamp, false, i);
                        for(int x=0; x<Asset.Frames.Length; x++) {
                            float[] X = GetFeatureVector(Asset.Frames[x].Timestamp, false, i);
                            float distance = ArrayExtensions.MSE(Y,X);
                            LocalChannels[i].Values[y][x] = distance;
                            LocalChannels[i].Min = Mathf.Min(LocalChannels[i].Min, distance);
                            LocalChannels[i].Max = Mathf.Max(LocalChannels[i].Max, distance);
                        }
                    }
                }
            }
        }

        public static CurveSeries Curves;
        public class CurveSeries {
            public MotionAsset Asset;
            public Actor Actor;
            public TimeSeries TimeSeries;
            public Curve[] Curves;
            public float MinView = -2f;
            public float MaxView = 2f;

            public CurveSeries(MotionAsset asset, Actor actor, TimeSeries timeSeries) {
                Asset = asset;
                Actor = actor;
                TimeSeries = timeSeries;
                Compute();
            }

            public void Compute() {
                RootModule rootModule = Asset.GetModule<RootModule>();
                int[] mapping = Asset.Source.GetBoneIndices(Actor.GetBoneNames());

                Curves = new Curve[mapping.Length];
                for(int i=0; i<Curves.Length; i++) {
                    Curves[i] = new Curve(this);
                }

                void Compute(int i, bool mirrored) {
                    //Positions
                    // Vector3[] positions = mirrored ? Curves[i].MirroredValues : Curves[i].OriginalValues;
                    // {
                    //     for(int j=0; j<positions.Length; j++) {
                    //         Matrix4x4 spaceC = rootModule.GetRootTransformation(Asset.Frames[j].Timestamp, mirrored);
                    //         Vector3 posC = Asset.Frames[j].GetBoneTransformation(mapping[i], mirrored).GetPosition();
                    //         positions[j] = posC.PositionTo(spaceC);
                    //     }
                    // }

                    //Velocities
                    Vector3[] velocities = mirrored ? Curves[i].MirroredValues : Curves[i].OriginalValues;
                    {
                        for(int j=0; j<velocities.Length; j++) {
                            Matrix4x4 spaceP = rootModule.GetRootTransformation(Asset.Frames[Mathf.Max(j-1,0)].Timestamp, mirrored);
                            Matrix4x4 spaceC = rootModule.GetRootTransformation(Asset.Frames[j].Timestamp, mirrored);

                            // Matrix4x4 spaceP = Asset.Frames[Mathf.Max(j-1,0)].GetBoneTransformation(mapping[0], mirrored);
                            // Matrix4x4 spaceC = Asset.Frames[j].GetBoneTransformation(mapping[0], mirrored);

                            Vector3 posP = Asset.Frames[Mathf.Max(j-1,0)].GetBoneTransformation(mapping[i], mirrored).GetPosition();
                            Vector3 posC = Asset.Frames[j].GetBoneTransformation(mapping[i], mirrored).GetPosition();
                            velocities[j] = (posC.PositionTo(spaceC) - posP.PositionTo(spaceP)) / Asset.GetDeltaTime();
                        }
                    }
                    //Low-Pass Filter
                    {
                        float[] x = Utility.Butterworth(velocities.ToArrayX(), Asset.GetDeltaTime(), TimeSeries.MaximumFrequency);
                        float[] y = Utility.Butterworth(velocities.ToArrayY(), Asset.GetDeltaTime(), TimeSeries.MaximumFrequency);
                        float[] z = Utility.Butterworth(velocities.ToArrayZ(), Asset.GetDeltaTime(), TimeSeries.MaximumFrequency);
                        for(int j=0; j<velocities.Length; j++) {
                            velocities[j].x = x[j];
                            velocities[j].y = y[j];
                            velocities[j].z = z[j];
                        }
                    }
                }

                for(int i=0; i<mapping.Length; i++) {
                    Compute(i, false);
                    Compute(i, true);
                }
            }

            public float[] Collect(float timestamp, bool mirrored) {
                int count = Curves.Length;
                float[] values = new float[count * 3];
                int idx = 0;
                for(int i=0; i<count; i++) {
                    int curve = i;
                    values[idx] = Curves[curve].GetValue(timestamp, mirrored).x; idx += 1;
                    values[idx] = Curves[curve].GetValue(timestamp, mirrored).y; idx += 1;
                    values[idx] = Curves[curve].GetValue(timestamp, mirrored).z; idx += 1;
                }
                return values;
            }

            public float[] Sample(float timestamp, bool mirrored) {
                int count = Curves.Length;
                float[] values = new float[TimeSeries.Samples.Length * count * 3];
                int idx = 0;
                for(int i=0; i<count; i++) {
                    int curve = i;
                    for(int j=0; j<TimeSeries.Samples.Length; j++) {
                        values[idx] = Curves[curve].GetValue(timestamp + TimeSeries.Samples[j].Timestamp, mirrored).x;
                        idx += 1;
                    }
                    for(int j=0; j<TimeSeries.Samples.Length; j++) {
                        values[idx] = Curves[curve].GetValue(timestamp + TimeSeries.Samples[j].Timestamp, mirrored).y;
                        idx += 1;
                    }
                    for(int j=0; j<TimeSeries.Samples.Length; j++) {
                        values[idx] = Curves[curve].GetValue(timestamp + TimeSeries.Samples[j].Timestamp, mirrored).z;
                        idx += 1;
                    }
                }
                return values;
            }

            public class Curve {
                public CurveSeries Series;
                public Vector3[] OriginalValues;
                public Vector3[] MirroredValues;

                public Curve(CurveSeries series) {
                    Series = series;
                    OriginalValues = new Vector3[Series.Asset.Frames.Length];
                    MirroredValues = new Vector3[Series.Asset.Frames.Length];
                }

                public Vector3[] GetValues(bool mirrored) {
                    if(mirrored) {
                        return MirroredValues;
                    } else {
                        return OriginalValues;
                    }
                }

                public Vector3 GetValue(float timestamp, bool mirrored) {
                    return GetValues(mirrored)[Series.Asset.GetFrame(timestamp).Index-1];
                }

                public Vector3 GetValue(int index, bool mirrored) {
                    return GetValues(mirrored)[index];
                }
            }
        }

        [Serializable]
        public class Channel {
            public DeepPhaseModule Module;

            public float[] RegularPhaseValues;
            public float[] RegularFrequencies;
            public float[] RegularAmplitudes;
            public float[] RegularOffsets;

            public float[] MirroredPhaseValues;
            public float[] MirroredFrequencies;
            public float[] MirroredAmplitudes;
            public float[] MirroredOffsets;

            public Channel(DeepPhaseModule module) {
                Module = module;

                RegularPhaseValues = new float[module.Asset.Frames.Length];
                RegularFrequencies = new float[module.Asset.Frames.Length];
                RegularAmplitudes = new float[module.Asset.Frames.Length];
                RegularOffsets = new float[module.Asset.Frames.Length];

                MirroredPhaseValues = new float[module.Asset.Frames.Length];
                MirroredFrequencies = new float[module.Asset.Frames.Length];
                MirroredAmplitudes = new float[module.Asset.Frames.Length];
                MirroredOffsets = new float[module.Asset.Frames.Length];
            }

            public Vector2 GetManifoldVector(float timestamp, bool mirrored) {
                if(Module.UseOffsets) {
                    return GetAmplitude(timestamp, mirrored) * GetPhaseVector(timestamp, mirrored) + GetOffset(timestamp, mirrored) * Vector2.one;
                } else {
                    return GetAmplitude(timestamp, mirrored) * GetPhaseVector(timestamp, mirrored);
                }
            }

            public Vector2 GetPhaseVector(float timestamp, bool mirrored) {
                float start = Module.Asset.Frames.First().Timestamp;
                float end = Module.Asset.Frames.Last().Timestamp;
                if(timestamp < start || timestamp > end) {
                    float boundary = Mathf.Clamp(timestamp, start, end);
                    float pivot = 2f*boundary - timestamp;
                    float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                    return Utility.PhaseVector(GetPhaseValue(timestamp, mirrored));
                } else {
                    return Utility.PhaseVector(GetPhaseValue(timestamp, mirrored));
                }
            }

            public float GetPhaseValue(float timestamp, bool mirrored) {
                float start = Module.Asset.Frames.First().Timestamp;
                float end = Module.Asset.Frames.Last().Timestamp;
                if(timestamp < start || timestamp > end) {
                    float boundary = Mathf.Clamp(timestamp, start, end);
                    float pivot = 2f*boundary - timestamp;
                    float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                    return
                    Mathf.Repeat(
                        PhaseValue(boundary, mirrored) -
                        Utility.SignedPhaseUpdate(
                            PhaseValue(boundary, mirrored),
                            PhaseValue(repeated, mirrored)
                        ), 1f
                    );
                } else {
                    return PhaseValue(timestamp, mirrored);
                }
            }

            public float GetFrequency(float timestamp, bool mirrored) {
                float start = Module.Asset.Frames.First().Timestamp;
                float end = Module.Asset.Frames.Last().Timestamp;
                if(timestamp < start || timestamp > end) {
                    float boundary = Mathf.Clamp(timestamp, start, end);
                    float pivot = 2f*boundary - timestamp;
                    float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                    return Frequency(repeated, mirrored);
                } else {
                    return Frequency(timestamp, mirrored);
                }
            }

            public float GetAmplitude(float timestamp, bool mirrored) {
                float start = Module.Asset.Frames.First().Timestamp;
                float end = Module.Asset.Frames.Last().Timestamp;
                if(timestamp < start || timestamp > end) {
                    float boundary = Mathf.Clamp(timestamp, start, end);
                    float pivot = 2f*boundary - timestamp;
                    float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                    return Amplitude(repeated, mirrored);
                } else {
                    return Amplitude(timestamp, mirrored);
                }
            }

            public float GetOffset(float timestamp, bool mirrored) {
                float start = Module.Asset.Frames.First().Timestamp;
                float end = Module.Asset.Frames.Last().Timestamp;
                if(timestamp < start || timestamp > end) {
                    float boundary = Mathf.Clamp(timestamp, start, end);
                    float pivot = 2f*boundary - timestamp;
                    float repeated = Mathf.Repeat(pivot-start, end-start) + start;
                    return Offset(repeated, mirrored);
                } else {
                    return Offset(timestamp, mirrored);
                }
            }

            public float GetDelta(float timestamp, bool mirrored, float delta) {
                return Utility.PhaseUpdate(GetPhaseValue(timestamp-delta, mirrored), GetPhaseValue(timestamp, mirrored));
            }

            // private Vector2 PhaseVector(float timestamp, bool mirrored) {
            //     return (mirrored ? MirroredPhases : RegularPhases)[Module.Asset.GetFrame(timestamp).Index-1];
            // }
            private float PhaseValue(float timestamp, bool mirrored) {
                return (mirrored ? MirroredPhaseValues : RegularPhaseValues)[Module.Asset.GetFrame(timestamp).Index-1];
            }
            private float Frequency(float timestamp, bool mirrored) {
                return (mirrored ? MirroredFrequencies : RegularFrequencies)[Module.Asset.GetFrame(timestamp).Index-1];
            }
            private float Amplitude(float timestamp, bool mirrored) {
                return (mirrored ? MirroredAmplitudes : RegularAmplitudes)[Module.Asset.GetFrame(timestamp).Index-1];
            }
            private float Offset(float timestamp, bool mirrored) {
                return (mirrored ? MirroredOffsets : RegularOffsets)[Module.Asset.GetFrame(timestamp).Index-1];
            }
        }

		public override void DerivedResetPrecomputation() {

		}

        public static void ComputeCurves(MotionAsset asset, Actor actor, TimeSeries timeSeries) {
            Curves = new CurveSeries(asset, actor, timeSeries);
        }

        public static void ComputeDistances(MotionAsset asset, Actor actor, TimeSeries timeSeries, DeepPhaseModule module) {
            Distances = new DistanceSeries(asset, actor, timeSeries, module);
        }

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            Series instance = new Series(global, Channels.Length);
            for(int i=0; i<instance.Samples.Length; i++) {
                instance.Phases[i] = GetPhaseValues(timestamp + instance.Samples[i].Timestamp, mirrored);
                instance.Amplitudes[i] = GetAmplitudes(timestamp + instance.Samples[i].Timestamp, mirrored);
                instance.Frequencies[i] = GetFrequencies(timestamp + instance.Samples[i].Timestamp, mirrored);
            }
            // instance.Phases = GetPhaseVectors(timestamp, mirrored);
            // instance.Amplitudes = GetAmplitudes(timestamp, mirrored);
            // instance.Frequencies = GetFrequencies(timestamp, mirrored);
            // instance.ComputeAlignment();
            instance.DrawScene = DrawPhaseSpace;
            instance.DrawGUI = DrawPhaseSpace;
            return instance;
		}

		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
            VerifyChannels(Channels == null ? 0 : Channels.Length);
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}


		protected override void DerivedCallback(MotionEditor editor) {

		}

		protected override void DerivedGUI(MotionEditor editor) {

		}

		protected override void DerivedDraw(MotionEditor editor) {
            if(DrawWindowPoses) {
                float timestamp = editor.GetTimestamp();
                for(int i=0; i<Channels.Length; i++) {
                    float f = Channels[i].GetFrequency(timestamp, editor.Mirror);
                    float previous = timestamp - 0.5f/f;
                    float next = timestamp + 0.5f/f;
                    editor.GetSession().GetActor().Draw(
                        Asset.GetFrame(previous).GetBoneTransformations(editor.GetSession().GetBoneMapping(), editor.Mirror),
                        UltiDraw.GetRainbowColor(i, Channels.Length-1),
                        UltiDraw.White,
                        Actor.DRAW.Skeleton
                    );
                    editor.GetSession().GetActor().Draw(
                        Asset.GetFrame(next).GetBoneTransformations(editor.GetSession().GetBoneMapping(), editor.Mirror),
                        UltiDraw.GetRainbowColor(i, Channels.Length-1),
                        UltiDraw.White,
                        Actor.DRAW.Skeleton
                    );
                }
            }
		}

		protected override void DerivedInspector(MotionEditor editor) {
			using(new EditorGUILayout.VerticalScope ("Box")) {
                UseOffsets = EditorGUILayout.Toggle("Use Offset", UseOffsets);
                ShowParameters = EditorGUILayout.Toggle("Show Parameters", ShowParameters);
                ShowNormalized = EditorGUILayout.Toggle("Show Normalized", ShowNormalized);
                DrawWindowPoses = EditorGUILayout.Toggle("Draw Window Poses", DrawWindowPoses);
                DrawPhaseSpace = EditorGUILayout.Toggle("Draw Phase Space", DrawPhaseSpace);
                DrawPivot = EditorGUILayout.Toggle("Draw Pivot", DrawPivot);

                Vector3Int view = editor.GetView();
                float height = 50f;
                float min = -1f;
                float max = 1f;
                float maxAmplitude = 1f;
                float maxFrequency = editor.GetTimeSeries().MaximumFrequency;
                float maxOffset = 1f;
                if(ShowNormalized) {
                    maxAmplitude = 0f;
                    foreach(Channel c in Channels) {
                        maxAmplitude = Mathf.Max(maxAmplitude, (editor.Mirror ? c.MirroredAmplitudes : c.RegularAmplitudes).Max());
                    }
                }
                for(int i=0; i<Channels.Length; i++) {
                    Channel c = Channels[i];
                    EditorGUILayout.BeginHorizontal();

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
                    {
                        prevPos.x = rect.xMin;
                        prevPos.y = rect.yMax - (0f).Normalize(min, max, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + rect.width;
                        newPos.y = rect.yMax - (0f).Normalize(min, max, 0f, 1f) * rect.height;
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(0.5f));
                    }

                    //Phase 1D
                    for(int j=0; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - c.GetPhaseValue(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror) * rect.height;
                        float weight = c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f);
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Cyan.Opacity(weight));
                    }

                    //Phase 2D X
                    for(int j=1; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).x.Normalize(-1f, 1f, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).x.Normalize(-1f, 1f, 0f, 1f) * rect.height;
                        float weight = c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f);
                        // UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Orange.Opacity(weight));
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White.Opacity(weight));
                    }
                    //Phase 2D Y
                    for(int j=1; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).y.Normalize(-1f, 1f, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).y.Normalize(-1f, 1f, 0f, 1f) * rect.height;
                        float weight = c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f);
                        // UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(weight));
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White.Opacity(weight));
                    }

                    UltiDraw.End();

                    if(DrawPivot) {
                        editor.DrawPivot(rect);
                        editor.DrawWindow(editor.GetCurrentFrame(), 1f/Channels[i].GetFrequency(editor.GetTimestamp(), editor.Mirror), Color.green.Opacity(0.25f), rect);
                    }

                    EditorGUILayout.EndVertical();

                    EditorGUILayout.EndHorizontal();

                    // EditorGUILayout.HelpBox(Channels[i].GetFrequency(editor.GetTimestamp(), editor.Mirror).ToString(), MessageType.None, true);
                }
                if(ShowParameters) {
                    foreach(Channel c in Channels) {
                        EditorGUILayout.HelpBox("F: " + c.GetFrequency(editor.GetTimestamp(), editor.Mirror).ToString("F3") + " / " + "D: " + (editor.TargetFramerate * c.GetDelta(editor.GetTimestamp(), editor.Mirror, 1f/editor.TargetFramerate)).ToString("F3"), MessageType.None);
                    }
                    {
                        EditorGUILayout.BeginHorizontal();

                        EditorGUILayout.BeginVertical(GUILayout.Height(height));
                        Rect ctrl = EditorGUILayout.GetControlRect();
                        Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
                        EditorGUI.DrawRect(rect, UltiDraw.Black);

                        UltiDraw.Begin();

                        Vector3 prevPos = Vector3.zero;
                        Vector3 newPos = Vector3.zero;
                        Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
                        Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

                        for(int i=0; i<Channels.Length; i++) {
                            Channel c = Channels[i];
                            for(int j=1; j<view.z; j++) {
                                prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                                prevPos.y = rect.yMax - (float)c.GetAmplitude(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f) * rect.height;
                                newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                                newPos.y = rect.yMax - (float)c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f) * rect.height;
                                UltiDraw.DrawLine(prevPos, newPos, UltiDraw.GetRainbowColor(i, Channels.Length));
                            }
                        }

                        UltiDraw.End();

                        if(DrawPivot) {
                            editor.DrawPivot(rect);
                        }

                        EditorGUILayout.EndVertical();

                        EditorGUILayout.EndHorizontal();
                    }
                    {
                        EditorGUILayout.BeginHorizontal();

                        EditorGUILayout.BeginVertical(GUILayout.Height(height));
                        Rect ctrl = EditorGUILayout.GetControlRect();
                        Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
                        EditorGUI.DrawRect(rect, UltiDraw.Black);

                        UltiDraw.Begin();

                        Vector3 prevPos = Vector3.zero;
                        Vector3 newPos = Vector3.zero;
                        Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
                        Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

                        for(int i=0; i<Channels.Length; i++) {
                            Channel c = Channels[i];
                            for(int j=1; j<view.z; j++) {
                                prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                                prevPos.y = rect.yMax - (float)c.GetFrequency(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).Normalize(0f, maxFrequency, 0f, 1f) * rect.height;
                                newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                                newPos.y = rect.yMax - (float)c.GetFrequency(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxFrequency, 0f, 1f) * rect.height;
                                UltiDraw.DrawLine(prevPos, newPos, UltiDraw.GetRainbowColor(i, Channels.Length));
                            }
                        }

                        UltiDraw.End();

                        if(DrawPivot) {
                            editor.DrawPivot(rect);
                        }

                        EditorGUILayout.EndVertical();

                        EditorGUILayout.EndHorizontal();
                    }
                    {
                        EditorGUILayout.BeginHorizontal();

                        EditorGUILayout.BeginVertical(GUILayout.Height(height));
                        Rect ctrl = EditorGUILayout.GetControlRect();
                        Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
                        EditorGUI.DrawRect(rect, UltiDraw.Black);

                        UltiDraw.Begin();

                        Vector3 prevPos = Vector3.zero;
                        Vector3 newPos = Vector3.zero;
                        Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
                        Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

                        for(int i=0; i<Channels.Length; i++) {
                            Channel c = Channels[i];
                            for(int j=1; j<view.z; j++) {
                                prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                                prevPos.y = rect.yMax - (float)c.GetOffset(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).Normalize(-maxOffset, maxOffset, 0f, 1f) * rect.height;
                                newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                                newPos.y = rect.yMax - (float)c.GetOffset(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(-maxOffset, maxOffset, 0f, 1f) * rect.height;
                                UltiDraw.DrawLine(prevPos, newPos, UltiDraw.GetRainbowColor(i, Channels.Length));
                            }
                        }

                        UltiDraw.End();

                        if(DrawPivot) {
                            editor.DrawPivot(rect);
                        }

                        EditorGUILayout.EndVertical();

                        EditorGUILayout.EndHorizontal();
                    }
                }
            }

            //STATIC CURVE VISUALIZATION
            if(Curves != null && Curves.Asset != Asset) {
                Curves = null;
            }
            if(Utility.GUIButton("Compute Curves", UltiDraw.DarkGrey, UltiDraw.White)) {
                ComputeCurves(Asset, editor.GetSession().GetActor(), editor.GetTimeSeries());
            }
            if(Curves != null) {
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Curves.MinView = EditorGUILayout.FloatField("Min View", Curves.MinView);
                    Curves.MaxView = EditorGUILayout.FloatField("Max View", Curves.MaxView);

                    EditorGUILayout.HelpBox("Curves " + Curves.Curves.Length, MessageType.None);
                    TimeSeries timeSeries = editor.GetTimeSeries();
                    Actor actor = editor.GetSession().GetActor();

                    float height = 50f;
                    Vector3Int view = editor.GetView();
                    for(int i=0; i<Curves.Curves.Length; i++) {
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
                        {
                            prevPos.x = rect.xMin;
                            prevPos.y = rect.yMax - (0f).Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            newPos.x = rect.xMin + rect.width;
                            newPos.y = rect.yMax - (0f).Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(0.5f));
                        }

                        //Values
                        Vector3[] values = Curves.Curves[i].GetValues(editor.Mirror);
                        for(int j=1; j<view.z; j++) {
                            prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                            prevPos.y = rect.yMax - (float)values[view.x+j-1-1].x.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                            newPos.y = rect.yMax - (float)values[view.x+j-1].x.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Red);
                        }
                        for(int j=1; j<view.z; j++) {
                            prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                            prevPos.y = rect.yMax - (float)values[view.x+j-1-1].y.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                            newPos.y = rect.yMax - (float)values[view.x+j-1].y.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Green);
                        }
                        for(int j=1; j<view.z; j++) {
                            prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                            prevPos.y = rect.yMax - (float)values[view.x+j-1-1].z.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                            newPos.y = rect.yMax - (float)values[view.x+j-1].z.Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Blue);
                        }

                        UltiDraw.End();

                        if(DrawPivot) {
                            editor.DrawPivot(rect);
                        }

                        EditorGUILayout.EndVertical();
                    }
                }
            }

            if(Distances != null && Distances.Asset != Asset) {
                Distances = null;
            }
            if(Utility.GUIButton("Compute Distances", UltiDraw.DarkGrey, UltiDraw.White)) {
                ComputeDistances(Asset, editor.GetSession().GetActor(), editor.GetTimeSeries(), this);
            }
            if(Distances != null) {
                float height = 50f;
                Vector3Int view = editor.GetView();
                void DrawChannel(DistanceSeries.Map channel, Color color) {
                    EditorGUILayout.BeginVertical(GUILayout.Height(height));
                    Rect ctrl = EditorGUILayout.GetControlRect();
                    Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, height);
                    EditorGUI.DrawRect(rect, UltiDraw.Black);
                    UltiDraw.Begin();
                    Vector3 prevPos = Vector3.zero;
                    Vector3 newPos = Vector3.zero;
                    Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
                    Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

                    int pivot = editor.GetCurrentFrame().Index-1;
                    for(int j=1; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax - (float)channel.Values[pivot][view.x+j-1-1].Normalize(channel.Min, channel.Max, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)channel.Values[pivot][view.x+j-1].Normalize(channel.Min, channel.Max, 0f, 1f) * rect.height;
                        UltiDraw.DrawLine(prevPos, newPos, color);
                    }

                    UltiDraw.End();

                    if(DrawPivot) {
                        editor.DrawPivot(rect);
                    }

                    EditorGUILayout.EndVertical();
                }
                DrawChannel(Distances.GlobalChannel, Color.cyan);
                foreach(DistanceSeries.Map channel in Distances.LocalChannels) {
                    DrawChannel(channel, Color.white);
                }
            }
		}

        public Vector2[] GetManifold(float timestamp, bool mirrored, TimeSeries timeSeries) {
            Vector2[] values = new Vector2[timeSeries.KeyCount * Channels.Length];
            int pivot = 0;
            for(int i=0; i<timeSeries.KeyCount; i++) {
                for(int j=0; j<Channels.Length; j++) {
                    values[pivot] = Channels[j].GetManifoldVector(timestamp + timeSeries.GetKey(i).Timestamp, mirrored); pivot += 1;
                }
            }
            return values;
        }

        public Vector2[] GetManifold(float timestamp, bool mirrored) {
            Vector2[] values = new Vector2[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetManifoldVector(timestamp, mirrored);
            }
            return values;
        }

        public Vector2[] GetPhaseVectors(float timestamp, bool mirrored) {
            Vector2[] values = new Vector2[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetPhaseVector(timestamp, mirrored);
            }
            return values;
        }

        public float[] GetPhaseValues(float timestamp, bool mirrored) {
            float[] values = new float[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetPhaseValue(timestamp, mirrored);
            }
            return values;
        }

        public float[] GetFrequencies(float timestamp, bool mirrored) {
            float[] values = new float[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetFrequency(timestamp, mirrored);
            }
            return values;
        }

        public float[] GetAmplitudes(float timestamp, bool mirrored) {
            float[] values = new float[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetAmplitude(timestamp, mirrored);
            }
            return values;
        }

        public float[] GetOffsets(float timestamp, bool mirrored) {
            float[] values = new float[Channels.Length];
            for(int i=0; i<values.Length; i++) {
                values[i] = Channels[i].GetOffset(timestamp, mirrored);
            }
            return values;
        }

        public void CreateChannels(int length) {
            Channels = new Channel[length];
            for(int i=0; i<Channels.Length; i++) {
                Channels[i] = new Channel(this);
            }
            Asset.MarkDirty(true, false);
        }

        public bool VerifyChannels(int length) {
            if(Channels.Length != length || Channels.Any(null)) {
                // Debug.Log("Recreating phase channels in asset: " + Asset.name);
                return false;
            }
            foreach(Channel channel in Channels) {
                if(new object[]{
                    channel.RegularPhaseValues,
                    channel.RegularFrequencies,
                    channel.RegularAmplitudes,
                    channel.RegularOffsets,
                    channel.MirroredPhaseValues,
                    channel.MirroredFrequencies,
                    channel.MirroredAmplitudes,
                    channel.MirroredOffsets
                }.Any(null)) {
                    // Debug.Log("Recreating phase channels in asset: " + Asset.name);
                    return false;
                }
            }
            return true;
        }


        // public class Series : TimeSeries.Component {
        //     public int Channels;
        //     public Vector2[] Phases;
        //     public float[] Amplitudes;
        //     public float[] Frequencies;

        //     public Vector2[][] Manifold;

        //     private float Max = float.MinValue;

        //     private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.15f, 0.2f, 0.15f);

        //     public Series(TimeSeries global, int channels) : base(global) {
        //         Channels = channels;
        //         Phases = new Vector2[channels];
        //         Amplitudes = new float[channels];
        //         Frequencies = new float[channels];
        //         Phases.SetAll(Vector2.up);
        //         Amplitudes.SetAll(0f);
        //         Frequencies.SetAll(0f);
        //         Manifold = new Vector2[channels][];
        //         for(int i=0; i<channels; i++) {
        //             Manifold[i] = new Vector2[Samples.Length];
        //         }
        //     }

        //     public void UpdateAlignment(float[] values, float stability, float deltaTime) {
        //         int pivot = 0;
        //         for(int b=0; b<Channels; b++) {
        //             Vector2 p = Phases[b];
        //             float a = Amplitudes[b];
        //             float f = Frequencies[b];
        //             Vector2 update = Quaternion.AngleAxis(-f*360f*deltaTime, Vector3.forward) * p;

        //             Vector2 next = new Vector2(values[pivot+0], values[pivot+1]).normalized;
        //             float amp = Mathf.Abs(values[pivot+2]);
        //             float freq = Mathf.Abs(values[pivot+3]);
        //             // Debug.Log(next + " / " + amp + " / " + freq);
        //             pivot += 4;

        //             Phases[b] = Vector3.Slerp(update.normalized, next.normalized, stability).ZeroZ().normalized;
        //             Amplitudes[b] = amp;
        //             Frequencies[b] = freq;

        //             Max = Mathf.Max(Max, amp);
        //         }
        //         ComputeAlignment();
        //     }

        //     public void ComputeAlignment() {
        //         for(int i=0; i<Channels; i++) {
        //             Vector2 p = Phases[i];
        //             float a = Amplitudes[i];
        //             float f = Frequencies[i];
        //             for(int j=0; j<Samples.Length; j++) {
        //                 float t = Samples[j].Timestamp;
        //                 Manifold[i][j] = a * (Quaternion.AngleAxis(-f*360f*t, Vector3.forward) * p);
        //             }
        //         }
        //     }

        //     public float[] GetAlignment() {
        //         int pivot = 0;
        //         float[] alignment = new float[Channels * KeyCount * 2];
        //         for(int i=0; i<Channels; i++) {
        //             for(int j=0; j<KeyCount; j++) {
        //                 int index = GetKey(j).Index;
        //                 alignment[pivot] = Manifold[i][index].x; pivot += 1;
        //                 alignment[pivot] = Manifold[i][index].y; pivot += 1;
        //             }
        //         }
        //         return alignment;
        //     }

        //     public float[] GetUpdate() {
        //         int pivot = 0;
        //         float[] update = new float[Channels * 4];
        //         for(int b=0; b<Channels; b++) {
        //             Vector2 phase = Phases[b];
        //             float amp = Amplitudes[b];
        //             float freq = Frequencies[b];
        //             phase *= amp;
        //             update[pivot] = phase.x; pivot += 1;
        //             update[pivot] = phase.y; pivot += 1;
        //             update[pivot] = amp; pivot += 1;
        //             update[pivot] = freq; pivot += 1;
        //         }
        //         return update;
        //     }

        //     public override void Increment(int start, int end) {

        //     }

        //     public override void Interpolate(int start, int end) {

        //     }

        //     public override void GUI() {
        //         if(DrawGUI) {

        //         }
        //     }

        //     public override void Draw() {
        //         if(DrawScene) {
        //             UltiDraw.Begin();
        //             float min = 0.05f;
        //             float max = 0.2f;
        //             float amplitude = Max == float.MinValue ? 1f : Max;
        //             float h = (max-min)/Channels;
        //             // Vector2[][] phases = Phases.GetTranspose();
        //             // float[][] amplitudes = Amplitudes.GetTranspose();
        //             // float[][] frequencies = Frequencies.GetTranspose();
        //             for(int i=0; i<Channels; i++) {
        //                 float ratio = i.Ratio(0, Channels-1);

        //                 UltiDraw.PlotFunctions(new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)), new Vector2(0.75f, h), Manifold[i], -amplitude, amplitude, backgroundColor:UltiDraw.Black, lineColors:new Color[]{UltiDraw.Magenta, UltiDraw.Cyan});

        //                 //Phases
        //                 float[] a = new float[Samples.Length];
        //                 float[] p = new float[Samples.Length];
        //                 Color[] c = new Color[Samples.Length];
        //                 for(int j=0; j<Samples.Length; j++) {
        //                     p[j] = Utility.PhaseValue(Manifold[i][j]);
        //                     a[j] = Manifold[i][j].magnitude;
        //                     c[j] = UltiDraw.White.Opacity(a[j].Normalize(0f, amplitude, 0f, 1f));
        //                 }
        //                 UltiDraw.PlotBars(new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)), new Vector2(0.75f, h), p, 0f, 1f, backgroundColor:UltiDraw.Transparent, barColors:c);
        //                 // UltiDraw.PlotBars(new Vector2(0.25f, ratio.Normalize(0f, 1f, max+h/2f, max+(max-min)-h/2f)), new Vector2(0.25f, h), p, 0f, 1f, barColors:c);

        //                 // //Amplitudes
        //                 // UltiDraw.PlotFunction(new Vector2(0.3f, ratio.Normalize(0f, 1f, max+h/2f, max+(max-min)-h/2f)), new Vector2(0.35f, h), a, 0f, amplitude);

        //                 // //Frequencies
        //                 // UltiDraw.PlotFunction(new Vector2(0.7f, ratio.Normalize(0f, 1f, max+h/2f, max+(max-min)-h/2f)), new Vector2(0.35f, h), frequencies[i], 0f, 3.25f);
        //             }
        //             UltiDraw.End();
        //         }
        //     }
        // }




        public class Series : TimeSeries.Component {
            public int Channels;
            public float[][] Phases;
            public float[][] Amplitudes;
            public float[][] Frequencies;

            private float Peak = 1f;
            private Queue<float> Peaks = new Queue<float>();
            private int MaxPeakCount = 100;

            private List<float[]> GatingHistory = new List<float[]>();

            private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.125f, 0.2f, 0.125f);

            private float HeightOffset = 0f;
            private float TextScale = 0.0225f;
            private UltiDraw.GUIRect ExpertWindow = new UltiDraw.GUIRect(0.1f, 0.95f, 0.125f, 0.05f);
            private UltiDraw.GUIRect PhaseWindow = new UltiDraw.GUIRect(0.5f, 0.95f, 0.125f, 0.05f);
            private UltiDraw.GUIRect GatingWindow = new UltiDraw.GUIRect(0.1f, 0.8f, 0.125f, 0.125f);

            public Series(TimeSeries global, int channels) : base(global) {
                Channels = channels;
                Phases = new float[Samples.Length][];
                Amplitudes = new float[Samples.Length][];
                Frequencies = new float[Samples.Length][];
                for(int i=0; i<Samples.Length; i++) {
                    Phases[i] = new float[channels];
                    Amplitudes[i] = new float[channels];
                    Frequencies[i] = new float[channels];
                }
            }

            public float[] GetAlignment(int start, int end) {
                int pivot = 0;
                float[] alignment = new float[Channels * (end-start+1) * 2];
                for(int k=start; k<=end; k++) {
                    int index = GetKey(k).Index;
                    for(int b=0; b<Channels; b++) {
                        Vector2 vector = Amplitudes[index][b] * Utility.PhaseVector(Phases[index][b]);
                        alignment[pivot] = vector.x; pivot += 1;
                        alignment[pivot] = vector.y; pivot += 1;
                    }
                }
                return alignment;
            }

            public int GetAlignmentDimensionality() {
                return Channels * KeyCount * 2;
            }

            public float[] GetAlignment() {
                int pivot = 0;
                float[] alignment = new float[GetAlignmentDimensionality()];
                for(int k=0; k<KeyCount; k++) {
                    int index = GetKey(k).Index;
                    for(int b=0; b<Channels; b++) {
                        Vector2 vector = Amplitudes[index][b] * Utility.PhaseVector(Phases[index][b]);
                        alignment[pivot] = vector.x; pivot += 1;
                        alignment[pivot] = vector.y; pivot += 1;
                    }
                }
                return alignment;
            }

            public float[] GetUpdate() {
                int pivot = 0;
                float[] update = new float[Channels * (FutureKeys+1) * 4];
                for(int k=PivotKey; k<KeyCount; k++) {
                    for(int b=0; b<Channels; b++) {
                        Vector2 phase = Utility.PhaseVector(Phases[k][b]);
                        float amp = Amplitudes[k][b];
                        float freq = Frequencies[k][b];
                        phase *= amp;
                        update[pivot] = phase.x; pivot += 1;
                        update[pivot] = phase.y; pivot += 1;
                        update[pivot] = amp; pivot += 1;
                        update[pivot] = freq; pivot += 1;
                    }
                }
                return update;
            }

            public void UpdateAlignment(float[] values, float stability, float deltaTime, float minAmplitude=0f) {
                int pivot = 0;
                for(int i=PivotKey; i<KeyCount; i++) {
                    int index = GetKey(i).Index;
                    for(int b=0; b<Channels; b++) {
                        Vector2 current = Utility.PhaseVector(Phases[index][b]);
                        Vector2 next = new Vector2(values[pivot+0], values[pivot+1]).normalized;
                        float amp = Mathf.Abs(values[pivot+2]);
                        amp = Mathf.Max(amp, minAmplitude);
                        float freq = Mathf.Abs(values[pivot+3]);
                        pivot += 4;

                        Vector2 update = Quaternion.AngleAxis(-freq*360f*deltaTime, Vector3.forward) * current;

                        Phases[index][b] = Utility.PhaseValue(Vector3.Slerp(update.normalized, next.normalized, stability).ZeroZ().normalized);
                        Amplitudes[index][b] = amp;
                        Frequencies[index][b] = freq;

                        Peaks.Enqueue(amp);
                        while(Peaks.Count > MaxPeakCount) {
                            Peaks.Dequeue();
                        }
                        Peak = Peaks.ToArray().Max();
                    }
                }
            }

            public float[] GetContent() {
                int pivot = 0;
                float[] content = new float[Channels * KeyCount * 2];
                for(int k=0; k<KeyCount; k++) {
                    int index = GetKey(k).Index;
                    for(int b=0; b<Channels; b++) {
                        content[pivot] = Amplitudes[index][b]; pivot += 1;
                        content[pivot] = Frequencies[index][b]; pivot += 1;
                    }
                }
                return content;
            }

            public float[] GetCurrentAlignment() {
                int pivot = 0;
                float[] alignment = new float[Channels * 2];
                for(int b=0; b<Channels; b++) {
                    Vector2 vector = Amplitudes[Pivot][b] * Utility.PhaseVector(Phases[Pivot][b]);
                    alignment[pivot] = vector.x; pivot += 1;
                    alignment[pivot] = vector.y; pivot += 1;
                }
                return alignment;
            }

            public float[] GetCurrentUpdate() {
                int pivot = 0;
                float[] update = new float[Channels * 4];
                for(int b=0; b<Channels; b++) {
                    Vector2 phase = Utility.PhaseVector(Phases[Pivot][b]);
                    float amp = Amplitudes[Pivot][b];
                    float freq = Frequencies[Pivot][b];
                    phase *= amp;
                    update[pivot] = phase.x; pivot += 1;
                    update[pivot] = phase.y; pivot += 1;
                    update[pivot] = amp; pivot += 1;
                    update[pivot] = freq; pivot += 1;
                }
                return update;
            }

            public void UpdateCurrentAlignment(float[] values, float stability, float deltaTime, float minAmplitude=0f) {
                int pivot = 0;
                for(int b=0; b<Channels; b++) {
                    Vector2 current = Utility.PhaseVector(Phases[Pivot][b]);
                    Vector2 next = new Vector2(values[pivot+0], values[pivot+1]).normalized;
                    float amp = Mathf.Abs(values[pivot+2]);
                    amp = Mathf.Max(amp, minAmplitude);
                    float freq = Mathf.Abs(values[pivot+3]);
                    pivot += 4;

                    Vector2 update = Quaternion.AngleAxis(-freq*360f*deltaTime, Vector3.forward) * current;

                    Phases[Pivot][b] = Utility.PhaseValue(Vector3.Slerp(update.normalized, next.normalized, stability).ZeroZ().normalized);
                    Amplitudes[Pivot][b] = amp;
                    Frequencies[Pivot][b] = freq;

                    Peaks.Enqueue(amp);
                    while(Peaks.Count > MaxPeakCount) {
                        Peaks.Dequeue();
                    }
                    Peak = Peaks.ToArray().Max();
                }
            }

            public void AddGatingHistory(float[] values, int maxHistory=100) {
                GatingHistory.Add(values);
                if(GatingHistory.Count > maxHistory) {
                    GatingHistory.RemoveAt(0);
                }
            }

            public override void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
                    for(int j=0; j<Channels; j++) {
                        Phases[i][j] = Phases[i+1][j];
                        Amplitudes[i][j] = Amplitudes[i+1][j];
                        Frequencies[i][j] = Frequencies[i+1][j];
                    }
                }
            }

            public override void Interpolate(int start, int end) {
                for(int i=start; i<end; i++) {
                    float weight = (float)(i % Resolution) / (float)Resolution;
                    int prevIndex = GetPreviousKey(i).Index;
                    int nextIndex = GetNextKey(i).Index;
                    for(int j=0; j<Channels; j++) {
                        Vector2 prev = Amplitudes[prevIndex][j] * Utility.PhaseVector(Phases[prevIndex][j]);
                        Vector2 next = Amplitudes[nextIndex][j] * Utility.PhaseVector(Phases[nextIndex][j]);
                        float update = Utility.PhaseUpdate(Utility.PhaseValue(prev), Utility.PhaseValue(next));
                        Phases[i][j] = Utility.PhaseValue(Quaternion.AngleAxis(-update*360f*weight, Vector3.forward) * prev.normalized);
                        Amplitudes[i][j] = Mathf.Lerp(Amplitudes[prevIndex][j], Amplitudes[nextIndex][j], weight);
                        Frequencies[i][j] = Mathf.Lerp(Frequencies[prevIndex][j], Frequencies[nextIndex][j], weight);
                    }
                }
            }

            public override void GUI() {
				if(DrawGUI) {
					UltiDraw.Begin();
                    UltiDraw.OnGUILabel(PhaseWindow.GetCenter(), PhaseWindow.GetSize(), TextScale, "Phase State", Color.white);
					// UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.0875f), Rect.GetSize(), 0.0175f, "Motion Content", UltiDraw.Black);
                    UltiDraw.OnGUILabel(ExpertWindow.GetCenter(), ExpertWindow.GetSize(), TextScale, "Expert Activation", Color.white);
					UltiDraw.End();
				}
            }

            public static void DrawPhaseState(Vector2 center, float radius, Vector2[] manifold, int channels, float max) {
                float[] amplitudes = new float[manifold.Length];
                Vector2[] phases = new Vector2[manifold.Length];
                for(int i=0; i<manifold.Length; i++) {
                    amplitudes[i] = manifold[i].magnitude;
                    phases[i] = manifold[i].normalized;
                }
                DrawPhaseState(center, radius, amplitudes, phases, channels, max);
            }

            public static void DrawPhaseState(Vector2 center, float radius, float[] amplitudes, Vector2[] phases, int channels, float max) {
                float outerRadius = radius;
                float innerRadius = 2f*Mathf.PI*outerRadius/(channels+1);
                float amplitude = max == float.MinValue ? 1f : max;

                UltiDraw.Begin();
                UltiDraw.GUICircle(center, 1.05f*2f*outerRadius, UltiDraw.White);
                UltiDraw.GUICircle(center, 2f*outerRadius, UltiDraw.BlackGrey);
                for(int i=0; i<channels; i++) {
                    float activation = amplitudes[i].Normalize(0f, max, 0f, 1f);
                    Color color = UltiDraw.GetRainbowColor(i, channels).Darken(0.5f);
                    float angle = Mathf.Deg2Rad*360f*i.Ratio(0, channels);
                    Vector2 position = center + outerRadius * new Vector2(Mathf.Sin(angle), UltiDraw.AspectRatio() * Mathf.Cos(angle));
                    UltiDraw.GUILine(center, position, 0f, activation*innerRadius, UltiDraw.GetRainbowColor(i, channels).Opacity(activation));
                    UltiDraw.GUICircle(position, innerRadius, color);
                    UltiDraw.PlotCircularPivot(position, 0.9f*innerRadius, 360f*Utility.PhaseValue(phases[i]), amplitudes[i].Normalize(0f, amplitude, 0f, 1f), amplitudes[i] == 0f ? UltiDraw.Red : UltiDraw.White, UltiDraw.Black);
                }
                UltiDraw.End();
            }

            private Vector2[] Vectors;
            public override void Draw() {
                if(DrawScene) {
                    Vector2[] manifold = new Vector2[Channels];
                    for(int i=0; i<manifold.Length; i++) {
                        manifold[i] = Amplitudes[Pivot][i] * Utility.PhaseVector(Phases[Pivot][i]);
                    }
                    DrawPhaseState(new Vector2(PhaseWindow.X, 0.7875f + HeightOffset), 0.04f, Amplitudes[Pivot], manifold, Channels, Peak);
                    if(GatingHistory != null) {
                        UltiDraw.Begin();
                        UltiDraw.DrawInterpolationSpace(GatingWindow, GatingHistory);
                        UltiDraw.End();
                    }
                }

				// if(DrawGUI) {
				// 	UltiDraw.Begin();
                //     Vector2 size = Rect.GetSize();
                //     size.y *= 0.5f;
                //     Vector2 centerA = Rect.GetCenter();
                //     centerA.y += 0.5f*size.y;
                //     Vector2 centerB = Rect.GetCenter();
                //     centerB.y -= 0.5f*size.y;
				// 	UltiDraw.PlotFunctions(centerA, size, Amplitudes, UltiDraw.Dimension.Y, yMin: 0f, yMax: 2.5f, thickness: 0.0025f);
				// 	UltiDraw.PlotFunctions(centerB, size, Frequencies, UltiDraw.Dimension.Y, yMin: 0f, yMax: 5f, thickness: 0.0025f);
				// 	UltiDraw.End();
				// }

                // if(DrawScene) {
                //     UltiDraw.Begin();
                //     float min = 0.05f;
                //     float max = 0.2f;
                //     // float w = 0.25f;
                //     float amplitude = Peak;
                //     float h = (max-min)/Channels;
                //     // Vector2[][] phases = Phases.GetTranspose();
                //     // float[][] amplitudes = Amplitudes.GetTranspose();
                //     // float[][] frequencies = Frequencies.GetTranspose();
                //     for(int i=0; i<Channels; i++) {
                //         float ratio = i.Ratio(0, Channels-1);

                //         // Vectors = Vectors.Validate(Samples.Length);
                //         // for(int j=0; j<Vectors.Length; j++) {
                //         //     Vectors[j] = Amplitudes[j][i] * Phases[j][i];
                //         // }
                //         // UltiDraw.PlotFunctions(new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)), new Vector2(w, h), Vectors, -amplitude, amplitude, thickness:0.001f, backgroundColor:UltiDraw.Black, lineColors:new Color[]{Color.white, Color.white}); //lineColors:new Color[]{UltiDraw.Magenta, UltiDraw.Cyan});
                //         // UltiDraw.PlotFunctions(
                //         //     new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)),
                //         //     new Vector2(0.75f, h),
                //         //     vectors,
                //         //     -amplitude,
                //         //     amplitude,
                //         //     backgroundColor:UltiDraw.Black,
                //         //     lineColors:new Color[]{UltiDraw.White, UltiDraw.White}
                //         // );

                //         // //Phases DRAW HERE
                //         float[] a = new float[Samples.Length];
                //         float[] p = new float[Samples.Length];
                //         Color[] c = new Color[Samples.Length];
                //         for(int j=0; j<Samples.Length; j++) {
                //             p[j] = Phases[j][i];
                //             a[j] = Amplitudes[j][i];
                //             c[j] = UltiDraw.White.Opacity(a[j].Normalize(0f, amplitude, 0f, 1f));
                //         }
                //         // UltiDraw.PlotBars(new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)), new Vector2(0.75f, h), p, 0f, 1f, backgroundColor:UltiDraw.Transparent, barColors:c);

                //         UltiDraw.PlotBars(new Vector2(0.5f, ratio.Normalize(0f, 1f, min+h/2f, max-h/2f)), new Vector2(0.5f, h), p, 0f, 1f, backgroundColor:UltiDraw.Black, barColors:c);

                //         // //Amplitudes
                //         // UltiDraw.PlotFunction(new Vector2(0.3f, ratio.Normalize(0f, 1f, max+h/2f, max+(max-min)-h/2f)), new Vector2(0.35f, h), a, 0f, amplitude);

                //         // //Frequencies
                //         // UltiDraw.PlotFunction(new Vector2(0.7f, ratio.Normalize(0f, 1f, max+h/2f, max+(max-min)-h/2f)), new Vector2(0.35f, h), frequencies[i], 0f, 3.25f);
                //     }
                //     UltiDraw.End();
                // }
            }
        }

	}
}
#endif
