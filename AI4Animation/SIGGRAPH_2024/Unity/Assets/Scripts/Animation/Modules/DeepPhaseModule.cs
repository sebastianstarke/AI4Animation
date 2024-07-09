using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class DeepPhaseModule : Module {
        public Channel[] Channels = new Channel[0];
        [NonSerialized] private bool UseOffsets = false;
        [NonSerialized] private bool ShowParameters = false;
        [NonSerialized] private bool DrawWindowPoses = false;
        [NonSerialized] private bool DrawPhaseSpace = true;
        [NonSerialized] private bool DrawPivot = true;

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

            public bool[] Assigned;

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

                Assigned = new bool[module.Asset.Frames.Length];
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

            public void Interpolate() {
                for(int i=0; i<Module.Asset.Frames.Length; i++) {
                    if(!Assigned[i]) {
                        Frame current = Module.Asset.Frames[i];
                        Frame prev = current;
                        for(int j=i-1; j>0; j--) {
                            if(Assigned[j]) {
                                prev = Module.Asset.Frames[j];
                                break;
                            }
                        }
                        Frame next = current;
                        for(int j=i+1; j<Module.Asset.Frames.Length; j++) {
                            if(Assigned[j]) {
                                next = Module.Asset.Frames[j];
                                break;
                            }
                        }
                        float ratio = current.Index.Ratio(prev.Index, next.Index);
                        int a = prev.Index-1;
                        int b = next.Index-1;
                        
                        RegularPhaseValues[i] = Mathf.Repeat(RegularPhaseValues[a] + ratio * Utility.SignedPhaseUpdate(RegularPhaseValues[a], RegularPhaseValues[b]), 1f);
                        RegularFrequencies[i] = Mathf.Lerp(RegularFrequencies[a], RegularFrequencies[b], ratio);
                        RegularAmplitudes[i] = Mathf.Lerp(RegularAmplitudes[a], RegularAmplitudes[b], ratio);
                        RegularOffsets[i] = Mathf.Lerp(RegularOffsets[a], RegularOffsets[b], ratio);

                        MirroredPhaseValues[i] = Mathf.Repeat(MirroredPhaseValues[a] + ratio * Utility.SignedPhaseUpdate(MirroredPhaseValues[a], MirroredPhaseValues[b]), 1f);
                        MirroredFrequencies[i] = Mathf.Lerp(MirroredFrequencies[a], MirroredFrequencies[b], ratio);
                        MirroredAmplitudes[i] = Mathf.Lerp(MirroredAmplitudes[a], MirroredAmplitudes[b], ratio);
                        MirroredOffsets[i] = Mathf.Lerp(MirroredOffsets[a], MirroredOffsets[b], ratio);
                    }
                }
                Assigned.SetAll(true);
            }
        }

        public static void ComputeCurves(MotionAsset asset, Actor actor, TimeSeries timeSeries) {
            Curves = new CurveSeries(asset, actor, timeSeries);
        }

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            Series instance = new Series(global, Channels.Length);
            for(int i=0; i<instance.Samples.Length; i++) {
                instance.Phases[i] = GetPhaseValues(timestamp + instance.Samples[i].Timestamp, mirrored);
                instance.Amplitudes[i] = GetAmplitudes(timestamp + instance.Samples[i].Timestamp, mirrored);
                instance.Frequencies[i] = GetFrequencies(timestamp + instance.Samples[i].Timestamp, mirrored);
            }
            instance.DrawScene = DrawPhaseSpace;
            instance.DrawGUI = DrawPhaseSpace;
            return instance;
		}
#if UNITY_EDITOR
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
                DrawWindowPoses = EditorGUILayout.Toggle("Draw Window Poses", DrawWindowPoses);
                DrawPhaseSpace = EditorGUILayout.Toggle("Draw Phase Space", DrawPhaseSpace);
                DrawPivot = EditorGUILayout.Toggle("Draw Pivot", DrawPivot);

                Vector3Int view = editor.GetView();
                float height = 50f;
                float min = -1f;
                float max = 1f;
                float maxFrequency = editor.GetTimeSeries().MaximumFrequency;
                float maxOffset = 1f;
                float maxAmplitude = 0f;
                foreach(Channel c in Channels) {
                    maxAmplitude = Mathf.Max(maxAmplitude, (editor.Mirror ? c.MirroredAmplitudes : c.RegularAmplitudes).Max());
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
                        prevPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).x.Normalize(-maxAmplitude, maxAmplitude, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).x.Normalize(-maxAmplitude, maxAmplitude, 0f, 1f) * rect.height;
                        float weight = c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f);
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White.Opacity(weight));
                    }
                    //Phase 2D Y
                    for(int j=1; j<view.z; j++) {
                        prevPos.x = rect.xMin + (float)(j-1)/(view.z-1) * rect.width;
                        prevPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j-1).Timestamp, editor.Mirror).y.Normalize(-maxAmplitude, maxAmplitude, 0f, 1f) * rect.height;
                        newPos.x = rect.xMin + (float)(j)/(view.z-1) * rect.width;
                        newPos.y = rect.yMax - (float)c.GetManifoldVector(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).y.Normalize(-maxAmplitude, maxAmplitude, 0f, 1f) * rect.height;
                        float weight = c.GetAmplitude(Asset.GetFrame(view.x+j).Timestamp, editor.Mirror).Normalize(0f, maxAmplitude, 0f, 1f);
                        UltiDraw.DrawLine(prevPos, newPos, UltiDraw.White.Opacity(weight));
                    }

                    UltiDraw.End();

                    if(DrawPivot) {
                        editor.DrawPivot(rect);
                        editor.DrawWindow(editor.GetCurrentFrame(), 1f/Channels[i].GetFrequency(editor.GetTimestamp(), editor.Mirror), Color.green.Opacity(0.25f), rect);
                    }

                    EditorGUILayout.EndVertical();

                    EditorGUILayout.EndHorizontal();
                }

                if(ShowParameters) {
                    foreach(Channel c in Channels) {
                        EditorGUILayout.HelpBox("F: " + c.GetFrequency(editor.GetTimestamp(), editor.Mirror).ToString("F3") + " / " + "D: " + (editor.TargetFramerate * c.GetDelta(editor.GetTimestamp(), editor.Mirror, 1f/editor.TargetFramerate)).ToString("F3"), MessageType.None);
                    }
                    {
                        EditorGUILayout.LabelField("Amplitudes");
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
                        EditorGUILayout.LabelField("Frequencies");
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
                        EditorGUILayout.LabelField("Offsets");
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

                        //Zero
                        {
                            prevPos.x = rect.xMin;
                            prevPos.y = rect.yMax - (0f).Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            newPos.x = rect.xMin + rect.width;
                            newPos.y = rect.yMax - (0f).Normalize(Curves.MinView, Curves.MaxView, 0f, 1f) * rect.height;
                            UltiDraw.DrawLine(prevPos, newPos, UltiDraw.Magenta.Opacity(0.5f));
                        }

                        //Values
                        {
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
                        }

                        UltiDraw.End();

                        if(DrawPivot) {
                            editor.DrawPivot(rect);
                        }

                        EditorGUILayout.EndVertical();
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
                RootModule rootModule = Asset.GetModule<RootModule>("BodyWorld");
                int[] mapping = Asset.Source.GetBoneIndices(Actor.GetBoneNames());

                Curves = new Curve[mapping.Length];
                for(int i=0; i<Curves.Length; i++) {
                    Curves[i] = new Curve(this);
                }

                void Compute(int i, bool mirrored) {
                    //Velocities
                    Vector3[] velocities = mirrored ? Curves[i].MirroredValues : Curves[i].OriginalValues;
                    {
                        for(int j=0; j<velocities.Length; j++) {
                            Matrix4x4 spaceP = rootModule.GetRootTransformation(Asset.Frames[Mathf.Max(j-1,0)].Timestamp, mirrored);
                            Matrix4x4 spaceC = rootModule.GetRootTransformation(Asset.Frames[j].Timestamp, mirrored);
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

            public float[] Collect(float timestamp, bool mirrored, bool[] mask) {
                float[] values = new float[mask.Count(true) * 3];
                int idx = 0;
                for(int i=0; i<Curves.Length; i++) {
                    if(mask[i]) {
                        values[idx] = Curves[i].GetValue(timestamp, mirrored).x; idx += 1;
                        values[idx] = Curves[i].GetValue(timestamp, mirrored).y; idx += 1;
                        values[idx] = Curves[i].GetValue(timestamp, mirrored).z; idx += 1;
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
            }
        }
#endif

        public Vector2[] GetManifold(float timestamp, bool mirrored) {
            Vector2[] manifold = new Vector2[Channels.Length];
            for(int b=0; b<Channels.Length; b++) {
                manifold[b] = Channels[b].GetAmplitude(timestamp, mirrored) * Utility.PhaseVector(Channels[b].GetPhaseValue(timestamp, mirrored));
            }
            return manifold;
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
            #if UNITY_EDITOR
            Asset.MarkDirty(true, false);
            #endif
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

        public class Series : TimeSeries.Component {
            public int Channels;
            public float[][] Phases;
            public float[][] Amplitudes;
            public float[][] Frequencies;
            public Vector2[][] Manifold;

            public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.85f, 0.1f, 0.1f);
            public string ID = "Phase";
            private float TextScale = 0.0225f;

            private List<float[]> GatingHistory = new List<float[]>();

            public Series(TimeSeries global, int channels) : base(global) {
                Channels = channels;
                Phases = new float[Samples.Length][];
                Amplitudes = new float[Samples.Length][];
                Frequencies = new float[Samples.Length][];
                Manifold = new Vector2[Samples.Length][];
                for(int i=0; i<Samples.Length; i++) {
                    Phases[i] = new float[channels];
                    Amplitudes[i] = new float[channels];
                    Amplitudes[i].SetAll(1f);
                    Frequencies[i] = new float[channels];
                    Manifold[i] = new Vector2[channels];
                }
            }

            public float[] GetManifold(int start, int end, bool keys) {
                List<float> values = new List<float>();
                for(int i=start; i<end; i++) {
                    int index = keys ? GetKey(i).Index : i;
                    for(int b=0; b<Channels; b++) {
                        Vector2 phase = Amplitudes[index][b] * Utility.PhaseVector(Phases[index][b]);
                        values.Add(phase.x);
                        values.Add(phase.y);
                    }
                }
                return values.ToArray();
            }

            public float[] GetManifold(int index) {
                List<float> values = new List<float>();
                for(int b=0; b<Channels; b++) {
                    Vector2 phase = Amplitudes[index][b] * Utility.PhaseVector(Phases[index][b]);
                    values.Add(phase.x);
                    values.Add(phase.y);
                }
                return values.ToArray();
            }

            public float[] GetManifoldUpdate(int start, int end, bool keys) {
                List<float> values = new List<float>();
                for(int i=start; i<end; i++) {
                    int index = keys ? GetKey(i).Index : i;
                    for(int b=0; b<Channels; b++) {
                        Vector2 phase = Utility.PhaseVector(Phases[index][b]);
                        float frequency = Frequencies[index][b];
                        float amplitude = Amplitudes[index][b];
                        values.Add(amplitude * phase.x);
                        values.Add(amplitude * phase.y);
                        values.Add(frequency);
                        values.Add(amplitude);
                    }
                }
                return values.ToArray();
            }

            public void AdvanceManifold(float[] frequencies, float deltaTime, int steps) {
                for(int i=0; i<steps; i++) {
                    Increment(0, Pivot);

                    for(int b=0; b<Channels; b++) {
                        Phases[Pivot][b] = Mathf.Repeat(Phases[Pivot][b] + frequencies[b] * deltaTime, 1f);
                    }
                }
            }

            public void UpdateManifold(float[] values, float projection, float deltaTime) {
                Increment(0, Pivot);

                int pivot = 0;

                for(int i=PivotKey; i<KeyCount; i++) {
                    int index = GetKey(i).Index;
                    for(int b=0; b<Channels; b++) {
                        Vector2 current = Utility.PhaseVector(Phases[index][b]);
                        Vector2 next = new Vector2(values[pivot+0], values[pivot+1]).normalized;
                        float frequency = Mathf.Abs(values[pivot+2]);
                        float amplitude = values[pivot+3];
                        pivot += 4;

                        Vector2 updated = Quaternion.AngleAxis(-frequency*360f*deltaTime, Vector3.forward) * current;
                        Phases[index][b] = Utility.PhaseValue(Vector3.Slerp(updated.normalized, next.normalized, projection).ZeroZ().normalized);
                        Amplitudes[index][b] = amplitude;
                        Frequencies[index][b] = frequency;
                    }
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
                        Manifold[i][j] = Manifold[i+1][j];
                    }
                }
            }

            public void Interpolate(int start, int end) {
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

            public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.Begin();
                    UltiDraw.OnGUILabel(Rect.GetCenter(), Rect.GetSize(), TextScale, ID, Color.white);
					UltiDraw.End();
				}
            }

            public override void Draw(UltiDraw.GUIRect rect=null) {
                if(DrawScene) {
                    float peak = 0f;
                    for(int i=0; i<Samples.Length; i++) {
                        peak = Mathf.Max(peak, Amplitudes[i].Max());
                    }

                    // //Phase State
                    // {
                    //     Vector2[] manifold = new Vector2[Channels];
                    //     for(int i=0; i<manifold.Length; i++) {
                    //         manifold[i] = Amplitudes[Pivot][i] * Utility.PhaseVector(Phases[Pivot][i]);
                    //     }
                    //     DrawPhaseState(Rect.GetCenter(), Rect.GetSize(), manifold, peak);
                    // }

                    //Gating Window
                    if(GatingHistory != null && GatingHistory.Count > 0) {
                        UltiDraw.Begin();
                        UltiDraw.DrawInterpolationSpace(new UltiDraw.GUIRect(0.15f, 0.85f, 0.125f, 0.125f), GatingHistory);
                        UltiDraw.End();
                    }

                    // //Phase Window
                    // {
                    //     UltiDraw.Begin();
                    //     float[][] bars = new float[Channels][];
                    //     Color[][] colors = new Color[Channels][];
                    //     for(int i=0; i<Channels; i++) {
                    //         bars[i] = new float[SampleCount];
                    //         colors[i] = new Color[SampleCount];
                    //         for(int j=0; j<SampleCount; j++) {
                    //             float p = Phases[j][i];
                    //             float a = Amplitudes[j][i];
                    //             bars[i][j] = p;
                    //             colors[i][j] = UltiDraw.White.Opacity(a.Normalize(0f, peak, 0f, 1f));
                    //         }
                    //     }
                    //     float height = Rect.GetSize().y/2f;
                    //     UltiDraw.PlotBars(Rect.GetCenter() - new Vector2(0f, Rect.GetSize().y + height/2f), new Vector2(2f*Rect.GetSize().x, height), bars, 0f, 1f, backgroundColor:UltiDraw.Black, barColors:colors);
                    //     UltiDraw.End();
                    // }
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
            }

            public static Vector2[] KroneckerProduct(float phase, float[] features) {
                Vector2[] result = new Vector2[features.Length];
                Vector2 vector = Utility.PhaseVector(phase);
                for(int i=0; i<features.Length; i++) {
                    result[i] = features[i] * vector;
                }
                return result;
            }

            public static void DrawPhaseState(Vector2 center, Vector2 size, Vector2[] manifold, float peak) {
                float radius = size.x/2f;
                float frameScale = 1.1f;
                
                UltiDraw.Begin();
                UltiDraw.GUICircle(center, frameScale*2f*radius, UltiDraw.White);
                UltiDraw.GUICircle(center, 2f*radius, UltiDraw.BlackGrey);
                for(int i=0; i<manifold.Length; i++) {
                    float angle = Mathf.Deg2Rad*360f*i.Ratio(0, manifold.Length);
                    float activation = manifold[i].magnitude.Normalize(0f, peak, 0f, 1f);
                    Color color = UltiDraw.GetRainbowColor(i, manifold.Length).Darken(0.5f);
                    Vector2 position = center + radius * new Vector2(Mathf.Sin(angle), UltiDraw.AspectRatio() * Mathf.Cos(angle));
                    UltiDraw.GUILine(center, position, 0f, activation*radius, UltiDraw.GetRainbowColor(i, manifold.Length).Opacity(activation));
                    UltiDraw.GUICircle(position, frameScale*radius, color);
                    UltiDraw.PlotCircularPivot(position, radius, 360f*Utility.PhaseValue(manifold[i]), activation, UltiDraw.White, UltiDraw.Black);
                }
                // UltiDraw.GUIRectangle(center, 2f*size, Color.black.Opacity(0.25f));
                UltiDraw.End();
            }

            //LEGACY FUNCTIONS
            public float[] GetManifold(bool useFrequencies=false, bool useAmplitudes=true, int startKey=-1, int endKey=-1) {
                if(startKey==-1) {
                    startKey = 0;
                }
                if(endKey==-1) {
                    endKey = KeyCount;
                }
                int pivot = 0;
                int keys = endKey - startKey;
                int count = 2 * Channels * keys;
                if(useFrequencies) {
                    count += Channels * keys;
                }
                if(useAmplitudes) {
                    count += Channels * keys;
                }
                float[] alignment = new float[count];
                for(int k=startKey; k<endKey; k++) {
                    int index = GetKey(k).Index;
                    for(int b=0; b<Channels; b++) {
                        Vector2 phase = Utility.PhaseVector(Phases[index][b]);
                        float frequency = Frequencies[index][b];
                        float amplitude = Amplitudes[index][b];
                        alignment[pivot] = amplitude * phase.x; pivot += 1;
                        alignment[pivot] = amplitude * phase.y; pivot += 1;
                        if(useFrequencies) {
                            alignment[pivot] = frequency; pivot += 1;
                        }
                        if(useAmplitudes) {
                            alignment[pivot] = amplitude; pivot += 1;
                        }
                    }
                }
                return alignment;
            }
            //
        }
	}
}