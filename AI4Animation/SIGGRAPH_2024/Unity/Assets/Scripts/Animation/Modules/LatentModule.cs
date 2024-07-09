using System;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    public class LatentModule : Module {
        public int Dimensions = 0;
        public Vector[] Vectors = new Vector[0];
        public float Min = -1f;
        public float Max = 1f;

        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }

        public void SetValues(float timestamp, bool mirrored, float[] values) {
            if(mirrored) {
                Vectors[Asset.GetFrame(timestamp).Index-1].Mirrored = values;
            } else {
                Vectors[Asset.GetFrame(timestamp).Index-1].Standard = values;
            }
        }

        public float[] GetValues(float timestamp, bool mirrored) {
            Vector vector = Vectors[Asset.GetFrame(timestamp).Index-1];
            return mirrored ? vector.Mirrored : vector.Standard;
        }
#if UNITY_EDITOR
        protected override void DerivedInitialize() {
            Vectors = new Vector[Asset.Frames.Length];
            for(int i=0; i<Vectors.Length; i++) {
                Vectors[i] = new Vector();
            }
        }

        protected override void DerivedLoad(MotionEditor editor) {

        }

        protected override void DerivedUnload(MotionEditor editor) {

        }
    
        protected override void DerivedCallback(MotionEditor editor) {
            
        }

        protected override void DerivedGUI(MotionEditor editor) {
        
        }

        protected override void DerivedDraw(MotionEditor editor) {

        }

        protected override void DerivedInspector(MotionEditor editor) {
            EditorGUI.BeginDisabledGroup(true);
            Dimensions = EditorGUILayout.IntField("Dimensions", Dimensions);
            Min = EditorGUILayout.FloatField("Min", Min);
            Max = EditorGUILayout.FloatField("Max", Max);
            EditorGUI.EndDisabledGroup();

            Frame frame = editor.GetCurrentFrame();

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

            //Values
            Color[] colors = UltiDraw.GetRainbowColors(Dimensions);
            float[] from = null;
            float[] to = null;
            for(int j=start; j<end; j++) {
                int prev = j;
                int next = j+1;
                for(int d=0; d<Dimensions; d++) {
                    float[] p = GetValues(Asset.GetFrame(Mathf.Clamp(prev, start, end)).Timestamp, editor.Mirror);
                    float[] n = GetValues(Asset.GetFrame(Mathf.Clamp(next, start, end)).Timestamp, editor.Mirror);
                    if(p.Length == Dimensions) {
                        from = p;
                    }
                    if(n.Length == Dimensions) {
                        to = n;
                    }
                    if(from != null && to != null) {
                        float prevValue = from[d];
                        float nextValue = to[d];
                        prevValue = prevValue.Normalize(Min, Max, 0f, 1f);
                        nextValue = nextValue.Normalize(Min, Max, 0f, 1f);
                        float _start = (float)(Mathf.Clamp(prev, start, end)-start) / (float)elements;
                        float _end = (float)(Mathf.Clamp(next, start, end)-start) / (float)elements;
                        float xStart = rect.x + _start * rect.width;
                        float xEnd = rect.x + _end * rect.width;
                        float yStart = rect.y + (1f - prevValue) * rect.height;
                        float yEnd = rect.y + (1f - nextValue) * rect.height;
                        UltiDraw.DrawLine(new Vector3(xStart, yStart, 0f), new Vector3(xEnd, yEnd, 0f), colors[d]);
                    }
                    if(n.Length == Dimensions) {
                        from = to;
                    }
                }
            }

            UltiDraw.End();

            editor.DrawPivot(rect);
            EditorGUILayout.EndVertical();
        }
#endif

        [Serializable]
        public class Vector {
            public float[] Standard;
            public float[] Mirrored;
        }
    }
}
