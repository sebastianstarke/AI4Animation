using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AI4Animation {
    public class MotionTrailDrawer : MonoBehaviour {

        public int Horizon = 60;
        public Color LineColor = Color.magenta;
        public float LineWidth = 0.1f;
        [Range(0f,1f)] public float LineFade = 0f;
        public List<Transform> Transforms = new List<Transform>();
        
        public float YMin = -1f;
        public float YMax = 1f;
        public bool DrawGraph = true;
        public float Thickness = 0.05f;
        public Color BackgroundColor = Color.white;

        private Dictionary<Transform, List<Vector3>> History = new Dictionary<Transform, List<Vector3>>();

        void LateUpdate() {
            List<Transform> keys = new List<Transform>(History.Keys);
            foreach(Transform key in keys) {
                if(!History.ContainsKey(key)) {
                    History.Remove(key);
                }
            }

            foreach(Transform t in Transforms) {
                if(!History.ContainsKey(t)) {
                    History.Add(t, new List<Vector3>());
                }
            }

            foreach(Transform t in Transforms) {
                List<Vector3> trail = History[t];
                trail.Add(t.position);
                while(trail.Count > Horizon) {
                    trail.RemoveAt(0);
                }
            }
        }

        void OnRenderObject() {
            UltiDraw.Begin();
            int pivot = 0;
            foreach(List<Vector3> trail in History.Values) {
                for(int i=1; i<trail.Count; i++) {
                    float fade = i.Ratio(0, trail.Count-1);
                    fade = Mathf.Pow(fade, LineFade);
                    UltiDraw.DrawLine(trail[i-1], trail[i], LineWidth, Transforms.Count == 1 ? LineColor.Opacity(fade) : UltiDraw.GetRainbowColor(pivot, History.Count).Opacity(fade));
                }
                pivot += 1;
            }
            if(DrawGraph) {
                List<float[]> values = new List<float[]>();
                foreach(List<Vector3> trail in History.Values) {
                    float[] magnitudes = new float[trail.Count-1];
                    for(int i=1; i<trail.Count; i++) {
                        magnitudes[i-1] = (trail[i] - trail[i-1]).magnitude;
                    }
                    values.Add(magnitudes);
                }
                UltiDraw.PlotFunctions(
                    new Vector2(0.5f, 0.15f),
                    new Vector2(0.75f, 0.25f),
                    values.ToArray(),
                    UltiDraw.Dimension.X,
                    YMin, YMax,
                    thickness:Thickness,
                    backgroundColor:BackgroundColor,
                    lineColor:LineColor
                );
            }
            UltiDraw.End();
        }

    }
}