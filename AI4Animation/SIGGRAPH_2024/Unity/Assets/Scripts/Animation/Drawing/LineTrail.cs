using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AI4Animation {
    public class LineTrail : MonoBehaviour {

        public int Horizon = 60;
        public Color LineColor = Color.magenta;
        public bool AutomaticColors = false;
        public float LineWidth = 0.1f;
        [Range(0f,1f)] public float LineFade = 0f;
        public List<Transform> Transforms = new List<Transform>();
        
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
                    Color color = LineColor;
                    if(Transforms.Count > 1 && AutomaticColors) {
                        color = UltiDraw.GetRainbowColor(pivot, History.Count);
                    }
                    float fade = Mathf.Pow(i.Ratio(0, trail.Count-1), LineFade);
                    UltiDraw.DrawLine(trail[i-1], trail[i], LineWidth, color.Opacity(fade));
                }
                pivot += 1;
            }
            UltiDraw.End();
        }

    }
}