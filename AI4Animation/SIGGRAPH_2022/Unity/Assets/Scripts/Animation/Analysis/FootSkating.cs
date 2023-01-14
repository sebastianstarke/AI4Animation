using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AI4Animation {
    public class FootSkating : MonoBehaviour {

        public float Framerate = 60f;
        public int Horizon = 60;
        public List<Actor> Actors = new List<Actor>();
        
        public bool AutoMax = true;
        public float LineStrength = 0.00125f;
        public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.9f, 0.5f, 0.2f);

        private Dictionary<Actor, Data> History = new Dictionary<Actor, Data>();
	    private float Max = 0f;

        public class Data {
            public List<float> History = new List<float>();
            public Vector3[] LastPositions = null;
        }

        void LateUpdate() {
            List<Actor> keys = new List<Actor>(History.Keys);
            foreach(Actor key in keys) {
                if(!History.ContainsKey(key)) {
                    History.Remove(key);
                }
            }

            foreach(Actor key in Actors) {
                if(!History.ContainsKey(key)) {
                    History.Add(key, new Data());
                }
            }

            foreach(Actor key in Actors) {
                Data data = History[key];
                //
                List<Vector4> contacts = new List<Vector4>();
                key.SendMessage("RetrieveContacts", contacts);
                List<Vector3> positions = new List<Vector3>();
                List<float> weights = new List<float>();
                foreach(Vector4 contact in contacts) {
                    positions.Add(new Vector3(contact.x, contact.y, contact.z));
                    weights.Add(contact.w);
                }
                float skating = 0f;
                if(data.LastPositions != null) {
                    for(int i=0; i<positions.Count; i++) {
                        skating += weights[i] * Framerate * (positions[i] - data.LastPositions[i]).magnitude;
                    }
                }
                data.History.Add(skating);
                data.LastPositions = positions.ToArray();
                //
                while(data.History.Count > Horizon) {
                    data.History.RemoveAt(0);
                }
            }
        }

        public float[][] GetValues() {
            float[][] values = new float[History.Count][];
            int pivot = 0;
            foreach(Data data in History.Values) {
                values[pivot] = data.History.ToArray(); pivot += 1;
            }
            return values;
        }

        void OnRenderObject() {
            UltiDraw.Begin();
            if(AutoMax) {
                UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), GetValues(), UltiDraw.Dimension.X, 0f, Max, thickness:LineStrength, backgroundColor:UltiDraw.Black);
            } else {
                UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), GetValues(), UltiDraw.Dimension.X, thickness:LineStrength, backgroundColor:UltiDraw.Black);
            }
            UltiDraw.End();
        }

        void OnGUI() {
            float size = 0.05f;
            UltiDraw.Begin();
            float[][] values = GetValues();
            for(int i=0; i<Actors.Count; i++) {
                float mean = values[i].Mean();
                float sigma = values[i].Sigma();
                UltiDraw.OnGUILabel(new Vector2(Rect.X - 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, mean.Round(3).ToString(), UltiDraw.Black);
                UltiDraw.OnGUILabel(new Vector2(Rect.X, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/4f, Actors[i].name, UltiDraw.GetRainbowColor(i, Actors.Count));
                UltiDraw.OnGUILabel(new Vector2(Rect.X + 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, sigma.Round(3).ToString(), UltiDraw.Black);
            }
            UltiDraw.End();
        }

    }
}