using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    [Serializable]
    public abstract class NeuralNetwork {

        public abstract class Inference {
            public int Pivot = 0;
            public double Time = 0.0;
            public abstract void Dispose();
            public abstract int GetFeedSize();
            public abstract int GetReadSize();
            public abstract void Feed(float value);
            public abstract float Read();
            public abstract void Run();
        }
        protected abstract Inference BuildInference();
        #if UNITY_EDITOR
        public abstract void Inspect();
        #endif

        private Inference Session = null;

        public Inference GetSession() {
            // if(Session == null) {
            //     Debug.Log("No session found.");
            // }
            return Session;
        }

        #if UNITY_EDITOR
        public bool Inspector() {
            EditorGUI.BeginChangeCheck();
            Utility.SetGUIColor(UltiDraw.White);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                Inspect();
            }
            return EditorGUI.EndChangeCheck();
        }
        #endif

        public void CreateSession() {
            if(Session != null) {
                Debug.Log("Session is already active.");
            } else {
                Session = BuildInference();
            }
        }

        public void CloseSession() {
            if(Session == null) { 
                Debug.Log("No session currently active.");
            } else {
                Session.Dispose();
                Session = null;
            }
        }

        public void RunSession() {
            if(GetSession() != null) {
                if(Session.Pivot != Session.GetFeedSize()) {
                    Debug.Log("Running prediction without all inputs given to the network: " + (Session.Pivot+1) + " / " + Session.GetFeedSize());
                }
                DateTime timestamp = Utility.GetTimestamp();
                Session.Run();
                Session.Time = Utility.GetElapsedTime(timestamp);
                Session.Pivot = Session.GetReadSize();
            }
        }

		public void Feed(float value) {
            if(GetSession() != null) {
                if(Session.Pivot >= Session.GetFeedSize()) {
                    Debug.Log("Attempting to feed more values than inputs available: " + (Session.Pivot+1) + " / " + Session.GetFeedSize());
                } else {
                    Session.Feed(value);
                }
                Session.Pivot += 1;
            }
		}

        public void Feed(bool value) {
            Feed(value ? 1f : 0f);
        }

        public void Feed(float[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(bool[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(Vector2 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void Feed(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.y);
            Feed(vector.z);
        }

        public void FeedXY(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void FeedXZ(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.z);
        }

        public void FeedYZ(Vector3 vector) {
            Feed(vector.y);
            Feed(vector.z);
        }

        public void Feed(Quaternion vector) {
            Feed(vector.x);
            Feed(vector.y);
            Feed(vector.z);
            Feed(vector.w);
        }

		public float Read() {
            float value = 0f;
            if(GetSession() != null) {
                if(Session.Pivot <= 0) {
                    Debug.Log("Attempting to read more values than outputs available: " + (Session.GetReadSize() - Session.Pivot) + " / " + Session.GetReadSize());
                } else {
                    value = Session.Read();
                }
                Session.Pivot -= 1;
            }
            return value;
		}

    	public float Read(float min, float max) {
            return Mathf.Clamp(Read(), min, max);
		}

        public float[] Read(int count) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read();
            }
            return values;
        }

        public float[] Read(int count, float min, float max) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read(min, max);
            }
            return values;
        }

    	// public float ReadBinary() {
        //     return Mathf.Round(Read(0f,1f));
		// }

        // public float[] ReadBinary(int count) {
        //     float[] values = new float[count];
        //     for(int i=0; i<count; i++) {
        //         values[i] = ReadBinary();
        //     }
        //     return values;
        // }

        // public float[] ReadOneHot(int count) {
        //     float[] values = Read(count);
        //     int argmax = values.ArgMax();
        //     for(int i=0; i<values.Length; i++) {
        //         values[i] = i == argmax ? 1f : 0f;
        //     }
        //     return values;
        // }

        public Vector3 ReadVector2() {
            return new Vector2(Read(), Read());
        }

        public Vector3 ReadVector3() {
            return new Vector3(Read(), Read(), Read());
        }

        public Vector3 ReadXY() {
            return new Vector3(Read(), Read(), 0f);
        }

        public Vector3 ReadXZ() {
            return new Vector3(Read(), 0f, Read());
        }

        public Vector3 ReadYZ() {
            return new Vector3(0f, Read(), Read());
        }

        public Quaternion Read6DRotation() {
            return Quaternion.LookRotation(
                ReadVector3().normalized,
                ReadVector3().normalized
            );
        }

    }
}