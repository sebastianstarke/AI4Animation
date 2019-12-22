using System.Collections.Generic;
using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public abstract class NeuralNetwork : MonoBehaviour {

        public Matrix X {get; protected set;} = null;
        public Matrix Y {get; protected set;} = null;
        public List<Matrix> Matrices {get; private set;} = new List<Matrix>();

        public double PredictionTime {get; set;} = 0.0;
        public bool Setup {get; set;} = false;

        public bool Inspect {get; set;} = false;

        private int Pivot = -1;

        protected abstract bool SetupDerived();
        protected abstract bool ShutdownDerived();
        protected abstract void PredictDerived();

        void OnEnable() {
            Setup = SetupDerived();
        }

        void OnDisable() {
            Setup = ShutdownDerived();
        }

        public void Predict() {
            System.DateTime timestamp = Utility.GetTimestamp();
            PredictDerived();
            PredictionTime = Utility.GetElapsedTime(timestamp);
        }

        public Matrix CreateMatrix(int rows, int cols, string id) {
            if(Matrices.Exists(x => x != null && x.ID == id)) {
                Debug.Log("Matrix with ID " + id + " already contained.");
                return GetMatrix(id);
            }
            Matrix M = new Matrix(rows, cols, id);
            Matrices.Add(M);
            return M;
        }

        public void DeleteMatrix(Matrix M) {
            int index = Matrices.IndexOf(M);
            if(index == -1) {
                Debug.Log("Matrix not found.");
                return;
            }
            Matrices.RemoveAt(index);
            M.Delete();
        }

        public void DeleteMatrices() {
            foreach(Matrix m in Matrices) {
                m.Delete();
            }
            Matrices.Clear();
        }

        public Matrix GetMatrix(string id) {
            int index = Matrices.FindIndex(x => x != null && x.ID == id);
            if(index == -1) {
                Debug.Log("Matrix with ID " + id + " not contained.");
                return null;
            }
            return Matrices[index];
        }

        public void SetPivot(int index) {
            Pivot = index;
        }

        public int GetPivot() {
            return Pivot;
        }

        public void ResetPivot() {
            Pivot = -1;
        }

        public void ResetPredictionTime() {
            PredictionTime = 0f;
        }

		public void SetInput(int index, float value) {
            if(Setup) {
		        X.SetValue(index, 0, value);
            }
		}

        public float GetInput(int index) {
            if(Setup) {
                return X.GetValue(index, 0);
            } else {
                return 0f;
            }
        }

        public void SetOutput(int index, float value) {
            if(Setup) {
                Y.SetValue(index, 0, value);
            }
        }

		public float GetOutput(int index) {
            if(Setup) {
			    return Y.GetValue(index, 0);
            } else {
                return 0f;
            }
		}

		public void Feed(float value) {
            if(Setup) {
                Pivot += 1;
			    SetInput(Pivot, value);
            }
		}

        public void Feed(float[] values) {
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

		public float Read() {
            Pivot += 1;
			return GetOutput(Pivot);
		}

    	public float Read(float min, float max) {
            Pivot += 1;
			return Mathf.Clamp(GetOutput(Pivot), min, max);
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

    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(NeuralNetwork), true)]
	public class NeuralNetwork_Editor : Editor {

		public NeuralNetwork Target;

		void Awake() {
			Target = (NeuralNetwork)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

            DrawDefaultInspector();

            EditorGUILayout.HelpBox("Prediction: " + 1000f*Target.PredictionTime + "ms", MessageType.None);

            if(Application.isPlaying) {
                Utility.SetGUIColor(Target.Setup ? UltiDraw.DarkGreen : UltiDraw.Grey);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    if(Target.enabled && !Target.Setup) {
                        if(Utility.GUIButton("Setup", UltiDraw.Mustard, UltiDraw.Black)) {
                            Target.enabled = false;
                            Target.enabled = true;
                        }
                    } else {
                        EditorGUILayout.LabelField(Target.Setup ? "Active" : "Inactive");
                    }
                }
            }

            if(Target.Matrices.Count > 0) {
                Utility.SetGUIColor(UltiDraw.Grey);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    if(Utility.GUIButton("Inspect", UltiDraw.DarkGrey, UltiDraw.White)) {
                        Target.Inspect = !Target.Inspect;
                    }
                    if(Target.Inspect) {
                        EditorGUI.BeginDisabledGroup(true);
                        for(int i=0; i<Target.Matrices.Count; i++) {
                            EditorGUILayout.IntField(Target.Matrices[i].ID, Target.Matrices[i].GetRows() * Target.Matrices[i].GetCols());
                        }
                        EditorGUI.EndDisabledGroup();
                    }
                }
            }

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
        
	}
	#endif

}
