using UnityEngine;
using System.IO;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public abstract class NeuralNetwork : MonoBehaviour {

		public string Folder = "";
        public Parameters Parameters = null;

        public Tensor X, Y;
        private int Pivot = -1;

        private List<Tensor> Tensors = new List<Tensor>();

        protected abstract void StoreParametersDerived();

        protected abstract void LoadParametersDerived();
        
        public abstract void Predict();

        public void StoreParameters() {
            Parameters = ScriptableObject.CreateInstance<Parameters>();
            StoreParametersDerived();
			if(!Parameters.Validate()) {
				Parameters = null;
			} else {
                #if UNITY_EDITOR
                string[] files = Directory.GetFiles(Folder);
                string directory = new FileInfo(files[0]).Directory.Name;
                Parameters asset = ScriptableObject.CreateInstance<Parameters>();
                string path = AssetDatabase.GenerateUniqueAssetPath("Assets/" + directory + ".asset");
				AssetDatabase.CreateAsset(Parameters, path);
                #endif
			}
        }
        public void LoadParameters() {
			if(Parameters == null) {
				Debug.Log("Building PFNN failed because no parameters were saved.");
			} else {
                LoadParametersDerived();
            }
        }

        public Tensor CreateTensor(int rows, int cols, string id) {
            if(Tensors.Exists(x => x.ID == id)) {
                Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(rows, cols, id);
            Tensors.Add(T);
            return T;
        }

        public Tensor CreateTensor(Parameters.Matrix matrix) {
            if(Tensors.Exists(x => x.ID == matrix.ID)) {
                Debug.Log("Tensor with ID " + matrix.ID + " already contained.");
                return null;
            }
            Tensor T = new Tensor(matrix.Rows, matrix.Cols, matrix.ID);
            for(int x=0; x<matrix.Rows; x++) {
                for(int y=0; y<matrix.Cols; y++) {
                    T.SetValue(x, y, matrix.Values[x].Values[y]);
                }
            }
            Tensors.Add(T);
            return T;
        }

        public void DeleteTensor(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                Debug.Log("Tensor not found.");
                return;
            }
            Tensors.RemoveAt(index);
            T.Delete();
        }

        public Tensor GetTensor(string id) {
            int index = Tensors.FindIndex(x => x.ID == id);
            if(index == -1) {
                return null;
            }
            return Tensors[index];
        }

        public string GetID(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                return null;
            }
            return Tensors[index].ID;
        }

        public void ResetPivot() {
            Pivot = -1;
        }

		public void SetInput(int index, float value) {
			Pivot = index;
			X.SetValue(index, 0, value);
		}

        public float GetInput(int index) {
            Pivot = index;
            return X.GetValue(index, 0);
        }

        public void SetOutput(int index, float value) {
            Pivot = index;
            Y.SetValue(index, 0, value);
        }

		public float GetOutput(int index) {
			Pivot = index;
			return Y.GetValue(index, 0);
		}

		public void Feed(float value) {
			SetInput(Pivot+1, value);
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

		public float Read() {
			return GetOutput(Pivot+1);
		}

        public float[] Read(int count) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read();
            }
            return values;
        }

        public Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            if(IN.GetRows() != mean.GetRows() || IN.GetRows() != std.GetRows() || IN.GetCols() != mean.GetCols() || IN.GetCols() != std.GetCols()) {
                Debug.Log("Incompatible dimensions for normalisation.");
                return IN;
            } else {
                Eigen.Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
                return OUT;
            }
        }
        
        public Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            if(IN.GetRows() != mean.GetRows() || IN.GetRows() != std.GetRows() || IN.GetCols() != mean.GetCols() || IN.GetCols() != std.GetCols()) {
                Debug.Log("Incompatible dimensions for renormalisation.");
                return IN;
            } else {
                Eigen.Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
                return OUT;
            }
        }

        public Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
            if(IN.GetRows() != W.GetCols() || W.GetRows() != b.GetRows() || IN.GetCols() != b.GetCols()) {
                Debug.Log("Incompatible dimensions for feed-forward.");
                return IN;
            } else {
                Eigen.Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
                return OUT;
            }
        }

        public Tensor Blend(Tensor T, Tensor W, float w) {
            if(T.GetRows() != W.GetRows() || T.GetCols() != W.GetCols()) {
                Debug.Log("Incompatible dimensions for blending.");
                return T;
            } else {
                Eigen.Blend(T.Ptr, W.Ptr, w);
                return T;
            }
        }

        public Tensor ELU(Tensor T) {
            Eigen.ELU(T.Ptr);
            return T;
        }

        public Tensor Sigmoid(Tensor T) {
            Eigen.Sigmoid(T.Ptr);
            return T;
        }

        public Tensor TanH(Tensor T) {
            Eigen.TanH(T.Ptr);
            return T;
        }

        public Tensor SoftMax(Tensor T) {
            Eigen.SoftMax(T.Ptr);
            return T;
        }

        public Tensor LogSoftMax(Tensor T) {
            Eigen.LogSoftMax(T.Ptr);
            return T;
        }

        public Tensor SoftSign(Tensor T) {
            Eigen.SoftSign(T.Ptr);
            return T;
        }

        public Tensor Exp(Tensor T) {
            Eigen.Exp(T.Ptr);
            return T;
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
            if(Utility.GUIButton("Store Parameters", UltiDraw.DarkGrey, UltiDraw.White)) {
                Target.StoreParameters();
            }

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
        
	}
	#endif

}
