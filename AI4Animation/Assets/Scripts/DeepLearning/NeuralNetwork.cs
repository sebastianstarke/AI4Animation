using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public abstract class NeuralNetwork : MonoBehaviour {

		public string Folder = "Assets/";
        public Parameters Parameters = null;

        private List<Tensor> Tensors = new List<Tensor>();

        public void StoreParameters() {
            Parameters = ScriptableObject.CreateInstance<Parameters>();
            StoreParametersDerived();
			if(!Parameters.Validate()) {
				Parameters = null;
			} else {
                #if UNITY_EDITOR
				AssetDatabase.CreateAsset(Parameters, Folder + "/Parameters.asset");
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
        protected abstract void StoreParametersDerived();
        protected abstract void LoadParametersDerived();
        public abstract void Predict();
        public abstract void SetInput(int index, float value);
        public abstract float GetOutput(int index);

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

        public Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }
        
        public Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
            Eigen.Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Blend(Tensor T, Tensor W, float w) {
            Eigen.Blend(T.Ptr, W.Ptr, w);
            return T;
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
