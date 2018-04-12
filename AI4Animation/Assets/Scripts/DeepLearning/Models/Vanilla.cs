using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class Vanilla : Model {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor W0, W1, W2, b0, b1, b2;
		private Tensor X, Y;

		public override void StoreParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1);
			Parameters.Store(Folder+"/W0_000.bin", HDim, XDim);
			Parameters.Store(Folder+"/W1_000.bin", HDim, HDim);
			Parameters.Store(Folder+"/W2_000.bin", YDim, HDim);
			Parameters.Store(Folder+"/b0_000.bin", HDim, 1);
			Parameters.Store(Folder+"/b1_000.bin", HDim, 1);
			Parameters.Store(Folder+"/b2_000.bin", YDim, 1);
		}

		public override void LoadParameters() {
			if(Parameters == null) {
				Debug.Log("Building MLP failed because no parameters are available.");
				return;
			}

			Xmean = CreateTensor(Parameters.Load(0), "Xmean");
			Xstd = CreateTensor(Parameters.Load(1), "Xstd");
			Ymean = CreateTensor(Parameters.Load(2), "Ymean");
			Ystd = CreateTensor(Parameters.Load(3), "Ystd");

			W0 = CreateTensor(Parameters.Load(4), "W0");
			W1 = CreateTensor(Parameters.Load(5), "W1");
			W2 = CreateTensor(Parameters.Load(6), "W2");
			b0 = CreateTensor(Parameters.Load(7), "b0");
			b1 = CreateTensor(Parameters.Load(8), "b1");
			b2 = CreateTensor(Parameters.Load(9), "b2");

			X = CreateTensor(XDim, 1, "X");
			Y = CreateTensor(YDim, 1, "Y");
		}

		public override void Predict() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process Vanilla Network
			ELU(Layer(Y, W0, b0, Y));
			ELU(Layer(Y, W1, b1, Y));
			Layer(Y, W2, b2, Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}

		public override void SetInput(int index, float value) {
			X.SetValue(index, 0, value);
		}

		public override float GetOutput(int index) {
			return Y.GetValue(index, 0);
		}

		#if UNITY_EDITOR
		public override void Inspector() {
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Folder = EditorGUILayout.TextField("Folder", Folder);
				EditorGUILayout.BeginHorizontal();
				if(GUILayout.Button("Store Parameters")) {
					StoreParameters();
				}
				Parameters = (Parameters)EditorGUILayout.ObjectField(Parameters, typeof(Parameters), true);
				EditorGUILayout.EndHorizontal();

				XDim = EditorGUILayout.IntField("XDim", XDim);
				HDim = EditorGUILayout.IntField("HDim", HDim);
				YDim = EditorGUILayout.IntField("YDim", YDim);
			}
		}
		#endif

	}

}