using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	[System.Serializable]
	public class MLP : Model {

		public bool Inspect = false;

		public string Folder = string.Empty;
		public int XDim = 0;
		public int HDim = 512;
		public int YDim = 0;
		public Parameters Parameters;

		private Tensor Xmean, Xstd, Ymean, Ystd;

		private Tensor W0, W1, W2, b0, b1, b2;

		private Tensor X, Y;

		public void LoadParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();
			Parameters.Save(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Save(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Save(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Save(Folder+"/Ystd.bin", YDim, 1);
			Parameters.Save(Folder+"/W0_000.bin", HDim, XDim);
			Parameters.Save(Folder+"/W1_000.bin", HDim, HDim);
			Parameters.Save(Folder+"/W2_000.bin", YDim, HDim);
			Parameters.Save(Folder+"/b0_000.bin", HDim, 1);
			Parameters.Save(Folder+"/b1_000.bin", HDim, 1);
			Parameters.Save(Folder+"/b2_000.bin", YDim, 1);
		}

		public void Initialise() {
			if(Parameters == null) {
				Debug.Log("Building MLP failed because no parameters were loaded.");
				return;
			}

			Xmean = Parameters.Load(0).MakeTensor();
			Xstd = Parameters.Load(1).MakeTensor();
			Ymean = Parameters.Load(2).MakeTensor();
			Ystd = Parameters.Load(3).MakeTensor();

			W0 = Parameters.Load(4).MakeTensor();
			W1 = Parameters.Load(5).MakeTensor();
			W2 = Parameters.Load(6).MakeTensor();
			b0 = Parameters.Load(7).MakeTensor();
			b1 = Parameters.Load(8).MakeTensor();
			b2 = Parameters.Load(9).MakeTensor();

			X = new Tensor(XDim, 1);
			Y = new Tensor(YDim, 1);
		}

		public void SetInput(int i, float value) {
			X.SetValue(i, 0, value);
		}

		public float GetOutput(int i) {
			return Y.GetValue(i, 0);
		}

		public void Predict() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process Vanilla Network
			ELU(Layer(Y, W0, b0, Y));
			ELU(Layer(Y, W1, b1, Y));
			Layer(Y, W2, b2, Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}

		#if UNITY_EDITOR
		public void Inspector() {
			Utility.SetGUIColor(Color.grey);
			using(new GUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				if(Utility.GUIButton("MLP", UltiDraw.DarkGrey, UltiDraw.White)) {
					Inspect = !Inspect;
				}

				if(Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Folder = EditorGUILayout.TextField("Folder", Folder);
						XDim = EditorGUILayout.IntField("XDim", XDim);
						HDim = EditorGUILayout.IntField("HDim", HDim);
						YDim = EditorGUILayout.IntField("YDim", YDim);
						EditorGUILayout.BeginHorizontal();
						if(GUILayout.Button("Load Parameters")) {
							LoadParameters();
						}
						Parameters = (Parameters)EditorGUILayout.ObjectField(Parameters, typeof(Parameters), true);
						EditorGUILayout.EndHorizontal();
					}
				}
			}
		}
		#endif

	}

}