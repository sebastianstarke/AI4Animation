using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class Vanilla : NeuralNetwork {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor W0, W1, W2, b0, b1, b2;

		protected override void StoreParametersDerived() {
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1, "Xmean");
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1, "Xstd");
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1, "Ymean");
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1, "Ystd");
			Parameters.Store(Folder+"/W0_000.bin", HDim, XDim, "W0");
			Parameters.Store(Folder+"/W1_000.bin", HDim, HDim, "W1");
			Parameters.Store(Folder+"/W2_000.bin", YDim, HDim, "W2");
			Parameters.Store(Folder+"/b0_000.bin", HDim, 1, "b0");
			Parameters.Store(Folder+"/b1_000.bin", HDim, 1, "b1");
			Parameters.Store(Folder+"/b2_000.bin", YDim, 1, "b2");
		}

		protected override void LoadParametersDerived() {
			Xmean = CreateTensor(Parameters.Load("Xmean"));
			Xstd = CreateTensor(Parameters.Load("Xstd"));
			Ymean = CreateTensor(Parameters.Load("Ymean"));
			Ystd = CreateTensor(Parameters.Load("Ystd"));
			W0 = CreateTensor(Parameters.Load("W0"));
			W1 = CreateTensor(Parameters.Load("W1"));
			W2 = CreateTensor(Parameters.Load("W2"));
			b0 = CreateTensor(Parameters.Load("b0"));
			b1 = CreateTensor(Parameters.Load("b1"));
			b2 = CreateTensor(Parameters.Load("b2"));
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

	}

}