using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class MANN : NeuralNetwork {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		public int XDimBlend = 0;
		public int HDimBlend = 0;
		public int YDimBlend = 0;
		public int[] ControlNeurons = new int[0];

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor BX, BY;
		private Tensor BW0, BW1, BW2, Bb0, Bb1, Bb2;
		private Tensor[] CW;
		private Tensor W0, W1, W2, b0, b1, b2;

		void OnValidate() {
			ArrayExtensions.Resize(ref ControlNeurons, XDimBlend);
		}

		protected override void StoreParametersDerived() {
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1, "Xmean");
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1, "Xstd");
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1, "Ymean");
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1, "Ystd");
			Parameters.Store(Folder+"/wc0_w.bin", HDimBlend, XDimBlend, "wc0_w");
			Parameters.Store(Folder+"/wc0_b.bin", HDimBlend, 1, "wc0_b");
			Parameters.Store(Folder+"/wc1_w.bin", HDimBlend, HDimBlend, "wc1_w");
			Parameters.Store(Folder+"/wc1_b.bin", HDimBlend, 1, "wc1_b");
			Parameters.Store(Folder+"/wc2_w.bin", YDimBlend, HDimBlend, "wc2_w");
			Parameters.Store(Folder+"/wc2_b.bin", YDimBlend, 1, "wc2_b");
			for(int i=0; i<YDimBlend; i++) {
				Parameters.Store(Folder+"/cp0_a"+i.ToString("D1")+".bin", HDim, XDim, "cp0_a"+i.ToString("D1"));
				Parameters.Store(Folder+"/cp0_b"+i.ToString("D1")+".bin", HDim, 1, "cp0_b"+i.ToString("D1"));
				Parameters.Store(Folder+"/cp1_a"+i.ToString("D1")+".bin", HDim, HDim, "cp1_a"+i.ToString("D1"));
				Parameters.Store(Folder+"/cp1_b"+i.ToString("D1")+".bin", HDim, 1, "cp1_b"+i.ToString("D1"));
				Parameters.Store(Folder+"/cp2_a"+i.ToString("D1")+".bin", YDim, HDim, "cp2_a"+i.ToString("D1"));
				Parameters.Store(Folder+"/cp2_b"+i.ToString("D1")+".bin", YDim, 1, "cp2_b"+i.ToString("D1"));
			}
		}

		protected override void LoadParametersDerived() {
			Xmean = CreateTensor(Parameters.Load("Xmean"));
			Xstd = CreateTensor(Parameters.Load("Xstd"));
			Ymean = CreateTensor(Parameters.Load("Ymean"));
			Ystd = CreateTensor(Parameters.Load("Ystd"));
			BW0 = CreateTensor(Parameters.Load("wc0_w"));
			Bb0 = CreateTensor(Parameters.Load("wc0_b"));
			BW1 = CreateTensor(Parameters.Load("wc1_w"));
			Bb1 = CreateTensor(Parameters.Load("wc1_b"));
			BW2 = CreateTensor(Parameters.Load("wc2_w"));
			Bb2 = CreateTensor(Parameters.Load("wc2_b"));
			CW = new Tensor[6*YDimBlend];
			for(int i=0; i<YDimBlend; i++) {
				CW[6*i+0] = CreateTensor(Parameters.Load("cp0_a"+i.ToString("D1")));
				CW[6*i+1] = CreateTensor(Parameters.Load("cp0_b"+i.ToString("D1")));
				CW[6*i+2] = CreateTensor(Parameters.Load("cp1_a"+i.ToString("D1")));
				CW[6*i+3] = CreateTensor(Parameters.Load("cp1_b"+i.ToString("D1")));
				CW[6*i+4] = CreateTensor(Parameters.Load("cp2_a"+i.ToString("D1")));
				CW[6*i+5] = CreateTensor(Parameters.Load("cp2_b"+i.ToString("D1")));
			}
			X = CreateTensor(XDim, 1, "X");
			Y = CreateTensor(YDim, 1, "Y");
			BX = CreateTensor(XDimBlend, 1, "BX");
			BY = CreateTensor(YDimBlend, 1, "BY");
			W0 = CreateTensor(HDim, XDim, "W0");
			W1 = CreateTensor(HDim, HDim, "W1");
			W2 = CreateTensor(YDim, HDim, "W2");
			b0 = CreateTensor(HDim, 1, "b0");
			b1 = CreateTensor(HDim, 1, "b1");
			b2 = CreateTensor(YDim, 1, "b2");
		}

		public override void Predict() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process Gating Network
			for(int i=0; i<ControlNeurons.Length; i++) {
				BX.SetValue(i, 0, Y.GetValue(ControlNeurons[i], 0));
			}
			ELU(Layer(BX, BW0, Bb0, BY));
			ELU(Layer(BY, BW1, Bb1, BY));
			SoftMax(Layer(BY, BW2, Bb2, BY));

			//Generate Network Weights
			W0.SetZero(); b0.SetZero();
			W1.SetZero(); b1.SetZero();
			W2.SetZero(); b2.SetZero();
			for(int i=0; i<YDimBlend; i++) {
				float weight = BY.GetValue(i, 0);
				Blend(W0, CW[6*i + 0], weight);
				Blend(b0, CW[6*i + 1], weight);
				Blend(W1, CW[6*i + 2], weight);
				Blend(b1, CW[6*i + 3], weight);
				Blend(W2, CW[6*i + 4], weight);
				Blend(b2, CW[6*i + 5], weight);
			}

			//Process Motion-Prediction Network
			ELU(Layer(Y, W0, b0, Y));
			ELU(Layer(Y, W1, b1, Y));
			Layer(Y, W2, b2, Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}
		
	}

}