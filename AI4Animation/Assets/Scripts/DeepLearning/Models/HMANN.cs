using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class HMANN : NeuralNetwork {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		public int XDimBlend = 0;
		public int HDimBlend = 0;
		public int YDimBlend = 0;
		public int[] ControlNeurons = new int[0];

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor X, Y;
		private Tensor BX, BY;
		private Tensor BW0_1, BW1_1, BW2_1, Bb0_1, Bb1_1, Bb2_1;
		private Tensor BW0_2, BW1_2, BW2_2, Bb0_2, Bb1_2, Bb2_2;
		private Tensor BW0_3, BW1_3, BW2_3, Bb0_3, Bb1_3, Bb2_3;
		private Tensor[] CW;
		private Tensor W0, W1, W2, b0, b1, b2;

		protected override void StoreParametersDerived() {
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1, "Xmean");
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1, "Xstd");
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1, "Ymean");
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1, "Ystd");
			Parameters.Store(Folder+"/wc0_w0.bin", HDimBlend, XDimBlend, "wc0_w0");
			Parameters.Store(Folder+"/wc0_b0.bin", HDimBlend, 1, "wc0_b0");
			Parameters.Store(Folder+"/wc1_w0.bin", HDimBlend, HDimBlend, "wc1_w0");
			Parameters.Store(Folder+"/wc1_b0.bin", HDimBlend, 1, "wc1_b0");
			Parameters.Store(Folder+"/wc2_w0.bin", YDimBlend, HDimBlend, "wc2_w0");
			Parameters.Store(Folder+"/wc2_b0.bin", YDimBlend, 1, "wc2_b0");
			Parameters.Store(Folder+"/wc0_w1.bin", HDimBlend, XDimBlend, "wc0_w1");
			Parameters.Store(Folder+"/wc0_b1.bin", HDimBlend, 1, "wc0_b1");
			Parameters.Store(Folder+"/wc1_w1.bin", HDimBlend, HDimBlend, "wc1_w1");
			Parameters.Store(Folder+"/wc1_b1.bin", HDimBlend, 1, "wc1_b1");
			Parameters.Store(Folder+"/wc2_w1.bin", YDimBlend, HDimBlend, "wc2_w1");
			Parameters.Store(Folder+"/wc2_b1.bin", YDimBlend, 1, "wc2_b1");
			Parameters.Store(Folder+"/wc0_w2.bin", HDimBlend, XDimBlend, "wc0_w2");
			Parameters.Store(Folder+"/wc0_b2.bin", HDimBlend, 1, "wc0_b2");
			Parameters.Store(Folder+"/wc1_w2.bin", HDimBlend, HDimBlend, "wc1_w2");
			Parameters.Store(Folder+"/wc1_b2.bin", HDimBlend, 1, "wc1_b2");
			Parameters.Store(Folder+"/wc2_w2.bin", YDimBlend, HDimBlend, "wc2_w2");
			Parameters.Store(Folder+"/wc2_b2.bin", YDimBlend, 1, "wc2_b2");
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
			BW0_1 = CreateTensor(Parameters.Load("wc0_w0"));
			Bb0_1 = CreateTensor(Parameters.Load("wc0_b0"));
			BW1_1 = CreateTensor(Parameters.Load("wc1_w0"));
			Bb1_1 = CreateTensor(Parameters.Load("wc1_b0"));
			BW2_1 = CreateTensor(Parameters.Load("wc2_w0"));
			Bb2_1 = CreateTensor(Parameters.Load("wc2_b0"));
			BW0_2 = CreateTensor(Parameters.Load("wc0_w1"));
			Bb0_2 = CreateTensor(Parameters.Load("wc0_b1"));
			BW1_2 = CreateTensor(Parameters.Load("wc1_w1"));
			Bb1_2 = CreateTensor(Parameters.Load("wc1_b1"));
			BW2_2 = CreateTensor(Parameters.Load("wc2_w1"));
			Bb2_2 = CreateTensor(Parameters.Load("wc2_b1"));
			BW0_3 = CreateTensor(Parameters.Load("wc0_w2"));
			Bb0_3 = CreateTensor(Parameters.Load("wc0_b2"));
			BW1_3 = CreateTensor(Parameters.Load("wc1_w2"));
			Bb1_3 = CreateTensor(Parameters.Load("wc1_b2"));
			BW2_3 = CreateTensor(Parameters.Load("wc2_w2"));
			Bb2_3 = CreateTensor(Parameters.Load("wc2_b2"));
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

			//Process Gating Network 1
			for(int i=0; i<ControlNeurons.Length; i++) {
				BX.SetValue(i, 0, Y.GetValue(ControlNeurons[i], 0));
			}
			ELU(Layer(BX, BW0_1, Bb0_1, BY));
			ELU(Layer(BY, BW1_1, Bb1_1, BY));
			SoftMax(Layer(BY, BW2_1, Bb2_1, BY));
			W0.SetZero(); b0.SetZero();
			for(int i=0; i<YDimBlend; i++) {
				float weight = BY.GetValue(i, 0);
				Blend(W0, CW[6*i + 0], weight);
				Blend(b0, CW[6*i + 1], weight);
			}

			//Process Gating Network 2
			for(int i=0; i<ControlNeurons.Length; i++) {
				BX.SetValue(i, 0, Y.GetValue(ControlNeurons[i], 0));
			}
			ELU(Layer(BX, BW0_2, Bb0_2, BY));
			ELU(Layer(BY, BW1_2, Bb1_2, BY));
			SoftMax(Layer(BY, BW2_2, Bb2_2, BY));
			W1.SetZero(); b1.SetZero();
			for(int i=0; i<YDimBlend; i++) {
				float weight = BY.GetValue(i, 0);
				Blend(W1, CW[6*i + 2], weight);
				Blend(b1, CW[6*i + 3], weight);
			}

			//Process Gating Network 3
			for(int i=0; i<ControlNeurons.Length; i++) {
				BX.SetValue(i, 0, Y.GetValue(ControlNeurons[i], 0));
			}
			ELU(Layer(BX, BW0_3, Bb0_3, BY));
			ELU(Layer(BY, BW1_3, Bb1_3, BY));
			SoftMax(Layer(BY, BW2_3, Bb2_3, BY));
			W2.SetZero(); b2.SetZero();
			for(int i=0; i<YDimBlend; i++) {
				float weight = BY.GetValue(i, 0);
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

		public override void SetInput(int index, float value) {
			X.SetValue(index, 0, value);
		}

		public override float GetOutput(int index) {
			return Y.GetValue(index, 0);
		}
		
	}

}