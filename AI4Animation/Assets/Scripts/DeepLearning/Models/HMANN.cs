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

		public override void StoreParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();

			Parameters.Store(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1);

			Parameters.Store(Folder+"/wc0_w0.bin", HDimBlend, XDimBlend);
			Parameters.Store(Folder+"/wc0_b0.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc1_w0.bin", HDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc1_b0.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc2_w0.bin", YDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc2_b0.bin", YDimBlend, 1);

			Parameters.Store(Folder+"/wc0_w1.bin", HDimBlend, XDimBlend);
			Parameters.Store(Folder+"/wc0_b1.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc1_w1.bin", HDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc1_b1.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc2_w1.bin", YDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc2_b1.bin", YDimBlend, 1);

			Parameters.Store(Folder+"/wc0_w2.bin", HDimBlend, XDimBlend);
			Parameters.Store(Folder+"/wc0_b2.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc1_w2.bin", HDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc1_b2.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc2_w2.bin", YDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc2_b2.bin", YDimBlend, 1);

			for(int i=0; i<YDimBlend; i++) {
				Parameters.Store(Folder+"/cp0_a"+i.ToString("D1")+".bin", HDim, XDim);
				Parameters.Store(Folder+"/cp0_b"+i.ToString("D1")+".bin", HDim, 1);
				Parameters.Store(Folder+"/cp1_a"+i.ToString("D1")+".bin", HDim, HDim);
				Parameters.Store(Folder+"/cp1_b"+i.ToString("D1")+".bin", HDim, 1);
				Parameters.Store(Folder+"/cp2_a"+i.ToString("D1")+".bin", YDim, HDim);
				Parameters.Store(Folder+"/cp2_b"+i.ToString("D1")+".bin", YDim, 1);
			}
		}

		public override void LoadParameters() {
			if(Parameters == null) {
				Debug.Log("Building HMANN failed because no parameters are available.");
				return;
			}
			Xmean = CreateTensor(Parameters.Load(0), "Xmean");
			Xstd = CreateTensor(Parameters.Load(1), "Xstd");
			Ymean = CreateTensor(Parameters.Load(2), "Ymean");
			Ystd = CreateTensor(Parameters.Load(3), "Ystd");

			BW0_1 = CreateTensor(Parameters.Load(4), "BW0_1");
			Bb0_1 = CreateTensor(Parameters.Load(5), "Bb0_1");
			BW1_1 = CreateTensor(Parameters.Load(6), "BW1_1");
			Bb1_1 = CreateTensor(Parameters.Load(7), "Bb1_1");
			BW2_1 = CreateTensor(Parameters.Load(8), "BW2_1");
			Bb2_1 = CreateTensor(Parameters.Load(9), "Bb2_1");

			BW0_2 = CreateTensor(Parameters.Load(10), "BW0_2");
			Bb0_2 = CreateTensor(Parameters.Load(11), "Bb0_2");
			BW1_2 = CreateTensor(Parameters.Load(12), "BW1_2");
			Bb1_2 = CreateTensor(Parameters.Load(13), "Bb1_2");
			BW2_2 = CreateTensor(Parameters.Load(14), "BW2_2");
			Bb2_2 = CreateTensor(Parameters.Load(15), "Bb2_2");

			BW0_3 = CreateTensor(Parameters.Load(16), "BW0_3");
			Bb0_3 = CreateTensor(Parameters.Load(17), "Bb0_3");
			BW1_3 = CreateTensor(Parameters.Load(18), "BW1_3");
			Bb1_3 = CreateTensor(Parameters.Load(19), "Bb1_3");
			BW2_3 = CreateTensor(Parameters.Load(20), "BW2_3");
			Bb2_3 = CreateTensor(Parameters.Load(21), "Bb2_3");

			CW = new Tensor[YDimBlend*6];
			for(int i=0; i<YDimBlend*6; i++) {
				CW[i] = CreateTensor(Parameters.Load(22+i), "CW"+i);
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