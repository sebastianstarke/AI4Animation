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
		private Tensor X, Y;
		private Tensor BX, BY;
		private Tensor BW0, BW1, BW2, Bb0, Bb1, Bb2;
		private Tensor[] CW;
		private Tensor W0, W1, W2, b0, b1, b2;

		public override void StoreParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();

			Parameters.Store(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1);

			Parameters.Store(Folder+"/wc0_w.bin", HDimBlend, XDimBlend);
			Parameters.Store(Folder+"/wc0_b.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc1_w.bin", HDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc1_b.bin", HDimBlend, 1);
			Parameters.Store(Folder+"/wc2_w.bin", YDimBlend, HDimBlend);
			Parameters.Store(Folder+"/wc2_b.bin", YDimBlend, 1);

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
				Debug.Log("Building MANN failed because no parameters are available.");
				return;
			}
			Xmean = CreateTensor(Parameters.Load(0), "Xmean");
			Xstd = CreateTensor(Parameters.Load(1), "Xstd");
			Ymean = CreateTensor(Parameters.Load(2), "Ymean");
			Ystd = CreateTensor(Parameters.Load(3), "Ystd");

			BW0 = CreateTensor(Parameters.Load(4), "BW0");
			Bb0 = CreateTensor(Parameters.Load(5), "Bb0");
			BW1 = CreateTensor(Parameters.Load(6), "BW1");
			Bb1 = CreateTensor(Parameters.Load(7), "Bb1");
			BW2 = CreateTensor(Parameters.Load(8), "BW2");
			Bb2 = CreateTensor(Parameters.Load(9), "Bb2");

			CW = new Tensor[YDimBlend*6];
			for(int i=0; i<YDimBlend*6; i++) {
				CW[i] = CreateTensor(Parameters.Load(10+i), "CW"+i);
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

		public override void SetInput(int index, float value) {
			X.SetValue(index, 0, value);
		}

		public override float GetOutput(int index) {
			return Y.GetValue(index, 0);
		}
		
	}

}