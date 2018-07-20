using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class SPFNN : NeuralNetwork {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		public int SDim = 0;
		public int[] StyleNeurons = new int[0];

		public int PhaseIndex = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor[] W0, W1, W2, b0, b1, b2;
		private Tensor[] SW, Sb;

		private Tensor WS, bS;

		private float Phase;
		private float Damping;

		private const float M_PI = 3.14159265358979323846f;

		public SPFNN() {
			
		}

		protected override void StoreParametersDerived() {
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1, "Xmean");
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1, "Xstd");
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1, "Ymean");
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1, "Ystd");
			for(int i=0; i<50; i++) {
				Parameters.Store(Folder+"/W0_"+i.ToString("D3")+".bin", HDim, XDim, "W0_"+i.ToString("D3"));
				Parameters.Store(Folder+"/W1_"+i.ToString("D3")+".bin", HDim, HDim, "W1_"+i.ToString("D3"));
				Parameters.Store(Folder+"/W2_"+i.ToString("D3")+".bin", YDim, HDim, "W2_"+i.ToString("D3"));
				Parameters.Store(Folder+"/b0_"+i.ToString("D3")+".bin", HDim, 1, "b0_"+i.ToString("D3"));
				Parameters.Store(Folder+"/b1_"+i.ToString("D3")+".bin", HDim, 1, "b1_"+i.ToString("D3"));
				Parameters.Store(Folder+"/b2_"+i.ToString("D3")+".bin", YDim, 1, "b2_"+i.ToString("D3"));
			}
			for(int i=0; i<SDim; i++) {
				Parameters.Store(Folder+"/cp3_a"+i.ToString()+".bin", YDim, YDim, "cp3_a"+i.ToString());
				Parameters.Store(Folder+"/cp3_b"+i.ToString()+".bin", YDim, 1, "cp3_b"+i.ToString());
			}
		}

		protected override void LoadParametersDerived() {
			Xmean = CreateTensor(Parameters.Load("Xmean"));
			Xstd = CreateTensor(Parameters.Load("Xstd"));
			Ymean = CreateTensor(Parameters.Load("Ymean"));
			Ystd = CreateTensor(Parameters.Load("Ystd"));
			W0 = new Tensor[50];
			W1 = new Tensor[50];
			W2 = new Tensor[50];
			b0 = new Tensor[50];
			b1 = new Tensor[50];
			b2 = new Tensor[50];
			for(int i=0; i<50; i++) {
				W0[i] = CreateTensor(Parameters.Load("W0"+i.ToString("D3")));
				W1[i] = CreateTensor(Parameters.Load("W1"+i.ToString("D3")));
				W2[i] = CreateTensor(Parameters.Load("W2"+i.ToString("D3")));
				b0[i] = CreateTensor(Parameters.Load("b0"+i.ToString("D3")));
				b1[i] = CreateTensor(Parameters.Load("b1"+i.ToString("D3")));
				b2[i] = CreateTensor(Parameters.Load("b2"+i.ToString("D3")));
			}
			SW = new Tensor[SDim];
			Sb = new Tensor[SDim];
			for(int i=0; i<SDim; i++) {
				SW[i] = CreateTensor(Parameters.Load("cp3_a"+i.ToString()));
				Sb[i] = CreateTensor(Parameters.Load("cp3_b"+i.ToString()));
			}
			WS = CreateTensor(YDim, YDim, "WS");
			bS = CreateTensor(YDim, 1, "bS");
			X = CreateTensor(XDim, 1, "X");
			Y = CreateTensor(YDim, 1, "Y");
			Phase = 0f;
			Damping = 0f;
		}
		
		public override void Predict() {
			float[] S = new float[SDim];
			for(int i=0; i<SDim; i++) {
				S[i] = X.GetValue(StyleNeurons[i], 0);
			}

			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process PFNN-Network
			int index = (int)((Phase / (2f*M_PI)) * 50f);
			ELU(Layer(Y, W0[index], b0[index], Y));
			ELU(Layer(Y, W1[index], b1[index], Y));
			ELU(Layer(Y, W2[index], b2[index], Y));

			//Process S-Layer
			WS.SetZero();
			bS.SetZero();
			for(int i=0; i<SDim; i++) {
				Blend(WS, SW[i], S[i]);
				Blend(bS, Sb[i], S[i]);
			}
			Layer(Y, WS, bS, Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);

			//Update Phase
			Phase = Mathf.Repeat(Phase + (1f-Damping)*GetOutput(PhaseIndex)*2f*Mathf.PI, 2f*Mathf.PI);
		}

		/*
		private Matrix Linear(ref Matrix y0, ref Matrix y1, float mu) {
			return (1.0f-mu) * y0 + (mu) * y1;
		}

		private Matrix Cubic(ref Matrix y0, ref Matrix y1, ref Matrix y2, ref Matrix y3, float mu) {
			return
			(-0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3)*mu*mu*mu + 
			(y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3)*mu*mu + 
			(-0.5f*y0 + 0.5f*y2)*mu + 
			(y1);
		}
		*/

		public void SetDamping(float value) {
			Damping = value;
		}

		public float GetPhase() {
			return Phase;
		}

	}

}