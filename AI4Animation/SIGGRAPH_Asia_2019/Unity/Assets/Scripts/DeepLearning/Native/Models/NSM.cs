using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class NSM : NativeNetwork {
		public int Features;

		public Component[] Components = new Component[0];
		public Encoder[] encoders = new Encoder[0];
		public bool Threading = true;

		private Matrix Xmean, Xstd, Ymean, Ystd;

		protected override void LoadDerived() {
			Xmean = CreateMatrix(Features, 1, "Xmean", Folder+"/Xmean.bin");
			Xstd = CreateMatrix(Features, 1, "Xstd", Folder+"/Xstd.bin");
			Ymean = CreateMatrix(Components[Components.Length-1].YDim, 1, "Ymean", Folder+"/Ymean.bin");
			Ystd = CreateMatrix(Components[Components.Length-1].YDim, 1, "Ystd", Folder+"/Ystd.bin");
			for(int i=0; i<Components.Length; i++) {
				Components[i].Load(this, i);
			}
			for(int i=0; i<encoders.Length; i++){
				encoders[i].Load(this, i);
			}

			X = CreateMatrix(Features, 1, "X");
			Y = GetMatrix("C"+(Components.Length-1)+"Y");
		}

		protected override void UnloadDerived() {
			
		}

		protected override void PredictDerived() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, X);

			if(Threading) {
				List<Task> tasks = new List<Task>();
				for(int i=0; i<encoders.Length; i++) {
					int index = i;
					tasks.Add(Task.Factory.StartNew(() => encoders[index].Process(this, index)));
				}
				Task.WaitAll(tasks.ToArray());
			} else {
				for(int i=0; i<encoders.Length; i++) {
					encoders[i].Process(this, i);
				}
			}

			for(int i=0; i<Components.Length; i++) {
				Components[i].Process(this, i);
			}
			
			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}

		[System.Serializable]
		public class Encoder {
			public int XDim, H1Dim, H2Dim;
			public int EncodePivot;

			private Matrix W0, W1, b0, b1;
			private Matrix X, Y;

			public void Load(NSM nn, int index) {
				if(H1Dim>0){
					W0 = nn.CreateMatrix(H1Dim, XDim, "encoder"+index+"0"+"_w", nn.Folder+"/encoder"+index+"_w"+"0"+".bin");
					b0 = nn.CreateMatrix(H1Dim, 1, "encoder"+index+"0"+"_b", nn.Folder+"/encoder"+index+"_b"+"0"+".bin");
					W1 = nn.CreateMatrix(H2Dim, H1Dim, "encoder"+index+"1"+"_w", nn.Folder+"/encoder"+index+"_w"+"1"+".bin");
					b1 = nn.CreateMatrix(H2Dim, 1, "encoder"+index+"1"+"_b", nn.Folder+"/encoder"+index+"_b"+"1"+".bin");
				}

				X = nn.CreateMatrix(XDim, 1, "encoder"+index+"X");
				Y = nn.CreateMatrix(H2Dim, 1, "encoder"+index+"Y");

			}

			public void Process(NSM nn, int index) {
				//Process Network
				if(H1Dim>0){
					for(int i=0; i<XDim; i++) {
						X.SetValue(i, 0, nn.GetInput(EncodePivot+i));
					}
					nn.Layer(X, W0, b0, Y).ELU();
					nn.Layer(Y, W1, b1, Y).ELU();
				}
				else if(XDim!=H2Dim){
					Debug.Log("XDim need to be equal to H2Dim");
				}
				else{
					for(int i=0; i<XDim; i++) {
						Y.SetValue(i, 0, nn.GetInput(EncodePivot+i));
					}
				}
			}
			
			public float GetLatent(int index) {
				return Y.GetValue(index, 0);
			}
		}

		[System.Serializable]
		public class Component {
			public int XDim, H1Dim, H2Dim, YDim;
			public int GatingPivot;
			public bool Bias = true;
			
			private List<Matrix[]> Experts;
			private Matrix W0, W1, W2, b0, b1, b2;
			private Matrix X, Y, X_copy;
			private Matrix Xmean_Main, Xstd_Main;

			public void Load(NSM nn, int index) {
				Experts = new List<Matrix[]>();
				for(int i=0; i<6; i++) {
					Experts.Add(new Matrix[GetExperts(nn, index)]);
				}
				for(int i=0; i<GetExperts(nn, index); i++) {
					Experts[0][i] = nn.CreateMatrix(H1Dim, XDim, "wc"+index+"0"+i.ToString("D1")+"_w", nn.Folder+"/wc"+index+"0"+i.ToString("D1")+"_w.bin");
					Experts[1][i] = nn.CreateMatrix(H1Dim, 1, "wc"+index+"0"+i.ToString("D1")+"_b", nn.Folder+"/wc"+index+"0"+i.ToString("D1")+"_b.bin");
					Experts[2][i] = nn.CreateMatrix(H2Dim, H1Dim, "wc"+index+"1"+i.ToString("D1")+"_w", nn.Folder+"/wc"+index+"1"+i.ToString("D1")+"_w.bin");
					Experts[3][i] = nn.CreateMatrix(H2Dim, 1, "wc"+index+"1"+i.ToString("D1")+"_b", nn.Folder+"/wc"+index+"1"+i.ToString("D1")+"_b.bin");
					Experts[4][i] = nn.CreateMatrix(YDim, H2Dim, "wc"+index+"2"+i.ToString("D1")+"_w", nn.Folder+"/wc"+index+"2"+i.ToString("D1")+"_w.bin");
					Experts[5][i] = nn.CreateMatrix(YDim, 1, "wc"+index+"2"+i.ToString("D1")+"_b", nn.Folder+"/wc"+index+"2"+i.ToString("D1")+"_b.bin");
				}
				W0 = nn.CreateMatrix(H1Dim, XDim, "C"+index+"W0");
				b0 = nn.CreateMatrix(H1Dim, 1, "C"+index+"b0");
				W1 = nn.CreateMatrix(H2Dim, H1Dim, "C"+index+"W1");
				b1 = nn.CreateMatrix(H2Dim, 1, "C"+index+"b1");
				W2 = nn.CreateMatrix(YDim, H2Dim, "C"+index+"W2");
				b2 = nn.CreateMatrix(YDim, 1, "C"+index+"b2");
				X = nn.CreateMatrix(XDim, 1, "C"+index+"X");
				Y = nn.CreateMatrix(YDim, 1, "C"+index+"Y");
			}

			public void Process(NSM nn, int index) {
				//Generate Weights
				if(index == 0) {
					W0 = Experts[0][0];
					b0 = Experts[1][0];
					W1 = Experts[2][0];
					b1 = Experts[3][0];
					W2 = Experts[4][0];
					b2 = Experts[5][0];
				} else {
					float[] weights = nn.Components[index-1].Y.Flatten();
					if(nn.Threading) {
						Task.WaitAll(
							Task.Factory.StartNew(() => nn.BlendAll(W0, Experts[0], weights, weights.Length)),
							Task.Factory.StartNew(() => nn.BlendAll(b0, Experts[1], weights, weights.Length)),
							Task.Factory.StartNew(() => nn.BlendAll(W1, Experts[2], weights, weights.Length)),
							Task.Factory.StartNew(() => nn.BlendAll(b1, Experts[3], weights, weights.Length)),
							Task.Factory.StartNew(() => nn.BlendAll(W2, Experts[4], weights, weights.Length)),
							Task.Factory.StartNew(() => nn.BlendAll(b2, Experts[5], weights, weights.Length))
						);
					} else {
						nn.BlendAll(W0, Experts[0], weights, weights.Length);
						nn.BlendAll(b0, Experts[1], weights, weights.Length);
						nn.BlendAll(W1, Experts[2], weights, weights.Length);
						nn.BlendAll(b1, Experts[3], weights, weights.Length);
						nn.BlendAll(W2, Experts[4], weights, weights.Length);
						nn.BlendAll(b2, Experts[5], weights, weights.Length);
					}
				}

				//Set Input
				if(index == nn.Components.Length-1 && nn.encoders.Length > 0) {
					int dim_accumulated = 0;
					for(int i=0; i<nn.encoders.Length; i++){
						for(int j=0; j<nn.encoders[i].H2Dim; j++){
							X.SetValue(j+dim_accumulated, 0, nn.encoders[i].GetLatent(j));
						}
						dim_accumulated += nn.encoders[i].H2Dim;
					}
				} else {
					for(int i=0; i<XDim; i++){
						X.SetValue(i, 0, nn.GetInput(GatingPivot+i));
					}
				}

				//Process Network
				if(!Bias) {
					b0.SetZero();
					b1.SetZero();
					b2.SetZero();
				}
				nn.Layer(X, W0, b0, Y).ELU();
				nn.Layer(Y, W1, b1, Y).ELU();
				nn.Layer(Y, W2, b2, Y);
				if(index < nn.Components.Length-1) {
					Y.SoftMax();
				}
			}

			private int GetExperts(NSM nn, int index) {
				return index == 0 ? 1 : nn.Components[index-1].YDim;
			}

		}
	}

}