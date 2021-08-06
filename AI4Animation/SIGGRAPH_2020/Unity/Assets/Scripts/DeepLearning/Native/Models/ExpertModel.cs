using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class ExpertModel : NativeNetwork {
		public Component[] Components = new Component[0];

		public int Features;

		// public bool Bias = true;
		public bool Threading = true;

		private Matrix Xmean, Xstd, Ymean, Ystd;

		protected override void LoadDerived() {
			Xmean = CreateMatrix(Features, 1, "Xmean", Folder+"/Xmean.bin");
			Xstd = CreateMatrix(Features, 1, "Xstd", Folder+"/Xstd.bin");
			Ymean = CreateMatrix(Components[Components.Length-1].YDim, 1, "Ymean", Folder+"/Ymean.bin");
			Ystd = CreateMatrix(Components[Components.Length-1].YDim, 1, "Ystd", Folder+"/Ystd.bin");
			for(int i=0; i<Components.Length; i++) {
				Components[i].Setup(this, i);
				Components[i].Load();
			}
			X = CreateMatrix(Features, 1, "X");
			Y = GetMatrix("C"+(Components.Length-1)+"Y");
		}

		protected override void UnloadDerived() {

		}

		protected override void PredictDerived() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, X);

			for(int i=0; i<Components.Length; i++) {
				Components[i].Process();
			}

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}

		[System.Serializable]
		public class Component {
			public int XDim, H1Dim, H2Dim, YDim;
			public int GatingPivot;
			
			private ExpertModel Model;
			private int Index;
			private List<Matrix[]> Experts;
			private Matrix W0, W1, W2, b0, b1, b2;
			private Matrix X, Y;

			public void Setup(ExpertModel nn, int index) {
				Model = nn;
				Index = index;
			}

			public void Load() {
				Experts = new List<Matrix[]>();
				for(int i=0; i<6; i++) {
					Experts.Add(new Matrix[GetExperts()]);
				}
				for(int i=0; i<GetExperts(); i++) {
					Experts[0][i] = Model.CreateMatrix(H1Dim, XDim, "wc"+Index+"0"+i.ToString("D1")+"_w", Model.Folder+"/wc"+Index+"0"+i.ToString("D1")+"_w.bin");
					Experts[1][i] = Model.CreateMatrix(H1Dim, 1, "wc"+Index+"0"+i.ToString("D1")+"_b", Model.Folder+"/wc"+Index+"0"+i.ToString("D1")+"_b.bin");
					Experts[2][i] = Model.CreateMatrix(H2Dim, H1Dim, "wc"+Index+"1"+i.ToString("D1")+"_w", Model.Folder+"/wc"+Index+"1"+i.ToString("D1")+"_w.bin");
					Experts[3][i] = Model.CreateMatrix(H2Dim, 1, "wc"+Index+"1"+i.ToString("D1")+"_b", Model.Folder+"/wc"+Index+"1"+i.ToString("D1")+"_b.bin");
					Experts[4][i] = Model.CreateMatrix(YDim, H2Dim, "wc"+Index+"2"+i.ToString("D1")+"_w", Model.Folder+"/wc"+Index+"2"+i.ToString("D1")+"_w.bin");
					Experts[5][i] = Model.CreateMatrix(YDim, 1, "wc"+Index+"2"+i.ToString("D1")+"_b", Model.Folder+"/wc"+Index+"2"+i.ToString("D1")+"_b.bin");
				}
				W0 = Model.CreateMatrix(H1Dim, XDim, "C"+Index+"W0");
				b0 = Model.CreateMatrix(H1Dim, 1, "C"+Index+"b0");
				W1 = Model.CreateMatrix(H2Dim, H1Dim, "C"+Index+"W1");
				b1 = Model.CreateMatrix(H2Dim, 1, "C"+Index+"b1");
				W2 = Model.CreateMatrix(YDim, H2Dim, "C"+Index+"W2");
				b2 = Model.CreateMatrix(YDim, 1, "C"+Index+"b2");
				X = Model.CreateMatrix(XDim, 1, "C"+Index+"X");
				Y = Model.CreateMatrix(YDim, 1, "C"+Index+"Y");
			}

			public void Process() {
				//Generate Parameters
				if(Index == 0) {
					W0 = Experts[0].First();
					b0 = Experts[1].First();
					W1 = Experts[2].First();
					b1 = Experts[3].First();
					W2 = Experts[4].First();
					b2 = Experts[5].First();
				} else {
					float[] weights = Model.Components[Index-1].Y.Flatten();
					if(Model.Threading) {
						Task.WaitAll(
							Task.Factory.StartNew(() => Model.BlendAll(W0, Experts[0], weights, weights.Length)),
							Task.Factory.StartNew(() => Model.BlendAll(b0, Experts[1], weights, weights.Length)),
							Task.Factory.StartNew(() => Model.BlendAll(W1, Experts[2], weights, weights.Length)),
							Task.Factory.StartNew(() => Model.BlendAll(b1, Experts[3], weights, weights.Length)),
							Task.Factory.StartNew(() => Model.BlendAll(W2, Experts[4], weights, weights.Length)),
							Task.Factory.StartNew(() => Model.BlendAll(b2, Experts[5], weights, weights.Length))
						);
					} else {
						Model.BlendAll(W0, Experts[0], weights, weights.Length);
						Model.BlendAll(b0, Experts[1], weights, weights.Length);
						Model.BlendAll(W1, Experts[2], weights, weights.Length);
						Model.BlendAll(b1, Experts[3], weights, weights.Length);
						Model.BlendAll(W2, Experts[4], weights, weights.Length);
						Model.BlendAll(b2, Experts[5], weights, weights.Length);
					}
				}

				//Process Inputs
				for(int i=0; i<XDim; i++) {
					X.SetValue(i, 0, Model.GetInput(GatingPivot+i));
				}

				//Process Network
				// if(!IsFinal() && !Model.Bias) {
				// 	b0.SetZero();
				// 	b1.SetZero();
				// 	b2.SetZero();
				// }
				Model.Layer(X, W0, b0, Y).ELU();
				Model.Layer(Y, W1, b1, Y).ELU();
				Model.Layer(Y, W2, b2, Y);
				if(!IsFinal()) {
					Y.SoftMax();
				}
			}

			private int GetExperts() {
				return Index == 0 ? 1 : Model.Components[Index-1].YDim;
			}

			private bool IsFinal() {
				return Index == Model.Components.Length-1;
			}
		}
	}

}