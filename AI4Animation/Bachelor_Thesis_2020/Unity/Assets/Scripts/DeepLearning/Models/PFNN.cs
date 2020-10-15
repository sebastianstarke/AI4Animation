using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class PFNN : NeuralNetwork {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		public int PhaseIndex = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor[] W0, W1, W2, b0, b1, b2;

		private float Phase;
		private float Damping;

		private const float M_PI = 3.14159265358979323846f;

		public PFNN() {
			
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
				W0[i] = CreateTensor(Parameters.Load("W0_"+i.ToString("D3")));
				W1[i] = CreateTensor(Parameters.Load("W1_"+i.ToString("D3")));
				W2[i] = CreateTensor(Parameters.Load("W2_"+i.ToString("D3")));
				b0[i] = CreateTensor(Parameters.Load("b0_"+i.ToString("D3")));
				b1[i] = CreateTensor(Parameters.Load("b1_"+i.ToString("D3")));
				b2[i] = CreateTensor(Parameters.Load("b2_"+i.ToString("D3")));
			}
			X = CreateTensor(XDim, 1, "X");
			Y = CreateTensor(YDim, 1, "Y");
			Phase = 0f;
			Damping = 0f;
		}
		
		public override void Predict() {
			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process PFNN
			int index = (int)((Phase / (2f*M_PI)) * 50f);
			ELU(Layer(Y, W0[index], b0[index], Y));
			ELU(Layer(Y, W1[index], b1[index], Y));
			Layer(Y, W2[index], b2[index], Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);

			//Update Phase
			Phase = Mathf.Repeat(Phase + (1f-Damping)*GetOutput(PhaseIndex)*2f*Mathf.PI, 2f*Mathf.PI);
		}

		public void SetDamping(float value) {
			Damping = value;
		}

		public float GetPhase() {
			return Phase / (2f*Mathf.PI);
		}

	}

}