using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class GenerativeControl : NativeNetwork {

		public bool Draw = true;
		public int NoiseSeed = 0;
		public float NoiseScale = 0f;
		public int XDim = 221;
		public int H1Dim = 256;
		public int H2Dim = 110;
		public int H3Dim = 256;
		public int YDim = 221;

		private float YPos = 0.1f;

		private Matrix Xmean, Xstd, Ymean, Ystd, Lstd;
		private Matrix W0, W1, W2, W3, b0, b1, b2, b3;

		private float Amplitude = 0f;
		private float SeedTimer = 0f;
		private float NoiseTimer = 0f;

		void Update() {
			if(Input.GetKeyDown(KeyCode.LeftArrow)) {
				DecreaseSeed();
				SeedTimer = 1f;
			}
			if(Input.GetKeyDown(KeyCode.RightArrow)) {
				IncreaseSeed();
				SeedTimer = 1f;
			}
			if(Input.GetKeyDown(KeyCode.DownArrow)) {
				DecreaseNoise();
				NoiseTimer = 1f;
			}
			if(Input.GetKeyDown(KeyCode.UpArrow)) {
				IncreaseNoise();
				NoiseTimer = 1f;
			}
			NoiseTimer -= Time.deltaTime;
			SeedTimer -= Time.deltaTime;
		}

		void OnGUI() {
			if(Draw) {
				UltiDraw.Begin();
				UltiDraw.OnGUILabel(new Vector2(1f/3f, YPos+0.075f), new Vector2(0.5f, 0.25f), 0.02f, "Seed: " + NoiseSeed, Color.Lerp(UltiDraw.DarkGrey, Color.white, SeedTimer));
				UltiDraw.OnGUILabel(new Vector2(2f/3f, YPos+0.075f), new Vector2(0.5f, 0.25f), 0.02f, "Scale: " + Utility.Round(NoiseScale, 1), Color.Lerp(UltiDraw.DarkGrey, Color.white, NoiseTimer));
				UltiDraw.OnGUILabel(new Vector2(0.5f, YPos+0.075f), new Vector2(0.5f, 0.2f), 0.02f, "Generative Control", Color.black);
				UltiDraw.End();
			}
		}

		void OnRenderObject() {
			if(Draw) {
				float x = 0.5f;
				float y = YPos;
				float w = 0.5f;
				float h = 0.1f;

				float[] deltas = new float[GetOutputDimensionality()];
				for(int i=0; i<deltas.Length; i++) {
					deltas[i] = GetOutput(i) - GetInput(i);
				}

				float lambda = 0.9f;
				float dAmp = deltas.Amp();
				Amplitude = lambda * dAmp + (1f-lambda) * Mathf.Max(Amplitude, dAmp);
				UltiDraw.Begin();
				UltiDraw.PlotFunction(new Vector2(x, y), new Vector2(w, h), deltas, yMin: -Amplitude, yMax: Amplitude);
				UltiDraw.End();
			}
		}

		protected override void LoadDerived() {
			Xmean = CreateMatrix(XDim, 1, "X_mean", Folder+"/X_mean.bin");
			Xstd = CreateMatrix(XDim, 1, "X_std", Folder+"/X_std.bin");
			Ymean = CreateMatrix(YDim, 1, "Y_mean", Folder+"/Y_mean.bin");
			Ystd = CreateMatrix(YDim, 1, "Y_std", Folder+"/Y_std.bin");
			Lstd = CreateMatrix(H2Dim, 1, "Latent_std", Folder+"/Latent_std.bin");
			W0 = CreateMatrix(H1Dim, XDim, "ln1_weight", Folder+"/ln1_weight.bin");
			W1 = CreateMatrix(H2Dim, H1Dim, "ln2_weight", Folder+"/ln2_weight.bin");
			W2 = CreateMatrix(H3Dim, H2Dim, "ln3_weight", Folder+"/ln3_weight.bin");
			W3 = CreateMatrix(YDim, H3Dim, "ln4_weight", Folder+"/ln4_weight.bin");
			b0 = CreateMatrix(H1Dim, 1, "ln1_bias", Folder+"/ln1_bias.bin");
			b1 = CreateMatrix(H2Dim, 1, "ln2_bias", Folder+"/ln2_bias.bin");
			b2 = CreateMatrix(H3Dim, 1, "ln3_bias", Folder+"/ln3_bias.bin");
			b3 = CreateMatrix(YDim, 1, "ln4_bias", Folder+"/ln4_bias.bin");

			X = CreateMatrix(XDim, 1, "X");
			Y = CreateMatrix(YDim, 1, "Y");
		}

		protected override void UnloadDerived() {
			
		}

		public void UpdateSeed(int value) {
			NoiseSeed = Mathf.Max(NoiseSeed + value, 0);
		}

		public void UpdateScale(float value) {
			NoiseScale = Mathf.Clamp(NoiseScale + value, 0f, 2f);
		}

		public void IncreaseSeed() {
			UpdateSeed(1);
		}

		public void DecreaseSeed() {
			UpdateSeed(-1);
		}

		public void IncreaseNoise() {
			UpdateScale(0.1f);
		}

		public void DecreaseNoise() {
			UpdateScale(-0.1f);
		}

		public static double SampleGaussian(System.Random random, double mean, double stddev) {
				// The method requires sampling from a uniform random of (0,1]
				// but Random.NextDouble() returns a sample of [0,1).
				double x1 = 1 - random.NextDouble();
				double x2 = 1 - random.NextDouble();

				double y1 = System.Math.Sqrt(-2.0 * System.Math.Log(x1)) * System.Math.Cos(2.0 * System.Math.PI * x2);
				return y1 * stddev + mean;
		}

		protected override void PredictDerived() {
			System.Random rd = new System.Random(NoiseSeed);

			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process Vanilla Network
			Layer(Y, W0, b0, Y).RELU();
			Layer(Y, W1, b1, Y).RELU();

			// Add Random noise
			for(int i = 0; i < H2Dim; i++) {
				Y.SetValue(i, 0, Y.GetValue(i, 0) + NoiseScale * (float)SampleGaussian(rd, 0f, Lstd.GetValue(i, 0)));
			}

			Layer(Y, W2, b2, Y).RELU();
			Layer(Y, W3, b3, Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}
		/*void OnRenderObject() {
			System.Random rd = new System.Random();
			int SampleLength = 10000;
			int SlotSize = 100;
			float SlotWidth = 0.05f;
			float[] count = new float[SlotSize];
			float[] array = new float[SampleLength];
			for(int i = 0; i < SampleLength; i++) {
				array[i] = (float)SampleGaussian(rd, 0f, 1f);
			}
			for(int i = 0; i < SampleLength; i++) {
				int index = (int)Mathf.Ceil(array[i] / SlotWidth) + SlotSize/2;
				index = Mathf.Clamp(index, 0, SlotSize-1);
				count[index] += 1f;
			}

			UltiDraw.Begin();
			UltiDraw.PlotFunction(new Vector2(.5f, .5f), new Vector2(.5f, .5f), count, lineColor:UltiDraw.Red);
			UltiDraw.End();
		}*/
	}

}