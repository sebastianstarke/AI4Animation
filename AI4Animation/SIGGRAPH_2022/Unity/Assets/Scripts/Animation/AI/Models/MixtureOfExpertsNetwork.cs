using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Eigen;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    [Serializable]
    public class MixtureOfExpertsNetwork : NeuralNetwork {

        public Parameters Parameters = null;
        public string Folder = string.Empty;

        public int GatingInput = 0;
        public int GatingHidden = 128;
        public int GatingOutput = 8;
        public int MainInput = 0;
        public int MainHidden = 512;
        public int MainOutput = 0;

        public enum DRAWMODE {Function, Graph}
        public DRAWMODE DrawMode = DRAWMODE.Function;
        public UltiDraw.GUIRect Rect;

        public class MixtureOfExpertsInference : Inference {

            public MixtureOfExpertsNetwork Network;
            public List<Matrix> Matrices;

            public Matrix X, Y;
            public Matrix Xmean, Xstd, Ymean, Ystd;

            public Matrix GX, MX;
            public Matrix GW0, GW1, GW2, Gb0, Gb1, Gb2;
            public Matrix MW0, MW1, MW2, Mb0, Mb1, Mb2;
            public List<Matrix[]> Experts;

            public float WeightHorizon = 100;
            public List<float[]> WeightHistory = new List<float[]>();

            public MixtureOfExpertsInference(MixtureOfExpertsNetwork network) {
                Network = network;
                Matrices = new List<Matrix>();
                X = CreateMatrix(GetFeedSize(), 1, "X");
			    Y = CreateMatrix(GetReadSize(), 1, "Y");
                Xmean = CreateMatrix(GetFeedSize(), 1, "Xmean", network.Folder+"/Xmean.bin");
                Xstd = CreateMatrix(GetFeedSize(), 1, "Xstd", network.Folder+"/Xstd.bin");
                Ymean = CreateMatrix(GetReadSize(), 1, "Ymean", network.Folder+"/Ymean.bin");
                Ystd = CreateMatrix(GetReadSize(), 1, "Ystd", network.Folder+"/Ystd.bin");

                GX = CreateMatrix(Network.GatingInput, 1, "GX");
                MX = CreateMatrix(Network.MainInput, 1, "MX");

                GW0 = CreateMatrix(Network.GatingHidden, Network.GatingInput, "wc000_w", Network.Folder+"/"+"wc000_w"+".bin");
			    Gb0 = CreateMatrix(Network.GatingHidden, 1, "wc000_b", Network.Folder+"/"+"wc000_b"+".bin");
                GW1 = CreateMatrix(Network.GatingHidden, Network.GatingHidden, "wc010_w", Network.Folder+"/"+"wc010_w"+".bin");
                Gb1 = CreateMatrix(Network.GatingHidden, 1, "wc010_b", Network.Folder+"/"+"wc010_b"+".bin");
                GW2 = CreateMatrix(Network.GatingOutput, Network.GatingHidden, "wc020_w", Network.Folder+"/"+"wc020_w"+".bin");
                Gb2 = CreateMatrix(Network.GatingOutput, 1, "wc020_b", Network.Folder+"/"+"wc020_b"+".bin");

                MW0 = CreateMatrix(Network.MainHidden, Network.MainInput, "MW0");
			    Mb0 = CreateMatrix(Network.MainHidden, 1, "Mb0");
                MW1 = CreateMatrix(Network.MainHidden, Network.MainHidden, "MW1");
                Mb1 = CreateMatrix(Network.MainHidden, 1, "Mb1");
                MW2 = CreateMatrix(Network.MainOutput, Network.MainHidden, "MW2");
                Mb2 = CreateMatrix(Network.MainOutput, 1, "Mb2");

				Experts = new List<Matrix[]>();
				for(int i=0; i<6; i++) {
					Experts.Add(new Matrix[network.GatingOutput]);
				}
				for(int i=0; i<network.GatingOutput; i++) {
					Experts[0][i] = CreateMatrix(Network.MainHidden, network.MainInput, "wc10"+i.ToString("D1")+"_w", Network.Folder+"/wc10"+i.ToString("D1")+"_w.bin");
					Experts[1][i] = CreateMatrix(Network.MainHidden, 1, "wc10"+i.ToString("D1")+"_b", Network.Folder+"/wc10"+i.ToString("D1")+"_b.bin");
					Experts[2][i] = CreateMatrix(Network.MainHidden, Network.MainHidden, "wc11"+i.ToString("D1")+"_w", Network.Folder+"/wc11"+i.ToString("D1")+"_w.bin");
					Experts[3][i] = CreateMatrix(Network.MainHidden, 1, "wc11"+i.ToString("D1")+"_b", Network.Folder+"/wc11"+i.ToString("D1")+"_b.bin");
					Experts[4][i] = CreateMatrix(Network.MainOutput, Network.MainHidden, "wc12"+i.ToString("D1")+"_w", Network.Folder+"/wc12"+i.ToString("D1")+"_w.bin");
					Experts[5][i] = CreateMatrix(Network.MainOutput, 1, "wc12"+i.ToString("D1")+"_b", Network.Folder+"/wc12"+i.ToString("D1")+"_b.bin");
				}
            }

            public override void Dispose() {
                foreach(Matrix m in Matrices) {
                    m.Delete();
                }
            }

            public override int GetFeedSize() {
                return Network.GatingInput + Network.MainInput;
            }

            public override int GetReadSize() {
                return Network.MainOutput;
            }

            public override void Feed(float value) {
                X.SetValue(Pivot, 0, value);
            }

            public override float Read() {
                return Y.GetValue(GetReadSize()-Pivot, 0);
            }

            public override void Run() {
                Matrix.Normalize(X, Xmean, Xstd, X);

                //Gating Network
                for(int i=0; i<Network.GatingInput; i++) {
                    GX.SetValue(i, 0, X.GetValue(Network.MainInput+i, 0));
                }
				Matrix.Layer(GX, GW0, Gb0, Y).ELU();
				Matrix.Layer(Y, GW1, Gb1, Y).ELU();
				Matrix.Layer(Y, GW2, Gb2, Y).SoftMax();
                
                //Expert Blending
                float[] weights = Y.Flatten();
                Task.WaitAll(
                    Task.Factory.StartNew(() => Matrix.BlendAll(MW0, Experts[0], weights, weights.Length)),
                    Task.Factory.StartNew(() => Matrix.BlendAll(Mb0, Experts[1], weights, weights.Length)),
                    Task.Factory.StartNew(() => Matrix.BlendAll(MW1, Experts[2], weights, weights.Length)),
                    Task.Factory.StartNew(() => Matrix.BlendAll(Mb1, Experts[3], weights, weights.Length)),
                    Task.Factory.StartNew(() => Matrix.BlendAll(MW2, Experts[4], weights, weights.Length)),
                    Task.Factory.StartNew(() => Matrix.BlendAll(Mb2, Experts[5], weights, weights.Length))
                );

                //Main Network
                for(int i=0; i<Network.MainInput; i++) {
                    MX.SetValue(i, 0, X.GetValue(i, 0));
                }
                Matrix.Layer(MX, MW0, Mb0, Y).ELU();
				Matrix.Layer(Y, MW1, Mb1, Y).ELU();
				Matrix.Layer(Y, MW2, Mb2, Y);

                Matrix.Renormalize(Y, Ymean, Ystd, Y);

                //Blending History
                WeightHistory.Add(weights);
                while(WeightHistory.Count > WeightHorizon) {
                    WeightHistory.RemoveAt(0);
                }
            }

            private Matrix CreateMatrix(int rows, int cols, string id, string binary=null) {
                Matrix M = binary == null ? new Matrix(rows, cols, id) : Matrix.FromBinary(rows, cols, id, binary);
                Matrices.Add(M);
                return M;
            }

        }

        public void DrawGatingSpace() {
            if(GetSession() != null) {
                UltiDraw.Begin();
                switch(DrawMode) {
                    case DRAWMODE.Function:
                    UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), ((MixtureOfExpertsInference)GetSession()).WeightHistory.ToArray(), UltiDraw.Dimension.Y, yMin: 0f, yMax: 1f, thickness: 0.001f);
                    break;

                    case DRAWMODE.Graph:
                    MixtureOfExpertsInference inference = (MixtureOfExpertsInference)GetSession();
                    int experts = GatingOutput;
                    Color[] colors = UltiDraw.GetRainbowColors(GatingOutput);
                    Vector2 pivot = Rect.GetCenter();
                    float radius = 0.2f * Rect.W;
                    UltiDraw.GUICircle(pivot, Rect.W*1.05f, UltiDraw.Gold);
                    UltiDraw.GUICircle(pivot, Rect.W, UltiDraw.White);
                    Vector2[] anchors = new Vector2[experts];
                    for(int i=0; i<experts; i++) {
                        float step = (float)i / (float)experts;
                        anchors[i] = Rect.ToScreen(new Vector2(Mathf.Cos(step*2f*Mathf.PI), Mathf.Sin(step*2f*Mathf.PI)));
                    }
                    float[] Transform(float[] weights) {
                        float[] w = weights == null ? new float[GatingOutput] : weights.Copy();
                        if(weights == null) {
                            w.SetAll(1f);
                        }
                        for(int i=0; i<w.Length; i++) {
                            w[i] = Mathf.Pow(w[i], 2f);
                        }
                        float sum = w.Sum();
                        for(int i=0; i<w.Length; i++) {
                            w[i] /= sum;
                        }
                        return w;
                    }
                    Vector2 Blend(float[] weights) {
                        Vector2 position = Vector2.zero;
                        for(int i=0; i<experts; i++) {
                            position += weights[i] * anchors[i];
                        }
                        return position;
                    }
                    //Variables
                    Vector2[] positions = new Vector2[inference.WeightHistory.Count];
                    for(int i=0; i<positions.Length; i++) {
                        positions[i] = Blend(Transform(inference.WeightHistory[i]));
                    }
                    float[] weights = Transform(inference.WeightHistory.Last());
                    //Anchors
                    for(int i=0; i<anchors.Length; i++) {
                        UltiDraw.GUILine(pivot + positions.Last(), pivot + anchors[i], 0.1f*radius, colors[i].Opacity(weights[i]));
                    }
                    for(int i=0; i<anchors.Length; i++) {
                        UltiDraw.GUICircle(pivot + anchors[i], weights[i].Normalize(0f, 1f, 0.5f, 1f) * radius, Color.Lerp(UltiDraw.Black, colors[i], weights[i]));
                    }
                    //Lines
                    for(int i=1; i<positions.Length; i++) {
                        UltiDraw.GUILine(pivot + positions[i-1], pivot + positions[i], 0.1f*radius, UltiDraw.Black.Opacity((float)(i+1)/(float)positions.Length));
                    }
                    //Head
                    UltiDraw.GUICircle(pivot + positions.Last(), 0.5f*radius, UltiDraw.Purple);
                    break;
                }
                UltiDraw.End();
            }
        }

        protected override Inference BuildInference() {
            return new MixtureOfExpertsInference(this);
        }

        #if UNITY_EDITOR
        public override void Inspect() {
            Parameters = EditorGUILayout.ObjectField("Parameters", Parameters, typeof(Parameters), true) as Parameters;
            Folder = EditorGUILayout.TextField("Folder", Folder);
            GatingInput = EditorGUILayout.IntField("Gating Input", GatingInput);
            GatingHidden = EditorGUILayout.IntField("Gating Hidden", GatingHidden);
            GatingOutput = EditorGUILayout.IntField("Gating Output", GatingOutput);
            MainInput = EditorGUILayout.IntField("Main Input", MainInput);
            MainHidden = EditorGUILayout.IntField("Main Hidden", MainHidden);
            MainOutput = EditorGUILayout.IntField("Main Output", MainOutput);
            DrawMode = (DRAWMODE)EditorGUILayout.EnumPopup("Draw Mode", DrawMode);
            Rect.Inspector();
        }
        #endif

    }
}