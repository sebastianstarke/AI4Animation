// using DeepLearning;
// using UnityEngine;

// public class MatrixVisualisation : MonoBehaviour {

//     public NeuralNetwork Model;
//     public enum AXIS {X, Y};
//     public enum PLOTTING {Curve, Bars};

//     public float Max = 0f;
//     public UltiDraw.GUIRect Rect;
//     public string ID;
//     public AXIS Axis;
//     public PLOTTING Plotting;
//     public Color Background = Color.black.Opacity(0.5f);

//     public string Label = string.Empty;
//     public float Offset = 0f;

//     private Matrix M;
//     private float[] Values;

//     private bool Setup() {
//         if(M == null) {
//             M = new Matrix(1, 1);
//         }
// 		if(Model == null) {
// 			return false;
// 		}
// 		if(!Model.enabled) {
// 			return false;
// 		}
// 		if(ID == "") {
// 			return false;
// 		}
// 		Matrix matrix = Model.GetMatrix(ID);
// 		if(matrix == null) {
// 			return false;
// 		}
//         return true;
//     }

// 	void FixedUpdate() {
// 		if(GetComponent<AI4Animation.AnimationController>().SkipVisualizer) {
// 			return;
// 		}
        
// 		if(!Application.isPlaying) {
// 			return;
// 		}
        
//         if(!Setup()) {
//             return;
//         }

//         M = Matrix.PointwiseAbsolute(Model.GetMatrix(ID), M);
        
//         if(Axis == AXIS.X) {
//             Values = new float[M.GetRows()];
//             for(int i=0; i<Values.Length; i++) {
//                 Values[i] = M.RowMean(i);
//             }
//         }
//         if(Axis == AXIS.Y) {
//             Values = new float[M.GetCols()];
//             for(int i=0; i<Values.Length; i++) {
//                 Values[i] = M.ColMean(i);
//             }
//         }
//     }

//     void OnGUI() {
//         UltiDraw.Begin();
//         UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.175f + Offset), new Vector2(0f, 0.2f), 0.02f, Label, Color.black);
//         UltiDraw.End();
//     }

// 	void OnRenderObject() {
// 		UltiDraw.Begin();
//         UltiDraw.GUIRectangle(
//             new Vector2(Rect.X, Rect.Y),
//             new Vector2(Rect.W + 0.01f/Screen.width*Screen.height, Rect.H + 0.01f),
//             Background
//         );
//         if(Plotting == PLOTTING.Curve) {
//             UltiDraw.PlotFunction(
//                 new Vector2(Rect.X, Rect.Y),
//                 new Vector2(Rect.W, Rect.H),
//                 Values,
//                 0f,
//                 Max == 0f ? Values.Max() : Max
//             );
//         }
//         if(Plotting == PLOTTING.Bars) {
//             UltiDraw.PlotBars(
//                 new Vector2(Rect.X, Rect.Y),
//                 new Vector2(Rect.W, Rect.H),
//                 Values,
//                 0f,
//                 Max == 0f ? Values.Max() : Max
//             );
//         }
// 		UltiDraw.End();
// 	}

//     public struct Feature {
//         public int Index;
//         public float Value;
//         public Feature(int index, float value) {
//             Index = index;
//             Value = value;
//         }
//     }

// }