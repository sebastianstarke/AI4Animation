using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class MatrixVisualisation : MonoBehaviour {

    public NeuralNetwork Model;
    public enum AXIS {X, Y};
    public enum PLOTTING {Curve, Bars};

    public UltiDraw.GUIRect Rect;
    public string ID;
    public AXIS Axis;
    public PLOTTING Plotting;

    private Matrix M;
    private float[] Values;

    private bool Setup() {
        if(M == null) {
            M = new Matrix(1, 1);
        }
		if(Model == null) {
			return false;
		}
		if(!Model.enabled) {
			return false;
		}
		if(ID == "") {
			return false;
		}
		Matrix matrix = Model.GetMatrix(ID);
		if(matrix == null) {
			return false;
		}
        return true;
    }

	void OnRenderObject() {
		if(!Application.isPlaying) {
			return;
		}
        
        if(!Setup()) {
            return;
        }

        M = Matrix.PointwiseAbsolute(Model.GetMatrix(ID), M);
        float minimum = float.MaxValue;
        float maximum = float.MinValue;
        
        if(Axis == AXIS.X) {
            Values = new float[M.GetRows()];
            for(int i=0; i<M.GetRows(); i++) {
                Values[i] = M.RowSum(i);
                minimum = Mathf.Min(minimum, Values[i]);
                maximum = Mathf.Max(maximum, Values[i]);
            }
        }
        if(Axis == AXIS.Y) {
            Values = new float[M.GetCols()];
            for(int i=0; i<M.GetCols(); i++) {
                Values[i] = M.ColSum(i);
                minimum = Mathf.Min(minimum, Values[i]);
                maximum = Mathf.Max(maximum, Values[i]);
            }
        }

		UltiDraw.Begin();
        UltiDraw.DrawGUIRectangle(
            new Vector2(Rect.X, Rect.Y),
            new Vector2(Rect.W + 0.01f/Screen.width*Screen.height, Rect.H + 0.01f),
            UltiDraw.Black.Transparent(0.5f)
        );
        if(Plotting == PLOTTING.Curve) {
            UltiDraw.DrawGUIFunction(
                new Vector2(Rect.X, Rect.Y),
                new Vector2(Rect.W, Rect.H),
                Values,
                0f,
                maximum,
                UltiDraw.White.Transparent(0.5f),
                UltiDraw.Black
            );
        }
        if(Plotting == PLOTTING.Bars) {
            UltiDraw.DrawGUIBars(
                new Vector2(Rect.X, Rect.Y),
                new Vector2(Rect.W, Rect.H),
                Values,
                0f,
                maximum,
                0.75f * Rect.W / Values.Length,
                UltiDraw.White.Transparent(0.5f),
                UltiDraw.Black
            );
        }
		UltiDraw.End();
	}

    public struct Feature {
        public int Index;
        public float Value;
        public Feature(int index, float value) {
            Index = index;
            Value = value;
        }
    }

}