using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class TensorActivation : MonoBehaviour {

    public enum AXIS {X, Y};
    public enum OPERATOR {AbsSum, AbsDiff};
    public enum PLOTTING {Curve, Bars};

    public UltiDraw.GUIRect Rect;
    public string ID;
    public AXIS Axis;
    public OPERATOR Operator;
    public PLOTTING Plotting;

    private NeuralNetwork NN;
    private Tensor T;
    private float[] Values;

    private float Minimum;
    private float Maximum;

	void Awake() {
		NN = GetComponent<NeuralNetwork>();
        Minimum = float.MaxValue;
        Maximum = float.MinValue;
    }

    void Start() {
        T = new Tensor(1, 1, "Activation");
    }

    void OnEnable() {
        Awake();
        Start();
    }

	void OnRenderObject() {
        Tensor t = NN.GetTensor(ID);
        if(t == null) {
            return;
        }

        T = Tensor.PointwiseAbsolute(t, T);
        //float minimum = float.MaxValue;
        //float maximum = float.MinValue;
        
        if(Operator == OPERATOR.AbsSum) {
            if(Axis == AXIS.X) {
                Values = new float[T.GetRows()];
                for(int i=0; i<T.GetRows(); i++) {
                    Values[i] = T.RowSum(i);
                    Minimum = Mathf.Min(Minimum, Values[i]);
                    Maximum = Mathf.Max(Maximum, Values[i]);
                }
            }
            if(Axis == AXIS.Y) {
                Values = new float[T.GetCols()];
                for(int i=0; i<T.GetCols(); i++) {
                    Values[i] = T.ColSum(i);
                    Minimum = Mathf.Min(Minimum, Values[i]);
                    Maximum = Mathf.Max(Maximum, Values[i]);
                }
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
                Maximum,
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
                Maximum,
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
