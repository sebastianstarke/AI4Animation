using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class TensorActivation : MonoBehaviour {

    public enum AXIS {X, Y};
    public enum OPERATOR {AbsSum, AbsDiff};
    public enum PLOTTING {Curve, Bars};

    public GUIRect Rect;
    public string ID;
    public AXIS Axis;
    public OPERATOR Operator;
    public PLOTTING Plotting;

    private Model Model;
    private Tensor T;
    private float[] Values;

	void Awake() {
		Model = GetComponent<BioAnimation>().NN.Model;
	}

    void Start() {
        T = new Tensor(1, 1);
    }

	void OnRenderObject() {
        Tensor t = Model.GetTensor(ID);
        if(t == null) {
            return;
        }

        T = Tensor.PointwiseAbsolute(t, T);
        float minimum = float.MaxValue;
        float maximum = float.MinValue;
        
        if(Operator == OPERATOR.AbsSum) {
            if(Axis == AXIS.X) {
                Values = new float[T.GetRows()];
                for(int i=0; i<T.GetRows(); i++) {
                    Values[i] = T.RowSum(i);
                    minimum = Mathf.Min(minimum, Values[i]);
                    maximum = Mathf.Max(maximum, Values[i]);
                }
            }
            if(Axis == AXIS.Y) {
                Values = new float[T.GetCols()];
                for(int i=0; i<T.GetCols(); i++) {
                    Values[i] = T.ColSum(i);
                    minimum = Mathf.Min(minimum, Values[i]);
                    maximum = Mathf.Max(maximum, Values[i]);
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
            UltiDraw.DrawFunction(
                new Vector2(Rect.X, Rect.Y),
                new Vector2(Rect.W, Rect.H),
                Values,
                minimum,
                maximum,
                UltiDraw.White.Transparent(0.5f),
                UltiDraw.Black
            );
        }
        if(Plotting == PLOTTING.Bars) {
            UltiDraw.DrawBars(
                new Vector2(Rect.X, Rect.Y),
                new Vector2(Rect.W, Rect.H),
                Values,
                minimum,
                maximum,
                0.75f * Rect.W / Values.Length,
                UltiDraw.White.Transparent(0.5f),
                UltiDraw.Black
            );
        }
		UltiDraw.End();

        /*
        Feature[] features = new Feature[Values.Length];
        for(int i=0; i<Values.Length; i++) {
            features[i] = new Feature(i, Values[i]);
        }
        System.Array.Sort(features,
			delegate(Feature a, Feature b) {
				return b.Value.CompareTo(a.Value);
			}
		);
        for(int i=0; i<30; i++) {
            Debug.Log(i + " - " + "Value: " + features[i].Value + " Index: " + features[i].Index);
        }
        */
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
