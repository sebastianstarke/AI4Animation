using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class TensorActivation : MonoBehaviour {

    public enum AXIS {X, Y};

    public GUIRect Rect;
    public string ID;
    public AXIS Axis;

    private MFNN Model;
    private Tensor T;
    private float[] Values;

	void Awake() {
		//Model = GetComponent<BioAnimation>().MFNN;
	}

    void Start() {
       // T = new Tensor(1, 1);
    }

	void OnRenderObject() {
        /*
        Tensor t = Model.GetTensor(ID);
        if(t == null) {
            return;
        }
        T = Tensor.PointwiseAbsolute(t, T);
        float minimum = float.MaxValue;
        float maximum = float.MinValue;
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
		UltiDraw.Begin();
        UltiDraw.DrawGUIRectangle(
            new Vector2(Rect.X, Rect.Y),
            new Vector2(Rect.W + 0.01f/Screen.width*Screen.height, Rect.H + 0.01f),
            UltiDraw.Black.Transparent(0.5f)
        );
        UltiDraw.DrawFunction(
            new Vector2(Rect.X, Rect.Y),
            new Vector2(Rect.W, Rect.H),
            Values,
            minimum,
            maximum,
            UltiDraw.White.Transparent(0.5f),
            UltiDraw.Black
        );
		UltiDraw.End();
        */
	}

}
