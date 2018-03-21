using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class TensorActivation : MonoBehaviour {

    [Range(0f, 1f)] public float Y = 0.1f;

	private BioAnimation Animation;
	private float[] Values;

    private Tensor T;
    private float Minimum = float.MaxValue;
    private float Maximum = float.MinValue;

	void Awake() {
		Animation = GetComponent<BioAnimation>();
	}

    void Start() {
        Values = new float[Animation.MFNN.XDim];
        T = new Tensor(1, 1);
    }

	void OnRenderObject() {
        T = Tensor.PointwiseAbsolute(Animation.MFNN.GetW0(), T);
        for(int i=0; i<Animation.MFNN.XDim; i++) {
            Values[i] = T.ColSum(i);
            Minimum = Mathf.Min(Minimum, Values[i]);
            Maximum = Mathf.Max(Maximum, Values[i]);
            //if(Values[i] < 1f) {
            //    Debug.Log(i + " is inactive.");
            //}
        }
		UltiDraw.Begin();
        UltiDraw.DrawGUIRectangle(
            new Vector2(0.5f, Y),
            new Vector2(0.95f + 0.01f/Screen.width*Screen.height, 0.2f + 0.01f),
            UltiDraw.Black.Transparent(0.5f)
        );
        UltiDraw.DrawFunction(
            new Vector2(0.5f, Y),
            new Vector2(0.95f, 0.2f),
            Values,
            Minimum,
            Maximum,
            UltiDraw.White.Transparent(0.5f),
            UltiDraw.Black
        );
		UltiDraw.End();
	}

}
