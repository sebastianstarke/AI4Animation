using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class PFNN : MonoBehaviour {

	void Update() {
		Matrix<double> m = Matrix<double>.Build.Random(3, 4);
		Vector<double> v = Vector<double>.Build.Random(4);
		Vector<double> result = m*v;
		Debug.Log(result);
	}

}
