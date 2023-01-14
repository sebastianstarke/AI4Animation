using UnityEngine;
using UnityEditor;
using AI4Animation;

public class Example_ONNX : MonoBehaviour {

    public ONNXNetwork NeuralNetwork;

    void Awake() {
        //Create a new inference session before running the network at each frame.
        NeuralNetwork.CreateSession();
    }

    void OnDestroy() {
        //Close the session which disposes allocated memory.
        NeuralNetwork.CloseSession();
    }

    void Update() {
        //Give your inputs to the network. You can directly feed your inputs to the network without allocating the inputs array,
        //which is faster. If not enough or too many inputs are given to the network, it will throw warnings.
        float[] input = new float[NeuralNetwork.GetSession().GetFeedSize()];
        for(int i=0; i<NeuralNetwork.GetSession().GetFeedSize(); i++) {
            NeuralNetwork.Feed(input[i]);
        }

        //Run the inference.
        NeuralNetwork.RunSession();

        //Read your outputs from the network. You can directly read all outputs from the network without allocating the outputs array,
        //which is faster. If not enough or too many outputs are read from the network, it will throw warnings.
        float[] output = new float[NeuralNetwork.GetSession().GetReadSize()];
        for(int i=0; i<NeuralNetwork.GetSession().GetReadSize(); i++) {
            output[i] = NeuralNetwork.Read();
        }

        output.Print(true);
        Debug.Log("Computing inference took " + NeuralNetwork.GetSession().Time + "s.");
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(Example_ONNX), true)]
	public class Example_ONNX_Editor : Editor {

		public Example_ONNX Target;

		void Awake() {
			Target = (Example_ONNX)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			DrawDefaultInspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif
}
