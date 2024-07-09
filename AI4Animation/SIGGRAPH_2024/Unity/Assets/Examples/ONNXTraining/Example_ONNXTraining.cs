using UnityEngine;
using System.Collections.Generic;

public class Example_ONNXTraining : MonoBehaviour {
    
    private PyTorchSocket Socket;

    void Awake() {
        Socket = GetComponent<PyTorchSocket>();
    }

    void Update() {
        for(int b=0; b<Socket.BatchSize; b++) {
            float[] input = new float[Socket.InputSize];
            float[] output = new float[Socket.OutputSize];
            for(int i=0; i<Socket.InputSize; i++) {
                input[i] = Random.Range(-5f, 5f);
                output[i] = Function(input[i]);
            }
            Socket.Feed(input);
            Socket.Feed(output);
        }
        Socket.RunSession();
    }

    private float Function(float x) {
        return Mathf.Pow(x, 2f);
    }

    void OnRenderObject() {
        UltiDraw.Begin();
        List<float> targets = new List<float>();
        for(int b=0; b<Socket.BatchSize; b++) {
            float[] input = new float[Socket.InputSize];
            float[] output = new float[Socket.OutputSize];
            for(int i=0; i<Socket.InputSize; i++) {
                input[i] = i.Ratio(0, Socket.InputSize-1).Normalize(0f,1f,-5f,5f);
                output[i] = Function(input[i]);
                targets.Add(output[i]);
            }
            Socket.Feed(input);
            Socket.Feed(output);
        }
        Socket.RunSession();
        float[] predictions = Socket.GetOutput();
        UltiDraw.PlotFunctions(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new float[][]{targets.ToArray(), predictions}, UltiDraw.Dimension.X, targets.Min(), targets.Max(), 0.001f, UltiDraw.Black, lineColors:new Color[]{UltiDraw.Red, UltiDraw.Green});
        UltiDraw.End();
    }
}