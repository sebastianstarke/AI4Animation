using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Example_PhaseActivation : MonoBehaviour {

    public int Experts = 4;
    [Range(0f,1f)] public List<float> Phases;

    public float Amplitude = 1f;
    public float Thickness = 0.0025f;

    private const int Resolution = 100;

    void OnRenderObject() {
        UltiDraw.Begin();
        for(int i=0; i<Phases.Count; i++) {
            float w = 0.75f;
            float hMin = 0.25f;
            float hMax = 0.75f;
            float step = (hMax-hMin) / Phases.Count;
            Vector2 center = new Vector2(0.5f, i.Ratio(0, Phases.Count-1).Normalize(0f, 1f, hMin+step/2f, hMax-step/2f));
            Vector2 size = new Vector2(w, step);
            List<float[]> functions = new List<float[]>();
            for(int k=0; k<Resolution; k++) {
                functions.Add(GetExpertActivation(Phases[i] + k.Ratio(0, Resolution-1), Experts));
            }
            UltiDraw.PlotFunctions(center, size, functions.ToArray(), UltiDraw.Dimension.Y, 0f, 1f, thickness:Thickness);
        }
        UltiDraw.End();
    }

    public float[] GetExpertActivation(float phase, int experts) {
        float[] activation = new float[experts];
        for(int j=0; j<experts; j++) {
            float amplitude = 1f;
            float angle = 2f*Mathf.PI*(phase + (float)j/(float)experts);
            float value = amplitude * Mathf.Sin(angle);
            value = 0.5f * value + 0.5f;
            value = Mathf.Pow(value, experts-1);
            activation[j] = value;
        }
        return activation;
    }
}
