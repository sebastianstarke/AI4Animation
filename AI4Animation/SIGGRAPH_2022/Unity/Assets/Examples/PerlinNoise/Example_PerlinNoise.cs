using UnityEngine;

public class Example_PerlinNoise : MonoBehaviour {

    public float Window = 1f;
    public int Samples = 121;
    public float Offset = 0f;

    public int Seed = 0;
    public int Octaves = 1;
    public float Amplitude = 1f;
    public float Frequency = 1f;

    void OnRenderObject() {
        UltiDraw.Begin();
        float[] coordinates = new float[Samples];
        for(int i=0; i<Samples; i++) {
            coordinates[i] = Window * i.Ratio(0, Samples-1);
        }
        float[] perlin = Perlin.Sample(coordinates, Seed, Octaves, Amplitude, Frequency, Offset);
        UltiDraw.PlotFunction(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), perlin, thickness:0.01f, yMin:-2.5f, yMax:2.5f);
        UltiDraw.End();
    }

}
