using UnityEngine;

public class PhaseSeries : ComponentSeries {
    public string[] Bones;
    public float[][] Amplitudes;
    public float[][] Phases;
    public bool[] Active;

    private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.775f, 0.125f, 0.2f, 0.15f);
    // private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.1f, 0.075f, 0.8f, 0.125f);

    public PhaseSeries(TimeSeries global, params string[] bones) : base(global) {
        Bones = bones;
        Amplitudes = new float[Samples.Length][];
        Phases = new float[Samples.Length][];
        for(int i=0; i<Samples.Length; i++) {
            Amplitudes[i] = new float[bones.Length];
            Phases[i] = new float[bones.Length];
        }
        Active = new bool[bones.Length];
        Active.SetAll(true);
    }

    public float[] GetAlignment() {
        int pivot = 0;
        float[] alignment = new float[Bones.Length * KeyCount * 2];
		for(int k=0; k<KeyCount; k++) {
			int index = GetKey(k).Index;
			for(int b=0; b<Bones.Length; b++) {
				Vector2 phase = Active[b] ? Utility.PhaseVector(Phases[index][b], Amplitudes[index][b]) : Vector2.zero;
				alignment[pivot] = phase.x; pivot += 1;
                alignment[pivot] = phase.y; pivot += 1;
			}
		}
        return alignment;
    }

    public bool IsActive(params string[] bones) {
        for(int i=0; i<bones.Length; i++) {
            if(!Active[System.Array.FindIndex(Bones, x => x == bones[i])]) {
                return false;
            }
        }
        return true;
    }

    public float GetPhase(int index, int bone) {
        return Active[bone] ? Phases[index][bone] : 0f;
    }

    public float GetAmplitude(int index, int bone) {
        return Active[bone] ? Amplitudes[index][bone] : 0f;
    }

    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            for(int j=0; j<Bones.Length; j++) {
                Phases[i][j] = Phases[i+1][j];
                Amplitudes[i][j] = Amplitudes[i+1][j];
            }
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            for(int j=0; j<Bones.Length; j++) {
                Phases[i][j] = Utility.PhaseValue(Vector2.Lerp(Utility.PhaseVector(Phases[prevIndex][j]), Utility.PhaseVector(Phases[nextIndex][j]), weight).normalized);
                Amplitudes[i][j] = Mathf.Lerp(Amplitudes[prevIndex][j], Amplitudes[nextIndex][j], weight);
            }
        }
    }

    public override void GUI(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(Rect.W/2f, Rect.H+0.04f), Rect.ToScreen(new Vector2(1f, 0.25f)), 0.02f, "Phases", UltiDraw.Black);

            float xMin = Rect.X;
            float xMax = Rect.X + Rect.W;
            float yMin = Rect.Y;
            float yMax = Rect.Y + Rect.H;
            for(int b=0; b<Bones.Length; b++) {
                float w = (float)b/(float)(Bones.Length-1);
                float vertical = w.Normalize(0f, 1f, yMax, yMin);
                float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
                float border = 0.025f*(yMax-yMin)/(Bones.Length-1);
                if(!Active[b]) {
                    UltiDraw.OnGUILabel(new Vector2(0.5f*(xMin+xMax), vertical), new Vector2(xMax-xMin, height), 0.015f, "Disabled", UltiDraw.White, UltiDraw.None);
                }
            }
            UltiDraw.End();
        }
    }

    public override void Draw(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);

            //Vector-Space
            // {
            //     float xMin = Rect.X;
            //     float xMax = Rect.X + Rect.W;
            //     float yMin = Rect.Y;
            //     float yMax = Rect.Y + Rect.H;
            //     float amp = Amplitudes.Flatten().Max();
            //     for(int b=0; b<Bones.Length; b++) {
            //         float w = (float)b/(float)(Bones.Length-1);
            //         float vertical = w.Normalize(0f, 1f, yMax, yMin);
            //         float border = 0.025f*(yMax-yMin)/(Bones.Length-1);
            //         Color phaseColor = UltiDraw.White;
            //         for(int i=0; i<KeyCount; i++) {
            //             float ratio = (float)(i) / (float)(KeyCount-1);
            //             Vector2 center = new Vector2(xMin + ratio * (xMax - xMin), vertical);
            //             float size = 0.95f*(xMax-xMin)/(KeyCount-1);
            //             if(i < PivotKey) {
            //                 float phase = Phases[GetKey(i).Index][b];
            //                 float amplitude = Amplitudes[GetKey(i).Index][b];
            //                 Color color = phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f));
            //                 UltiDraw.PlotCircularPivot(center, size, 360f * phase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.DarkGrey, pivotColor: color);
            //             }
            //             if(i == PivotKey) {
            //                 float phase = Phases[GetKey(i).Index][b];
            //                 float amplitude = Amplitudes[GetKey(i).Index][b];
            //                 Color color = phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f));
            //                 UltiDraw.PlotCircularPivot(center, size, 360f * phase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.DarkGrey, pivotColor: color);

            //                 Vector2[] vectors = GetPhaseStabilization(Pivot, b);
            //                 float[] phases = new float[vectors.Length];
            //                 float[] amplitudes = new float[vectors.Length];
            //                 for(int v=0; v<vectors.Length; v++) {
            //                     phases[v] = 360f * Utility.PhaseValue(vectors[v]);
            //                     amplitudes[v] = vectors[v].magnitude.Normalize(0f, amp, 0f, 1f);
            //                 }
            //                 UltiDraw.PlotCircularPivots(center, size, phases, amplitudes, backgroundColor: UltiDraw.None, pivotColors: UltiDraw.GetRainbowColors(vectors.Length));
                            
            //                 // float stabilizedPhase = GetStabilizedPhase(Pivot, b, vectors, 0.5f);
            //                 // UltiDraw.PlotCircularPivot(center, size, 360f * stabilizedPhase, amplitude.Normalize(0f, amp, 0f, 1f), backgroundColor: UltiDraw.None, pivotColor: UltiDraw.Magenta);
            //             }
            //             if(i > PivotKey) {
            //                 float[] phases = new float[2]{
            //                     360f * Phases[GetKey(i).Index][b], 
            //                     360f * GetUpdateRate(b, Pivot, GetKey(i).Index, 1f/30f)
            //                 };
            //                 float[] amplitudes = new float[2]{
            //                     Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f), 
            //                     Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f)
            //                 };
            //                 Color[] colors = new Color[2]{
            //                     phaseColor.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f)),
            //                     UltiDraw.Red.Opacity(Amplitudes[GetKey(i).Index][b].Normalize(0f, amp, 0f, 1f))
            //                 };
            //                 UltiDraw.PlotCircularPivots(center, size, phases, amplitudes, backgroundColor: UltiDraw.DarkGrey, pivotColors: colors);
            //             }
            //         }
            //     }
            // }

            //Phase-Space
            {
                Color inactive = UltiDraw.Red.Opacity(0.5f);
                float xMin = Rect.X;
                float xMax = Rect.X + Rect.W;
                float yMin = Rect.Y;
                float yMax = Rect.Y + Rect.H;
                float[][] amplitudes;
                if(Active.All(true)) {
                    amplitudes = Amplitudes;
                } else {
                    amplitudes = (float[][])Amplitudes.Clone();
                    for(int i=0; i<Bones.Length; i++) {
                        if(!Active[i]) {
                            for(int j=0; j<SampleCount; j++) {
                                amplitudes[j][i] = 0f;
                            }
                        }
                    }
                }
                float amp = Amplitudes.Flatten().Max();
                for(int b=0; b<Bones.Length; b++) {
                    float[] values = new float[Samples.Length];
                    Color[] colors = new Color[Samples.Length];
                    for(int i=0; i<Samples.Length; i++) {
                        values[i] = GetPhase(i,b);
                        colors[i] = UltiDraw.Black.Opacity(GetAmplitude(i,b).Normalize(0f, amp, 0f, 1f));
                    }
                    float w = (float)b/(float)(Bones.Length-1);
                    float vertical = w.Normalize(0f, 1f, yMax, yMin);
                    float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
                    float border = 0.025f*(yMax-yMin)/(Bones.Length-1);
                    UltiDraw.PlotBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, yMin: 0f, yMax: 1f, barColors: colors, backgroundColor: Active[b] ? UltiDraw.White : inactive);
                    UltiDraw.PlotCircularPivot(new Vector2(0.5f * (xMin + xMax), vertical), 0.8f * height/2f, 360f*GetPhase(Pivot,b), GetAmplitude(Pivot,b).Normalize(0f, amp, 0f, 1f), backgroundColor: Active[b] ? UltiDraw.DarkGrey : inactive, pivotColor: colors[Pivot].Invert());
                    // UltiDraw.PlotCircularPivot(new Vector2(xMax + 0.8f * height/2f, vertical), 0.8f * height/2f, 360f*GetPhase(Pivot,b), GetAmplitude(Pivot,b).Normalize(0f, amp, 0f, 1f), backgroundColor: Active[b] ? UltiDraw.DarkGrey : inactive, pivotColor: colors[Pivot].Invert());
                }
                {
                    float vertical = yMin - 1f*(yMax-yMin)/(Bones.Length-1);
                    float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
                    UltiDraw.PlotFunctions(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), amplitudes, UltiDraw.Dimension.Y, yMin: 0f, yMax: amp);
                }
            }

            UltiDraw.End();
        }
    }
}