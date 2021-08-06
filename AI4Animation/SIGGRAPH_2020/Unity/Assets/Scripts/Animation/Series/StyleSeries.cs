using UnityEngine;

public class StyleSeries : ComponentSeries {
    public string[] Styles;
    public float[][] Values;

    private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.9f, 0.2f, 0.1f);

    public StyleSeries(TimeSeries global, params string[] styles) : base(global) {
        Styles = styles;
        Values = new float[Samples.Length][];
        for(int i=0; i<Values.Length; i++) {
            Values[i] = new float[Styles.Length];
        }
    }

    public StyleSeries(TimeSeries global, string[] styles, float[] seed) : base(global) {
        Styles = styles;
        Values = new float[Samples.Length][];
        for(int i=0; i<Values.Length; i++) {
            Values[i] = new float[Styles.Length];
        }
        if(styles.Length != seed.Length) {
            Debug.Log("Given number of styles and seed do not match.");
            return;
        }
        for(int i=0; i<Values.Length; i++) {
            for(int j=0; j<Styles.Length; j++) {
                Values[i][j] = seed[j];
            }
        }
    }
    
    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            for(int j=0; j<Styles.Length; j++) {
                Values[i][j] = Values[i+1][j];
            }
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            for(int j=0; j<Styles.Length; j++) {
                Values[i][j] = Mathf.Lerp(Values[prevIndex][j], Values[nextIndex][j], weight);
            }
        }
    }

    public override void GUI(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.075f), Rect.GetSize(), 0.0175f, "Actions", UltiDraw.Black);
            Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
            for(int i=0; i<Styles.Length; i++) {
                float value = Values[Pivot][i];
                UltiDraw.OnGUILabel(new Vector2(Rect.X, value.Normalize(0f, 1f, Rect.Y-Rect.H/2f, Rect.Y+Rect.H/2f)), Rect.GetSize(), 0.0175f, Styles[i], colors[i]);
            }
            UltiDraw.End();
        }
    }

    public override void Draw(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), Values, UltiDraw.Dimension.Y, yMin: 0f, yMax: 1f, thickness: 0.0025f);
            // UltiDraw.GUIRectangle(new Vector2(0.875f, 0.685f), new Vector2(0.005f, 0.1f), UltiDraw.White.Opacity(0.5f));
            UltiDraw.End();
        }
    }

    public void SetStyle(int index, string name, float value) {
        int idx = ArrayExtensions.FindIndex(ref Styles, name);
        if(idx == -1) {
            // Debug.Log("Style " + name + " could not be found.");
            return;
        }
        Values[index][idx] = value;
    }

    public float GetStyle(int index, string name) {
        int idx = ArrayExtensions.FindIndex(ref Styles, name);
        if(idx == -1) {
            // Debug.Log("Style " + name + " could not be found.");
            return 0f;
        }
        return Values[index][idx];
    }

    public float[] GetStyles(int index, params string[] names) {
        float[] values = new float[names.Length];
        for(int i=0; i<names.Length; i++) {
            values[i] = GetStyle(index, names[i]);
        }
        return values;
    }
}
