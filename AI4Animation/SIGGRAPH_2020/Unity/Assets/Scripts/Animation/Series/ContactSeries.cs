using UnityEngine;

public class ContactSeries : ComponentSeries {
    public string[] Bones;
    public float[][] Values;
    
    private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.7f, 0.2f, 0.1f);

    public ContactSeries(TimeSeries global, params string[] bones) : base(global) {
        Bones = bones;
        Values = new float[Samples.Length][];
        for(int i=0; i<Values.Length; i++) {
            Values[i] = new float[Bones.Length];
        }
    }

    public float[] GetContacts(int index) {
        return GetContacts(index, Bones);
    }

    public float[] GetContacts(string bone) {
        for(int i=0; i<Bones.Length; i++) {
            if(Bones[i] == bone) {
                float[] contacts = new float[Values.Length];
                for(int j=0; j<contacts.Length; j++) {
                    contacts[j] = Values[j][i];
                }
                return contacts;
            }
        }
        return null;
    }

    public float[] GetContacts(int index, params string[] bones) {
        float[] values = new float[bones.Length];
        for(int i=0; i<bones.Length; i++) {
            values[i] = GetContact(index, bones[i]);
        }
        return values;
    }

    public float GetContact(int index, string bone) {
        int idx = ArrayExtensions.FindIndex(ref Bones, bone);
        if(idx == -1) {
            Debug.Log("Contact " + bone + " could not be found.");
            return 0f;
        }
        return Values[index][idx];
    }

    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            for(int j=0; j<Bones.Length; j++) {
                Values[i][j] = Values[i+1][j];
            }
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            for(int j=0; j<Bones.Length; j++) {
                Values[i][j] = Mathf.Lerp(Values[prevIndex][j], Values[nextIndex][j], weight);
            }
        }
    }

    public override void GUI(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.13f), Rect.GetSize(), 0.0175f, "Contacts", UltiDraw.Black);
            UltiDraw.End();
        }
    }

    public override void Draw(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            float[] function = new float[Samples.Length];
            Color[] colors = UltiDraw.GetRainbowColors(Bones.Length);
            for(int i=0; i<Bones.Length; i++) {
                for(int j=0; j<function.Length; j++) {
                    function[j] = Values[j][Bones.Length-1-i];
                }
                UltiDraw.PlotBars(new Vector2(Rect.X, Rect.Y + (i+1)*Rect.H/Bones.Length), new Vector2(Rect.W, Rect.H/Bones.Length), function, yMin: 0f, yMax: 1f, barColor : colors[i]);
            }
            UltiDraw.End();
        }
    }
}