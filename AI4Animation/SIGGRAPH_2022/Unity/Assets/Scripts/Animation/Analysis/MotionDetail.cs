#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MotionDetail : MonoBehaviour {

	public int Frames = 100;
	public float LineStrength = 0.00125f;
	public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.9f, 0.5f, 0.2f);
	public Character[] Characters = new Character[0];
	public bool AutoMax = true;

	public int Framerate = 60;

    [System.Serializable]
    public class Character {
        public Actor Actor;
        public Pair[] Transforms;
        public void Update() {
            if(Actor == null) {
                Transforms = new Pair[0];
                return;
            }
            if(Transforms.Length != Actor.Bones.Length) {
                Transforms = new Pair[Actor.Bones.Length];
                for(int i=0; i<Actor.Bones.Length; i++) {
                    Transforms[i] = new Pair();
                    Transforms[i].Bone = Actor.Bones[i].GetTransform();
                    Transforms[i].Active = true;
                }
            }
        }

        [System.Serializable]
        public class Pair {
            public Transform Bone;
            public bool Active;
        }
    }

	private List<Matrix4x4[]> PreviousTransformations;
	private List<List<float>> Values;

	private float Max = 0f;

    void OnValidate() {
        foreach(Character character in Characters) {
            character.Update();
        }
    }

	void Start() {
		PreviousTransformations = new List<Matrix4x4[]>();
		Values = new List<List<float>>();
		for(int i=0; i<Characters.Length; i++) {
			PreviousTransformations.Add(new Matrix4x4[Characters[i].Actor.Bones.Length]);
			for(int j=0; j<PreviousTransformations[i].Length; j++) {
				PreviousTransformations[i][j] = Characters[i].Actor.Bones[j].GetTransform().GetLocalMatrix();
			}
			Values.Add(new List<float>());
		}
	}

	void LateUpdate() {
		for(int i=0; i<Characters.Length; i++) {
			float value = 0f;
			for(int j=0; j<Characters[i].Actor.Bones.Length; j++) {
                if(Characters[i].Transforms[j].Active) {
                    Matrix4x4 transformation = Characters[i].Actor.Bones[j].GetTransform().GetLocalMatrix();
                    value += Quaternion.Angle(PreviousTransformations[i][j].GetRotation(), transformation.GetRotation());
                    PreviousTransformations[i][j] = transformation;
                }
			}
			value /= Characters[i].Actor.Bones.Length;
			value *= Framerate;
            Max = Mathf.Max(Max, value);
            Values[i].Add(value);
			while(Values[i].Count > Frames) {
				Values[i].RemoveAt(0);
			}
		}
	}

	public float[][] GetValues() {
		float[][] values = new float[Values.Count][];
		for(int i=0; i<values.Length; i++) {
			values[i] = Values[i].ToArray();
		}
		return values;
	}

	public float[] GetValues(int actor) {
		return Values[actor].ToArray();
	}

	void OnRenderObject() {
		UltiDraw.Begin();
        if(AutoMax) {
		    UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), GetValues(), UltiDraw.Dimension.X, 0f, Max, thickness:LineStrength, backgroundColor:UltiDraw.Black);
        } else {
		    UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), GetValues(), UltiDraw.Dimension.X, thickness:LineStrength, backgroundColor:UltiDraw.Black);
        }
		UltiDraw.End();
	}

	void OnGUI() {
		float size = 0.05f;
		UltiDraw.Begin();
		float[][] values = GetValues();
		for(int i=0; i<Characters.Length; i++) {
			float mean = values[i].Mean();
			float sigma = values[i].Sigma();
			UltiDraw.OnGUILabel(new Vector2(Rect.X - 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, mean.Round(1).ToString(), UltiDraw.Black);
			UltiDraw.OnGUILabel(new Vector2(Rect.X, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/4f, Characters[i].Actor.name, UltiDraw.GetRainbowColor(i, Characters.Length));
			UltiDraw.OnGUILabel(new Vector2(Rect.X + 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, sigma.Round(1).ToString(), UltiDraw.Black);
		}
		UltiDraw.End();
	}
}
#endif