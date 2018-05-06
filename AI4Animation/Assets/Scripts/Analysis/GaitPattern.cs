using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GaitPattern : MonoBehaviour {

	public int Frames = 100;
	public float Threshold = 0.025f;
	public float Thickness = 0.01f;
	public Transform[] Feet;
	public UltiDraw.GUIRect Rect;

	private List<Vector3>[] Positions;

	void Awake() {
		Positions = new List<Vector3>[Feet.Length];
		for(int i=0; i<Positions.Length; i++) {
			Positions[i] = new List<Vector3>();
		}
	}

	void Start() {

	}

	void Update() {
		for(int i=0; i<Positions.Length; i++) {
			while(Positions[i].Count >= Frames) {
				Positions[i].RemoveAt(0);
			}
			Positions[i].Add(Feet[i].position);
		}
	}

	void OnRenderObject() {
		UltiDraw.Begin();

		Color[] colors = UltiDraw.GetRainbowColors(Feet.Length);
		for(int i=0; i<colors.Length; i++) {
			colors[i] = colors[i].Darken(0.25f).Transparent(0.75f);
		}
		for(int i=0; i<Feet.Length; i++) {
			UltiDraw.DrawSphere(Feet[i].transform.position, Quaternion.identity, 0.075f, colors[i]);
		}

		float border = 0.01f;
		float width = Rect.W;
		float height = Feet.Length * Rect.H + (Feet.Length-1) * border/2f; 

		UltiDraw.DrawGUIRectangle(Rect.GetPosition(), new Vector2(width, height), UltiDraw.DarkGrey.Transparent(0.75f), 0.5f*border, UltiDraw.BlackGrey);
		float pivot = 0.5f * height;
		for(int i=1; i<=Feet.Length; i++) {
			pivot -= Rect.H/2f;
			UltiDraw.DrawGUIRectangle(Rect.GetPosition() + new Vector2(0f, pivot), new Vector2(Rect.W, Rect.H), UltiDraw.White.Transparent(0.5f));
			for(int j=0; j<Positions[i-1].Count; j++) {
				float p = (float)j/(float)(Positions[i-1].Count-1);
				p = Utility.Normalise(p, 0f, 1f, 0.5f*Thickness/Rect.W, (Rect.W-0.5f*Thickness)/Rect.W);
				float x = Rect.X - 0.5f*Rect.W + p*Rect.W;
				float yTop = pivot + Rect.H/2f;
				float yBot = pivot - Rect.H/2f;
				float h = Positions[i-1][j].y;
				if(h < Threshold) {
					UltiDraw.DrawGUILine(new Vector2(x, Rect.Y + yTop), new Vector2(x, Rect.Y + yBot), Thickness, colors[i-1]);
				}
			}
			pivot -= border/2f;
			pivot -= Rect.H/2f;
		}

		UltiDraw.End();
	}

}