using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GaitProfile : MonoBehaviour {


	public int Frames = 150;
	public Transform[] Feet;
	public float Threshold = 0.03f;

	public Vector2 Center = new Vector2(0.5f, 0.5f);
	public float Width = 600f;
	public float Height = 50f;
	public float Border = 10f;
	public float Thickness = 10f;

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
		float totalHeight = Feet.Length * Height + (Feet.Length+1) * Border/2f; 
		float totalWidth = Width + Border;
		for(int i=0; i<Feet.Length; i++) {
			UltiDraw.DrawSphere(Feet[i].transform.position, Quaternion.identity, 0.075f, colors[i]);
		}
		UltiDraw.DrawGUIRectangle(Center, new Vector2(totalWidth/Screen.width, totalHeight/Screen.height), UltiDraw.DarkGrey);
		float pivot = 0.5f * totalHeight;
		for(int i=1; i<=Feet.Length; i++) {
			pivot -= Height/2f + Border/2f;
			UltiDraw.DrawGUIRectangle(Center + new Vector2(0f, pivot)/Screen.height, new Vector2(Width/Screen.width, Height/Screen.height), UltiDraw.White);
			for(int j=0; j<Positions[i-1].Count; j++) {
				float p = (float)j/(float)(Positions[i-1].Count-1);
				p = Utility.Normalise(p, 0f, 1f, 0.5f*Thickness/Width, (Width-0.5f*Thickness)/Width);
				float x = Center.x - 0.5f*Width/Screen.width + p*Width/Screen.width;
				float yTop = pivot + Height/2f;
				float yBot = pivot - Height/2f;
				float h = Positions[i-1][j].y;
				if(h < Threshold) {
					float weight = 1f; //Utility.Exponential01(1f - h / Threshold);
					UltiDraw.DrawGUILine(new Vector2(x, Center.y + yTop/Screen.height), new Vector2(x, Center.y + yBot/Screen.height), Thickness/Screen.width, colors[i-1].Transparent(weight));
				}
			}
			pivot -= Height/2f;
		}

		UltiDraw.End();
	}

}