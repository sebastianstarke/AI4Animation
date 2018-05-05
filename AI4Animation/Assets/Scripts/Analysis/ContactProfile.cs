#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class ContactProfile : MonoBehaviour {

	public float Time = 1f;	
	public float[] Thresholds = new float[4];
	public int[] Screenshots = new int[4];
	public int[] Indices = new int[4];

	public float Width = 1600f;
	public float Height = 100f;
	public float Border = 15f;
	public float Thickness = 30f;

	private MotionEditor Editor;

	private MotionEditor GetEditor() {
		if(Editor == null) {
			Editor = GetComponent<MotionEditor>();
		}
		return Editor;
	}

	void OnRenderObject() {
		MotionData.Frame[] frames = GetEditor().Data.GetFrames(GetEditor().GetState().Timestamp, Mathf.Clamp(GetEditor().GetState().Timestamp + Time, 0f, GetEditor().Data.GetTotalTime()));
		if(frames.Length == 1) {
			return;
		}

		Screenshots = new int[4];
		Screenshots[0] = GetEditor().Data.GetFrame(Mathf.Clamp(GetEditor().GetState().Timestamp + 0f*Time, 0f, GetEditor().Data.GetTotalTime())).Index;
		Screenshots[1] = GetEditor().Data.GetFrame(Mathf.Clamp(GetEditor().GetState().Timestamp + 0.25f*Time, 0f, GetEditor().Data.GetTotalTime())).Index;
		Screenshots[2] = GetEditor().Data.GetFrame(Mathf.Clamp(GetEditor().GetState().Timestamp + 0.5f*Time, 0f, GetEditor().Data.GetTotalTime())).Index;
		Screenshots[3] = GetEditor().Data.GetFrame(Mathf.Clamp(GetEditor().GetState().Timestamp + 0.75f*Time, 0f, GetEditor().Data.GetTotalTime())).Index;

		UltiDraw.Begin();

		UltiDraw.DrawGUIRectangle(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), UltiDraw.White);

		Indices[0] = GetEditor().Data.Source.FindBone("LeftHandSite").Index;
		Indices[1] = GetEditor().Data.Source.FindBone("RightHandSite").Index;
		Indices[2] = GetEditor().Data.Source.FindBone("LeftFootSite").Index;
		Indices[3] = GetEditor().Data.Source.FindBone("RightFootSite").Index;
		Color[] colors = UltiDraw.GetRainbowColors(Indices.Length);

		float totalHeight = Indices.Length * Height + (Indices.Length+1) * Border/2f; 
		float totalWidth = Width + Border;
		Vector2 center = new Vector2(0.5f, 0.5f);

		for(int i=0; i<Indices.Length; i++) {
			UltiDraw.DrawSphere(GetEditor().GetState().BoneTransformations[Indices[i]].GetPosition(), Quaternion.identity, 0.075f, colors[i]);
		}

		UltiDraw.DrawGUIRectangle(center, new Vector2(totalWidth/Screen.width, totalHeight/Screen.height), UltiDraw.DarkGrey);

		float pivot = 0.5f * totalHeight;
		for(int i=1; i<=Indices.Length; i++) {
			pivot -= Height/2f + Border/2f;
			UltiDraw.DrawGUIRectangle(center + new Vector2(0f, pivot)/Screen.height, new Vector2(Width/Screen.width, Height/Screen.height), UltiDraw.White);
			for(int j=0; j<frames.Length; j++) {
				float p = (float)j/(float)(frames.Length-1);
				p = Utility.Normalise(p, 0f, 1f, 0.5f*Thickness/Width, (Width-0.5f*Thickness)/Width);
				float x = center.x - 0.5f*Width/Screen.width + p*Width/Screen.width;
				float yTop = pivot + Height/2f;
				float yBot = pivot - Height/2f;
				float h = frames[j].GetBoneTransformation(Indices[i-1], GetEditor().IsMirror()).GetPosition().y;
				if(h < Thresholds[i-1]) {
					UltiDraw.DrawGUILine(new Vector2(x, center.y + yTop/Screen.height), new Vector2(x, center.y + yBot/Screen.height), Thickness/Screen.width, colors[i-1]);
				}
			}
			pivot -= Height/2f;
		}

		UltiDraw.End();
	}

}
#endif