#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class HeightMapModule : DataModule {

	public int Sensor = 0;
	public float Size = 0.25f;
	public LayerMask Mask = -1;
	public string[] Names = new string[0];

	public override TYPE Type() {
		return TYPE.HeightMap;
	}

	public override DataModule Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		MotionData.Hierarchy.Bone bone = Data.Source.FindBoneContains("Hip");
		if(bone == null) {
			Debug.Log("Could not find height map sensor.");
		} else {
			Sensor = bone.Index;
		}
		Names = new string[Data.Source.Bones.Length];
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			Names[i] = Data.Source.Bones[i].Name;
		}
		return this;
	}

	public HeightMap GetHeightMap(Frame frame, bool mirrored) {
		HeightMap heightMap = new HeightMap(Size);
		Matrix4x4 pivot = frame.GetBoneTransformation(Sensor, mirrored);
		heightMap.Sense(pivot, Mask);
		return heightMap;
	}

	//public void Draw() {
		/*
		UltiDraw.Begin();
		UltiDraw.DrawGUIRectangle(Vector2.one/2f, Vector2.one, UltiDraw.Mustard);
		UltiDraw.End();
		if(ShowDepthImage) {
			UltiDraw.Begin();
			Vector2 size = new Vector2(0.5f, 0.5f*Screen.width/Screen.height);
			for(int x=0; x<GetState().DepthMap.GetResolution(); x++) {
				for(int y=0; y<GetState().DepthMap.GetResolution(); y++) {
					float distance = Vector3.Distance(GetState().DepthMap.Points[GetState().DepthMap.GridToArray(x,y)], GetState().DepthMap.Pivot.GetPosition());
					float intensity = 1f - distance / GetState().DepthMap.GetDistance();
					UltiDraw.DrawGUIRectangle(Vector2.one/2f - size/2f + new Vector2((float)x*size.x, (float)y*size.y) / (GetState().DepthMap.GetResolution()-1), size / (GetState().DepthMap.GetResolution()-1), Color.Lerp(Color.black, Color.white, intensity));
				}
			}
			UltiDraw.End();
		}
		*/
	//}

	public override void Draw(MotionEditor editor) {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Names);
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
