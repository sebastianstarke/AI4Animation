#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class HeightMapModule : Module {

	public int Sensor = 0;
	public float Size = 0.25f;
	public LayerMask Mask = -1;
	public string[] Names = new string[0];

	public override TYPE Type() {
		return TYPE.HeightMap ;
	}

	public override Module Initialise(MotionData data) {
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
		Matrix4x4 pivot = frame.GetRootTransformation(mirrored);
		heightMap.Sense(pivot, Mask);
		return heightMap;
	}

	public override void Draw(MotionEditor editor) {
		GetHeightMap(editor.GetCurrentFrame(), editor.Mirror).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Names);
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
