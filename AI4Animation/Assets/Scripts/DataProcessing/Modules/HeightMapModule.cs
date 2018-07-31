#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class HeightMapModule : Module {

	public int Sensor = 0;
	public float Size = 1f;
	public LayerMask Mask = -1;

	public override TYPE Type() {
		return TYPE.HeightMap;
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
		return this;
	}

	public HeightMap GetHeightMap(Actor actor) {
		HeightMap heightMap = new HeightMap(Size, Mask);
		heightMap.Sense(actor.GetRoot().GetWorldMatrix());
		return heightMap;
	}

	public override void Draw(MotionEditor editor) {
		GetHeightMap(editor.GetActor()).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Data.Source.GetNames());
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
