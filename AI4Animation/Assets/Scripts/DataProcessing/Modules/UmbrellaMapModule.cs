#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class UmbrellaMapModule : Module {

	public int Sensor = 0;
	public float Size = 0.25f;
	public LayerMask Mask = -1;

	public override TYPE Type() {
		return TYPE.UmbrellaMap;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		MotionData.Hierarchy.Bone bone = Data.Source.FindBoneContains("Hip");
		if(bone == null) {
			Debug.Log("Could not find umbrella map sensor.");
		} else {
			Sensor = bone.Index;
		}
		return this;
	}

	public UmbrellaMap GetUmbrellaMap(Frame frame, bool mirrored) {
		UmbrellaMap umbrellaMap = new UmbrellaMap(Size);
		Matrix4x4 pivot = frame.GetBoneTransformation(Sensor, mirrored);
		umbrellaMap.Sense(pivot, Mask);
		return umbrellaMap;
	}

	public override void Draw(MotionEditor editor) {
		GetUmbrellaMap(editor.GetCurrentFrame(), editor.Mirror).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Data.Source.GetNames());
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
