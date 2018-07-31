#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class KeypointModule : Module {

	public float Size = 1f;
	public LayerMask Mask = -1;

	public override TYPE Type() {
		return TYPE.Keypoint;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		return this;
	}

	public KeypointField GetKeypointField(Actor actor) {
		KeypointField keypointField = new KeypointField(actor, Size/2f, Mask);
		keypointField.Sense();
		return keypointField;
	}

	public override void Draw(MotionEditor editor) {
		GetKeypointField(editor.GetActor()).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
