#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class SphereMapModule : Module {

    public float Radius = 1f;
	public LayerMask Mask = -1;

	public override ID GetID() {
		return ID.SphereMap;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		
	}

	public SphereMap GetSphereMap(Frame frame, bool mirrored) {
		SphereMap sensor = new SphereMap(Radius, Mask);
		RootModule module = (RootModule)Data.GetModule(ID.Root);
		sensor.Sense(module == null ? Matrix4x4.identity : module.GetRootTransformation(frame, mirrored));
		return sensor;
	}

	protected override void DerivedDraw(MotionEditor editor) {
		SphereMap sensor = GetSphereMap(editor.GetCurrentFrame(), editor.Mirror);
		sensor.Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
        Radius = EditorGUILayout.FloatField("Radius", Radius);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
