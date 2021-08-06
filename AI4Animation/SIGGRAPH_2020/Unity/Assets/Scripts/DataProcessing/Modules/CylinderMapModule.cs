#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class CylinderMapModule : Module {

	public float Size = 1f;
	public int Resolution = 10;
	public int Layers = 10;
	public bool Overlap = true;
	public LayerMask Mask = -1;
	public Color Color = UltiDraw.Cyan;
	public bool DrawReferences = false;
	public bool DrawDistribution = false;

	private int Samples = 0;

	public override ID GetID() {
		return ID.CylinderMap;
	}

    public override void DerivedResetPrecomputation() {

    }

    public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
		return null;
    }

	protected override void DerivedInitialize() {

	}

	protected override void DerivedLoad(MotionEditor editor) {
		
    }

	protected override void DerivedCallback(MotionEditor editor) {
		
	}

	protected override void DerivedGUI(MotionEditor editor) {
		
	}

	protected override void DerivedDraw(MotionEditor editor) {
		CylinderMap sensor = GetCylinderMap(editor.GetCurrentFrame(), editor.Mirror);
		sensor.Draw(Color, DrawReferences, DrawDistribution);
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.FloatField("Size", Size);
		Resolution = Mathf.Clamp(EditorGUILayout.IntField("Resolution", Resolution), 1, 25);
		Layers = Mathf.Clamp(EditorGUILayout.IntField("Layers", Layers), 1, 25);
		Overlap = EditorGUILayout.Toggle("Overlap", Overlap);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
		Color = EditorGUILayout.ColorField("Color", Color);
		DrawReferences = EditorGUILayout.Toggle("Draw References", DrawReferences);
		DrawDistribution = EditorGUILayout.Toggle("Draw Distribution", DrawDistribution);
		EditorGUILayout.LabelField("Samples: " + Samples);
	}

	public CylinderMap GetCylinderMap(Frame frame, bool mirrored) {
		CylinderMap sensor = new CylinderMap(Size, Resolution, Layers, Overlap);
		RootModule module = Data.GetModule<RootModule>();
		if(module != null) {
			Vector3 position = module.GetRootPosition(frame.Timestamp, mirrored);
			Quaternion rotation = module.GetRootRotation(frame.Timestamp, mirrored);
			sensor.Sense(Matrix4x4.TRS(position + new Vector3(0f, 0f, 0f), rotation, Vector3.one), Mask);
		} else {
			sensor.Sense(frame.GetBoneTransformation(0, mirrored), Mask);
		}
		Samples = sensor.Points.Length;
		return sensor;
	}

}
#endif
