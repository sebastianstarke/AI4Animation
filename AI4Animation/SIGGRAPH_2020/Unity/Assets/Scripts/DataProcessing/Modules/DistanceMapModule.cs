#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class DistanceMapModule : Module {

	public Vector3 Size = Vector3.one;
	public Vector3Int Resolution = new Vector3Int(10, 10, 10);
	public LayerMask Mask = -1;
	public Color Color = UltiDraw.Cyan;
	public bool DrawReferences = false;
	public bool DrawDistribution = false;
    public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.1f, 0.9f, 0.1f);

	public override ID GetID() {
		return ID.DistanceMap;
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
		DistanceMap sensor = GetDistanceMap(editor.GetCurrentFrame(), editor.Mirror);
		sensor.Draw(Color);
		if(DrawReferences) {
			sensor.DrawReferences();
		}
		if(DrawDistribution) {
			sensor.DrawDistribution(Color, Rect);
		}
    }

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.Vector3Field("Size", Size);
		Resolution = EditorGUILayout.Vector3IntField("Resolution", Resolution);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
		Color = EditorGUILayout.ColorField("Color", Color);
		DrawReferences = EditorGUILayout.Toggle("Draw References", DrawReferences);
		DrawDistribution = EditorGUILayout.Toggle("Draw Distribution", DrawDistribution);
        Rect.Inspector();
		EditorGUILayout.LabelField("Samples: " + 0);
	}

	public DistanceMap GetDistanceMap(Frame frame, bool mirrored) {
		DistanceMap sensor = new DistanceMap(Resolution);
		RootModule module = Data.GetModule<RootModule>();
		if(module != null) {
			Vector3 position = module.GetRootPosition(frame.Timestamp, mirrored);
			Quaternion rotation = module.GetRootRotation(frame.Timestamp, mirrored);
			sensor.Sense(Matrix4x4.TRS(position + new Vector3(0f, 0.5f*Size.y, 0f), rotation, Vector3.one), Mask, Size);
		} else {
			sensor.Sense(frame.GetBoneTransformation(0, mirrored), Mask, Size);
		}
		return sensor;
	}

}
#endif
