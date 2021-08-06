#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class SphereMapModule : Module {

    public float Radius = 1f;
	public LayerMask Mask = -1;

	public override ID GetID() {
		return ID.SphereMap;
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
		SphereMap sensor = GetSphereMap(editor.GetCurrentFrame(), editor.Mirror);
		sensor.Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
        Radius = EditorGUILayout.FloatField("Radius", Radius);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

	public SphereMap GetSphereMap(Frame frame, bool mirrored) {
		SphereMap sensor = new SphereMap(Radius, Mask);
		RootModule module = Data.GetModule<RootModule>();
		sensor.Sense(module == null ? Matrix4x4.identity : module.GetRootTransformation(frame.Timestamp, mirrored));
		return sensor;
	}

}
#endif
