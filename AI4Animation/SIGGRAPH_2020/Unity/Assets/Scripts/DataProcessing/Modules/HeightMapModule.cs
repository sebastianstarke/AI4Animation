#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class HeightMapModule : Module {

	public float Size = 1f;
	public int Resolution = 25;
	public LayerMask Mask = -1;

	public override ID GetID() {
		return ID.HeightMap;
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
		HeightMap sensor = GetHeightMap(editor.GetActor());
		sensor.Draw();
		sensor.Render(new Vector2(0.1f, 0.25f), new Vector2(0.3f*Screen.height/Screen.width, 0.3f), Resolution, Resolution, 1f);
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.FloatField("Size", Size);
		Resolution = EditorGUILayout.IntField("Resolution", Resolution);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

	public HeightMap GetHeightMap(Actor actor) {
		HeightMap sensor = new HeightMap(Size, Resolution, Mask);
		sensor.Sense(actor.GetRoot().GetWorldMatrix());
		return sensor;
	}

}
#endif
