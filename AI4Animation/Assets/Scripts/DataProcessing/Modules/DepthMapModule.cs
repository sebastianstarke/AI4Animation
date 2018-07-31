#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class DepthMapModule : Module {

	public int Sensor = 0;
	public MotionData.AXIS Axis = MotionData.AXIS.ZPositive;
	public int Resolution = 20;
	public float Size = 10f;
	public float Distance = 10f;
	public LayerMask Mask = -1;

	public override TYPE Type() {
		return TYPE.DepthMap;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		MotionData.Hierarchy.Bone bone = Data.Source.FindBoneContains("Head");
		if(bone == null) {
			Debug.Log("Could not find depth map sensor.");
		} else {
			Sensor = bone.Index;
		}
		return this;
	}

	public DepthMap GetDepthMap(Frame frame, bool mirrored) {
		DepthMap depthMap = new DepthMap(Resolution, Size, Distance);
		Matrix4x4 pivot = frame.GetBoneTransformation(Sensor, mirrored);
		pivot *= Matrix4x4.TRS(Vector3.zero, Quaternion.FromToRotation(Vector3.forward, Data.GetAxis(Axis)), Vector3.one);
		depthMap.Sense(pivot, Mask);
		return depthMap;
	}

	public void Draw() {
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
	}

	public override void Draw(MotionEditor editor) {
		GetDepthMap(editor.GetCurrentFrame(), editor.Mirror).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Data.Source.GetNames());
		Axis = (MotionData.AXIS)EditorGUILayout.EnumPopup("Axis", Axis);
		Resolution = EditorGUILayout.IntField("Resolution", Resolution);
		Size = EditorGUILayout.FloatField("Size", Size);
		Distance = EditorGUILayout.FloatField("Distance", Distance);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
