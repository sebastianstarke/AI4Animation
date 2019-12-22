#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class DepthMapModule : Module {

	public int Sensor = 0;
	public Axis Axis = Axis.ZPositive;
	public int Resolution = 20;
	public float Size = 10f;
	public float Distance = 10f;
	public LayerMask Mask = -1;
	public bool ShowImage = false;

	public override ID GetID() {
		return ID.DepthMap;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		MotionData.Hierarchy.Bone bone = Data.Source.FindBoneContains("Head");
		if(bone == null) {
			Debug.Log("Could not find depth map sensor.");
		} else {
			Sensor = bone.Index;
		}
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		
	}

	public DepthMap GetDepthMap(Frame frame, bool mirrored) {
		DepthMap sensor = new DepthMap(Resolution, Size, Distance);
		Matrix4x4 pivot = frame.GetBoneTransformation(Sensor, mirrored);
		pivot *= Matrix4x4.TRS(Vector3.zero, Quaternion.FromToRotation(Vector3.forward, Axis.GetAxis()), Vector3.one);
		sensor.Sense(pivot, Mask);
		return sensor;
	}

	protected override void DerivedDraw(MotionEditor editor) {
		DepthMap sensor = GetDepthMap(editor.GetCurrentFrame(), editor.Mirror);
		sensor.Draw();
		if(ShowImage) {
			UltiDraw.Begin();
			UltiDraw.DrawGUIRectangle(Vector2.one/2f, Vector2.one, UltiDraw.Mustard);
			Vector2 size = new Vector2(0.5f, 0.5f*Screen.width/Screen.height);
			for(int x=0; x<sensor.GetResolution(); x++) {
				for(int y=0; y<sensor.GetResolution(); y++) {
					float distance = Vector3.Distance(sensor.Points[sensor.GridToArray(x,y)], sensor.Pivot.GetPosition());
					float intensity = 1f - distance / sensor.GetDistance();
					UltiDraw.DrawGUIRectangle(
						Vector2.one/2f - size/2f + new Vector2((float)x*size.x, 
						(float)y*size.y) / (sensor.GetResolution()-1), 
						size / (sensor.GetResolution()-1), 
						Color.Lerp(Color.black, Color.white, intensity)
					);
				}
			}
			UltiDraw.End();
		}
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Sensor = EditorGUILayout.Popup("Sensor", Sensor, Data.Source.GetBoneNames());
		Axis = (Axis)EditorGUILayout.EnumPopup("Axis", Axis);
		Resolution = EditorGUILayout.IntField("Resolution", Resolution);
		Size = EditorGUILayout.FloatField("Size", Size);
		Distance = EditorGUILayout.FloatField("Distance", Distance);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
		ShowImage = EditorGUILayout.Toggle("Show Image", ShowImage);
	}

}
#endif
