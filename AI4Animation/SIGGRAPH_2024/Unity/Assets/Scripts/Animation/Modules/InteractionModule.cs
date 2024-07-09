using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class InteractionModule : Module {
		public string ObjectName = string.Empty;
		public int ObjectBone = 0;
		public Matrix4x4[] Override = null;

		private int Start;
		private int End;
		private int Frame;

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
		}
#if UNITY_EDITOR
		protected override void DerivedInitialize() {
			
		}

		protected override void DerivedLoad(MotionEditor editor) {

		} 

		protected override void DerivedUnload(MotionEditor editor) {

		}
		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
			
		}
		protected override void DerivedCallback(MotionEditor editor) {
			GameObject instance = Asset.GetSceneGameObject(ObjectName);
			if(instance != null) {
				Matrix4x4 m = GetTransformation(editor.GetTimestamp(), editor.Mirror);
				instance.transform.position = m.GetPosition();
				instance.transform.rotation = m.GetRotation();
			}
		}

		protected override void DerivedInspector(MotionEditor editor) {
			ObjectName = EditorGUILayout.TextField("Object Name", ObjectName);
			ObjectBone = EditorGUILayout.Popup("Object Bone", ObjectBone, Asset.Source.GetBoneNames());

			Start = EditorGUILayout.IntField("Start", Start);
			End = EditorGUILayout.IntField("End", End);
			Frame = EditorGUILayout.IntField("Frame", Frame);
			if(Utility.GUIButton("Override", UltiDraw.DarkGrey, UltiDraw.White)) {
				Override = new Matrix4x4[Asset.Frames.Length];
				for(int i=0; i<Asset.Frames.Length; i++) {
					Frame frame = Asset.Frames[i];
					if(frame.Index >= Start && frame.Index <= End) {
						Override[i] = GetRawTransformation(Asset.GetFrame(Frame).Timestamp, editor.Mirror);
					} else {
						Override[i] = GetRawTransformation(Asset.Frames[i].Timestamp, editor.Mirror);
					}
				}
			}
			if(Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White)) {
				Override = null;
			}
		}
#endif

		private Matrix4x4 GetTransformation(float timestamp, bool mirrored) {
			if(Override.Verify(Asset.Frames.Length)) {
				return Override[Asset.GetFrame(timestamp).Index-1];
			}
			return GetRawTransformation(timestamp, mirrored);
		}

		private Matrix4x4 GetRawTransformation(float timestamp, bool mirrored) {
			SmoothingModule module = Asset.GetModule<SmoothingModule>();
			if(module != null) {
				return module.GetBoneTransformation(timestamp, mirrored, ObjectBone);
			} else {
				return Asset.GetFrame(timestamp).GetBoneTransformation(ObjectBone, mirrored);
			}
		}
	}
}
