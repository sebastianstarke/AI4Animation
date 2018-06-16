#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class ContactModule : DataModule {

	public ContactFunction[] Functions = new ContactFunction[0];
	public string[] Names = new string[0];

	public override TYPE Type() {
		return TYPE.Contact;
	}

	public override DataModule Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		Names = new string[Data.Source.Bones.Length];
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			Names[i] = Data.Source.Bones[i].Name;
		}
		return this;
	}

	public void Compute() {
		for(int i=0; i<Functions.Length; i++) {
			Functions[i].Compute();
		}
	}
	
	public override void Draw() {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		EditorGUILayout.BeginHorizontal();
		if(Utility.GUIButton("Add Contact", UltiDraw.DarkGrey, UltiDraw.White)) {
			ArrayExtensions.Add(ref Functions, new ContactFunction(this));
		}
		if(Utility.GUIButton("Remove Contact", UltiDraw.DarkGrey, UltiDraw.White)) {
			ArrayExtensions.Shrink(ref Functions);
		}
		EditorGUILayout.EndHorizontal();
		if(Utility.GUIButton("Compute", UltiDraw.DarkGrey, UltiDraw.White)) {
			Compute();
		}
		for(int i=0; i<Functions.Length; i++) {
			Functions[i].Inspector(editor);
		}
	}

	[System.Serializable]
	public class ContactFunction {
		public ContactModule Module = null;
		public int Sensor = 0;
		public float Threshold = 0.1f;
		public Vector3 Normal = Vector3.down;
		public LayerMask Mask = -1;
		public bool[] RegularContacts = new bool[0];
		public bool[] MirroredContacts = new bool[0];

		public ContactFunction(ContactModule module) {
			Module = module;
			Sensor = 0;
			RegularContacts = new bool[Module.Data.GetTotalFrames()];
			MirroredContacts = new bool[Module.Data.GetTotalFrames()];
			Compute();
		}

		public void SetSensor(int index) {
			if(Sensor != index) {
				Sensor = index;
				Compute();
			}
		}

		public void SetThreshold(float value) {
			if(Threshold != value) {
				Threshold = value;
				Compute();
			}
		}

		public void SetNormal(Vector3 normal) {
			if(Normal != normal) {
				Normal = normal;
				Compute();
			}
		}

		public void SetMask(LayerMask mask) {
			if(Mask != mask) {
				Mask = mask;
				Compute();
				Debug.Log("COMPUTING MASK");
			}
		}

		public void Compute() {
			for(int i=0; i<Module.Data.GetTotalFrames(); i++) {
				Vector3 position = Module.Data.Frames[i].GetBoneTransformation(Sensor, false).GetPosition();
				RegularContacts[i] = Physics.CheckSphere(position, Threshold, Mask.value);
			}
		}

		public void Inspector(MotionEditor editor) {
			UltiDraw.Begin();

			UltiDraw.DrawSphere(Vector3.zero, Quaternion.identity, 1f, Color.red);

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Frame frame = Module.Data.GetFrame(editor.GetState().Index);

				SetSensor(EditorGUILayout.Popup("Sensor", Sensor, Module.Names));
				SetThreshold(EditorGUILayout.FloatField("Threshold", Threshold));
				SetNormal(EditorGUILayout.Vector3Field("Normal", Normal));
				SetMask(InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers)));

				EditorGUILayout.BeginVertical(GUILayout.Height(50f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				int start = 1;
				int end = Module.Data.GetTotalFrames();
				int elements = end-start;
				Vector2 top = new Vector3(0f, rect.yMax - rect.height, 0f);
				Vector2 bottom = new Vector3(0f, rect.yMax, 0f);

				//Contacts
				for(int i=start; i<=end; i++) {
					top.x = rect.xMin + (float)(i-start)/elements * rect.width;
					bottom.x = rect.xMin + (float)(i-start)/elements * rect.width;
					UltiDraw.DrawLine(top, bottom, RegularContacts[i-1] ? UltiDraw.Green : UltiDraw.Red);
				}

				//Sequences
				for(int i=0; i<Module.Data.Sequences.Length; i++) {
					float left = rect.x + (float)(Module.Data.Sequences[i].Start-1)/(float)(Module.Data.GetTotalFrames()-1) * rect.width;
					float right = rect.x + (float)(Module.Data.Sequences[i].End-1)/(float)(Module.Data.GetTotalFrames()-1) * rect.width;
					Vector3 a = new Vector3(left, rect.y, 0f);
					Vector3 b = new Vector3(right, rect.y, 0f);
					Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
					Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
					UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
					UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
				}

				//Current Pivot
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
				UltiDraw.DrawCircle(top, 3f, UltiDraw.Green);
				UltiDraw.DrawCircle(bottom, 3f, UltiDraw.Green);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...

				EditorGUILayout.EndVertical();

			}
			UltiDraw.End();
		}
	}

}
#endif
