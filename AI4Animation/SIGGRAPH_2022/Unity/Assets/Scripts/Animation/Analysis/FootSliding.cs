#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEngine;

namespace AI4Animation {
	public class FootSliding : MonoBehaviour {

		public int Frames = 100;
		public float LineStrength = 0.00125f;
		public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.75f, 0.6f, 0.2f);
		public MotionEditor[] Editors = new MotionEditor[0];
		public AnimationController[] Animations = new AnimationController[0];
		public bool AutoYMax = true;
		public float YMax = 1f;

		public int Framerate = 30;
		public bool UseDeltaTime = false;

		private List<Matrix4x4[]> PreviousTransformations;
		private List<List<float>> Values;

		void Start() {
			PreviousTransformations = new List<Matrix4x4[]>();
			Values = new List<List<float>>();
			for(int i=0; i<Animations.Length; i++) {
				PreviousTransformations.Add(new Matrix4x4[2]);
				for(int j=0; j<PreviousTransformations[i].Length; j++) {
					PreviousTransformations[i][j] = Matrix4x4.identity;
				}
				Values.Add(new List<float>());
			}
			for(int i=0; i<Editors.Length; i++) {
				PreviousTransformations.Add(new Matrix4x4[2]);
				for(int j=0; j<PreviousTransformations[i].Length; j++) {
					PreviousTransformations[i][j] = Matrix4x4.identity;
				}
				Values.Add(new List<float>());
			}
		}

		void LateUpdate () {
			// for(int i=0; i<Animations.Length; i++) {
			//     //Compute Value
			// 	Matrix4x4 left = Animations[i].Actor.GetBoneTransformation("Player 01:LeftFoot");
			// 	Matrix4x4 right = Animations[i].Actor.GetBoneTransformation("Player 01:RightFoot");
			// 	TimeSeries.Contact contact = (TimeSeries.Contact)Animations[i].GetTimeSeries(TimeSeries.ID.Contact);
			// 	float value =
			// 		contact.Values[contact.Pivot][0] * Vector3.Distance(PreviousTransformations[i][0].GetPosition(), left.GetPosition())
			// 		+ contact.Values[contact.Pivot][1] * Vector3.Distance(PreviousTransformations[i][1].GetPosition(), right.GetPosition());
			// 	value *= UseDeltaTime ? (1f/Time.deltaTime) : Framerate;
			// 	Values[i].Add(value);
			// 	PreviousTransformations[i][0] = left;
			// 	PreviousTransformations[i][1] = right;
			//     //
			// 	while(Values[i].Count > Frames) {
			// 		Values[i].RemoveAt(0);
			// 	}
			// }

			for(int i=0; i<Editors.Length; i++) {
				//Compute Value
				Frame frame = Editors[i].GetCurrentFrame();
				Matrix4x4 left = frame.GetBoneTransformation("Player 01:LeftFoot", Editors[i].Mirror);
				Matrix4x4 right = frame.GetBoneTransformation("Player 01:RightFoot", Editors[i].Mirror);
				ContactModule.Sensor[] sensors = (Editors[i].GetSession().Asset.GetModule<ContactModule>()).GetSensors("Player 01:LeftFootEnd", "Player 01:RightFootEnd");
				float value =
					sensors[0].GetContact(frame.Timestamp, Editors[i].Mirror) * Vector3.Distance(PreviousTransformations[i][0].GetPosition(), left.GetPosition())
					+ sensors[1].GetContact(frame.Timestamp, Editors[i].Mirror) * Vector3.Distance(PreviousTransformations[i][1].GetPosition(), right.GetPosition());
				value *= UseDeltaTime ? (1f/Time.deltaTime) : Framerate;
				Values[i].Add(value);
				PreviousTransformations[i][0] = left;
				PreviousTransformations[i][1] = right;
				//
				while(Values[i].Count > Frames) {
					Values[i].RemoveAt(0);
				}
			}
		}

		public float[][] GetValues() {
			float[][] values = new float[Values.Count][];
			for(int i=0; i<values.Length; i++) {
				values[i] = Values[i].ToArray();
			}
			return values;
		}

		public float[] GetValues(int actor) {
			return Values[actor].ToArray();
		}

		void OnRenderObject() {
			UltiDraw.Begin();
			UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), GetValues(), UltiDraw.Dimension.X, lineColors: GetColors(), backgroundColor: Color.white, thickness:LineStrength);
			UltiDraw.End();
		}

		void OnGUI() {
			float size = 0.05f;
			UltiDraw.Begin();
			float[][] values = GetValues();
			Color[] colors = GetColors();
			for(int i=0; i<Animations.Length; i++) {
				float mean = values[i].Mean();
				float sigma = values[i].Sigma();
				UltiDraw.OnGUILabel(new Vector2(Rect.X - 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, mean.Round(3).ToString(), colors[i]);
				UltiDraw.OnGUILabel(new Vector2(Rect.X, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, Animations[i].name, UltiDraw.Black);
				UltiDraw.OnGUILabel(new Vector2(Rect.X + 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, sigma.Round(3).ToString(), colors[i]);
			}
			for(int i=0; i<Editors.Length; i++) {
				float mean = values[i].Mean();
				float sigma = values[i].Sigma();
				UltiDraw.OnGUILabel(new Vector2(Rect.X - 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, mean.Round(3).ToString(), colors[i]);
				UltiDraw.OnGUILabel(new Vector2(Rect.X, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, Editors[i].name, UltiDraw.Black);
				UltiDraw.OnGUILabel(new Vector2(Rect.X + 0.5f * Rect.W, Rect.Y - 0.5f*Rect.H - (i+1)*size), Rect.GetSize(), size/2f, sigma.Round(3).ToString(), colors[i]);
			}
			UltiDraw.End();
		}

		private Color[] GetColors() {
			return UltiDraw.GetRainbowColors(Values.Count);
		}

	}
}
#endif

// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class FootSliding : MonoBehaviour {

// 	public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.175f, 0.1f, 0.3f, 0.15f);

// 	public LayerMask Ground = ~0;
// 	public float HeightThreshold = 0.025f;
// 	public int Frames = 100;
// 	public Transform[] Bones;

// 	private Vector3[] PreviousPositions;
// 	private List<float>[] Values;

// 	private int Index = 0;

// 	void OnValidate() {
// 		HeightThreshold = Mathf.Max(HeightThreshold, 0.001f);
// 	}

// 	void Start() {
// 		PreviousPositions = new Vector3[Bones.Length];
// 		for(int i=0; i<PreviousPositions.Length; i++) {
// 			PreviousPositions[i] = Bones[i].position;
// 		}
// 		Values = new List<float>[Bones.Length];
// 		for(int i=0; i<Values.Length; i++) {
// 			Values[i] = new List<float>();
// 		}
// 	}

// 	void LateUpdate () {
// 		int index = Index;
// 		Index = MotionEditor.GetInstance().GetCurrentFrame().Index;
// 		bool skip = index > Index;
// 		for(int i=0; i<Bones.Length; i++) {
// 			Vector3 position = Bones[i].position;	
// 			float height = Mathf.Clamp(position.y - Utility.ProjectGround(position, Ground).y, 0f, HeightThreshold);
// 			float globalWeight = 2f - Mathf.Pow(2f, height / HeightThreshold); //This is the quadratic weight between 0 and 1 when the foot is within the defined height threshold.
// 			float horizontal = new Vector3(position.x - PreviousPositions[i].x, 0f, position.z - PreviousPositions[i].z).magnitude; //This is the horizontal foot movement.
// 			float total = (position - PreviousPositions[i]).magnitude; //This is the absolute foot movement.
// 			if(!skip) {
// 				if(total > 0f) {
// 					float localWeight = horizontal / total; //This is the linear weight measuring the amount of horizontal food movement.
// 					Values[i].Add(1f/Time.deltaTime * globalWeight * localWeight * horizontal);
// 				} else {
// 					Values[i].Add(0f);
// 				}
// 			}
// 			while(Values[i].Count > Frames) {
// 				Values[i].RemoveAt(0);
// 			}
// 			PreviousPositions[i] = Bones[i].position;
// 		}
// 	}

// 	void OnRenderObject() {
// 		// List<float[]> values = new List<float[]>();
// 		// for(int i=0; i<Values.Length; i++) {
// 		// 	float[] v = Values[i].ToArray();
// 		// 	values.Add(v);
// 		// }
// 		// UltiDraw.Begin();
// 		// UltiDraw.DrawGUIFunctions(Rect.GetCenter(), Rect.GetSize(), values, 0f, 1f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
// 		// UltiDraw.End();
// 	}

// 	private float GetAverage() {
// 		float value = 0f;
// 		for(int i=0; i<Values.Length; i++) {
// 			value += Values[i].ToArray().Mean();
// 		}
// 		return value / Values.Length;
// 	}

// 	private float GetSigma() {
// 		float value = 0f;
// 		for(int i=0; i<Values.Length; i++) {
// 			value += Values[i].ToArray().Sigma();
// 		}
// 		return value / Values.Length;
// 	}

// 	void OnGUI() {
// 		GUI.color = UltiDraw.Black;
// 		GUI.backgroundColor = UltiDraw.White;
// 		GUI.Box(Utility.GetGUIRect(Rect.X - 0.5f*Rect.W, 1f - (Rect.Y + 0.5f*Rect.H) - 0.1f, Rect.W, 0.1f),"Average: " + GetAverage() + " Sigma: " + GetSigma());
// 	}
// }
