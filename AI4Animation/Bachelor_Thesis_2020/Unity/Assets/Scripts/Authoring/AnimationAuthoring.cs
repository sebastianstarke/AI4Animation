using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using ColorHexUtility;
#if UNITY_EDITOR
using UnityEditorInternal;
#endif
using System;

[ExecuteInEditMode]
public class AnimationAuthoring : MonoBehaviour
{
	//public LayerMask Ground = 0;
	public float TimeDelta = 1f/60f;
	public float TimeInterval = 1f;
	public List<ControlPoint> ControlPoints = new List<ControlPoint>();
	public Point[] LookUpPoints;
	public bool isLooping = false;

	public float DebugTimestamp = 0f;

	public bool Inspect = true;
	public bool CreateCP = false;
	public KeyCode CreateCPKey = KeyCode.Space;
	public StyleNames Stylenames;
	public float RefTimestamp = 0f;
	public enum StyleNames : int
	{
		Idle,
		Move,
		Jump,
		Sit,
		Stand,
		Lie,
		Sneak,
		Eat,
		Hydrate
	};


	public Color32 Idle = new ColorHex("#C3C603");
	public Color32 Move = new ColorHex("#4DA800");
	public Color32 Jump = new ColorHex("#F2840B");
	public Color32 Sit = new ColorHex("#F4FE37");
	public Color32 Lie = new ColorHex("#AD01BC");
	public Color32 Stand = new ColorHex("#0005FF");
	public Color32 Sneak = new ColorHex("#FF0600");
	public Color32 Eat = new ColorHex("#800080");
	public Color32 Hydrate = new ColorHex("#00FFF9");

	public Color[] StyleColors = new Color[System.Enum.GetNames(typeof(StyleNames)).Length];

	public static float[] StyleValues = new float[System.Enum.GetNames(typeof(StyleNames)).Length];

	private bool Clicked;

	void Awake()
	{
		StyleColors = new Color[] { Idle, Move, Jump, Sit, Stand, Lie, Sneak, Eat, Hydrate };
		if (ControlPoints.Count == 1 && !isLooping) isLooping = true;
	}
	public ControlPoint CreateControlPoint(Vector3 pos)
	{
		ControlPoint p = new ControlPoint(this, System.Enum.GetNames(typeof(StyleNames)), StyleValues);
		ControlPoints.Add(p);
		p.SetPosition(GetGroundPosition(pos, p.Ground));
		return p;
	}

	public ControlPoint InsertControlPoint(ControlPoint cp)
	{
		ControlPoint newCP = new ControlPoint(this, "hidden");
		newCP.SetPosition(GetGroundPosition(cp.GetPosition(), newCP.Ground));
		int styleCount = cp.GetStyles().Length;
		string[] names = new string[styleCount];
		float[] values = new float[styleCount];
		for(int i=0; i< styleCount; i++)
		{
			names[i] = cp.Styles[i].Name;
			values[i] = cp.Styles[i].Value;
		}
		newCP.SetStyles(names, values);
		newCP.Inspector = false;
		newCP.Ground = cp.Ground;
		ControlPoints.Insert(ControlPoints.IndexOf(cp)+1, newCP);
		if (Application.isPlaying)
		{
			UpdateLookUpPoints(TimeDelta);
		}
		return newCP;
	}

	public void RemoveControlPoint(ControlPoint cp)
	{
		ControlPoints.RemoveAt(ControlPoints.IndexOf(cp));
		if (Application.isPlaying)
		{
			UpdateLookUpPoints(TimeDelta);
		}
	}



	//sets y value of pointPos to y value of GroundVector
	public Vector3 GetGroundPosition(Vector3 pointPos, LayerMask layer)
	{
		return Utility.ProjectGround(pointPos, layer);
	}

	public Point CreatePoint(float timestamp)
	{
		Point point = new Point();
		point.SetPosition(GetPointPositon(timestamp));
		//point.SetDirection(GetPointDirection(timestamp).normalized);
		point.SetVelocity(GetPointVelocity(timestamp, TimeDelta));
		//1s in future 
		//DESIRED VELOCITY
		point.SetSpeed(GetPointSpeed(timestamp, TimeInterval));
		float[] styles = new float[System.Enum.GetNames(typeof(StyleNames)).Length];
		for(int i=0; i< System.Enum.GetNames(typeof(StyleNames)).Length; i++)
		{
			styles[i] = GetPointStyleValue(timestamp, i);
		}
		point.SetStyle(styles);
		return point;
	}

	public Vector3 GetPointPositon(float timestamp)
	{
		//get all 4 CP's for CatmullRom calculation
		ControlPoint c0 = GetControlPoint(timestamp, -1);
		ControlPoint c1 = GetControlPoint(timestamp, 0);
		ControlPoint c2 = GetControlPoint(timestamp, 1);
		ControlPoint c3 = GetControlPoint(timestamp, 2);
		//get timestamp
		float t = FilterTimestamp(timestamp);
		float pivot = GetPivotTimestamp(timestamp);
		if (t < pivot)
		{
			t += GetTotalTime();
		}
		float tCatmull = (t - pivot) / TimeInterval; // [0,1]
		return GetGroundPosition(GetCatmullRomVector(tCatmull, c0.GetPosition(), c1.GetPosition(), c2.GetPosition(), c3.GetPosition()), c1.Ground);
	}


	public float GetPointStyleValue(float timestamp, int styleOffset)
	{
		//get all 4 CP's for CatmullRom calculation
		ControlPoint c0 = GetControlPoint(timestamp, -1);
		ControlPoint c1 = GetControlPoint(timestamp, 0);
		ControlPoint c2 = GetControlPoint(timestamp, 1);
		ControlPoint c3 = GetControlPoint(timestamp, 2);
		//get timestamp
		float t = FilterTimestamp(timestamp);
		float pivot = GetPivotTimestamp(timestamp);
		if (t < pivot)
		{
			t += GetTotalTime();
		}
		float tCatmull = (t - pivot) / TimeInterval; // [0,1]
		return GetCatmullRomValue(tCatmull, c0.GetStyles()[styleOffset].Value, c1.GetStyles()[styleOffset].Value, c2.GetStyles()[styleOffset].Value, c3.GetStyles()[styleOffset].Value);
	}

	//return the maximum time for each loop or round
	public float GetTotalTime() {
		if(isLooping) {
			return TimeInterval * ControlPoints.Count;
		} else {
			return TimeInterval * (ControlPoints.Count-1);
		}
	}

	// make sure that the timestamp is in correct interval between 0f and TotalTime
	private float FilterTimestamp(float timestamp)
	{
		//timestamp mod totaltime, if we loop
		if (isLooping)
		{
			return Mathf.Repeat(timestamp, GetTotalTime());
		}
		else
		{
			//returns 0f, if timestamp < 0f
			//return TotalTime, if timestamp > TotalTime
			//else return timestamp
			return Mathf.Clamp(timestamp, 0f, GetTotalTime());
		}
	}

	//get PivotIndex of given timestamp for ControlPoint
	private int GetPivotIndex(float timestamp) {
		return Mathf.FloorToInt(FilterTimestamp(timestamp) / TimeInterval);
	}

	//get PivotTimestamp of given timestamp for ???
	private float GetPivotTimestamp(float timestamp) {
		return TimeInterval * GetPivotIndex(timestamp);
	}

	private Vector3 GetPointVelocity(float timestamp, float delta)
	{
		return (GetPointPositon(timestamp) - GetPointPositon(timestamp-delta)) / delta;
	}

	private float GetPointSpeed(float timestamp, float timeinterval)
	{
		float length = 0f;
		for (float i = timestamp; i <= timestamp + timeinterval; i += TimeDelta)
		{
			length += Vector3.Distance(GetPointPositon(i), GetPointPositon(i + TimeDelta));
		}
		//length = Vector3.Distance(GetPointPositon(timestamp), GetPointPositon(timestamp + timeinterval));
		return length;
	}

	public ControlPoint GetControlPoint(float timestamp, int offset) {
		//clamp (timestamp+offset mod CP Size), if we loop
		if(isLooping) {
			return ControlPoints[Mathf.Clamp((int)Mathf.Repeat(GetPivotIndex(timestamp) + offset, ControlPoints.Count), 0, ControlPoints.Count-1)];
		} else {
			return ControlPoints[Mathf.Clamp(GetPivotIndex(timestamp) + offset, 0, ControlPoints.Count-1)];
		}
	}
	public float GetControlPointTimestamp(float timestamp, int offset)
	{
		if (isLooping)
		{
			return Mathf.Clamp((int)Mathf.Repeat(GetPivotIndex(timestamp) + offset, ControlPoints.Count), 0, ControlPoints.Count - 1);
		}
		else
		{
			return Mathf.Clamp(GetPivotIndex(timestamp) + offset, 0, ControlPoints.Count - 1);
		}
	}

	public Vector3 GetCatmullRomVector(float t, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3)
	{
		Vector3 a = 2f * v1;
		Vector3 b = v2 - v0;
		Vector3 c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		Vector3 d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
	}
	private float GetCatmullRomValue(float t, float v0, float v1, float v2, float v3)
	{
		float a = 2f * v1;
		float b = v2 - v0;
		float c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		float d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
	}

	public void GenerateLookUpPoints(float deltaTime)
	{
		LookUpPoints = new Point[(int)Math.Round(GetTotalTime() / deltaTime)];

		for (int i = 0; i < LookUpPoints.Length; i++)
		{
			LookUpPoints[i] = CreatePoint(i * deltaTime);
		}
	}

	public void UpdateLookUpPoint(float timestamp)
	{
		var startTime = System.DateTime.Now;
		//Debug.Log( (int)timestamp);
		ControlPoint cp = ControlPoints[(int)timestamp];
		if (cp.MotionTime > 0)
		{
			for (int i = 1; i <= cp.MotionTime; i++)
			{
				//Debug.Log(cp.Transform.position);
				ControlPoints[(int)timestamp + i].SetPosition(cp.Transform.position);
				UpdateLookUpPoint((int)timestamp + i);
			}
		}

		int start = (int)Math.Round(GetControlPointTimestamp(timestamp, -2) / TimeDelta);
		int end = (int)Math.Round(GetControlPointTimestamp(timestamp, 2) / TimeDelta);

		if (end > start)
		{
			for (int i = start; i < end; i++)
			{
				LookUpPoints[i] = CreatePoint(i * TimeDelta);
			}
		}
		else
		{
			for (int i = start; i < (int)Math.Round(GetTotalTime() / TimeDelta); i++)
			{
				LookUpPoints[i] = CreatePoint(i * TimeDelta);
			}

			for (int i = 0; i < end; i++)
			{
				LookUpPoints[i] = CreatePoint(i * TimeDelta);
			}
		}
		/*
		if (AnimationAuthoring.ControlPoints[(int)timestamp].MotionTime > 0)
		{
			UpdateLookUpPoint((int)timestamp + 1);
		}
		*/


		var elapsed = (DateTime.Now - startTime).Milliseconds;
		Debug.Log("update Authoring: " + elapsed + "ms");
	}



	public void UpdateLookUpPoints(float deltaTime)
	{
		var startTime = DateTime.Now;

		GenerateLookUpPoints(deltaTime);

		var elapsed = (DateTime.Now - startTime).Milliseconds;

		Debug.Log("update Authoring: " + elapsed + "ms");
	}


	public Point GetLookUpPoint(float timestamp)
	{

		if (isLooping)
		{
			timestamp = Mathf.Repeat(timestamp, GetTotalTime());
		}
		return LookUpPoints[(int)Math.Round(Mathf.Clamp(timestamp / TimeDelta, 0, LookUpPoints.Length-1))];
	}

	public float GetClosestPointTimestamp(Vector3 pivot, float timestamp)
	{
		float distance = Vector3.Distance(pivot, GetLookUpPoint(timestamp).GetPosition());

		int pointCount = 2;

		for (int i = 1; i < pointCount; i++)
		{
			float distanceFuture = Vector3.Distance(pivot, GetLookUpPoint(timestamp + (i * TimeDelta)).GetPosition());
			float distancePast = Vector3.Distance(pivot, GetLookUpPoint(timestamp - (i * TimeDelta)).GetPosition());
			float d = Mathf.Min(distanceFuture, distancePast);
			if (d < distance)
			{
				distance = d;
				timestamp += (i * TimeDelta);
			}
		}
		if(timestamp == GetTotalTime() && !isLooping)
		{
			return timestamp;
		}

		if (timestamp == RefTimestamp)
		{
			return timestamp + (TimeDelta);
		}

		return timestamp;
	}


#if UNITY_EDITOR
	private void OnEnable()
	{
		SceneView.onSceneGUIDelegate -= CustomUpdate;
		SceneView.onSceneGUIDelegate += CustomUpdate;
	}

	void CustomUpdate(UnityEditor.SceneView sv)
	{
		Event e = Event.current;

		if (CreateCP && !Application.isPlaying && Application.isEditor)
		{
			RaycastHit hit;
			Vector3 screenPosition = Event.current.mousePosition;
			screenPosition.y = Camera.current.pixelHeight - screenPosition.y;

			if (Physics.Raycast(Camera.current.ScreenPointToRay(screenPosition), out hit))
			{

				Clicked = false;

				CreateControlPoint(hit.point);
				UltiDraw.Begin();
				UltiDraw.DrawSphere(hit.point, Quaternion.identity, 0.1f, Color.red);
				for (float i = GetTotalTime() - 2 * TimeInterval; i < GetTotalTime(); i += TimeDelta)
				{
					//UltiDraw.DrawLine(GetPointPositon(i), GetPointPositon(i + TimeDelta), 0.05f, UltiDraw.DarkGreen);
					UltiDraw.DrawSphere(GetPointPositon(i), Quaternion.identity, 0.03f, UltiDraw.DarkGreen);
				}

				//UltiDraw.DrawLine(GetPointPositon(GetTotalTime()-TimeInterval), hit.point, 0.05f, UltiDraw.DarkGreen);
				UltiDraw.End();

				if (((Event.current.type == EventType.KeyUp) || (Event.current.type == EventType.MouseDown)) && Event.current.keyCode == CreateCPKey)
				{
					Clicked = true;
					//Event.current.Use();
				}

				if (!Clicked)
				{
					Utility.Destroy(ControlPoints[ControlPoints.Count - 1].GameObject);
					ControlPoints.RemoveAt(ControlPoints.Count - 1);
					//Event.current.Use();
				}
			}

		}
	}
#endif
	// Visualize the points
	void OnDrawGizmos()
	{
		if (Application.isEditor && !Application.isPlaying)
		{
			DrawAuthoring();
		}
	}

	public void LabelCP(ControlPoint cp, string label)
	{
		if (!cp.Inspector) return;
		GUIStyle style = new GUIStyle();
		style.normal.textColor = UltiDraw.Red;
		style.fontSize = 15;
		Vector3 pos = GetGroundPosition(cp.GetPosition(), cp.Ground);
#if UNITY_EDITOR
		Handles.Label(new Vector3(pos.x, pos.y + 0.2f, pos.z), label, style);
#endif   
	}

	public static Color CombineColors(Dictionary<Color, float> colors)
	{
		Color result = new Color(0, 0, 0, 0);

		foreach (KeyValuePair<Color, float> c in colors)
		{
			result += c.Key * c.Value;
		}

		//result /= colors.Count;
		return result;
	}

	public void DrawAuthoring()
	{
		UltiDraw.Begin();
		int countCpInspector = 0;
		foreach (ControlPoint cp in ControlPoints)
		{
			if (cp.Inspector) countCpInspector++;
			LabelCP(cp, (countCpInspector - 1).ToString());
			UltiDraw.DrawSphere(cp.GetPosition(), Quaternion.identity, 0.1f, Color.red);
		}

		if (TimeDelta > 0 && ControlPoints.Count > 0)
		{
			for (float i = 0f; i < GetTotalTime(); i += TimeDelta)
			{
				if (!GetControlPoint(i, +1).Inspector) continue;

				StyleColors = new Color[] { Idle, Move, Jump, Sit, Stand, Lie, Sneak, Eat, Hydrate};
				float r = 0f;
				float g = 0f;
				float b = 0f;
				for (int j = 0; j < System.Enum.GetNames(typeof(StyleNames)).Length; j++)
				{
					float styleValue = GetPointStyleValue(i, j);
					r += styleValue * StyleColors[j].r;
					g += styleValue * StyleColors[j].g;
					b += styleValue * StyleColors[j].b;
				}

				Color color = new Color(r, g, b, 1f);
				UltiDraw.DrawLine(GetPointPositon(i), GetPointPositon(i + TimeDelta), 0.05f, color);
				//UltiDraw.DrawSphere(GetPointPositon(i), Quaternion.identity, 0.02f, color);
			}
		}
		UltiDraw.End();
	}

#if UNITY_EDITOR
	[CustomEditor(typeof(AnimationAuthoring))]
	public class AnimationAuthoring_Editor : Editor
	{

		public AnimationAuthoring Target;

		void Awake()
		{
			Target = (AnimationAuthoring)target;
		}

		public override void OnInspectorGUI()
		{
			Undo.RecordObject(Target, Target.name);

			Inspector();

			

			if (GUI.changed)
			{
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector()
		{
			Utility.SetGUIColor(UltiDraw.Grey);
			using (new EditorGUILayout.VerticalScope("Box"))
			{
				Utility.ResetGUIColor();
				if (Utility.GUIButton("Create ControlPoint", UltiDraw.DarkGreen, UltiDraw.White))
				{
					Target.CreateCP = !Target.CreateCP;
				}

				Target.CreateCP = EditorGUILayout.Toggle("Create Controlpoint", Target.CreateCP);
				Target.CreateCPKey = (KeyCode)EditorGUILayout.EnumPopup("Key to create Controlpoint", Target.CreateCPKey);

				if (Target.CreateCP)
				{
					Utility.SetGUIColor(UltiDraw.Grey);
					using (new EditorGUILayout.VerticalScope("Box"))
					{
						Utility.ResetGUIColor();

						string[] styles = System.Enum.GetNames(typeof(StyleNames));
						for (int i = 0; i < styles.Length; i++)
						{
							StyleValues[i] = EditorGUILayout.Slider(styles[i], StyleValues[i], 0f, 1f);
						}
					}
				}

				if (Utility.GUIButton("ControlPoints", UltiDraw.Yellow, UltiDraw.Black))
				{
					Target.Inspect = !Target.Inspect;
				}

				if (Target.Inspect)
				{
					using (new EditorGUILayout.VerticalScope("Box"))
					{
						EditorGUI.BeginChangeCheck();
						//LayerMask tempMask = EditorGUILayout.MaskField("Ground",InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.Ground), InternalEditorUtility.layers);
						//Target.Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(tempMask);
						Target.isLooping = EditorGUILayout.Toggle("Looping", Target.isLooping);
						Target.TimeDelta = EditorGUILayout.FloatField("TimeDelta", Target.TimeDelta);
						Target.TimeInterval = EditorGUILayout.FloatField("Time between Points", Target.TimeInterval);
						using (new EditorGUILayout.VerticalScope("Box"))
						{
							Target.Idle = (Color)EditorGUILayout.ColorField("Idle", Target.Idle);
							Target.Move = (Color)EditorGUILayout.ColorField("Move", Target.Move);
							Target.Jump = (Color)EditorGUILayout.ColorField("Jump", Target.Jump);
							Target.Sit = (Color)EditorGUILayout.ColorField("Sit", Target.Sit);
							Target.Stand = (Color)EditorGUILayout.ColorField("Stand", Target.Stand);
							Target.Lie = (Color)EditorGUILayout.ColorField("Lie", Target.Lie);
							Target.Sneak = (Color)EditorGUILayout.ColorField("Sneak", Target.Sneak);
							Target.Eat = (Color)EditorGUILayout.ColorField("Eat", Target.Eat);
							Target.Hydrate = (Color)EditorGUILayout.ColorField("Hydrate", Target.Hydrate);
						}

						if (EditorGUI.EndChangeCheck())
						{
							if (Application.isPlaying)
							{
								Target.UpdateLookUpPoints(Target.TimeDelta);
							}

						}
						int countCpInspector = 0;
						for (int i=0; i<Target.ControlPoints.Count; i++)
						{
							if (!Target.ControlPoints[i].Inspector) continue;
							if (Target.ControlPoints[i].Inspector) countCpInspector++;


							int m = Target.ControlPoints[i].MotionTime;

							EditorGUILayout.BeginHorizontal();
							if (Utility.GUIButton("ControlPoint " + (countCpInspector-1), UltiDraw.DarkGrey, UltiDraw.White) )
							{
								Target.ControlPoints[i].Inspect = !Target.ControlPoints[i].Inspect;
							}

							EditorGUI.BeginChangeCheck();
							if (Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 20f, 20f))
							{
								for (int j = 0; j < m; j++)
								{
									Utility.Destroy(Target.ControlPoints[i + 1].GameObject);
									Target.RemoveControlPoint(Target.ControlPoints[i + 1]);
								}

								Utility.Destroy(Target.ControlPoints[i].GameObject);
								Target.ControlPoints.RemoveAt(i);
								i--;
							}
							if (EditorGUI.EndChangeCheck())
							{
								if (Application.isPlaying)
								{
									Target.UpdateLookUpPoints(Target.TimeDelta);
								}

							}

							EditorGUILayout.EndHorizontal();

							if (Target.ControlPoints[i].Inspect )
							{
								Utility.SetGUIColor(UltiDraw.Grey);
								using (new EditorGUILayout.VerticalScope("Box"))
								{
									

									EditorGUI.BeginChangeCheck();
									Utility.ResetGUIColor();
									Target.ControlPoints[i].GameObject = (GameObject)EditorGUILayout.ObjectField("Controlpoint " + (countCpInspector - 1), Target.ControlPoints[i].GameObject, typeof(GameObject), true);
									Target.ControlPoints[i].Transform = EditorGUILayout.ObjectField("Transform", Target.ControlPoints[i].Transform, typeof(Transform), true) as Transform;
									LayerMask tempMask = EditorGUILayout.MaskField("Ground", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Target.ControlPoints[i].Ground), InternalEditorUtility.layers);
									Target.ControlPoints[i].Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(tempMask);

									for (int j = 0; j < Target.ControlPoints[i].GetStyles().Length; j++)
									{
										var style = Target.ControlPoints[i].GetStyles()[j];
										style.Value = EditorGUILayout.Slider(style.Name, style.Value, 0f, 1f);
									}
									if (EditorGUI.EndChangeCheck())
									{
										//update paused CP (motiontime)
										for (int j = 0; j < m; j++)
										{
											Utility.Destroy(Target.ControlPoints[i + 1].GameObject);
											Target.RemoveControlPoint(Target.ControlPoints[i + 1]);
										}

										for (int j = 0; j < Target.ControlPoints[i].MotionTime; j++)
										{
											Target.InsertControlPoint(Target.ControlPoints[i]);
										}

										if (Application.isPlaying)
										{
											Target.UpdateLookUpPoint(i * Target.TimeInterval);
										}

									}
									
									EditorGUI.BeginChangeCheck();
									Target.ControlPoints[i].MotionTime = EditorGUILayout.IntField("MotionTime", Target.ControlPoints[i].MotionTime);
									if (EditorGUI.EndChangeCheck())
									{
										//update paused CP (motiontime)
										for (int j = 0; j < m; j++)
										{
											Utility.Destroy(Target.ControlPoints[i+1].GameObject);
											Target.RemoveControlPoint(Target.ControlPoints[i+1]);										
										}

										for (int j=0; j < Target.ControlPoints[i].MotionTime; j++)
										{
											Target.InsertControlPoint(Target.ControlPoints[i]);
										}

									}



								}
							}


						}
						EditorGUI.BeginChangeCheck();
						if (Utility.GUIButton("Remove All", UltiDraw.DarkRed, UltiDraw.White))
						{
							for (int i = 0; i < Target.ControlPoints.Count; i++)
							{
								Utility.Destroy(Target.ControlPoints[i].GameObject);
								Target.ControlPoints.RemoveAt(i);
								i--;
							}
						}
						if (EditorGUI.EndChangeCheck())
						{
							if (Application.isPlaying)
							{
								Target.UpdateLookUpPoints(Target.TimeDelta);
							}

						}
					}
				}

			}
		}
	}
#endif

}
