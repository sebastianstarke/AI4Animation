using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BioVisualisation : MonoBehaviour {

	public bool Show = true;

	public CameraController CameraController;

	public GameObject Canvas;

	public Button CanvasToggle;

	public Button StraightForward, StraightBack, StraightLeft, StraightRight, TurnLeft, TurnRight, Idle, Move, Jump, Sit, Lie, Stand;
	public Color Active = Utility.Orange;
	public Color Inactive = Utility.DarkGrey;

	public Button Skeleton, Transforms, Velocities, Trajectory, CyclicWeights, InverseKinematics, MotionTrails;
	public Color VisualisationEnabled = Utility.Cyan;
	public Color VisualisationDisabled = Utility.Grey;

	public Button SmoothFollow, ConstantView, Static;
	public Color CameraEnabled = Utility.Mustard;
	public Color CameraDisabled = Utility.LightGrey;

	public Slider Yaw, Pitch, FOV;

	public bool DrawTrails = false;
	public Trail[] Trails = new Trail[0];

	private int Frames = 150;
	private Queue<float>[] CW;
	public bool DrawCW = true;

	private BioAnimation_APFNN Animation;

	void Awake() {
		Animation = GetComponent<BioAnimation_APFNN>();
		CW = new Queue<float>[Animation.APFNN.ControlWeights];
		for(int i=0; i<CW.Length; i++) {
			CW[i] = new Queue<float>();
			for(int j=0; j<Frames; j++) {
				CW[i].Enqueue(0f);
			}
		}
		Skeleton.onClick.AddListener(ToggleSkeleton); UpdateColor(Skeleton, Animation.Character.DrawSkeleton ? VisualisationEnabled : VisualisationDisabled);
		Transforms.onClick.AddListener(ToggleTransforms); UpdateColor(Transforms, Animation.Character.DrawTransforms ? VisualisationEnabled : VisualisationDisabled);
		Velocities.onClick.AddListener(ToggleVelocities); UpdateColor(Velocities, Animation.ShowVelocities ? VisualisationEnabled : VisualisationDisabled);
		Trajectory.onClick.AddListener(ToggleTrajectory); UpdateColor(Trajectory, Animation.ShowTrajectory ? VisualisationEnabled : VisualisationDisabled);
		CyclicWeights.onClick.AddListener(ToggleCyclicWeights); UpdateColor(CyclicWeights, DrawCW ? VisualisationEnabled : VisualisationDisabled);
		InverseKinematics.onClick.AddListener(ToggleInverseKinematics); UpdateColor(InverseKinematics, Animation.SolveIK ? VisualisationEnabled : VisualisationDisabled);
		MotionTrails.onClick.AddListener(ToggleMotionTrails); UpdateColor(MotionTrails, DrawTrails ? VisualisationEnabled : VisualisationDisabled);

		SmoothFollow.onClick.AddListener(SetSmoothFollow); UpdateColor(SmoothFollow, CameraController.Mode == CameraController.MODE.SmoothFollow ? CameraEnabled : CameraDisabled);
		ConstantView.onClick.AddListener(SetConstantView); UpdateColor(ConstantView, CameraController.Mode == CameraController.MODE.ConstantView ? CameraEnabled : CameraDisabled);
		Static.onClick.AddListener(SetStatic); UpdateColor(Static, CameraController.Mode == CameraController.MODE.Static ? CameraEnabled : CameraDisabled);

		Yaw.onValueChanged.AddListener(SetYaw);
		Pitch.onValueChanged.AddListener(SetPitch);
		FOV.onValueChanged.AddListener(SetFOV);

		CanvasToggle.onClick.AddListener(ToggleShow);
	}

	void Start() {
		SetYaw(0f);
		SetPitch(0f);
		SetFOV(1f);
	}

	private void UpdateData() {
		for(int i=0; i<CW.Length; i++) {
			CW[i].Dequeue();
			CW[i].Enqueue(Animation.APFNN.GetControlPoint(i));
		}
	}

	private void DrawControlPoint(float x, float y, float width, float height, Queue<float> CW, Color color) {
		int _index = 0;
		float _x = 0f;
		float _xPrev = 0f;
		float _y = 0f;
		float _yPrev = 0f;
		foreach(float value in CW) {
			_x = x + (float)(_index)/(float)(Frames-1) * width;
			_y = y - height + value*height;
			if(_index > 0) {
				UnityGL.DrawGUILine(
					_xPrev,
					_yPrev, 
					_x,
					_y,
					2.5f,
					color
				);
			}
			_xPrev = _x; 
			_yPrev = _y;
			_index += 1;
		}
	}

	public void ToggleSkeleton() {
		Animation.Character.DrawSkeleton = !Animation.Character.DrawSkeleton;
		UpdateColor(Skeleton, Animation.Character.DrawSkeleton ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleTransforms() {
		Animation.Character.DrawTransforms = !Animation.Character.DrawTransforms;
		UpdateColor(Transforms, Animation.Character.DrawTransforms ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleVelocities() {
		Animation.ShowVelocities = !Animation.ShowVelocities;
		UpdateColor(Velocities, Animation.ShowVelocities ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleTrajectory() {
		Animation.ShowTrajectory = !Animation.ShowTrajectory;
		UpdateColor(Trajectory, Animation.ShowTrajectory ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleCyclicWeights() {
		DrawCW = !DrawCW;
		UpdateColor(CyclicWeights, DrawCW ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleInverseKinematics() {
		Animation.UseIK(!Animation.SolveIK);
		UpdateColor(InverseKinematics, Animation.SolveIK ? VisualisationEnabled : VisualisationDisabled);
	}

	public void ToggleMotionTrails() {
		DrawTrails = !DrawTrails;
		UpdateColor(MotionTrails, DrawTrails ? VisualisationEnabled : VisualisationDisabled);
	}

	public void SetSmoothFollow() {
		CameraController.SetMode(CameraController.MODE.SmoothFollow);
		UpdateColor(SmoothFollow, CameraEnabled); UpdateColor(ConstantView, CameraDisabled); UpdateColor(Static, CameraDisabled);
	}

	public void SetConstantView() {
		CameraController.SetMode(CameraController.MODE.ConstantView);
		UpdateColor(SmoothFollow, CameraDisabled); UpdateColor(ConstantView, CameraEnabled); UpdateColor(Static, CameraDisabled);
	}

	public void SetStatic() {
		CameraController.SetMode(CameraController.MODE.Static);
		UpdateColor(SmoothFollow, CameraDisabled); UpdateColor(ConstantView, CameraDisabled); UpdateColor(Static, CameraEnabled);
	}

	public void SetYaw(float value) {
		CameraController.Yaw = value;
		Yaw.transform.Find("Text").GetComponent<Text>().text = "YAW: " + Mathf.RoundToInt(value);
	}

	public void SetPitch(float value) {
		CameraController.Pitch = value;
		Pitch.transform.Find("Text").GetComponent<Text>().text = "PITCH: " + Mathf.RoundToInt(value);
	}

	public void SetFOV(float value) {
		CameraController.FOV = Mathf.RoundToInt(value*10f) / 10f;
		FOV.transform.Find("Text").GetComponent<Text>().text = "FOV: " + value.ToString("F1");
	}
	
	public void ToggleShow() {
		Show = !Show;
	}

	void OnGUI() {
		GameObject.Find("Trajectory_Circle").GetComponent<CatmullRomSpline>().DrawGUI = Show;
		GameObject.Find("Trajectory_Square").GetComponent<CatmullRomSpline>().DrawGUI = Show;
		GameObject.Find("Trajectory_Slalom").GetComponent<CatmullRomSpline>().DrawGUI = Show;
		Canvas.SetActive(Show);
		if(!Show) {
			return;
		}

		UpdateControl(StraightForward, Input.GetKey(Animation.Controller.MoveForward));
		UpdateControl(StraightBack, Input.GetKey(Animation.Controller.MoveBackward));
		UpdateControl(StraightLeft, Input.GetKey(Animation.Controller.MoveLeft));
		UpdateControl(StraightRight, Input.GetKey(Animation.Controller.MoveRight));
		UpdateControl(TurnLeft, Input.GetKey(Animation.Controller.TurnLeft));
		UpdateControl(TurnRight, Input.GetKey(Animation.Controller.TurnRight));

		UpdateStyle(Idle, 0);
		UpdateStyle(Move, 1);
		UpdateStyle(Jump, 2);
		UpdateStyle(Sit, 3);
		UpdateStyle(Lie, 4);
		UpdateStyle(Stand, 5);
	}

	private void UpdateColor(Button button, Color color) {
		if(button == null) {
			return;
		}
		button.GetComponent<Image>().color = color;
	}

	private void UpdateControl(Button button, bool active) {
		if(button == null) {
			return;
		}
		UpdateColor(button, active ? Active : Inactive);
	}

	private void UpdateStyle(Button button, int index) {
		if(button == null) {
			return;
		}
		float activation = Animation.GetTrajectory().Points[60].Styles[index];
		//if(index == 1) {
		//	activation += Animation.GetTrajectory().Points[60].Styles[index+1];
		//	activation = Mathf.Clamp(activation, 0f, 1f);
		//}
		button.GetComponent<Image>().color = Color.Lerp(Inactive, Active, activation);
		button.GetComponentInChildren<Text>().text = (100f*activation).ToString("F0")  + "%";
	}

	void OnRenderObject() {
		UpdateData();

		if(!Show) {
			return;
		}

		if(DrawCW) {
			UnityGL.Start();
			float x = 0.025f;
			float y = 0.15f;
			float width = 0.95f;
			float height = 0.1f;
			float border = 0.0025f;
			UnityGL.DrawGUIQuad(x-border/Screen.width*Screen.height, y-height-border, width+2f*border/Screen.width*Screen.height, height+2f*border, Utility.Black);
			UnityGL.DrawGUIQuad(x, y-height, width, height, Utility.White);

			Color[] colors = Utility.GetRainbowColors(Animation.APFNN.ControlWeights);
			for(int i=0; i<colors.Length; i++) {
				DrawControlPoint(x, y, width, height, CW[i], colors[i]);
			}
			/*
			DrawControlPoint(x, y, width, height, CW[0], Utility.Red);
			DrawControlPoint(x, y, width, height, CW[1], Utility.DarkGreen);
			DrawControlPoint(x, y, width, height, CW[2], Utility.Purple);
			DrawControlPoint(x, y, width, height, CW[3], Utility.Orange);
			*/
			UnityGL.Finish();
		}

		for(int i=0; i<Trails.Length; i++) {
			Trails[i].Update(Frames);
			if(DrawTrails) {
				Trails[i].Draw(0.01f, new Color(0f, 1f, 2f/3f, 1f));
			}
		}
	}

	[System.Serializable]
	public class Trail {
		public Transform Target;

		private Queue<Vector3> Positions = new Queue<Vector3>();

		public void Update(int length) {
			while(Positions.Count >= length) {
				Positions.Dequeue();
			}
			Positions.Enqueue(Target.position);
		}

		public void Draw(float width, Color color) {
			UnityGL.Start();
			int index = 0;
			Vector3 previous = Vector3.zero;
			foreach(Vector3 position in Positions) {
				if(index > 1) {
					UnityGL.DrawLine(previous, position, width, color);
				}
				previous = position;
				index += 1;
			}
			UnityGL.Finish();
		}
	}

}
