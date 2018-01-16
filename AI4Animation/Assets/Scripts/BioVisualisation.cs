using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(BioAnimation_APFNN))]
public class BioVisualisation : MonoBehaviour {

	public Button StraightForward, StraightBack, StraightLeft, StraightRight, TurnLeft, TurnRight, Sprint, Jump, Sit, Lie;
	public Color Active = Utility.Orange;
	public Color Inactive = Utility.DarkGrey;

	public Button Skeleton, Transforms, Velocities, Trajectory, CyclicWeights, InverseKinematics;
	public Color Enabled = Utility.Cyan;
	public Color Disabled = Utility.Grey; 

	private int Frames = 150;
	private Queue<float>[] CW;
	private bool DrawCW = true;

	private BioAnimation_APFNN Animation;

	void Awake() {
		Animation = GetComponent<BioAnimation_APFNN>();
		CW = new Queue<float>[4];
		for(int i=0; i<4; i++) {
			CW[i] = new Queue<float>();
			for(int j=0; j<Frames; j++) {
				CW[i].Enqueue(0f);
			}
		}
		Skeleton.onClick.AddListener(ToggleSkeleton); UpdateColor(Skeleton, Animation.Character.DrawSkeleton);
		Transforms.onClick.AddListener(ToggleTransforms); UpdateColor(Transforms, Animation.Character.DrawTransforms);
		Velocities.onClick.AddListener(ToggleVelocities); UpdateColor(Velocities, Animation.ShowVelocities);
		Trajectory.onClick.AddListener(ToggleTrajectory); UpdateColor(Trajectory, Animation.ShowTrajectory);
		CyclicWeights.onClick.AddListener(ToggleCyclicWeights); UpdateColor(CyclicWeights, DrawCW);
		InverseKinematics.onClick.AddListener(ToggleInverseKinematics); UpdateColor(InverseKinematics, Animation.SolveIK);
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
					color
				);
			}
			_xPrev = _x; 
			_yPrev = _y;
			_index += 1;
		}
	}

	public void UpdateColor(Button button, bool value) {
		if(value) {
			button.GetComponent<Image>().color = Enabled;
		} else {
			button.GetComponent<Image>().color = Disabled;
		}
	}

	public void ToggleSkeleton() {
		Animation.Character.DrawSkeleton = !Animation.Character.DrawSkeleton;
		UpdateColor(Skeleton, Animation.Character.DrawSkeleton);
	}

	public void ToggleTransforms() {
		Animation.Character.DrawTransforms = !Animation.Character.DrawTransforms;
		UpdateColor(Transforms, Animation.Character.DrawTransforms);
	}

	public void ToggleVelocities() {
		Animation.ShowVelocities = !Animation.ShowVelocities;
		UpdateColor(Velocities, Animation.ShowVelocities);
	}

	public void ToggleTrajectory() {
		Animation.ShowTrajectory = !Animation.ShowTrajectory;
		UpdateColor(Trajectory, Animation.ShowTrajectory);
	}

	public void ToggleCyclicWeights() {
		DrawCW = !DrawCW;
		UpdateColor(CyclicWeights, DrawCW);
	}

	public void ToggleInverseKinematics() {
		Animation.SolveIK = !Animation.SolveIK;
		UpdateColor(InverseKinematics, Animation.SolveIK);
	}

	void OnGUI() {
		if(Input.GetKey(Animation.Controller.MoveForward)) {
			StraightForward.GetComponent<Image>().color = Active;
		} else {
			StraightForward.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(Animation.Controller.MoveBackward)) {
			StraightBack.GetComponent<Image>().color = Active;
		} else {
			StraightBack.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(Animation.Controller.MoveLeft)) {
			StraightLeft.GetComponent<Image>().color = Active;
		} else {
			StraightLeft.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(Animation.Controller.MoveRight)) {
			StraightRight.GetComponent<Image>().color = Active;
		} else {
			StraightRight.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(Animation.Controller.TurnLeft)) {
			TurnLeft.GetComponent<Image>().color = Active;
		} else {
			TurnLeft.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(Animation.Controller.TurnRight)) {
			TurnRight.GetComponent<Image>().color = Active;
		} else {
			TurnRight.GetComponent<Image>().color = Inactive;
		}

		if(Input.GetKey(KeyCode.LeftShift)) {
			Sprint.GetComponent<Image>().color = Active;
		} else {
			Sprint.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(KeyCode.Space)) {
			Jump.GetComponent<Image>().color = Active;
		} else {
			Jump.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(KeyCode.LeftControl)) {
			Sit.GetComponent<Image>().color = Active;
		} else {
			Sit.GetComponent<Image>().color = Inactive;
		}
		if(Input.GetKey(KeyCode.LeftAlt)) {
			Lie.GetComponent<Image>().color = Active;
		} else {
			Lie.GetComponent<Image>().color = Inactive;
		}
	}

	void OnRenderObject() {
		UpdateData();

		if(DrawCW) {
			UnityGL.Start();
			float x = 0.025f;
			float y = 0.15f;
			float width = 0.95f;
			float height = 0.1f;
			float border = 0.005f;
			UnityGL.DrawGUIQuad(x-border/Screen.width*Screen.height, y-height-border, width+2f*border/Screen.width*Screen.height, height+2f*border, Utility.Black.Transparent(0.5f));
			UnityGL.DrawGUIQuad(x, y-height, width, height, Utility.Black.Transparent(0.75f));
			DrawControlPoint(x, y, width, height, CW[0], Utility.Red);
			DrawControlPoint(x, y, width, height, CW[1], Utility.Green);
			DrawControlPoint(x, y, width, height, CW[2], Utility.Cyan);
			DrawControlPoint(x, y, width, height, CW[3], Utility.Orange);
			UnityGL.Finish();
		}
	}

}
