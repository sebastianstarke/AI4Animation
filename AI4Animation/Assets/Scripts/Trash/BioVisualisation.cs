using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BioVisualisation : MonoBehaviour {

	public Button StraightForward, StraightBack, StraightLeft, StraightRight, TurnLeft, TurnRight, Idle, Move, Jump, Sit, Lie, Stand;
	public Color Active = UltiDraw.Orange;
	public Color Inactive = UltiDraw.DarkGrey;

	private SIGGRAPH_2018.BioAnimation Animation;

	private Actor Actor;

	void Awake() {
		Animation = GetComponent<SIGGRAPH_2018.BioAnimation>();
		Actor = GetComponent<Actor>();
	}

	void Start() {

	}


	void OnGUI() {
		//GameObject.Find("Trajectory_Circle").GetComponent<CatmullRomSpline>().DrawGUI = Show;
		//GameObject.Find("Trajectory_Square").GetComponent<CatmullRomSpline>().DrawGUI = Show;
		//GameObject.Find("Trajectory_Slalom").GetComponent<CatmullRomSpline>().DrawGUI = Show;

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
		if(index >= Animation.GetTrajectory().Points[0].Styles.Length) {
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
			UltiDraw.Begin();
			int index = 0;
			Vector3 previous = Vector3.zero;
			foreach(Vector3 position in Positions) {
				if(index > 1) {
					UltiDraw.DrawLine(previous, position, width, color);
				}
				previous = position;
				index += 1;
			}
			UltiDraw.End();
		}
	}

}
