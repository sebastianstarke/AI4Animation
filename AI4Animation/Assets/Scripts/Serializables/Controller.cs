using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Controller {

	public bool Inspect = false;

	public KeyCode MoveForward = KeyCode.W;
	public KeyCode MoveBackward = KeyCode.S;
	public KeyCode MoveLeft = KeyCode.A;
	public KeyCode MoveRight = KeyCode.D;
	public KeyCode TurnLeft = KeyCode.Q;
	public KeyCode TurnRight = KeyCode.E;
	public KeyCode Jog = KeyCode.LeftShift;
	public KeyCode Crouch = KeyCode.LeftControl;

	public Controller() {

	}
	
	public Vector3 QueryMove() {
		Vector3 move = Vector3.zero;
		if(Input.GetKey(MoveForward)) {
			move.z += 1f;
		}
		if(Input.GetKey(MoveBackward)) {
			move.z -= 1f;
		}
		if(Input.GetKey(MoveLeft)) {
			move.x -= 1f;
		}
		if(Input.GetKey(MoveRight)) {
			move.x += 1f;
		}
		return move;
	}

	public float QueryTurn() {
		float turn = 0f;
		if(Input.GetKey(TurnLeft)) {
			turn -= 1f;
		}
		if(Input.GetKey(TurnRight)) {
			turn += 1f;
		}
		return turn;
	}

	public float QueryJog() {
		return Input.GetKey(Jog) ? 1f : 0f;
	}

	public float QueryCrouch() {
		return Input.GetKey(Crouch) ? 1f : 0f;
	}

	#if UNITY_EDITOR
	public void Inspector() {
		using(new EditorGUILayout.VerticalScope ("Box")) {
			if(GUILayout.Button("Controller")) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					MoveForward = (KeyCode)EditorGUILayout.EnumPopup("Move Forward", MoveForward);
					MoveBackward = (KeyCode)EditorGUILayout.EnumPopup("Move Backward", MoveBackward);
					MoveLeft = (KeyCode)EditorGUILayout.EnumPopup("Move Left", MoveLeft);
					MoveRight = (KeyCode)EditorGUILayout.EnumPopup("Move Right", MoveRight);
					TurnLeft = (KeyCode)EditorGUILayout.EnumPopup("Turn Left", TurnLeft);
					TurnRight = (KeyCode)EditorGUILayout.EnumPopup("Turn Right", TurnRight);
					Jog = (KeyCode)EditorGUILayout.EnumPopup("Jog", Jog);
					Crouch = (KeyCode)EditorGUILayout.EnumPopup("Crouch", Crouch);
				}
			}
		}
	}
	#endif

}
