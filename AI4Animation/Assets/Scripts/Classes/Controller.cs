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

	public Style[] Styles = new Style[0];

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

	public void SetStyleCount(int count) {
		count = Mathf.Max(count, 0);
		if(Styles.Length != count) {
			int size = Styles.Length;
			System.Array.Resize(ref Styles, count);
			for(int i=size; i<count; i++) {
				Styles[i] = new Style();
			}
		}
	}

	[System.Serializable]
	public class Style {
		public string Name;
		public float Bias = 1f;
		public KeyCode[] Keys = new KeyCode[0];

		public float Query() {
			if(Keys.Length == 0) {
				return 0f;
			}

			//OR
			for(int i=0; i<Keys.Length; i++) {
				if(Keys[i] == KeyCode.None) {
					if(!Input.anyKey) {
						return 1f;
					}
				} else {
					if(Input.GetKey(Keys[i])) {
						return 1f;
					}
				}
			}

			//AND
			//(TODO)

			return 0f;
		}

		public void SetKeyCount(int count) {
			count = Mathf.Max(count, 0);
			if(Keys.Length != count) {
				System.Array.Resize(ref Keys, count);
			}
		}
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Controller", Utility.DarkGrey, Utility.White)) {
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
					SetStyleCount(EditorGUILayout.IntField("Styles", Styles.Length));
					for(int i=0; i<Styles.Length; i++) {
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Styles[i].Name = EditorGUILayout.TextField("Name", Styles[i].Name);
							Styles[i].Bias = EditorGUILayout.FloatField("Bias", Styles[i].Bias);
							Styles[i].SetKeyCount(EditorGUILayout.IntField("Keys", Styles[i].Keys.Length));
							for(int j=0; j<Styles[i].Keys.Length; j++) {
								Styles[i].Keys[j] = (KeyCode)EditorGUILayout.EnumPopup("Key", Styles[i].Keys[j]);
							}
						}
					}
				}
			}
		}
	}
	#endif

}
