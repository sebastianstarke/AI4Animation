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

	public bool QueryAny() {
		for(int i=0; i<Styles.Length; i++) {
			if(Styles[i].Query()) {
				return true;
			}
		}
		return false;
	}

	[System.Serializable]
	public class Style {
		public string Name;
		public float Bias = 1f;
		public KeyCode[] Keys = new KeyCode[0];
		public bool[] Negations = new bool[0];
		public Multiplier[] Multipliers = new Multiplier[0];

		public bool Query() {
			if(Keys.Length == 0) {
				return false;
			}

			bool active = false;

			for(int i=0; i<Keys.Length; i++) {
				if(!Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Input.anyKey) {
							active = true;
						}
					} else {
						if(Input.GetKey(Keys[i])) {
							active = true;
						}
					}
				}
			}

			for(int i=0; i<Keys.Length; i++) {
				if(Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Input.anyKey) {
							active = false;
						}
					} else {
						if(Input.GetKey(Keys[i])) {
							active = false;
						}
					}
				}
			}

			return active;
		}

		public void SetKeyCount(int count) {
			count = Mathf.Max(count, 0);
			if(Keys.Length != count) {
				System.Array.Resize(ref Keys, count);
				System.Array.Resize(ref Negations, count);
			}
		}

		public void AddMultiplier() {
			ArrayExtensions.Add(ref Multipliers, new Multiplier());
		}

		public void RemoveMultiplier() {
			ArrayExtensions.Shrink(ref Multipliers);
		}

		[System.Serializable]
		public class Multiplier {
			public KeyCode Key;
			public float Value;
		}
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Controller", UltiDraw.DarkGrey, UltiDraw.White)) {
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
						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {

							Utility.ResetGUIColor();
							Styles[i].Name = EditorGUILayout.TextField("Name", Styles[i].Name);
							Styles[i].Bias = EditorGUILayout.FloatField("Bias", Styles[i].Bias);
							Styles[i].SetKeyCount(EditorGUILayout.IntField("Keys", Styles[i].Keys.Length));

							for(int j=0; j<Styles[i].Keys.Length; j++) {
								EditorGUILayout.BeginHorizontal();
								Styles[i].Keys[j] = (KeyCode)EditorGUILayout.EnumPopup("Key", Styles[i].Keys[j]);
								Styles[i].Negations[j] = EditorGUILayout.Toggle("Negate", Styles[i].Negations[j]);
								EditorGUILayout.EndHorizontal();
							}

							for(int j=0; j<Styles[i].Multipliers.Length; j++) {
								Utility.SetGUIColor(Color.grey);
								using(new GUILayout.VerticalScope ("Box")) {
									Utility.ResetGUIColor();
									Styles[i].Multipliers[j].Key = (KeyCode)EditorGUILayout.EnumPopup("Key", Styles[i].Multipliers[j].Key);
									Styles[i].Multipliers[j].Value = EditorGUILayout.FloatField("Value", Styles[i].Multipliers[j].Value);
								}
							}
							
							if(Utility.GUIButton("Add Multiplier", UltiDraw.DarkGrey, UltiDraw.White)) {
								Styles[i].AddMultiplier();
							}
							if(Utility.GUIButton("Remove Multiplier", UltiDraw.DarkGrey, UltiDraw.White)) {
								Styles[i].RemoveMultiplier();
							}
						}
					}
				}
			}
		}
	}
	#endif

}
