using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Controller {

	public bool Inspect = false;

	public KeyCode Forward = KeyCode.W;
	public KeyCode Back = KeyCode.S;
	public KeyCode Left = KeyCode.A;
	public KeyCode Right = KeyCode.D;
	public KeyCode TurnLeft = KeyCode.Q;
	public KeyCode TurnRight = KeyCode.E;

	public Style[] Styles = new Style[0];

	public float[] GetStyle() {
		float[] style = new float[Styles.Length];
		for(int i=0; i<Styles.Length; i++) {
			style[i] = Styles[i].Query() ? 1f : 0f;
		}
		return style;
	}

	public string[] GetNames() {
		string[] names = new string[Styles.Length];
		for(int i=0; i<names.Length; i++) {
			names[i] = Styles[i].Name;
		}
		return names;
	}

	public Vector3 QueryMove() {
		Vector3 move = Vector3.zero;
		if(InputHandler.GetKey(Forward)) {
			move.z += 1f;
		}
		if(InputHandler.GetKey(Back)) {
			move.z -= 1f;
		}
		if(InputHandler.GetKey(Left)) {
			move.x -= 1f;
		}
		if(InputHandler.GetKey(Right)) {
			move.x += 1f;
		}
		return move;
	}

	public float QueryTurn() {
		float turn = 0f;
		if(InputHandler.GetKey(TurnLeft)) {
			turn -= 1f;
		}
		if(InputHandler.GetKey(TurnRight)) {
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

	public float PoolBias(float[] weights) {
		float bias = 0f;
		for(int i=0; i<weights.Length; i++) {
			float _bias = Styles[i].Bias;
			float max = 0f;
			for(int j=0; j<Styles[i].Multipliers.Length; j++) {
				if(InputHandler.GetKey(Styles[i].Multipliers[j].Key)) {
					max = Mathf.Max(max, Styles[i].Bias * Styles[i].Multipliers[j].Value);
				}
			}
			for(int j=0; j<Styles[i].Multipliers.Length; j++) {
				if(InputHandler.GetKey(Styles[i].Multipliers[j].Key)) {
					_bias = Mathf.Min(max, _bias * Styles[i].Multipliers[j].Value);
				}
			}
			bias += weights[i] * _bias;
		}
		return bias;
	}

	[System.Serializable]
	public class Style {
		public string Name;
		public float Bias = 1f;
		public float Transition = 0.1f;
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
						if(!InputHandler.anyKey) {
							active = true;
						}
					} else {
						if(InputHandler.GetKey(Keys[i])) {
							active = true;
						}
					}
				}
			}

			for(int i=0; i<Keys.Length; i++) {
				if(Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!InputHandler.anyKey) {
							active = false;
						}
					} else {
						if(InputHandler.GetKey(Keys[i])) {
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
					Forward = (KeyCode)EditorGUILayout.EnumPopup("Forward", Forward);
					Back = (KeyCode)EditorGUILayout.EnumPopup("Backward", Back);
					Left = (KeyCode)EditorGUILayout.EnumPopup("Left", Left);
					Right = (KeyCode)EditorGUILayout.EnumPopup("Right", Right);
					TurnLeft = (KeyCode)EditorGUILayout.EnumPopup("Turn Left", TurnLeft);
					TurnRight = (KeyCode)EditorGUILayout.EnumPopup("Turn Right", TurnRight);
					SetStyleCount(EditorGUILayout.IntField("Styles", Styles.Length));
					for(int i=0; i<Styles.Length; i++) {
						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {

							Utility.ResetGUIColor();
							Styles[i].Name = EditorGUILayout.TextField("Name", Styles[i].Name);
							Styles[i].Bias = EditorGUILayout.FloatField("Bias", Styles[i].Bias);
							Styles[i].Transition = EditorGUILayout.Slider("Transition", Styles[i].Transition, 0f, 1f);
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
