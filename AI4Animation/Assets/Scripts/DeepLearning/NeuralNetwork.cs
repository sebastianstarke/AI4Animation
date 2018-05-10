using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	[System.Serializable]
	public class NeuralNetwork {

		public bool Inspect = false;

		public TYPE Type;

		public Model Model;

		public NeuralNetwork(TYPE type) {
			SetModel(type);
		}

		public void SetModel(TYPE type) {
			if(Type == type && Model != null) {
				return;
			}
			Type = type;
			switch(Type) {
				case TYPE.Vanilla:
				Model = ScriptableObject.CreateInstance<Vanilla>();
				break;
				case TYPE.MANN:
				Model = ScriptableObject.CreateInstance<MANN>();
				break;
				case TYPE.PFNN:
				Model = ScriptableObject.CreateInstance<PFNN>();
				break;
				case TYPE.HMANN:
				Model = ScriptableObject.CreateInstance<HMANN>();
				break;
			}
		}
		
		#if UNITY_EDITOR
		public void Inspector() {
			Utility.SetGUIColor(Color.grey);
			using(new GUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				if(Utility.GUIButton("Neural Network", UltiDraw.DarkGrey, UltiDraw.White)) {
					Inspect = !Inspect;
				}

				if(Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						SetModel((TYPE)EditorGUILayout.EnumPopup(Type));
						Model.Inspector();
					}
				}
			}
		}
		#endif

	}

}
