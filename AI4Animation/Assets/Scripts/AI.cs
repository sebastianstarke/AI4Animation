using UnityEngine;

public class AI : MonoBehaviour {

	private PFNN Network;
	private Character Character;

	void Start() {
		Network = new PFNN(PFNN.MODE.CONSTANT);
		//Network.Load();
		//Network.Predict(0.5f);
	
		Character = new Character();
	}

	void Update() {
		PreUpdate();
		RegularUpdate();
		PostUpdate();
	}

	private void PreUpdate() {

	}

	private void RegularUpdate() {
		
	}

	private void PostUpdate() {
		//Character.Phase = 0.5f;	
	}

}
