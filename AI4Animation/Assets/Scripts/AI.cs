using UnityEngine;

public class AI : MonoBehaviour {

	private PFNN Network;

	void Start() {
		Network = new PFNN(PFNN.MODE.CONSTANT);
		Network.Load();
		Network.Predict(0.5f);
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
		
	}

}
