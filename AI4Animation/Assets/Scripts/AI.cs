using UnityEngine;

public class AI : MonoBehaviour {

	private PFNN Network;

	void Start() {
		Network = new PFNN(PFNN.MODE.CONSTANT);
		Network.Load();
	}

	void Update() {

	}

}
