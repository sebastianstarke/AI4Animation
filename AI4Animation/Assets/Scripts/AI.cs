using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class AI : MonoBehaviour {

	private PFNN Network;
	private Character Character;

	public float MoveSpeed = 1f;
	public float TurnSpeed = 10f;

	private const float M_PI = 3.14159265358979323846f;

	void Start() {
		//Network = new PFNN(PFNN.MODE.CONSTANT);
		//Network.Load();
		
		Character = new Character();
	}

	void Update() {
		PreUpdate();
		RegularUpdate();
		PostUpdate();
	}

	private void PreUpdate() {
		if(Input.GetKey(KeyCode.W)) {
			transform.position += MoveSpeed * Time.deltaTime * (transform.rotation * new Vector3(0,0,1));
		}
		if(Input.GetKey(KeyCode.S)) {
			transform.position += MoveSpeed * Time.deltaTime * (transform.rotation * new Vector3(0,0,-1));
		}
		if(Input.GetKey(KeyCode.A)) {
			transform.position += MoveSpeed * Time.deltaTime * (transform.rotation * new Vector3(-1,0,0));
		}
		if(Input.GetKey(KeyCode.D)) {
			transform.position += MoveSpeed * Time.deltaTime * (transform.rotation * new Vector3(1,0,0));
		}
		if(Input.GetKey(KeyCode.Q)) {
			transform.rotation *= Quaternion.Euler(0f, -TurnSpeed*Time.deltaTime, 0f);
		}
		if(Input.GetKey(KeyCode.E)) {
			transform.rotation *= Quaternion.Euler(0f, TurnSpeed*Time.deltaTime, 0f);
		}
	}

	private void RegularUpdate() {
		
	}

	private void PostUpdate() {
		//Character.Phase = GetPhase();
	}

	private float GetPhase() {
		float stand_amount = 0f;
		float factor = 0.9f;
		return Mathf.Repeat(Character.Phase + stand_amount*factor + (1f-factor), 2f*M_PI);
	}

}
