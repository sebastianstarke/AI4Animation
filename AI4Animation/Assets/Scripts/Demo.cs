using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class Demo : MonoBehaviour {

	public Transform ChainStart;

	//public float Scale = 1f;
	public float Speed = 0f;
	public float Phase = 0f;

	private PFNN Network;
	private Character Character;
	private Trajectory Trajectory;

	private const float M_PI = 3.14159265358979323846f;

	void Start() {
		Network = new PFNN(PFNN.MODE.CONSTANT);
		Character = new Character(transform, ChainStart);
		Trajectory = new Trajectory(transform, 120, 0.15f);

		//Predict();
	}

	void Update() {
		//Scale = Mathf.Max(1e-5f, Scale);

		PreUpdate();
		Predict();
		PostUpdate();
	}

	private void Predict() {
		////////////////////////////////////////
		////////// Input
		////////////////////////////////////////

		/* Input Trajectory Positions / Directions */
		//Debug.Log("Setting Trajectory Positions / Directions");
		for(int i=0; i<Trajectory.Length; i+=10) {
			int w = (Trajectory.Length)/10;
			Vector3 pos = Trajectory.Points[i].GetRelativeToRootPosition();
			Vector3 dir = Trajectory.Points[i].GetRelativeToRootDirection();
			Network.SetInput((w*0)+i/10, pos.x);
			Network.SetInput((w*1)+i/10, pos.z);
			Network.SetInput((w*2)+i/10, dir.x);
			Network.SetInput((w*3)+i/10, dir.z);
		}

		/* Input Trajectory Gaits */
		//Debug.Log("Setting Trajectory Gaits");
		for (int i=0; i<Trajectory.Length; i+=10) {
			int w = (Trajectory.Length)/10;
			Network.SetInput((w*4)+i/10, 0f);
			Network.SetInput((w*5)+i/10, 0f);
			Network.SetInput((w*6)+i/10, 0f);
			Network.SetInput((w*7)+i/10, 0f);
			Network.SetInput((w*8)+i/10, 0f);
			Network.SetInput((w*9)+i/10, 0f);
		}

		/* Input Joint Previous Positions / Velocities / Rotations */
		//Debug.Log("Setting Joint Positions / Velocities / Rotations");
		for(int i=0; i<Character.Joints.Length; i++) {
			int o = (((Trajectory.Length)/10)*10);  
			Vector3 pos = Character.Joints[i].GetRelativeToRootPosition(Character.Joints[i].Transform.position);
			Vector3 vel = Vector3.zero;
			Network.SetInput(o+(Character.Joints.Length*3*0)+i*3+0, pos.x);
			Network.SetInput(o+(Character.Joints.Length*3*0)+i*3+1, pos.y);
			Network.SetInput(o+(Character.Joints.Length*3*0)+i*3+2, pos.z);
			Network.SetInput(o+(Character.Joints.Length*3*1)+i*3+0, vel.x);
			Network.SetInput(o+(Character.Joints.Length*3*1)+i*3+1, vel.y);
			Network.SetInput(o+(Character.Joints.Length*3*1)+i*3+2, vel.z);
		}

		/* Input Trajectory Heights */
		//Debug.Log("Setting Trajectory Heights");
		for (int i=0; i<Trajectory.Length; i+=10) {
			int o = (((Trajectory.Length)/10)*10) + Character.Joints.Length*3*2;
			int w = Trajectory.Length/10;
			Network.SetInput(o+(w*0)+(i/10), (Trajectory.Points[i].ProjectLeft().y - Character.Root.position.y));
			Network.SetInput(o+(w*1)+(i/10), (Trajectory.Points[i].Position.y - Character.Root.position.y));
			Network.SetInput(o+(w*2)+(i/10), (Trajectory.Points[i].ProjectRight().y - Character.Root.position.y));
		}

		////////////////////////////////////////
		////////// Predict
		////////////////////////////////////////
		//Phase += Speed * Time.deltaTime;
		
		Character.Root.position = Vector3.zero;
		Character.Root.rotation = Quaternion.identity;
		for(int i=0; i<342; i++) {
			Network.SetInput(i, 0f);
		}
		Phase = 0f;

		Network.Predict(Mathf.Repeat(Phase, 2f*M_PI));

		/*
		string output = string.Empty;
		for(int i=0; i<Network.YDim; i++) {
			output += "OUTPUT " + i + ": " + Network.GetOutput(i) +  "\n";
		}
		Debug.Log(output);
		*/

		////////////////////////////////////////
		////////// Output
		////////////////////////////////////////

		int opos = 8+(((Trajectory.Length/2)/10)*4)+(Character.Joints.Length*3*0);
		int orot = 8+(((Trajectory.Length/2)/10)*4)+(Character.Joints.Length*3*2);
		for(int i=0; i<Character.Joints.Length; i++) {			
			Vector3 position = new Vector3(Network.GetOutput(opos+i*3+0), Network.GetOutput(opos+i*3+1), Network.GetOutput(opos+i*3+2));
			Quaternion rotation = quat_exp(new Vector3(Network.GetOutput(orot+i*3+0), Network.GetOutput(orot+i*3+1), Network.GetOutput(orot+i*3+2)));
			//Quaternion rotation = Quaternion.Euler(Mathf.Rad2Deg*new Vector3(Network.GetOutput(orot+i*3+0), Network.GetOutput(orot+i*3+1), Network.GetOutput(orot+i*3+2)));

			//Debug.Log("JOINT NUM: " + i + " X: " + position.x + " Y: " + position.y + " Z: " + position.z);

			Character.Joints[i].SetPosition(position);
			Character.Joints[i].SetRotation(rotation);

			//Debug.Log("QUAT: " + i + " X: " + rotation.x + " Y: " + rotation.y + " Z: " + rotation.z + " W: " + rotation.w);

			//Debug.Log("JOINT NUM: " + i + " X: " + Character.Joints[i].Transform.position.x + " Y: " + Character.Joints[i].Transform.position.y + " Z: " + Character.Joints[i].Transform.position.z);
		}

		Character.ForwardKinematics();
	}

	private Quaternion quat_exp(Vector3 l) {
		float w = l.magnitude;
		Quaternion q = w < 0.01f ? new Quaternion(0f, 0f, 0f, 1f) : new Quaternion(
			l.x * (Mathf.Sin(w) / w),
			l.y * (Mathf.Sin(w) / w),
			l.z * (Mathf.Sin(w) / w),
			Mathf.Cos(w)
			);
		float div = Mathf.Sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
		return new Quaternion(q.x/div, q.y/div, q.z/div, q.w/div);
	}

	private void PreUpdate() {
		float x = 0f;
		float y = 0f;
		if(Input.GetKey(KeyCode.W)) {
			y += 1f;
		}
		if(Input.GetKey(KeyCode.S)) {
			y -= 1f;
		}
		if(Input.GetKey(KeyCode.A)) {
			x -= 1f;
		}
		if(Input.GetKey(KeyCode.D)) {
			x += 1f;
		}

		Trajectory.Predict(new Vector3(x, 0f, y).normalized);
	}

	private void PostUpdate() {
		Trajectory.Correct();
	}

	private float GetPhase() {
		float stand_amount = 0f;
		float factor = 0.9f;
		return Mathf.Repeat(Character.Phase + stand_amount*factor + (1f-factor), 2f*M_PI);
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			return;
		}
		Gizmos.color = Color.black;
		for(int i=0; i<Trajectory.Points.Length-1; i++) {
			Gizmos.DrawLine(Trajectory.Points[i].Position, Trajectory.Points[i+1].Position);
		}
		Gizmos.color = Color.blue;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Vector3 center = Trajectory.Points[i].Position;
			Vector3 left = Trajectory.Points[i].ProjectLeft();
			Vector3 right = Trajectory.Points[i].ProjectRight();
			Gizmos.DrawLine(center, left);
			Gizmos.DrawLine(center, right);
			Gizmos.DrawSphere(left, 0.01f);
			Gizmos.DrawSphere(right, 0.01f);
		}
		Gizmos.color = Color.green;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Gizmos.DrawLine(Trajectory.Points[i].Position, Trajectory.Points[i].Position + Trajectory.Points[i].Velocity);
		}
		Gizmos.color = Color.red;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Gizmos.DrawSphere(Trajectory.Points[i].Position, 0.015f);
		}

		Gizmos.color = Color.cyan;
		Gizmos.DrawSphere(Trajectory.TargetPosition, 0.03f);
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetVelocity);
	}

}
