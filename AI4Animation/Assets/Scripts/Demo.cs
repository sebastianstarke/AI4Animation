using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class Demo : MonoBehaviour {

	public Transform Root;

	public float Scale = 1f;
	public float Speed = 0f;
	public float Phase = 0f;

	private PFNN Network;
	private Character Character;
	private Trajectory Trajectory;

	private const float M_PI = 3.14159265358979323846f;

	void Start() {
		//Network = new PFNN(PFNN.MODE.CONSTANT);
		Character = new Character(transform, Root);
		Trajectory = new Trajectory(transform);

		//Predict();
	}

	void Update() {
		Scale = Mathf.Max(1e-5f, Scale);

		PreUpdate();
		//Predict();
		//RegularUpdate();
		PostUpdate();
		
		//Vector3 angles = Quaternion.LookRotation(Trajectory.Velocities[Trajectory.Length/2], Vector3.up).eulerAngles;
		//angles.x = 0f;
		//transform.rotation = Quaternion.Euler(0f,90f,0f) * Quaternion.Euler(angles);
	}

	private void Predict() {
		/* Input Trajectory Positions / Directions */
		for(int i=0; i<Trajectory.Length; i+=10) {
			int w = (Trajectory.Length)/10;
			Vector3 pos = Quaternion.Inverse(Character.Transform.rotation) * (Trajectory.Positions[i] - Character.Transform.position);
			Vector3 dir = Quaternion.Inverse(Character.Transform.rotation) * Trajectory.Velocities[i].normalized;  
			Network.Xp[(w*0)+i/10, 0] = 0;
			Network.Xp[(w*1)+i/10, 0] = 0;
			Network.Xp[(w*2)+i/10, 0] = 0;
			Network.Xp[(w*3)+i/10, 0] = 0;
		}

		/* Input Trajectory Gaits */
		for (int i=0; i<Trajectory.Length; i+=10) {
			int w = (Trajectory.Length)/10;
			Network.Xp[(w*4)+i/10, 0] = 0;
			Network.Xp[(w*5)+i/10, 0] = 0;
			Network.Xp[(w*6)+i/10, 0] = 0;
			Network.Xp[(w*7)+i/10, 0] = 0;
			Network.Xp[(w*8)+i/10, 0] = 0;
			Network.Xp[(w*9)+i/10, 0] = 0;
		}

		//TODO: Maybe take previous state? But why?
		for(int i=0; i<Character.Joints.Length; i++) {
			int o = Trajectory.Length;
			Vector3 pos; Quaternion rot;
			Vector3 vel = Quaternion.Inverse(Character.Transform.rotation) * transform.forward;
			Character.Joints[i].GetConfiguration(out pos, out rot);
			pos = 1f/Scale * pos;
			//glm::vec3 prv = glm::inverse(prev_root_rotation) *  character->joint_velocities[i];
			Network.Xp[o+(Character.Joints.Length*3*0)+i*3+0, 0] = pos.x;
			Network.Xp[o+(Character.Joints.Length*3*0)+i*3+1, 0] = pos.y;
			Network.Xp[o+(Character.Joints.Length*3*0)+i*3+2, 0] = pos.z;
			Network.Xp[o+(Character.Joints.Length*3*1)+i*3+0, 0] = 0;
			Network.Xp[o+(Character.Joints.Length*3*1)+i*3+1, 0] = 0;
			Network.Xp[o+(Character.Joints.Length*3*1)+i*3+2, 0] = 0;
		}

		/* Input Trajectory Heights */
		for (int i=0; i<Trajectory.Length; i+=10) {
			int o = Trajectory.Length + Character.Joints.Length*3*2;
			int w = Trajectory.Length/10;
			/*
			glm::vec3 position_r = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3( trajectory->width, 0, 0));
			glm::vec3 position_l = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(-trajectory->width, 0, 0));
			pfnn->Xp(o+(w*0)+(i/10)) = heightmap->sample(glm::vec2(position_r.x, position_r.z)) - root_position.y;
			pfnn->Xp(o+(w*1)+(i/10)) = trajectory->positions[i].y - root_position.y;
			pfnn->Xp(o+(w*2)+(i/10)) = heightmap->sample(glm::vec2(position_l.x, position_l.z)) - root_position.y;
			*/
		}

		Phase += Speed * Time.deltaTime;
		Matrix<float> result = Network.Predict(Mathf.Repeat(Phase, 2f*M_PI));

		/*
		string output = string.Empty;
		for(int i=0; i<Network.YDim; i++) {
			output += result[i, 0] +  " ";
		}
		Debug.Log(output);
		*/

		for(int i=0; i<Character.Joints.Length; i++) {
			int opos = 8+(((Trajectory.Length/2)/10)*4)+(Character.Joints.Length*3*0);
			int orot = 8+(((Trajectory.Length/2)/10)*4)+(Character.Joints.Length*3*2);
			
			Vector3 position = Scale * new Vector3(result[opos+i*3+0, 0], result[opos+i*3+1, 0], result[opos+i*3+2, 0]);
			//Quaternion rotation = Quaternion.Euler(new Vector3(result[orot+i*3+2, 0], result[orot+i*3+0, 0], result[orot+i*3+1, 0]));

			Vector3 pos;
			Quaternion rot;
			Character.Joints[i].GetConfiguration(out pos, out rot);
			//Quaternion rotation = quat_exp(new Vector3(result[orot+i*3+0, 0], result[orot+i*3+1, 0], result[orot+i*3+2, 0]));

			Character.Joints[i].SetConfiguration(position, rot);

			//Debug.Log(position);
			//Debug.Log(rotation);
		}
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

	/*
	private void RegularUpdate() {
		//Character.Move(new Vector2(XAxis, YAxis));
		//Character.Turn(Turn);

		//int current = Trajectory.Length/2;
		//int last = Trajectory.Length-1;
		//transform.position = Trajectory.Positions[current];
		//transform.rotation = Quaternion.LookRotation(Trajectory.Directions[current], Vector3.up);
		
		//Network.Predict(0.5f);
	}
	*/

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
		for(int i=0; i<Trajectory.Positions.Length-1; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i+1]);
		}
		Gizmos.color = Color.blue;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Trajectory.Velocities[i];
			Vector3 left = Trajectory.Positions[i] - 0.15f * ortho.normalized;
			left.y = Utility.GetHeight(left.x, left.z, LayerMask.GetMask("Ground"));
			Vector3 right = Trajectory.Positions[i] + 0.15f * ortho.normalized;
			right.y = Utility.GetHeight(right.x, right.z, LayerMask.GetMask("Ground"));
			Gizmos.DrawLine(Trajectory.Positions[i], left);
			Gizmos.DrawLine(Trajectory.Positions[i], right);
			Gizmos.DrawSphere(left, 0.01f);
			Gizmos.DrawSphere(right, 0.01f);
		}
		Gizmos.color = Color.green;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i] + Trajectory.Velocities[i]);
		}
		Gizmos.color = Color.red;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawSphere(Trajectory.Positions[i], 0.015f);
		}

		Gizmos.color = Color.cyan;
		Gizmos.DrawSphere(Trajectory.TargetPosition, 0.03f);
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetVelocity);
	}

}
