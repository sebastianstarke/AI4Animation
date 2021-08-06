using UnityEngine;
using DeepLearning;

public class QuadrupedController : NeuralAnimation {

	public LayerMask CollisionMask = ~0;
	public Camera Camera = null;

	public Controller.TYPE ControlType = Controller.TYPE.Gamepad;

	public bool DrawGUI = true;
	public bool DrawDebug = true;

	public float WalkSpeed = 1.5f;
	public float SprintSpeed = 4f;

	public float PositionBias = 0.025f;
	public float RotationBias = 0.05f;
	public float VelocityBias = 0.1f;
	public float ActionBias = 0.25f;
	public float CorrectionBias = 0.25f;

	public float ControlStrength = 0.1f;
	public float CorrectionStrength = 1f;

	public float ContactPower = 3f;
	public float ContactThreshold = 0.7f;

	private Controller Controller;

	private TimeSeries TimeSeries;
	private RootSeries RootSeries;
	private StyleSeries StyleSeries;
	private ContactSeries ContactSeries;
	private PhaseSeries PhaseSeries;

	private UltimateIK.Model LeftFootIK;
	private UltimateIK.Model RightFootIK;
	private UltimateIK.Model LeftHandIK;
	private UltimateIK.Model RightHandIK;

	private CapsuleCollider Collider = null;

	private Camera GetCamera() {
		return Camera == null ? Camera.main : Camera;
	}

	protected override void Setup() {	
		Collider = GetComponent<CapsuleCollider>();
		Controller = new Controller(1);

		Controller.Logic idle = Controller.AddLogic("Idle", () => Controller.QueryLeftJoystickVector().magnitude < 0.1f);
		Controller.Function idleControl = Controller.AddFunction("IdleControl", (x) => TimeSeries.GetControl((int)x, ActionBias, 0.1f, ControlStrength));
		Controller.Function idleCorrection = Controller.AddFunction("IdleCorrection", (x) => TimeSeries.GetCorrection((int)x, CorrectionBias, CorrectionStrength, 0f));
		Controller.Logic move = Controller.AddLogic("Move", () => !idle.Query());
		Controller.Function moveControl = Controller.AddFunction("MoveControl", (x) => TimeSeries.GetControl((int)x, ActionBias, 0.1f, ControlStrength));
		Controller.Function moveCorrection = Controller.AddFunction("MoveCorrection", (x) => TimeSeries.GetCorrection((int)x, CorrectionBias, CorrectionStrength, 0f));
		Controller.Logic speed = Controller.AddLogic("Speed", () => true);
		Controller.Function speedControl = Controller.AddFunction("SpeedControl", (x) => TimeSeries.GetControl((int)x, ActionBias, 0.1f, ControlStrength));
		Controller.Function speedCorrection = Controller.AddFunction("SpeedCorrection", (x) => TimeSeries.GetCorrection((int)x, CorrectionBias, CorrectionStrength, 0f));

		Controller.Logic sprint = Controller.AddLogic("Sprint", () => move.Query() && Controller.QueryLeftJoystickVector().y > 0.25f);

		Controller.Function rootPositionControl = Controller.AddFunction("RootPositionControl", (x) => TimeSeries.GetControl((int)x, 
			PositionBias, 
			0.1f,
			ControlStrength
		));
		Controller.Function rootPositionCorrection = Controller.AddFunction("RootPositionCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			CorrectionBias, 
			CorrectionStrength, 
			0f
		));
		Controller.Function rootRotationControl = Controller.AddFunction("RootRotationControl", (x) => TimeSeries.GetControl((int)x, 
			RotationBias,
			0.1f,
			ControlStrength
		));
		Controller.Function rootRotationCorrection = Controller.AddFunction("RootRotationCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			CorrectionBias,
			CorrectionStrength, 
			0f
		));
		Controller.Function rootVelocityControl = Controller.AddFunction("RootVelocityControl", (x) => TimeSeries.GetControl((int)x, 
			VelocityBias,
			0.1f,
			ControlStrength
		));
		Controller.Function rootVelocityCorrection = Controller.AddFunction("RootVelocityCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			CorrectionBias,
			CorrectionStrength, 
			0f
		));

		Controller.Function phaseStability = Controller.AddFunction("PhaseStability", (x) => TimeSeries.GetCorrection((int)x, 
			2f,
			1f,
			0.5f
		));

		TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
		RootSeries = new RootSeries(TimeSeries, transform);
		StyleSeries = new StyleSeries(TimeSeries, new string[]{"Idle", "Move", "Speed"}, new float[]{1f, 0f, 0f});
		ContactSeries = new ContactSeries(TimeSeries, "Left Hand", "Right Hand", "Left Foot", "Right Foot");
		PhaseSeries = new PhaseSeries(TimeSeries, "Left Hand", "Right Hand", "Left Foot", "Right Foot");

		LeftHandIK = UltimateIK.BuildModel(Actor.FindTransform("LeftForeArm"), Actor.GetBoneTransforms("LeftHandSite"));
		RightHandIK = UltimateIK.BuildModel(Actor.FindTransform("RightForeArm"), Actor.GetBoneTransforms("RightHandSite"));
		LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("LeftLeg"), Actor.GetBoneTransforms("LeftFootSite"));
		RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("RightLeg"), Actor.GetBoneTransforms("RightFootSite"));

		RootSeries.DrawGUI = DrawGUI;
		StyleSeries.DrawGUI = DrawGUI;
		ContactSeries.DrawGUI = DrawGUI;
		PhaseSeries.DrawGUI = DrawGUI;
		GetComponent<ExpertActivation>().Draw = DrawGUI;
		RootSeries.DrawScene = DrawDebug;
		StyleSeries.DrawScene = DrawDebug;
		ContactSeries.DrawScene = DrawDebug;
		PhaseSeries.DrawScene = DrawDebug;
	}

	protected override void Feed() {
		//User Input Control
		Control();

		//Get Root
		Matrix4x4 root = Actor.GetRoot().GetWorldMatrix(true);

		//Input Timeseries
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			int index = TimeSeries.GetKey(i).Index;
			NeuralNetwork.FeedXZ(RootSeries.GetPosition(index).GetRelativePositionTo(root));
			NeuralNetwork.FeedXZ(RootSeries.GetDirection(index).GetRelativeDirectionTo(root));
			NeuralNetwork.FeedXZ(RootSeries.Velocities[index].GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(StyleSeries.Values[index]);
		}

		//Input Character
		for(int i=0; i<Actor.Bones.Length; i++) {
			NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
		}

		//Input Contacts
		for(int i=0; i<=TimeSeries.PivotKey; i++) {
			int index = TimeSeries.GetKey(i).Index;
			NeuralNetwork.Feed(ContactSeries.Values[index]);
		}

		//Input Gating Features
		NeuralNetwork.Feed(PhaseSeries.GetAlignment());
	}

	protected override void Read() {
		//Update Past States
		ContactSeries.Increment(0, TimeSeries.Pivot);
		PhaseSeries.Increment(0, TimeSeries.Pivot);
		
		//Update Root State
		Vector3 offset = NeuralNetwork.ReadVector3();
		offset = Vector3.Lerp(offset, Vector3.zero, StyleSeries.Values[TimeSeries.Pivot].First());

		Matrix4x4 root = Actor.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
		RootSeries.Transformations[TimeSeries.Pivot] = root;
		RootSeries.Velocities[TimeSeries.Pivot] = NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root);
		
		for(int j=0; j<StyleSeries.Styles.Length; j++) {
			StyleSeries.Values[TimeSeries.Pivot][j] = Mathf.Lerp(
				StyleSeries.Values[TimeSeries.Pivot][j], 
				NeuralNetwork.Read(), 
				Controller.QueryFunction(StyleSeries.Styles[j] + "Correction", TimeSeries.Pivot)
			);
		}
		
		//Read Future States
		for(int i=TimeSeries.PivotKey+1; i<TimeSeries.KeyCount; i++) {
			int index = TimeSeries.GetKey(i).Index;

			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root).normalized, Vector3.up), Vector3.one);
			RootSeries.Transformations[index] = Utility.Interpolate(RootSeries.Transformations[index], m, 
				Controller.QueryFunction("RootPositionCorrection", index),
				Controller.QueryFunction("RootRotationCorrection", index)
			);
			RootSeries.Velocities[index] = Vector3.Lerp(RootSeries.Velocities[index], NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root), 
				Controller.QueryFunction("RootVelocityCorrection", index)
			);
			
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[index][j] = Mathf.Lerp(StyleSeries.Values[index][j], NeuralNetwork.Read(), Controller.QueryFunction(StyleSeries.Styles[j] + "Correction", index));
			}
		}

		//Read Posture
		Vector3[] positions = new Vector3[Actor.Bones.Length];
		Vector3[] forwards = new Vector3[Actor.Bones.Length];
		Vector3[] upwards = new Vector3[Actor.Bones.Length];
		Vector3[] velocities = new Vector3[Actor.Bones.Length];
		for(int i=0; i<Actor.Bones.Length; i++) {
			Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
			velocities[i] = velocity;
			positions[i] = Vector3.Lerp(Actor.Bones[i].Transform.position + velocity / Framerate, position, 0.5f);
			forwards[i] = forward;
			upwards[i] = upward;
		}

		//Update Contacts
		float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length, 0f, 1f);
		for(int i=0; i<ContactSeries.Bones.Length; i++) {
			ContactSeries.Values[TimeSeries.Pivot][i] = contacts[i].SmoothStep(ContactPower, ContactThreshold);
		}

		//Update Phases
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			int index = TimeSeries.GetKey(i).Index;
			float stability = Controller.QueryFunction("PhaseStability", index);
			for(int b=0; b<PhaseSeries.Bones.Length; b++) {
				Vector2 update = NeuralNetwork.ReadVector2();
				Vector3 state = NeuralNetwork.ReadVector2();
				float phase = Utility.PhaseValue(
					Vector2.Lerp(
						Utility.PhaseVector(Mathf.Repeat(PhaseSeries.Phases[index][b] + Utility.PhaseValue(update), 1f)),
						Utility.PhaseVector(Mathf.Repeat(PhaseSeries.Phases[index][b] + Utility.SignedPhaseUpdate(PhaseSeries.Phases[index][b], Utility.PhaseValue(state)), 1f)),
						stability).normalized
					);
				PhaseSeries.Amplitudes[index][b] = update.magnitude;
				PhaseSeries.Phases[index][b] = phase;
			}
		}

		//Interpolate Timeseries
		RootSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
		StyleSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
		PhaseSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);

		//Assign Posture
		transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
		transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Velocity = velocities[i];
			Actor.Bones[i].Transform.position = positions[i];
			Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
		}
		// Correct Twist
		for(int i=0; i<Actor.Bones.Length; i++) {
			if(Actor.Bones[i].Childs.Length == 1) {
				Vector3 position = Actor.Bones[i].Transform.position;
				Quaternion rotation = Actor.Bones[i].Transform.rotation;
				Vector3 childPosition = Actor.Bones[i].GetChild(0).Transform.position;
				Quaternion childRotation = Actor.Bones[i].GetChild(0).Transform.rotation;
				Vector3 aligned = (position - childPosition).normalized;
				float[] angles = new float[] {
					Vector3.Angle(rotation.GetRight(), aligned),
					Vector3.Angle(rotation.GetUp(), aligned),
					Vector3.Angle(rotation.GetForward(), aligned),
					Vector3.Angle(-rotation.GetRight(), aligned),
					Vector3.Angle(-rotation.GetUp(), aligned),
					Vector3.Angle(-rotation.GetForward(), aligned)
				};
				float min = angles.Min();
				if(min == angles[0]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(rotation.GetRight(), aligned) * rotation;
				}
				if(min == angles[1]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(rotation.GetUp(), aligned) * rotation;
				}
				if(min == angles[2]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(rotation.GetForward(), aligned) * rotation;
				}
				if(min == angles[3]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(-rotation.GetRight(), aligned) * rotation;
				}
				if(min == angles[4]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(-rotation.GetRight(), aligned) * rotation;
				}
				if(min == angles[5]) {
					Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(-rotation.GetForward(), aligned) * rotation;
				}
				Actor.Bones[i].GetChild(0).Transform.position = childPosition;
				Actor.Bones[i].GetChild(0).Transform.rotation = childRotation;
			}
		}

		//Resolve Trajectory Collisions
		RootSeries.ResolveCollisions(Collider.radius, CollisionMask);

		//Process Contact States
		ProcessFootIK(LeftHandIK, ContactSeries.Values[TimeSeries.Pivot][0]);
		ProcessFootIK(RightHandIK, ContactSeries.Values[TimeSeries.Pivot][1]);
		ProcessFootIK(LeftFootIK, ContactSeries.Values[TimeSeries.Pivot][2]);
		ProcessFootIK(RightFootIK, ContactSeries.Values[TimeSeries.Pivot][3]);
	}

	private void Control() {
		float KeyboardTurnAngle() {
			return Controller.QueryLogic("Sprint") ? 45f : 60f;
		}

		Controller.ControlType = ControlType;

		//Update Past
		RootSeries.Increment(0, TimeSeries.Samples.Length-1);
		StyleSeries.Increment(0, TimeSeries.Samples.Length-1);

		//Update User Controller Inputs
		Controller.Update();

		//Locomotion
		Vector3 move = Controller.QueryLeftJoystickVector().ZeroY();
		float turn = Controller.QueryRadialController();
		float spin = Controller.QueryLogic("Move") ? Controller.QueryButtonController() : 0f;

		//Amplify Factors
		move = Quaternion.LookRotation(Vector3.ProjectOnPlane(transform.position - GetCamera().transform.position, Vector3.up).normalized, Vector3.up) * move;
		if(Controller.QueryLogic("Sprint")) {
			move *= SprintSpeed;
		} else {
			move *= WalkSpeed;
		}

		//Keyboard Adjustments
		if(ControlType == Controller.TYPE.Keyboard) {
			if(!Controller.QueryLogic("Sprint")) {
				move /= 1.25f;
			}
			float length = move.magnitude;
			if(move.x != 0f && move.z < 0f && Mathf.Abs(turn) > 0.1f) {
				move.z *= 0f;
				move.x *= 0.5f;
			}
			if(move.x != 0f && move.z == 0f && Mathf.Abs(turn) > 0.1f) {
				move.z = Mathf.Abs(move.x);
			}
			if(move.z < 0f && Mathf.Abs(turn) < 0.1f) {
				move.z *= 0.5f;
			}
			if(move.z < 0f && move.x == 0f && Mathf.Abs(turn) < 0.1f) {
				move.z = 0f;
			}
			if(Controller.QueryLogic("Sprint")) {
				move.z = Mathf.Max(move.z, 0f);
			}
			move = move.ClampMagnitudeXZ(length);
			move = Quaternion.AngleAxis(KeyboardTurnAngle() * turn, Vector3.up) * move.GetRelativeDirectionFrom(transform.GetWorldMatrix(true));
		}

		//Trajectory
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			//Root Positions
			RootSeries.SetPosition(i,
				Vector3.Lerp(
					RootSeries.GetPosition(i),
					Actor.GetRoot().position + i.Ratio(TimeSeries.Pivot, TimeSeries.Samples.Length-1) * move,
					Controller.QueryFunction("RootPositionControl", i)
				)
			);

			//Root Rotations
			if(ControlType == Controller.TYPE.Gamepad) {
				RootSeries.SetRotation(i,
					Quaternion.Slerp(
						RootSeries.GetRotation(i),
						Controller.QueryLogic("Move") && move != Vector3.zero ? Quaternion.LookRotation(move, Vector3.up) : Actor.GetRoot().rotation,
						Mathf.Clamp(move.magnitude, 0f, 1f) * Controller.QueryFunction("RootRotationControl", i)
					)
				);
			}
			if(ControlType == Controller.TYPE.Keyboard) {
				RootSeries.SetRotation(i,
					Quaternion.Slerp(
						RootSeries.GetRotation(i),
						Controller.QueryLogic("Move") && move != Vector3.zero ? Quaternion.LookRotation(move, Vector3.up) : Actor.GetRoot().rotation,
						Controller.QueryLogic("Sprint") ? Mathf.Pow(i.Ratio(TimeSeries.Pivot, TimeSeries.Samples.Length-1), 0.1f) : Mathf.Clamp(move.magnitude, 0f, 1f) * Controller.QueryFunction("RootRotationControl", i)
					)
				);
			}

			//Root Velocities
			RootSeries.SetVelocity(i,
				Vector3.Lerp(
					RootSeries.GetVelocity(i),
					move,
					Controller.QueryFunction("RootVelocityControl", i)
				)
			);
		}

		//Resolve Trajectory Collisions
		RootSeries.ResolveCollisions(Collider.radius, CollisionMask);

		//Action Values
		float[] actions = Controller.PoolLogics(StyleSeries.Styles);
		actions[StyleSeries.Styles.FindIndex("Speed")] *= move.magnitude;
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[i][j] = Mathf.Lerp(
					StyleSeries.Values[i][j], 
					actions[j],
					Controller.QueryFunction(StyleSeries.Styles[j] + "Control", i)
				);
			}
		}
	}

	private void ProcessFootIK(UltimateIK.Model ik, float contact) {
		ik.Activation = UltimateIK.ACTIVATION.Constant;
		for(int i=0; i<ik.Objectives.Length; i++) {
			ik.Objectives[i].SetTarget(Vector3.Lerp(ik.Objectives[i].TargetPosition, ik.Bones[ik.Objectives[i].Bone].Transform.position, 1f-contact));
			ik.Objectives[i].SetTarget(ik.Bones[ik.Objectives[i].Bone].Transform.rotation);
		}
		ik.Iterations = 25;
		ik.Solve();
	}

	protected override void OnGUIDerived() {
		RootSeries.DrawGUI = DrawGUI;
		StyleSeries.DrawGUI = DrawGUI;
		ContactSeries.DrawGUI = DrawGUI;
		PhaseSeries.DrawGUI = DrawGUI;
		GetComponent<ExpertActivation>().Draw = DrawGUI;
		RootSeries.GUI(GetCamera());
		StyleSeries.GUI(GetCamera());
		ContactSeries.GUI(GetCamera());
		PhaseSeries.GUI(GetCamera());
	}

	protected override void OnRenderObjectDerived() {
		RootSeries.DrawScene = DrawDebug;
		StyleSeries.DrawScene = DrawDebug;
		ContactSeries.DrawScene = DrawDebug;
		PhaseSeries.DrawScene = DrawDebug;
		RootSeries.Draw(GetCamera());
		StyleSeries.Draw(GetCamera());
		ContactSeries.Draw(GetCamera());
		PhaseSeries.Draw(GetCamera());

		//Debug Collider
		// UltiDraw.Begin();
		// UltiDraw.DrawWireCylinder(Collider.transform.position + new Vector3(0f, Collider.height/4f, 0f), Collider.transform.rotation, 2f*Collider.radius, Collider.height/2f, UltiDraw.Black);
		// UltiDraw.End();
	}

}