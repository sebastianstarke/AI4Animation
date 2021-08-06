using UnityEngine;
using DeepLearning;

public class BasketballController : NeuralAnimation {

	public LayerMask CollisionMask = ~0;
	public Camera Camera = null;
	public Ball Ball = null;

	public Controller.TYPE ControlType = Controller.TYPE.Gamepad;

	public bool GenerativeControl = false;

	public bool DrawGUI = true;
	public bool DrawDebug = true;

	private float WalkFactor = 3.75f;
	private float SprintFactor = 2.5f;
	private float TurnFactor = 1.25f;
	private float SpinFactor = 35f;

	private float ContactPower = 3f;
	private float BoneContactThreshold = 0.5f;
	private float BallContactThreshold = 0.1f;

	private float HeightDribbleThreshold = 1.5f;
	private float SpeedDribbleThreshold = 2.5f;

	private Controller Controller;

	private GenerativeControl GenerativeModel;

	private TimeSeries TimeSeries;
	private RootSeries RootSeries;
	private DribbleSeries DribbleSeries;
	private StyleSeries StyleSeries;
	private ContactSeries ContactSeries;
	private PhaseSeries PhaseSeries;

	private UltimateIK.Model BodyIK;
	private UltimateIK.Model LeftFootIK;
	private UltimateIK.Model RightFootIK;
	private UltimateIK.Model LeftHandIK;
	private UltimateIK.Model RightHandIK;
	private UltimateIK.Model HeadIK;

	private int PlayerID = 1;
	private bool Carrier = true;

	private CapsuleCollider Collider = null;

	private Camera GetCamera() {
		return Camera == null ? Camera.main : Camera;
	}

	protected override void Setup() {	
		Collider = GetComponent<CapsuleCollider>();
		Controller = new Controller(PlayerID);

		GenerativeModel = GetComponent<GenerativeControl>();

		Controller.Logic stand = Controller.AddLogic("Stand", () => Controller.QueryLeftJoystickVector().magnitude < 0.25f);
		Controller.Function standControl = Controller.AddFunction("StandControl", (x) => TimeSeries.GetControl((int)x, 0.5f, 0.1f, 1f));
		Controller.Function standCorrection = Controller.AddFunction("StandCorrection", (x) => TimeSeries.GetCorrection((int)x, 0.1f, 1f, 0f));

		Controller.Logic move = Controller.AddLogic("Move", () => !stand.Query() && !Controller.GetButton(Controller.Button.Y));
		Controller.Function moveControl = Controller.AddFunction("MoveControl", (x) => TimeSeries.GetControl((int)x, 0.5f, 0.1f, 1f));
		Controller.Function moveCorrection = Controller.AddFunction("MoveCorrection", (x) => TimeSeries.GetCorrection((int)x, 0.1f, 1f, 0f));

		Controller.Logic hold = Controller.AddLogic("Hold", () => Controller.GetButton(Controller.Button.B));
		Controller.Function holdControl = Controller.AddFunction("HoldControl", (x) => TimeSeries.GetControl((int)x, hold.Query() ? 0.5f : 1f, 0.1f, 1f));
		Controller.Function holdCorrection = Controller.AddFunction("HoldCorrection", (x) => TimeSeries.GetCorrection((int)x, hold.Query() ? 0.1f : 0f, 1f, 0f));

		Controller.Logic shoot = Controller.AddLogic("Shoot", () => Carrier && Controller.GetButton(Controller.Button.Y));
		Controller.Function shootControl = Controller.AddFunction("ShootControl", (x) => TimeSeries.GetControl((int)x, shoot.Query() ? 0.5f : 1f, 0.1f, 1f));
		Controller.Function shootCorrection = Controller.AddFunction("ShootCorrection", (x) => TimeSeries.GetCorrection((int)x, shoot.Query() ? 0.1f : 0f, 1f, 0f));

		Controller.Logic dribble = Controller.AddLogic("Dribble", () => Carrier && !hold.Query() && !shoot.Query());
		Controller.Function dribbleControl = Controller.AddFunction("DribbleControl", (x) => TimeSeries.GetControl((int)x, hold.Query() || shoot.Query() ? 0.5f : 1f, 0.1f, 1f));
		Controller.Function dribbleCorrection = Controller.AddFunction("DribbleCorrection", (x) => TimeSeries.GetCorrection((int)x, hold.Query() || shoot.Query() ? 0.1f : 0f, 1f, 0f));

		Controller.Logic sprint = Controller.AddLogic("Sprint", () => move.Query() && Controller.QueryLeftJoystickVector().y > 0.25f);

		Controller.Logic spin = Controller.AddLogic("Spin", () => move.Query() && Controller.QueryButtonController() != 0f);

		Controller.Logic horizontalControl = Controller.AddLogic("HorizontalControl", () => 
			!Carrier && hold.Query() || 
			Carrier && !spin.Query() && Controller.QueryRightJoystickVector().magnitude > 0.1f
		);
		Controller.Logic heightControl = Controller.AddLogic("HeightControl", () => 
			!Carrier && hold.Query() || 
			Carrier && Controller.QueryDPadController().z != 0f || 
			Carrier && hold.Query() && horizontalControl.Query()
		);
		Controller.Logic speedControl = Controller.AddLogic("SpeedControl", () => 
			!Carrier && hold.Query() || 
			Carrier && dribble.Query() && DribbleSeries.Pivots[TimeSeries.Pivot].y > HeightDribbleThreshold || 
			Carrier && dribble.Query() && DribbleSeries.Momentums[TimeSeries.Pivot].y < SpeedDribbleThreshold
		);

		Controller.Function phaseStability = Controller.AddFunction("PhaseStability", (x) => TimeSeries.GetCorrection((int)x, 
			1f,
			0.9f,
			0.1f
		));

		Controller.Logic rootControl = Controller.AddLogic("RootControl", () => true);

		Controller.Function rootPositionControl = Controller.AddFunction("RootPositionControl", (x) => TimeSeries.GetControl((int)x, 
			rootControl.Query() ? 0.25f : 0f, 
			0.1f, 
			1f
		));

		Controller.Function rootPositionCorrection = Controller.AddFunction("RootPositionCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			rootControl.Query() ? 0.25f : 1f, 
			1f, 
			0f
		));

		Controller.Function rootRotationControl = Controller.AddFunction("RootRotationControl", (x) => TimeSeries.GetControl((int)x, 
			rootControl.Query() ? 0.5f : 0f,
			0.1f, 
			1f
		));

		Controller.Function rootRotationCorrection = Controller.AddFunction("RootRotationCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			rootControl.Query() ? 0.25f : 1f,
			1f, 
			0f
		));

		Controller.Function rootVelocityControl = Controller.AddFunction("RootVelocityControl", (x) => TimeSeries.GetControl((int)x, 
			rootControl.Query() ? 0.75f : 0f,
			0.1f, 
			1f
		));

		Controller.Function rootVelocityCorrection = Controller.AddFunction("RootVelocityCorrection", (x) => TimeSeries.GetCorrection((int)x, 
			rootControl.Query() ? 0.25f : 1f,
			1f, 
			0f
		));

		Controller.Function ballHorizontalControl = Controller.AddFunction("BallHorizontalControl",	(x) => TimeSeries.GetControl((int)x, 
			horizontalControl.Query() ? 0.2f : 0f,
			0f,
			0.5f
		));

		Controller.Function ballHorizontalCorrection = Controller.AddFunction("BallHorizontalCorrection", (x) => TimeSeries.GetCorrection(
			(int)x, horizontalControl.Query() ? 0.2f : 1f, 
			0.5f, 
			0f
		));

		Controller.Function ballHeightControl = Controller.AddFunction("BallHeightControl",	(x) => TimeSeries.GetControl(
			(int)x, heightControl.Query() ? 0.1f : 0f,
			0f,
			0.5f
		));

		Controller.Function ballHeightCorrection = Controller.AddFunction("BallHeightCorrection", (x) => TimeSeries.GetCorrection(
			(int)x, heightControl.Query() ? 0.1f : 1f, 
			0.5f, 
			0f
		));

		Controller.Function ballSpeedControl = Controller.AddFunction("BallSpeedControl", (x) => TimeSeries.GetControl(
			(int)x, speedControl.Query() ? 0.1f : 0f,
			0f,
			0.5f
		));

		Controller.Function ballSpeedCorrection = Controller.AddFunction("BallSpeedCorrection", (x) => TimeSeries.GetCorrection(
			(int)x, speedControl.Query() ? 0.1f : 1f, 
			0.5f, 
			0f
		));

		TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
		RootSeries = new RootSeries(TimeSeries, transform);
		DribbleSeries = new DribbleSeries(TimeSeries, 2.5f, Ball.transform, Actor, RootSeries, null, null);
		StyleSeries = new StyleSeries(TimeSeries, new string[]{"Stand", "Move", "Dribble", "Hold", "Shoot"}, new float[]{1f, 0f, 1f, 0f, 0f});
		ContactSeries = new ContactSeries(TimeSeries, "Left Foot", "Right Foot", "Left Hand", "Right Hand", "Ball");
		PhaseSeries = new PhaseSeries(TimeSeries, "Left Foot", "Right Foot", "Left Hand", "Right Hand", "Ball");

		BodyIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:Hips"), Actor.GetBoneTransforms("Player 01:LeftFoot", "Player 01:RightFoot", "Player 01:LeftBallAux", "Player 01:RightBallAux"));
		LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:LeftUpLeg"), Actor.GetBoneTransforms("Player 01:LeftFoot"));
		RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:RightUpLeg"), Actor.GetBoneTransforms("Player 01:RightFoot"));
		LeftHandIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:LeftArm"), Actor.GetBoneTransforms("Player 01:LeftBallAux"));
		RightHandIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:RightArm"), Actor.GetBoneTransforms("Player 01:RightBallAux"));
		HeadIK = UltimateIK.BuildModel(Actor.FindTransform("Player 01:Neck"), Actor.GetBoneTransforms("Player 01:Head"));

		RootSeries.DrawGUI = DrawGUI;
		StyleSeries.DrawGUI = DrawGUI;
		DribbleSeries.DrawGUI = DrawGUI;
		ContactSeries.DrawGUI = DrawGUI;
		PhaseSeries.DrawGUI = DrawGUI;
		GetComponent<ExpertActivation>().Draw = DrawGUI;
		RootSeries.DrawScene = DrawDebug;
		StyleSeries.DrawScene = DrawDebug;
		DribbleSeries.DrawScene = DrawDebug;
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
			
			NeuralNetwork.Feed(DribbleSeries.Pivots[index]);
			NeuralNetwork.Feed(DribbleSeries.Momentums[index]);
			
			NeuralNetwork.Feed(StyleSeries.Values[index]);
		}

		//Input Character
		for(int i=0; i<Actor.Bones.Length; i++) {
			NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
		}
		
		//Input Ball
		for(int i=0; i<=TimeSeries.PivotKey; i++) {
			int index = TimeSeries.GetKey(i).Index;
			NeuralNetwork.Feed(DribbleSeries.GetControlWeight(index, root.GetPosition()));
			NeuralNetwork.Feed(DribbleSeries.GetWeightedBallPosition(index, root.GetPosition()).GetRelativePositionTo(root));
			NeuralNetwork.Feed(DribbleSeries.GetWeightedBallVelocity(index, root.GetPosition()).GetRelativeDirectionTo(root));
		}

		//Input Contacts
		for(int i=0; i<=TimeSeries.PivotKey; i++) {
			int index = TimeSeries.GetKey(i).Index;
			NeuralNetwork.Feed(ContactSeries.Values[index]);
		}

		//Input Rival
		//Not included in this demo. If you want to use it, you will need to create a second player and pass its reference to the DribbleSeries contructor in the Setup function.
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			NeuralNetwork.Feed(DribbleSeries.GetInteractorWeight(i));
		}
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			NeuralNetwork.FeedXZ(DribbleSeries.GetInteractorGradient(i));
			NeuralNetwork.FeedXZ(DribbleSeries.GetInteractorDirection(i));
			NeuralNetwork.FeedXZ(DribbleSeries.GetInteractorVelocity(i));
		}
		for(int i=0; i<Actor.Bones.Length; i++) {
			NeuralNetwork.Feed(DribbleSeries.GetInteractorBoneDistance(i));
		}

		//Input Gating Features
		NeuralNetwork.Feed(PhaseSeries.GetAlignment());

		//Generative Controller
		if(GenerativeControl) {
			for(int i=0; i<GenerativeModel.GetOutputDimensionality(); i++) {
				GenerativeModel.SetInput(i, NeuralNetwork.GetInput(i));
			}
			GenerativeModel.Predict();
			for(int i=0; i<GenerativeModel.GetOutputDimensionality(); i++) {
				NeuralNetwork.SetInput(i, GenerativeModel.GetOutput(i));
			}
		}
	}

	protected override void Read() {
		//Update Past States
		DribbleSeries.IncrementBall(0, TimeSeries.Pivot);
		ContactSeries.Increment(0, TimeSeries.Pivot);
		PhaseSeries.Increment(0, TimeSeries.Pivot);
		
		//Update Root State
		Vector3 offset = NeuralNetwork.ReadVector3();
		offset = Vector3.Lerp(offset, Vector3.zero, StyleSeries.Values[TimeSeries.Pivot].First());

		Matrix4x4 root = Actor.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
		RootSeries.Transformations[TimeSeries.Pivot] = root;
		RootSeries.Velocities[TimeSeries.Pivot] = NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root);
		
		DribbleSeries.Pivots[TimeSeries.Pivot] = DribbleSeries.InterpolatePivot(
			DribbleSeries.Pivots[TimeSeries.Pivot], 
			NeuralNetwork.ReadVector3(), 
			Controller.QueryFunction("BallHorizontalCorrection", TimeSeries.Pivot),
			Controller.QueryFunction("BallHeightCorrection", TimeSeries.Pivot)
		);
		DribbleSeries.Momentums[TimeSeries.Pivot] = DribbleSeries.InterpolateMomentum(
			DribbleSeries.Momentums[TimeSeries.Pivot], 
			NeuralNetwork.ReadVector3(), 
			Controller.QueryFunction("BallHorizontalCorrection", TimeSeries.Pivot),
			Controller.QueryFunction("BallSpeedCorrection", TimeSeries.Pivot)
		);
		
		for(int j=0; j<StyleSeries.Styles.Length; j++) {
			StyleSeries.Values[TimeSeries.Pivot][j] = Mathf.Lerp(
				StyleSeries.Values[TimeSeries.Pivot][j], 
				NeuralNetwork.Read(0f, 1f), 
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
			
			DribbleSeries.Pivots[index] = DribbleSeries.InterpolatePivot(DribbleSeries.Pivots[index], NeuralNetwork.ReadVector3(),
				Controller.QueryFunction("BallHorizontalCorrection", index),
				Controller.QueryFunction("BallHeightCorrection", index)
			);
			DribbleSeries.Momentums[index] = DribbleSeries.InterpolateMomentum(DribbleSeries.Momentums[index], NeuralNetwork.ReadVector3(),
				Controller.QueryFunction("BallHorizontalCorrection", index),
				Controller.QueryFunction("BallSpeedCorrection", index)
			);
			
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[index][j] = Mathf.Lerp(StyleSeries.Values[index][j], NeuralNetwork.Read(0f, 1f), Controller.QueryFunction(StyleSeries.Styles[j] + "Correction", index));
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

		//Compute Ball
		float controlWeight = NeuralNetwork.Read(0f, 1f);
		Vector3 ballPosition = NeuralNetwork.ReadVector3();
		Vector3 ballVelocity = NeuralNetwork.ReadVector3();
		Vector3 ballForward = NeuralNetwork.ReadVector3();
		Vector3 ballUp = NeuralNetwork.ReadVector3();
		if(Carrier && controlWeight > 0f) {
			ballPosition = (ballPosition / controlWeight).GetRelativePositionFrom(root);
			ballVelocity = (ballVelocity / controlWeight).GetRelativeDirectionFrom(root);
			Quaternion ballRotation = DribbleSeries.BallTransformations[TimeSeries.Pivot].GetRotation() * Quaternion.Slerp(
				Quaternion.LookRotation((ballForward / controlWeight).normalized, (ballUp / controlWeight).normalized), 
				Quaternion.identity, 
				StyleSeries.GetStyle(TimeSeries.Pivot, "Hold")
			);
			
			//Assign Ball
			Ball.SetPosition(
				DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition() + Vector3.ClampMagnitude(
					Vector3.Lerp(
						ballPosition, 
						DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition() + ballVelocity / Framerate, 
						0.5f
					) - DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition(),
					2f * ballVelocity.magnitude / Framerate
				)
			);
			Ball.SetRotation(ballRotation);
			Ball.SetVelocity(ballVelocity);
		}
		DribbleSeries.BallTransformations[TimeSeries.Pivot] = 
			Matrix4x4.TRS(
				Ball.GetPosition(),
				Ball.GetRotation(),
				Vector3.one
			);
		DribbleSeries.BallVelocities[TimeSeries.Pivot] = Ball.GetVelocity();

		//Update Contacts
		float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length, 0f, 1f);
		for(int i=0; i<ContactSeries.Bones.Length; i++) {
			if(i==4) {
				ContactSeries.Values[TimeSeries.Pivot][i] = contacts[i].SmoothStep(ContactPower, BallContactThreshold);
			} else {
				ContactSeries.Values[TimeSeries.Pivot][i] = contacts[i].SmoothStep(ContactPower, BoneContactThreshold);
			}
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
		DribbleSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
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
		//Correct Twist
		for(int i=0; i<Actor.Bones.Length; i++) {
			if(Actor.Bones[i].Childs.Length == 1) {
				Vector3 position = Actor.Bones[i].Transform.position;
				Quaternion rotation = Actor.Bones[i].Transform.rotation;
				Vector3 childPosition = Actor.Bones[i].GetChild(0).Transform.position;
				Quaternion childRotation = Actor.Bones[i].GetChild(0).Transform.rotation;
				Vector3 aligned = (position - childPosition).normalized;
				Actor.Bones[i].Transform.rotation = Quaternion.FromToRotation(rotation.GetRight(), aligned) * rotation;
				Actor.Bones[i].GetChild(0).Transform.position = childPosition;
				Actor.Bones[i].GetChild(0).Transform.rotation = childRotation;
			}
		}

		//Resolve Trajectory Collisions
		RootSeries.ResolveCollisions(Collider.radius, CollisionMask);

		//Refine Ball
		if(PhaseSeries.IsActive("Ball")) {
			if(Carrier) {
				Vector3 position = DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition();
				Vector3 velocity = DribbleSeries.BallVelocities[TimeSeries.Pivot];

				//Process Hold Ball
				if(Controller.QueryLogic("Hold") && Controller.QueryLogic("HorizontalControl")) {
					Vector3 target = DribbleSeries.Pivots.Last().GetRelativePositionFrom(RootSeries.Transformations[TimeSeries.Pivot]);
					Vector3 modified = Vector3.Lerp(position, target, 0.15f);
					velocity += (modified - position) * Framerate;
					position = modified;
				}

				//Process Ground Ball
				float weight = Physics.Raycast(
					position, 
					velocity, 
					Ball.Radius + velocity.magnitude, 
					LayerMask.GetMask("Ground")
				) ? ContactSeries.Values[TimeSeries.Pivot].Last() : 0f;
				float prev = position.y;
				float next = Mathf.Lerp(position.y, Ball.Radius, weight);
				position.y = next;
				velocity.y += Framerate * (next - prev);
				
				Ball.SetPosition(position);
				Ball.SetVelocity(velocity);
			} else {
				if(Controller.QueryLogic("Hold") && DribbleSeries.IsInsideControlRadius(Ball.GetPosition(), root.GetPosition())) {
					//Process Flying Ball
					float leftWeight = GetBallControlWeight(LeftHandIK.Bones.Last().Transform.position);
					float rightWeight = GetBallControlWeight(RightHandIK.Bones.Last().Transform.position);
					float weight = 0.5f * (leftWeight + rightWeight);
					Vector3 leftPos = LeftHandIK.Bones.Last().Transform.position;
					Vector3 rightPos = RightHandIK.Bones.Last().Transform.position;
					Vector3 target = 0.5f*(rightPos+leftPos);
					Vector3 position = Ball.GetPosition();
					Vector3 velocity = Ball.GetVelocity();
					float angle = Vector3.Angle(target-position, velocity);
					float correction = 1f - angle / 180f;
					correction = weight * correction.ActivateCurve(0.25f, 0f, 1f);
					Ball.SetVelocity(Vector3.Lerp(velocity, velocity.magnitude * (target-position).normalized, correction));
				}
			}
			
			//Resolve Ball-Body Penetrations
			BroadcastMessage("ResolveBallCollisions");

			//Synchronize Ball
			DribbleSeries.BallTransformations[TimeSeries.Pivot] = 
				Matrix4x4.TRS(
					Ball.GetPosition(),
					Ball.GetRotation(),
					Vector3.one
				);
			DribbleSeries.BallVelocities[TimeSeries.Pivot] = Ball.GetVelocity();
		}

		//Process Contact States
		ProcessBody();
		ProcessFootIK(LeftFootIK, ContactSeries.Values[TimeSeries.Pivot][0]);
		ProcessFootIK(RightFootIK, ContactSeries.Values[TimeSeries.Pivot][1]);
		ProcessHandIK(LeftHandIK, ContactSeries.Values[TimeSeries.Pivot][2]);
		ProcessHandIK(RightHandIK, ContactSeries.Values[TimeSeries.Pivot][3]);
		ProcessHeadIK();
	}

	private void Control() {
		Controller.ControlType = ControlType;

		//Update Past
		RootSeries.Increment(0, TimeSeries.Samples.Length-1);
		StyleSeries.Increment(0, TimeSeries.Samples.Length-1);
		DribbleSeries.Increment(0, TimeSeries.Samples.Length-1);

		//Update User Controller Inputs
		Controller.Update();

		//Ball Vectors
		Vector3 control = Vector3.Lerp(
			Controller.QueryRightJoystickVector(), //User Control
			(DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition() - transform.position).ZeroY().normalized.GetRelativeDirectionTo(Actor.GetRoot().GetWorldMatrix(true)), //Game State
			Carrier ? 0f : 1f
		);
		float height = DribbleSeries.GetBallAmplitude();
		float speed = Mathf.Abs(DribbleSeries.BallVelocities[TimeSeries.Pivot].y);
		if(Carrier && Controller.QueryLogic("Dribble")) {
			if(DribbleSeries.Pivots[TimeSeries.Pivot].y > HeightDribbleThreshold) {
				height -= 2f*HeightDribbleThreshold;
			}
			if(DribbleSeries.Momentums[TimeSeries.Pivot].y < SpeedDribbleThreshold) {
				speed += 2f*SpeedDribbleThreshold;
			}
		}
		if(Carrier && Controller.QueryLogic("Hold") && Controller.QueryLogic("HorizontalControl")) {
			control = ToHoldTarget(control);
			height = control.y;
			speed = (height - DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition().y) * Framerate;
		}

		//Locomotion
		Vector3 move = Controller.QueryLeftJoystickVector().ZeroY();
		float turn = Controller.QueryRadialController();
		float spin = Controller.QueryLogic("Move") ? Controller.QueryButtonController() : 0f;

		//Amplify Factors
		move = Quaternion.LookRotation(Vector3.ProjectOnPlane(transform.position - GetCamera().transform.position, Vector3.up).normalized, Vector3.up) * move;
		move *= WalkFactor;
		if(Controller.QueryLogic("Sprint")) {
			move *= SprintFactor;
		}
		spin *= SpinFactor;
		turn = Mathf.Sign(turn) * Mathf.Pow(Mathf.Abs(turn), 1f/TurnFactor);

		//Keyboard Adjustments
		if(ControlType == Controller.TYPE.Keyboard) {
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
				move.z = -1f;
				float left = ContactSeries.GetContacts("Left Hand").Mean();
				float right = ContactSeries.GetContacts("Right Hand").Mean();
				if(left > right) {
					move.x = 1f;
				}
				if(right > left) {
					move.x = -1f;
				}
			}
			move = move.ClampMagnitudeXZ(length);
			move = Quaternion.AngleAxis(60f * turn, Vector3.up) * move.GetRelativeDirectionFrom(transform.GetWorldMatrix(true));
			if(control.magnitude > 0.1f) {
				control = control.normalized;
			} else {
				control = Vector3.zero;
			}
		}

		//Ball Shooting
		if(Carrier && Controller.QueryLogic("Shoot")) {
			move = Vector3.zero;
			if(
				DribbleSeries.BallVelocities[TimeSeries.Pivot].magnitude < DribbleSeries.BallVelocities[TimeSeries.Pivot-1].magnitude && 
				DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition().y > 1.5f && ContactSeries.GetContacts(TimeSeries.Pivot, "Left Hand", "Right Hand").Sum() < 0.1f
			) {
				Carrier = false;
			}
		}

		//Holding, Picking & Catching
		if(Controller.QueryLogic("Hold")) {
			if(Carrier) {
				move = Vector3.zero;
				turn = 0f;
				spin = 0f;
			} else {
				if((HasBallContact(LeftHandIK.Bones.Last().Transform.position) || HasBallContact(RightHandIK.Bones.Last().Transform.position))) {
					Carrier = true;
				} else {
					move = Vector3.Lerp(move, move.magnitude * (DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition() - Actor.GetRoot().position).normalized, GetBallInteractionWeight(Actor.GetRoot().position));
				}
			}
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
			RootSeries.SetRotation(i,
				Quaternion.Slerp(
					RootSeries.GetRotation(i),
					Controller.QueryLogic("Move") && move != Vector3.zero ?  Quaternion.LookRotation(move, Vector3.up) : Actor.GetRoot().rotation,
					Mathf.Abs(turn) * Controller.QueryFunction("RootRotationControl", i)
				)
			);
			if(ControlType == Controller.TYPE.Keyboard && !Controller.QueryLogic("Sprint")) {
				if(turn != 0f) {
					float w = i.Ratio(TimeSeries.Pivot, TimeSeries.Samples.Length-1).ActivateCurve(0.75f, 0f, 1f);
					RootSeries.SetRotation(i, RootSeries.GetRotation(TimeSeries.Pivot) * Quaternion.AngleAxis(60f*turn*w, Vector3.up));
				}
			}

			//Spin Rotations
			if(spin != 0f) {
				float w = i.Ratio(TimeSeries.Pivot, TimeSeries.Samples.Length-1).ActivateCurve(0.25f, 0.75f, 0f);
				RootSeries.SetRotation(i, RootSeries.GetRotation(i) * Quaternion.AngleAxis(spin*w, Vector3.up));
			}

			//Root Velocities
			RootSeries.SetVelocity(i,
				Vector3.Lerp(
					RootSeries.GetVelocity(i),
					move,
					Controller.QueryFunction("RootVelocityControl", i)
				)
			);

			//Ball Control
			DribbleSeries.Target = new Vector4(control.x, height,	control.z, speed);
			DribbleSeries.Pivots[i] = DribbleSeries.InterpolatePivot(
				DribbleSeries.Pivots[i],
				new Vector3(DribbleSeries.Target.x, DribbleSeries.Target.y, DribbleSeries.Target.z),
				Controller.QueryFunction("BallHorizontalControl", i),
				Controller.QueryFunction("BallHeightControl", i)
			);
			DribbleSeries.Momentums[i] = DribbleSeries.InterpolateMomentum(
				DribbleSeries.Momentums[i],
				new Vector3(
					0.5f * Framerate * (DribbleSeries.Target.x - DribbleSeries.Pivots[i].x), 
					DribbleSeries.Target.w, 
					0.5f * Framerate * (DribbleSeries.Target.z - DribbleSeries.Pivots[i].z)
				),
				Controller.QueryFunction("BallHorizontalControl", i),
				Controller.QueryFunction("BallSpeedControl", i)
			);
		}

		//Resolve Trajectory Collisions
		RootSeries.ResolveCollisions(Collider.radius, CollisionMask);

		//Action Values
		float[] actions = Controller.PoolLogics(StyleSeries.Styles);
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

	private void ProcessBody() {
		if(!Carrier && Controller.QueryLogic("Hold") || Carrier && Controller.QueryLogic("Hold") && Controller.QueryLogic("HorizontalControl") && !Controller.QueryLogic("Move")) {
			BodyIK.Activation = UltimateIK.ACTIVATION.Square;
			BodyIK.Objectives[0].SetTarget(LeftFootIK.Bones.Last().Transform);
			BodyIK.Objectives[1].SetTarget(RightFootIK.Bones.Last().Transform);
			BodyIK.Objectives[2].SetTarget(Vector3.Lerp(
				LeftHandIK.Bones.Last().Transform.position,
				DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition(),
				GetBallControlWeight(LeftHandIK.Bones.Last().Transform.position)
			));
			BodyIK.Objectives[2].SetTarget(LeftHandIK.Bones.Last().Transform.rotation);
			BodyIK.Objectives[3].SetTarget(Vector3.Lerp(
				RightHandIK.Bones.Last().Transform.position,
				DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition(),
				GetBallControlWeight(RightHandIK.Bones.Last().Transform.position)
			));
			BodyIK.Objectives[3].SetTarget(RightHandIK.Bones.Last().Transform.rotation);
			BodyIK.AllowRootUpdateY = true;
			BodyIK.Iterations = 25;
			BodyIK.Solve();
		}
	}

	private void ProcessFootIK(UltimateIK.Model ik, float contact) {
		ik.Activation = UltimateIK.ACTIVATION.Constant;
		ik.Objectives.First().SetTarget(Vector3.Lerp(ik.Objectives[0].TargetPosition, ik.Bones.Last().Transform.position, 1f-contact));
		if(Carrier && Controller.QueryLogic("Hold")) {
			ik.Objectives.First().SetTarget(Quaternion.Slerp(ik.Objectives[0].TargetRotation, ik.Bones.Last().Transform.rotation, 1f-contact));
		} else {
			ik.Objectives.First().SetTarget(ik.Bones.Last().Transform.rotation);
		}
		ik.Iterations = 50;
		ik.Solve();
	}

	private void ProcessHandIK(UltimateIK.Model ik, float contact) {
		if(Carrier) {
			ik.Activation = UltimateIK.ACTIVATION.Linear;
			ik.Objectives.First().SetTarget(Vector3.Lerp(ik.Bones.Last().Transform.position, DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition(), contact));
			ik.Objectives.First().SetTarget(ik.Bones.Last().Transform.rotation);
			ik.Iterations = 50;
			ik.Solve();
		}
	}

	private void ProcessHeadIK() {
		if(!Carrier && Controller.QueryLogic("Hold")) {
			Vector3 target = DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition();
			Solve(target, ComputeWeight(target));
		}

		float ComputeWeight(Vector3 target) {
			float[] distances = new float[TimeSeries.KeyCount];
			float[] angles = new float[TimeSeries.KeyCount];
			for(int i=0; i<TimeSeries.KeyCount; i++) {
				distances[i] = 1f - Mathf.Clamp(Vector3.Distance(RootSeries.GetPosition(RootSeries.GetKey(i).Index), target) / DribbleSeries.GetInteractionRadius(), 0f, 1f);
				angles[i] = 1f - Vector3.Angle(RootSeries.GetDirection(RootSeries.GetKey(i).Index), target - RootSeries.GetPosition(RootSeries.GetKey(i).Index)) / 180f;
			}
			float distance = distances.Gaussian();
			float angle = angles.Gaussian();
			return Mathf.Min(distance*distance, angle*angle);
		}

		void Solve(Vector3 target, float weight) {
			HeadIK.Activation = UltimateIK.ACTIVATION.Square;
			Matrix4x4 self = Actor.GetBoneTransformation("Player 01:Head");
			Quaternion rotation = Quaternion.LookRotation(self.GetPosition() - target) * Quaternion.Euler(0f, 90f, -90f);
			HeadIK.Objectives.First().SetTarget(HeadIK.Bones.Last().Transform.position);
			HeadIK.Objectives.First().SetTarget(Quaternion.Slerp(HeadIK.Bones.Last().Transform.rotation, rotation, weight));
			HeadIK.Iterations = 50;
			HeadIK.Solve();
		}
	}

	private Vector3 ToHoldTarget(Vector3 target) {
		Vector3 scale = new Vector3(0.5f, 1f, 0.8f);
		Vector3 offset = new Vector3(0f, 1.5f, 0.15f);
		float angle = 65f;
		target.x *= -1f;
		return Quaternion.AngleAxis(angle, Vector3.right) * Vector3.Scale(scale, -target.ZeroY()) + offset;
	}

	private bool HasBallContact(Vector3 pivot) {
		return Vector3.Distance(pivot, DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition()) <= 1.25f*Ball.Radius;
	}

	private float GetBallControlWeight(Vector3 pivot) {
		float w = 1f - Mathf.Clamp(Vector3.Distance(pivot, DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition()) / DribbleSeries.GetControlRadius(), 0f, 1f);
		return w.ActivateCurve(Mathf.Lerp(1f/3f, 2f/3f, w), 0f, w);
	}

	private float GetBallInteractionWeight(Vector3 pivot) {
		float w = 1f - Mathf.Clamp(Vector3.Distance(pivot, DribbleSeries.BallTransformations[TimeSeries.Pivot].GetPosition()) / DribbleSeries.GetInteractionRadius(), 0f, 1f);
		return w.ActivateCurve(Mathf.Lerp(1f/3f, 2f/3f, w), 0f, w);
	}

	protected override void OnGUIDerived() {
		RootSeries.DrawGUI = DrawGUI;
		StyleSeries.DrawGUI = DrawGUI;
		DribbleSeries.DrawGUI = DrawGUI;
		ContactSeries.DrawGUI = DrawGUI;
		PhaseSeries.DrawGUI = DrawGUI;
		GetComponent<ExpertActivation>().Draw = DrawGUI;
		RootSeries.GUI(GetCamera());
		StyleSeries.GUI(GetCamera());
		DribbleSeries.GUI(GetCamera());
		ContactSeries.GUI(GetCamera());
		PhaseSeries.GUI(GetCamera());
	}

	protected override void OnRenderObjectDerived() {
		RootSeries.DrawScene = DrawDebug;
		StyleSeries.DrawScene = DrawDebug;
		DribbleSeries.DrawScene = DrawDebug;
		ContactSeries.DrawScene = DrawDebug;
		PhaseSeries.DrawScene = DrawDebug;
		RootSeries.Draw(GetCamera());
		StyleSeries.Draw(GetCamera());
		DribbleSeries.Draw(GetCamera());
		ContactSeries.Draw(GetCamera());
		PhaseSeries.Draw(GetCamera());
		if(DrawDebug) {
			if(Carrier && Controller.QueryLogic("Hold") && !Controller.QueryLogic("Move")) {
				UltiDraw.Begin();
				int resolution = 100;
				for(int i=0; i<resolution; i++) {
					Vector3 target = ToHoldTarget(Quaternion.AngleAxis(360f * (float)i / (float)resolution, Vector3.up) * Vector3.forward);
					UltiDraw.DrawSphere(target.GetRelativePositionFrom(transform.GetWorldMatrix(true)), Quaternion.identity, 0.05f, Color.cyan);
				}
				UltiDraw.DrawSphere(ToHoldTarget(Controller.QueryRightJoystickVector()).GetRelativePositionFrom(transform.GetWorldMatrix(true)), Quaternion.identity, 0.1f, Color.black);
				UltiDraw.End();
			}
		}

		//Debug Collider
		// UltiDraw.Begin();
		// UltiDraw.DrawWireCylinder(Collider.transform.position + new Vector3(0f, Collider.height/4f, 0f), Collider.transform.rotation, 2f*Collider.radius, Collider.height/2f, UltiDraw.Black);
		// UltiDraw.End();
	}

}