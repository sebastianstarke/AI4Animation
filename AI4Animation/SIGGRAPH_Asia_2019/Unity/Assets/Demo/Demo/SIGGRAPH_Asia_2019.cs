
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepLearning;

public class SIGGRAPH_Asia_2019 : NeuralAnimation {
	
	public bool ShowBiDirectional = true;
	public bool ShowRoot = true;
	public bool ShowGoal = true;
	public bool ShowCurrent = true;
	public bool ShowPhase = false;
	public bool ShowContacts = false;
	public bool ShowEnvironment = true;
	public bool ShowInteraction = true;
	public bool ShowGUI = true;

	private Controller Controller;
	private TimeSeries TimeSeries;
	private TimeSeries.Root RootSeries;
	private TimeSeries.Style StyleSeries;
	private TimeSeries.Goal GoalSeries;
	private TimeSeries.Contact ContactSeries;
	private TimeSeries.Phase PhaseSeries;
	private CylinderMap Environment;
	private CuboidMap Geometry;

	private Vector3[] PosePrediction;
	private Matrix4x4[] RootPrediction;
	private Matrix4x4[] GoalPrediction;

	private float[] Signals = new float[0];
    private float UserControl = 0f;
	private float NetworkControl = 0f;

	private float InteractionSmoothing = 0.9f;

	private bool IsInteracting = false;

	private UltimateIK.Model RightFootIK, LeftFootIK;

	public Controller GetController() {
		return Controller;
	}

	public TimeSeries GetTimeSeries() {
		return TimeSeries;
	}

	protected override void Setup() {
		Controller = new Controller();
		Controller.Signal idle = Controller.AddSignal("Idle");
		idle.Default = true;
		idle.Velocity = 0f;
		idle.AddKey(KeyCode.W, false);
		idle.AddKey(KeyCode.A, false);
		idle.AddKey(KeyCode.S, false);
		idle.AddKey(KeyCode.D, false);
		idle.AddKey(KeyCode.Q, false);
		idle.AddKey(KeyCode.E, false);
		idle.AddKey(KeyCode.V, true);
		idle.UserControl = 0.25f;
		idle.NetworkControl = 0.1f;

		Controller.Signal walk = Controller.AddSignal("Walk");
		walk.AddKey(KeyCode.W, true);
		walk.AddKey(KeyCode.A, true);
		walk.AddKey(KeyCode.S, true);
		walk.AddKey(KeyCode.D, true);
		walk.AddKey(KeyCode.Q, true);
		walk.AddKey(KeyCode.E, true);
		walk.AddKey(KeyCode.LeftShift, false);
		walk.AddKey(KeyCode.C, false);
		walk.Velocity = 1f;
		walk.UserControl = 0.25f;
		walk.NetworkControl = 0.25f;

		Controller.Signal run = Controller.AddSignal("Run");
		run.AddKey(KeyCode.LeftShift, true);
		run.Velocity = 3f;
		run.UserControl = 0.25f;
		run.NetworkControl = 0.25f;

		Controller.Signal carry = Controller.AddSignal("Carry");
		carry.AddKey(KeyCode.V, true);
		carry.Velocity = 0f;
		carry.UserControl = 0.1f;
		carry.NetworkControl = 0f;

		Controller.Signal open = Controller.AddSignal("Open");
		open.AddKey(KeyCode.F, true);
		open.Velocity = 0f;
		open.UserControl = 0.1f;
		open.NetworkControl = 0f;

		Controller.Signal sit = Controller.AddSignal("Sit");
		sit.AddKey(KeyCode.C, true);
		sit.Velocity = 0f;
		sit.UserControl = 0.25f;
		sit.NetworkControl = 0f;

		Environment = new CylinderMap(4f, 9, 9, true);
		Geometry = new CuboidMap(new Vector3Int(8, 8, 8));

		TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
		RootSeries = new TimeSeries.Root(TimeSeries);
		StyleSeries = new TimeSeries.Style(TimeSeries, "Idle", "Walk", "Run", "Carry", "Open", "Sit", "Climb");
		GoalSeries = new TimeSeries.Goal(TimeSeries, Controller.GetSignalNames());
		ContactSeries = new TimeSeries.Contact(TimeSeries, "Hips", "RightWrist", "LeftWrist", "RightAnkle", "LeftAnkle");
		PhaseSeries = new TimeSeries.Phase(TimeSeries);
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			RootSeries.Transformations[i] = transform.GetWorldMatrix(true);
			if(StyleSeries.Styles.Length > 0) {
				StyleSeries.Values[i][0] = 1f;
			}
			if(GoalSeries.Actions.Length > 0) {
				GoalSeries.Values[i][0] = 1f;
			}
			GoalSeries.Transformations[i] = transform.GetWorldMatrix(true);
			PhaseSeries.Values[i] = Mathf.Repeat((float)i/GetFramerate(), 1f);
			Geometry.Pivot = transform.GetWorldMatrix(true);
			Geometry.References[i] = transform.position;
		}

		PosePrediction = new Vector3[Actor.Bones.Length];
		RootPrediction = new Matrix4x4[7];
		GoalPrediction = new Matrix4x4[7];
		
		RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("RightHip"), Actor.GetBoneTransforms(ContactSeries.Bones[3]));
		LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("LeftHip"), Actor.GetBoneTransforms(ContactSeries.Bones[4]));
	}

    protected override void Feed() {
		Controller.Update();

		//Get Root
		Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];

		//Control Cycle
		Signals = Controller.PoolSignals();
		UserControl = Controller.PoolUserControl(Signals);
		NetworkControl = Controller.PoolNetworkControl(Signals);

		if(IsInteracting) {
			//Do nothing because coroutines have control.
		} else if(Controller.QuerySignal("Sit")) {
			StartCoroutine(Sit());
		} else if(Controller.QuerySignal("Carry")) {
			StartCoroutine(Carry());
		} else if(Controller.QuerySignal("Open")) {
			StartCoroutine(Open());
		} else {
			Default();
		}

		//Input Bone Positions / Velocities
		for(int i=0; i<Actor.Bones.Length; i++) {
			NeuralNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
		}

		//Input Trajectory Positions / Directions / Velocities / Styles
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			NeuralNetwork.FeedXZ(RootSeries.GetPosition(sample.Index).GetRelativePositionTo(root));
			NeuralNetwork.FeedXZ(RootSeries.GetDirection(sample.Index).GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(StyleSeries.Values[sample.Index]);
		}
		
		//Input Goals
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetPosition().GetRelativePositionTo(root));
			NeuralNetwork.Feed(GoalSeries.Transformations[sample.Index].GetForward().GetRelativeDirectionTo(root));
			NeuralNetwork.Feed(GoalSeries.Values[sample.Index]);
		}

		//Input Environment
		Environment.Sense(root, LayerMask.GetMask("Default", "Interaction"));
		NeuralNetwork.Feed(Environment.Occupancies);

		//Input Geometry
		for(int i=0; i<Geometry.Points.Length; i++) {
			NeuralNetwork.Feed(Geometry.References[i].GetRelativePositionTo(root));
			NeuralNetwork.Feed(Geometry.Occupancies[i]);
		}

		//Setup Gating Features
		NeuralNetwork.Feed(GenerateGating());
	}

	protected override void Read() {
		//Update Past State
		for(int i=0; i<TimeSeries.Pivot; i++) {
			TimeSeries.Sample sample = TimeSeries.Samples[i];
			PhaseSeries.Values[i] = PhaseSeries.Values[i+1];
			RootSeries.SetPosition(i, RootSeries.GetPosition(i+1));
			RootSeries.SetDirection(i, RootSeries.GetDirection(i+1));
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[i][j] = StyleSeries.Values[i+1][j];
			}
			for(int j=0; j<ContactSeries.Bones.Length; j++) {
				ContactSeries.Values[i][j] = ContactSeries.Values[i+1][j];
			}
			GoalSeries.Transformations[i] = GoalSeries.Transformations[i+1];
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				GoalSeries.Values[i][j] = GoalSeries.Values[i+1][j];
			}
		}

		//Get Root
		Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];

		//Read Posture
		Vector3[] positions = new Vector3[Actor.Bones.Length];
		Vector3[] forwards = new Vector3[Actor.Bones.Length];
		Vector3[] upwards = new Vector3[Actor.Bones.Length];
		Vector3[] velocities = new Vector3[Actor.Bones.Length];
		for(int i=0; i<Actor.Bones.Length; i++) {
			Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
			positions[i] = Vector3.Lerp(Actor.Bones[i].Transform.position + velocity / GetFramerate(), position, 0.5f);
			forwards[i] = forward;
			upwards[i] = upward;
			velocities[i] = velocity;
		}

		//Read Inverse Pose
		for(int i=0; i<Actor.Bones.Length; i++) {
			PosePrediction[i] = NeuralNetwork.ReadVector3().GetRelativePositionFrom(RootSeries.Transformations.Last());
			velocities[i] = Vector3.Lerp(velocities[i], GetFramerate() * (PosePrediction[i] - Actor.Bones[i].Transform.position), 1f/GetFramerate());
		}

		//Read Future Trajectory
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(root);
			Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root);
			RootSeries.SetPosition(sample.Index, pos);
			RootSeries.SetDirection(sample.Index, dir);
			float[] styles = NeuralNetwork.Read(StyleSeries.Styles.Length);
			for(int j=0; j<styles.Length; j++) {
				styles[j] = Mathf.Clamp(styles[j], 0f, 1f);
			}
			StyleSeries.Values[sample.Index] = styles;

			RootPrediction[i-6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
		}

		//Read Inverse Trajectory
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Matrix4x4 goal = GoalSeries.Transformations[TimeSeries.Pivot];
			goal[1,3] = 0f;
			Vector3 pos = NeuralNetwork.ReadXZ().GetRelativePositionFrom(goal);
			Vector3 dir = NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(goal);
			if(i > TimeSeries.PivotKey) {
				Matrix4x4 pivot = RootSeries.Transformations[sample.Index];
				pivot[1,3] = 0f;
				Matrix4x4 reference = GoalSeries.Transformations[sample.Index];
				reference[1,3] = 0f;
				float distance = Vector3.Distance(pivot.GetPosition(), reference.GetPosition());
				float weight = Mathf.Pow((float)(i-6)/7f, distance*distance);

				RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(sample.Index), pos, weight));
				RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(sample.Index), dir, weight));
			}
			
			GoalPrediction[i-6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
		}
		
		//Read and Correct Goals
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			float weight = TimeSeries.GetWeight1byN1(TimeSeries.GetKey(i).Index, 2f);
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Vector3 pos = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 dir = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			float[] actions = NeuralNetwork.Read(GoalSeries.Actions.Length);
			for(int j=0; j<actions.Length; j++) {
				actions[j] = Mathf.Clamp(actions[j], 0f, 1f);
			}
			GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[sample.Index], Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one), weight * NetworkControl);
			GoalSeries.Values[sample.Index] = Utility.Interpolate(GoalSeries.Values[sample.Index], actions, weight * NetworkControl);
		}

		//Read Future Contacts
		float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length);
		for(int i=0; i<contacts.Length; i++) {
			contacts[i] = Mathf.Clamp(contacts[i], 0f, 1f);
		}
		ContactSeries.Values[TimeSeries.Pivot] = contacts;

		//Read Phase Update
		float phase = PhaseSeries.Values[TimeSeries.Pivot];
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			PhaseSeries.Values[TimeSeries.GetKey(i).Index] = Mathf.Repeat(phase + NeuralNetwork.Read(), 1f);
		}

		//Interpolate Current to Future Trajectory
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = (float)(i % TimeSeries.Resolution) / TimeSeries.Resolution;
			TimeSeries.Sample sample = TimeSeries.Samples[i];
			TimeSeries.Sample prevSample = TimeSeries.GetPreviousKey(i);
			TimeSeries.Sample nextSample = TimeSeries.GetNextKey(i);
			//PhaseSeries.Values[sample.Index] = Mathf.Lerp(PhaseSeries.Values[prevSample.Index], PhaseSeries.Values[nextSample.Index], weight);
			RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(prevSample.Index), RootSeries.GetPosition(nextSample.Index), weight));
			RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(prevSample.Index), RootSeries.GetDirection(nextSample.Index), weight));
			GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[prevSample.Index], GoalSeries.Transformations[nextSample.Index], weight);
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[i][j] = Mathf.Lerp(StyleSeries.Values[prevSample.Index][j], StyleSeries.Values[nextSample.Index][j], weight);
			}
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				GoalSeries.Values[i][j] = Mathf.Lerp(GoalSeries.Values[prevSample.Index][j], GoalSeries.Values[nextSample.Index][j], weight);
			}
		}

		//Assign Posture
		transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
		transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Velocity = velocities[i];
			Actor.Bones[i].Transform.position = positions[i];
			Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
			Actor.Bones[i].ApplyLength();
		}
	}

	private void Default() {
		if(Controller.ProjectionActive) {
			ApplyStaticGoal(Controller.Projection.point, Vector3.ProjectOnPlane(Controller.Projection.point-transform.position, Vector3.up).normalized, Signals);
			/*
			Vector3 direction = (Controller.Projection.point - transform.position).GetRelativeDirectionTo(transform.GetWorldMatrix()).normalized;
			ApplyDynamicGoal(
				transform.GetWorldMatrix(),
				direction,
				Vector3.SignedAngle(transform.forward, Controller.Projection.point, Vector3.up) / 2f,
				Signals
			);
			*/
		} else {
			ApplyDynamicGoal(
				RootSeries.Transformations[TimeSeries.Pivot],
				Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
				Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f), 
				Signals
			);
		}
		Geometry.Setup(Geometry.Resolution);
		Geometry.Sense(RootSeries.Transformations[TimeSeries.Pivot], LayerMask.GetMask("Interaction"), Vector3.zero, InteractionSmoothing);
	}

	private IEnumerator Sit() {
		Controller.Signal signal = Controller.GetSignal("Sit");
		Interaction interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);

		float threshold = 0.25f;

		if(interaction != null) {
			Controller.ActiveInteraction = interaction;
			IsInteracting = true;
			while(signal.Query()) {
				ApplyStaticGoal(interaction.GetContact("Hips").GetPosition(), interaction.GetContact("Hips").GetForward(), Signals);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				yield return new WaitForSeconds(0f);
			}
			while(StyleSeries.GetStyle(TimeSeries.Pivot, "Sit") > threshold) {
				ApplyDynamicGoal(
					RootSeries.Transformations[TimeSeries.Pivot],
					Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
					Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f), 
					Signals
				);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				yield return new WaitForSeconds(0f);
			}
			IsInteracting = false;
			Controller.ActiveInteraction = null;
		}
	}

	private IEnumerator Open() {
		Controller.Signal signal = Controller.GetSignal("Open");
		Interaction interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);

		if(interaction != null) {
			Controller.ActiveInteraction = interaction;
			IsInteracting = true;
			while(signal.Query()) {
				ApplyStaticGoal(interaction.GetCenter().GetPosition(), interaction.GetCenter().GetForward(), Signals);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				yield return new WaitForSeconds(0f);
			}
			IsInteracting = false;
			Controller.ActiveInteraction = null;
		}
	}

	private IEnumerator Carry() {
		Controller.Signal signal = Controller.GetSignal("Carry");
		Interaction interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);
		if(interaction != null) {
			Controller.ActiveInteraction = interaction;
			// Debug.Log("Carrying started...");
			IsInteracting = true;

			float duration = 0.5f;
			float threshold = 0.2f;

			Vector3 deltaPos = new Vector3(0f, -0.15f, 0.2f);
			Quaternion deltaRot = Quaternion.Euler(-30f, 0f, 0f);

			float height = 1f;

			Matrix4x4 GetObjectMatrix(float staticness) {
				Matrix4x4 right = Actor.GetBoneTransformation("RightWrist");
				Matrix4x4 left = Actor.GetBoneTransformation("LeftWrist");

				Quaternion rotation = Quaternion.Slerp(
					Quaternion.LookRotation(left.GetRight(), left.GetForward()), 
					Quaternion.LookRotation(-right.GetRight(), right.GetForward()), 
					0.5f
				);
				rotation *= deltaRot;
				rotation = Quaternion.Slerp(
					rotation,
					transform.rotation,
					Utility.Normalise(staticness, 0f, 1f, 0.5f, 1f)
				);

				return Matrix4x4.TRS(
					Vector3.Lerp(left.GetPosition(), right.GetPosition(), 0.5f) + rotation*deltaPos, 
					rotation,
					interaction.transform.lossyScale
				);
			}

			bool HasContact() {
				float left = ContactSeries.GetContact(TimeSeries.Pivot, "LeftWrist");
				float right = ContactSeries.GetContact(TimeSeries.Pivot, "RightWrist");
				return right > 0.5f && left > 0.5f || (left+right) > 1f;
			}

			//Move to the target location
			// Debug.Log("Approaching to lift object...");
			while(signal.Query()) {
				ApplyStaticGoal(interaction.GetCenter().GetPosition(), interaction.GetCenter().GetForward(), Controller.PoolSignals());
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				Geometry.Retransform(interaction.GetOrigin(GoalSeries.Transformations[TimeSeries.Pivot]));
				if(Vector3.Distance(GetObjectMatrix(0f).GetPosition(), interaction.transform.position) < threshold) {
					break;
				}
				yield return new WaitForSeconds(0f);
			}

			//Move the object from the surface to the hands
			// Debug.Log("Picking object...");
			float tPick = Time.time;
			Vector3 pos = interaction.transform.position;
			Quaternion rot = interaction.transform.rotation;
			while(signal.Query() && HasContact()) {
				float ratio = Mathf.Clamp((Time.time - tPick) / duration, 0f, 1f);
				Matrix4x4 m = GetObjectMatrix(1f-ratio);
				interaction.transform.position = Vector3.Lerp(pos, m.GetPosition(), ratio);
				interaction.transform.rotation = Quaternion.Slerp(rot, m.GetRotation(), ratio);
				ApplyStaticGoal(interaction.GetCenter().GetPosition(), interaction.GetCenter().GetForward(), Controller.PoolSignals());
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				Geometry.Retransform(interaction.GetOrigin(GoalSeries.Transformations[TimeSeries.Pivot]));
				if(ratio >= 1f) {
					break;
				}
				yield return new WaitForSeconds(0f);
			}
			
			//Move around with the object
			// Debug.Log("Carrying object and moving...");
			while(signal.Query() && HasContact()) {
				Matrix4x4 m = GetObjectMatrix(0f);
				interaction.transform.position = m.GetPosition();
				interaction.transform.rotation = m.GetRotation();
				ApplyDynamicGoal(
					interaction.GetCenter(),
					Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals) + new Vector3(0f, height-interaction.transform.position.y, 0f),
					Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f), 
					Signals
				);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				Geometry.Retransform(interaction.GetOrigin(GoalSeries.Transformations[TimeSeries.Pivot]));
				yield return new WaitForSeconds(0f);
			}

			//Perform motions to start placing the object
			// Debug.Log("Transitioning to placing down object...");
			while(HasContact()) {
				Matrix4x4 m = GetObjectMatrix(0f);
				interaction.transform.position = m.GetPosition();
				interaction.transform.rotation = m.GetRotation();
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				Geometry.Retransform(interaction.GetOrigin(GoalSeries.Transformations[TimeSeries.Pivot]));
				if(Controller.ProjectionActive) {
					ApplyStaticGoal(Controller.Projection.point, Vector3.ProjectOnPlane(Controller.Projection.point-transform.position, Vector3.up).normalized, Signals);
					if(Vector3.Distance(Controller.Projection.point, interaction.transform.position) < threshold) {
						break;
					}
				} else {
					Vector3 surface = Utility.ProjectGround(interaction.transform.position, LayerMask.GetMask("Default", "Ground"));
					ApplyStaticGoal(surface, Vector3.ProjectOnPlane(m.GetForward(), Vector3.up).normalized, Signals);
					if(Vector3.Distance(surface, interaction.transform.position) < threshold) {
						break;
					}
				}
				yield return new WaitForSeconds(0f);
			}

			//Make sure the object is again placed on the surface
			// Debug.Log("Placing object...");
			float tPlace = Time.time;
			Vector3 aPosPlace = interaction.transform.position;
			Vector3 bPosPlace = Utility.ProjectGround(aPosPlace, LayerMask.GetMask("Default", "Ground"));
			Quaternion aRotPlace = interaction.transform.rotation;
			Quaternion bRotPlace = Quaternion.LookRotation(Vector3.ProjectOnPlane(aRotPlace.GetForward(), Vector3.up), Vector3.up);
			while(true) {
				float ratio = Mathf.Clamp((Time.time - tPlace) / duration, 0f, 1f);
				interaction.transform.position = Vector3.Lerp(aPosPlace, bPosPlace, ratio);
				interaction.transform.rotation = Quaternion.Slerp(aRotPlace, bRotPlace, ratio);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				Geometry.Retransform(interaction.GetOrigin(GoalSeries.Transformations[TimeSeries.Pivot]));
				ApplyStaticGoal(transform.position, transform.forward, Signals);
				if(ratio >= 1f) {
					break;
				}
				yield return new WaitForSeconds(0f);
			}

			while(StyleSeries.GetStyle(TimeSeries.Pivot, "Carry") > 0.1) {
				ApplyDynamicGoal(
					RootSeries.Transformations[TimeSeries.Pivot],
					Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
					Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f), 
					Signals
				);
				Geometry.Setup(Geometry.Resolution);
				Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
				yield return new WaitForSeconds(0f);
			}

			IsInteracting = false;
			// Debug.Log("Carrying finished...");
			Controller.ActiveInteraction = null;
		}
	}

    protected override void Postprocess() {
		Matrix4x4 rightFoot = Actor.GetBoneTransformation(ContactSeries.Bones[3]);
		Matrix4x4 leftFoot = Actor.GetBoneTransformation(ContactSeries.Bones[4]);
		RightFootIK.Objectives[0].SetTarget(rightFoot.GetPosition(), 1f-ContactSeries.Values[TimeSeries.Pivot][3]);
		RightFootIK.Objectives[0].SetTarget(rightFoot.GetRotation());
		LeftFootIK.Objectives[0].SetTarget(leftFoot.GetPosition(), 1f-ContactSeries.Values[TimeSeries.Pivot][4]);
		LeftFootIK.Objectives[0].SetTarget(leftFoot.GetRotation());
		RightFootIK.Solve();
		LeftFootIK.Solve();

		Transform rightToe = Actor.FindBone("RightToe").Transform;
		Vector3 rightPos = rightToe.transform.position;
		rightPos.y = Mathf.Max(rightPos.y, 0.02f);
		rightToe.position = rightPos;

		Transform leftToe = Actor.FindBone("LeftToe").Transform;
		Vector3 leftPos = leftToe.transform.position;
		leftPos.y = Mathf.Max(leftPos.y, 0.02f);
		leftToe.position = leftPos;
    }

	protected override void OnGUIDerived() {
		if(!ShowGUI) {
			return;
		}
		if(ShowGoal) {
			GoalSeries.GUI();
		}
		if(ShowCurrent) {
			StyleSeries.GUI();
		}
		if(ShowPhase) {
			PhaseSeries.GUI();
		}
		if(ShowContacts) {
			ContactSeries.GUI();
		}
	}

	protected override void OnRenderObjectDerived() {
		Controller.Draw();

		if(ShowRoot) {
			RootSeries.Draw();
		}
		if(ShowGoal) {
			GoalSeries.Draw();
		}
		if(ShowCurrent) {
			StyleSeries.Draw();
		}
		if(ShowPhase) {
			PhaseSeries.Draw();
		}
		if(ShowContacts) {
			ContactSeries.Draw();
		}
		if(ShowEnvironment) {
			Environment.Draw(UltiDraw.Mustard.Transparent(0.25f));
		}
		if(ShowInteraction) {
			Geometry.Draw(UltiDraw.Cyan.Transparent(0.25f));
		}

		if(ShowBiDirectional) {
			UltiDraw.Begin();
			for(int i=0; i<PosePrediction.Length; i++) {
				UltiDraw.DrawSphere(PosePrediction[i], Quaternion.identity, 0.05f, UltiDraw.Magenta);
			}
			for(int i=0; i<RootPrediction.Length; i++) {
				UltiDraw.DrawCircle(RootPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkRed.Darken(0.5f));
				UltiDraw.DrawArrow(RootPrediction[i].GetPosition(), RootPrediction[i].GetPosition() + 0.1f*RootPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkRed);
				if(i<RootPrediction.Length-1) {
					UltiDraw.DrawLine(RootPrediction[i].GetPosition(), RootPrediction[i+1].GetPosition(), UltiDraw.Black);
				}
			}
			for(int i=0; i<GoalPrediction.Length; i++) {
				UltiDraw.DrawCircle(GoalPrediction[i].GetPosition(), 0.05f, UltiDraw.DarkGreen.Darken(0.5f));
				UltiDraw.DrawArrow(GoalPrediction[i].GetPosition(), GoalPrediction[i].GetPosition() + 0.1f*GoalPrediction[i].GetForward(), 0f, 0f, 0.025f, UltiDraw.DarkGreen);
				if(i<GoalPrediction.Length-1) {
					UltiDraw.DrawLine(GoalPrediction[i].GetPosition(), GoalPrediction[i+1].GetPosition(), UltiDraw.Black);
				}
			}
			UltiDraw.End();
		}
    }

	private float[] GenerateGating() {
		List<float> values = new List<float>();

		for(int k=0; k<TimeSeries.KeyCount; k++) {
			int index = TimeSeries.GetKey(k).Index;
			Vector2 phase = Utility.PhaseVector(PhaseSeries.Values[index]);
			for(int i=0; i<StyleSeries.Styles.Length; i++) {
				float magnitude = StyleSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f ,1f);
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
			}
			for(int i=0; i<GoalSeries.Actions.Length; i++) {
				float magnitude = GoalSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f ,1f);
				Matrix4x4 root = RootSeries.Transformations[index];
				root[1,3] = 0f;
				Matrix4x4 goal = GoalSeries.Transformations[index];
				goal[1,3] = 0f;
				float distance = Vector3.Distance(root.GetPosition(), goal.GetPosition());
				float angle = Quaternion.Angle(root.GetRotation(), goal.GetRotation());
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
				values.Add(magnitude * distance * phase.x);
				values.Add(magnitude * distance * phase.y);
				values.Add(magnitude * angle * phase.x);
				values.Add(magnitude * angle * phase.y);
			}
		}
		return values.ToArray();
	}

	private void ApplyStaticGoal(Vector3 position, Vector3 direction, float[] actions) {
		//Transformations
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = TimeSeries.GetWeight1byN1(i, 2f);
			float positionBlending = weight * UserControl;
			float directionBlending = weight * UserControl;
			Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), position, positionBlending));
			Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.LookRotation(Vector3.Slerp(GoalSeries.Transformations[i].GetForward(), direction, directionBlending), Vector3.up));
		}

		//Actions
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			float w = (float)(i-TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
			w = Utility.Normalise(w, 0f, 1f, 1f/TimeSeries.FutureKeyCount, 1f);
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				float weight = GoalSeries.Values[i][j];
				weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
				weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f-UserControl);
				if(actions[j] != GoalSeries.Values[i][j]) {
					GoalSeries.Values[i][j] = Mathf.Lerp(
						GoalSeries.Values[i][j], 
						Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
						w);
				}
			}
		}
	}

	private void ApplyDynamicGoal(Matrix4x4 root, Vector3 move, float turn, float[] actions) {
		//Transformations
		Vector3[] positions_blend = new Vector3[TimeSeries.Samples.Length];
		Vector3[] directions_blend = new Vector3[TimeSeries.Samples.Length];
		float time = 2f;
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = TimeSeries.GetWeight1byN1(i, 0.5f);
			float bias_pos = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			float bias_dir = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			directions_blend[i] = Quaternion.AngleAxis(bias_dir * turn, Vector3.up) * Vector3.ProjectOnPlane(root.GetForward(), Vector3.up).normalized;
			if(i==0) {
				positions_blend[i] = root.GetPosition() + 
					Vector3.Lerp(
					GoalSeries.Transformations[i+1].GetPosition() - GoalSeries.Transformations[i].GetPosition(), 
					time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			} else {
				positions_blend[i] = positions_blend[i-1] + 
					Vector3.Lerp(
					GoalSeries.Transformations[i].GetPosition() - GoalSeries.Transformations[i-1].GetPosition(), 
					time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			}
		}
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), positions_blend[i], UserControl));
			Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.Slerp(GoalSeries.Transformations[i].GetRotation(), Quaternion.LookRotation(directions_blend[i], Vector3.up), UserControl));
		}
		
		//Actions
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			float w = (float)(i-TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
			w = Utility.Normalise(w, 0f, 1f, 1f/TimeSeries.FutureKeyCount, 1f);
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				float weight = GoalSeries.Values[i][j];
				weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
				weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f-UserControl);
				if(actions[j] != GoalSeries.Values[i][j]) {
					GoalSeries.Values[i][j] = Mathf.Lerp(
						GoalSeries.Values[i][j], 
						Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
						w);
				}
			}
		}
	}

}