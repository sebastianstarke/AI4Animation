#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class RootModule : Module {

	public enum TOPOLOGY {Biped, Quadruped, Custom};

	public int Root = 0;
	public TOPOLOGY Topology = TOPOLOGY.Biped;
	public int RightShoulder, LeftShoulder, RightHip, LeftHip, Neck, Hips;
	public LayerMask Ground = 0;
	public Axis ForwardAxis = Axis.ZPositive;
	public bool Smooth = true;

	//Precomputed
	private float[] SamplingWindow = null;
	private Quaternion[] TmpRotations = null;
	private float[] TmpAngles = null;
	private Precomputable<Matrix4x4>[] PrecomputedRegularTransformations = null;
	private Precomputable<Matrix4x4>[] PrecomputedInverseTransformations = null;
	private Precomputable<Vector3>[] PrecomputedRegularPositions = null;
	private Precomputable<Vector3>[] PrecomputedInversePositions = null;
	private Precomputable<Quaternion>[] PrecomputedRegularRotations = null;
	private Precomputable<Quaternion>[] PrecomputedInverseRotations = null;
	private Precomputable<Vector3>[] PrecomputedRegularVelocities = null;
	private Precomputable<Vector3>[] PrecomputedInverseVelocities = null;

	public override ID GetID() {
		return ID.Root;
	}

	public override void DerivedResetPrecomputation() {
		SamplingWindow = null;
		TmpRotations = null;
		TmpAngles = null;
		PrecomputedRegularTransformations = Data.ResetPrecomputable(PrecomputedRegularTransformations);
		PrecomputedInverseTransformations = Data.ResetPrecomputable(PrecomputedInverseTransformations);
		PrecomputedRegularPositions = Data.ResetPrecomputable(PrecomputedRegularPositions);
		PrecomputedInversePositions = Data.ResetPrecomputable(PrecomputedInversePositions);
		PrecomputedRegularRotations = Data.ResetPrecomputable(PrecomputedRegularRotations);
		PrecomputedInverseRotations = Data.ResetPrecomputable(PrecomputedInverseRotations);
		PrecomputedRegularVelocities = Data.ResetPrecomputable(PrecomputedRegularVelocities);
		PrecomputedInverseVelocities = Data.ResetPrecomputable(PrecomputedInverseVelocities);
	}

	public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
		RootSeries instance = new RootSeries(global);
		for(int i=0; i<instance.Samples.Length; i++) {
			instance.Transformations[i] = GetRootTransformation(timestamp + instance.Samples[i].Timestamp, mirrored);
			instance.Velocities[i] = GetRootVelocity(timestamp + instance.Samples[i].Timestamp, mirrored);
		}
		return instance;
	}

	protected override void DerivedInitialize() {
		MotionData.Hierarchy.Bone rs = Data.Source.FindBoneContains("RightShoulder");
		RightShoulder = rs == null ? 0 : rs.Index;
		MotionData.Hierarchy.Bone ls = Data.Source.FindBoneContains("LeftShoulder");
		LeftShoulder = ls == null ? 0 : ls.Index;
		MotionData.Hierarchy.Bone rh = Data.Source.FindBoneContains("RightHip", "RightUpLeg");
		RightHip = rh == null ? 0 : rh.Index;
		MotionData.Hierarchy.Bone lh = Data.Source.FindBoneContains("LeftHip", "LeftUpLeg");
		LeftHip = lh == null ? 0 : lh.Index;
		MotionData.Hierarchy.Bone n = Data.Source.FindBoneContains("Neck");
		Neck = n == null ? 0 : n.Index;
		MotionData.Hierarchy.Bone h = Data.Source.FindBoneContains("Hips");
		Hips = h == null ? 0 : h.Index;
		Ground = LayerMask.GetMask("Ground");
	}

	protected override void DerivedLoad(MotionEditor editor) {
		
	}

	protected override void DerivedCallback(MotionEditor editor) {
		Frame frame = editor.GetCurrentFrame();
		editor.GetActor().transform.OverridePositionAndRotation(GetRootPosition(frame.Timestamp, editor.Mirror), GetRootRotation(frame.Timestamp, editor.Mirror));
	}

	protected override void DerivedGUI(MotionEditor editor) {
	
	}

	protected override void DerivedDraw(MotionEditor editor) {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		Root = EditorGUILayout.Popup("Root", Root, Data.Source.GetBoneNames());
		Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
		RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Data.Source.GetBoneNames());
		LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Data.Source.GetBoneNames());
		RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Data.Source.GetBoneNames());
		LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Data.Source.GetBoneNames());
		Neck = EditorGUILayout.Popup("Neck", Neck, Data.Source.GetBoneNames());
		Hips = EditorGUILayout.Popup("Hips", Hips, Data.Source.GetBoneNames());
		ForwardAxis = (Axis)EditorGUILayout.EnumPopup("Forward Axis", ForwardAxis);
		Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
		Smooth = EditorGUILayout.Toggle("Smooth", Smooth);
	}

	public Matrix4x4 GetRootTransformation(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseTransformations[index] == null) {
				PrecomputedInverseTransformations[index] = new Precomputable<Matrix4x4>(Compute());
			}
			if(!mirrored && PrecomputedRegularTransformations[index] == null) {
				PrecomputedRegularTransformations[index] = new Precomputable<Matrix4x4>(Compute());
			}
			return mirrored ? PrecomputedInverseTransformations[index].Value : PrecomputedRegularTransformations[index].Value;
		}

		return Compute();
		Matrix4x4 Compute() {
			return Matrix4x4.TRS(GetRootPosition(timestamp, mirrored), GetRootRotation(timestamp, mirrored), Vector3.one);
		}
	}

	public Vector3 GetRootPosition(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInversePositions[index] == null) {
				PrecomputedInversePositions[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularPositions[index] == null) {
				PrecomputedRegularPositions[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInversePositions[index].Value : PrecomputedRegularPositions[index].Value;
		}

		return Compute();
		Vector3 Compute() {
			return RootPosition(timestamp, mirrored);
		}
	}

	public Quaternion GetRootRotation(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseRotations[index] == null) {
				PrecomputedInverseRotations[index] = new Precomputable<Quaternion>(Compute());
			}
			if(!mirrored && PrecomputedRegularRotations[index] == null) {
				PrecomputedRegularRotations[index] = new Precomputable<Quaternion>(Compute());
			}
			return mirrored ? PrecomputedInverseRotations[index].Value : PrecomputedRegularRotations[index].Value;
		}

		return Compute();
		Quaternion Compute() {
			if(!Smooth)  {
				return RootRotation(timestamp, mirrored);
			}
			
			SamplingWindow = SamplingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SamplingWindow;
			TmpRotations = TmpRotations.Validate(SamplingWindow.Length);
			TmpAngles = TmpAngles.Validate(TmpRotations.Length-1);
			for(int i=0; i<SamplingWindow.Length; i++) {
				TmpRotations[i] = RootRotation(timestamp + SamplingWindow[i], mirrored);
			}
			for(int i=0; i<TmpAngles.Length; i++) {
				TmpAngles[i] = Vector3.SignedAngle(TmpRotations[i].GetForward(), TmpRotations[i+1].GetForward(), Vector3.up) / (SamplingWindow[i+1] - SamplingWindow[i]);
			}
			float power = Mathf.Deg2Rad*Mathf.Abs(TmpAngles.Gaussian());

			return TmpRotations.Gaussian(power);
		}
	}

	public Vector3 GetRootVelocity(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseVelocities[index] == null) {
				PrecomputedInverseVelocities[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularVelocities[index] == null) {
				PrecomputedRegularVelocities[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInverseVelocities[index].Value : PrecomputedRegularVelocities[index].Value;
		}
		
		return Compute();
		Vector3 Compute() {
			return (GetRootPosition(timestamp, mirrored) - GetRootPosition(timestamp - Data.GetDeltaTime(), mirrored)) / Data.GetDeltaTime();
		}
	}

	private Vector3 RootPosition(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return 2f*RootPosition(Data.GetFrame(boundary)) - RootPosition(Data.GetFrame(clamped));
		} else {
			return RootPosition(Data.GetFrame(timestamp));
		}

		Vector3 RootPosition(Frame frame) {
			Vector3 position = frame.GetBoneTransformation(Root, mirrored).GetPosition();
			if(Ground == 0) {
				position.y = 0f;
			} else {
				position = Utility.ProjectGround(position, Ground);
			}
			return position;
		}
	}

	private Quaternion RootRotation(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return RootRotation(Data.GetFrame(clamped));
		} else {
			return RootRotation(Data.GetFrame(timestamp));
		}

		Quaternion RootRotation(Frame frame) {
			if(Topology == TOPOLOGY.Biped) {
				Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightHip, mirrored).GetPosition() - frame.GetBoneTransformation(LeftHip, mirrored).GetPosition(), Vector3.up).normalized;
				Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightShoulder, mirrored).GetPosition() - frame.GetBoneTransformation(LeftShoulder, mirrored).GetPosition(), Vector3.up).normalized;
				Vector3 v = (v1+v2).normalized;
				Vector3 forward = Vector3.ProjectOnPlane(-Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
				return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
			}
			if(Topology == TOPOLOGY.Quadruped) {
				Vector3 neck = frame.GetBoneTransformation(Neck, mirrored).GetPosition();
				Vector3 hips = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
				Vector3 forward = Vector3.ProjectOnPlane(neck - hips, Vector3.up).normalized;;
				return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward.normalized, Vector3.up);
			}
			if(Topology == TOPOLOGY.Custom) {
				return Quaternion.LookRotation(
					Vector3.ProjectOnPlane(Quaternion.FromToRotation(Vector3.forward, ForwardAxis.GetAxis()) * frame.GetBoneTransformation(Root, mirrored).GetForward(), Vector3.up).normalized, 
					Vector3.up
				);
			}
			return Quaternion.identity;
		}
	}

}
#endif
