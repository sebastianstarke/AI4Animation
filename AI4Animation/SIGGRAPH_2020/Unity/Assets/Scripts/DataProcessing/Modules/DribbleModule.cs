#if UNITY_EDITOR
using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class DribbleModule : Module {

    public Actor Rival = null;

    public int Ball = 0;
    
    public int LeftHand = 0;
    public int RightHand = 0;

    public int Hips = 0;
    public int Head = 0;
    public Axis Axis = Axis.ZPositive;

    public float Area = 2.5f;
    public float Radius = 0.125f;

    public string[] Interactors = new string[0];

    private Transform BallInstance = null;

    [NonSerialized] private bool ShowSource = false;
    [NonSerialized] private bool ShowSequence = false;

    private RootModule RootModule = null;
    private ContactModule ContactModule = null;

    //Precomputed
    private float[] SmoothingWindow = null;
    private Vector3[] SmoothingVectors = null;
    private float[] SmoothingAngles = null;
    private float[] SmoothingContacts = null;
    private float[] SmoothingHeights = null;
    private float[] SmoothingSpeeds = null;
    private Precomputable<Vector3>[] PrecomputedRegularPivots = null;
    private Precomputable<Vector3>[] PrecomputedInversePivots = null;
    private Precomputable<Vector3>[] PrecomputedRegularMomentums = null;
    private Precomputable<Vector3>[] PrecomputedInverseMomentums = null;

	public override ID GetID() {
		return ID.Dribble;
	}

    public override void DerivedResetPrecomputation() {
        SmoothingWindow = null;
        SmoothingVectors = null;
        SmoothingAngles = null;
        SmoothingContacts = null;
        SmoothingHeights = null;
        SmoothingSpeeds = null;
        PrecomputedRegularPivots = Data.ResetPrecomputable(PrecomputedRegularPivots);
        PrecomputedInversePivots = Data.ResetPrecomputable(PrecomputedInversePivots);
        PrecomputedRegularMomentums = Data.ResetPrecomputable(PrecomputedRegularMomentums);
        PrecomputedInverseMomentums = Data.ResetPrecomputable(PrecomputedInverseMomentums);
    }

    public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
        DribbleSeries instance = new DribbleSeries(
            global,
            Area,
            BallInstance,
            MotionEditor.GetInstance().GetActor(),
            (RootSeries)Data.GetModule<RootModule>().ExtractSeries(global, timestamp, mirrored),
            GetRival(),
            GetRivalRoot(global, timestamp, mirrored)
        );
        for(int i=0; i<instance.Samples.Length; i++) {
            instance.BallTransformations[i] = GetBallTransformation(timestamp + instance.Samples[i].Timestamp, mirrored);
            instance.BallVelocities[i] = GetBallVelocity(timestamp + instance.Samples[i].Timestamp, mirrored);
            instance.Pivots[i] = GetBallPivot(timestamp + instance.Samples[i].Timestamp, mirrored);
            instance.Momentums[i] = GetBallMomentum(timestamp + instance.Samples[i].Timestamp, mirrored);
        }
        return instance;
    }

	protected override void DerivedInitialize() {
        Ball = Data.Source.FindBone("Player 01:Ball") != null ? Data.Source.FindBone("Player 01:Ball").Index : 0;
        LeftHand = Data.Source.FindBone("Player 01:LeftHand") != null ? Data.Source.FindBone("Player 01:LeftHand").Index : 0;
        RightHand = Data.Source.FindBone("Player 01:RightHand") != null ? Data.Source.FindBone("Player 01:RightHand").Index : 0;
        Hips = Data.Source.FindBone("Player 01:Hips") != null ? Data.Source.FindBone("Player 01:Hips").Index : 0;
        Head = Data.Source.FindBone("Player 01:Head") != null ? Data.Source.FindBone("Player 01:Head").Index : 0;
        Interactors = new string[Data.Frames.Length];
        for(int i=0; i<Interactors.Length; i++) {
            Interactors[i] = null;
        }
	}

	protected override void DerivedLoad(MotionEditor editor) {
		
    }

	protected override void DerivedCallback(MotionEditor editor) {
        if(BallInstance == null) {
            BallInstance = editor.GetActor().FindTransform(Data.Source.Bones[Ball].Name);
        }
        if(BallInstance != null) {
            Matrix4x4 m = ShowSource ? editor.GetCurrentFrame().GetSourceTransformation(Ball, editor.Mirror) : editor.GetCurrentFrame().GetBoneTransformation(Ball, editor.Mirror);
            BallInstance.transform.position = m.GetPosition();
            BallInstance.transform.rotation = m.GetRotation();
        }
        if(GetRival() != null) {
            float t = editor.GetCurrentFrame().Timestamp;
            MotionData asset = GetRivalAsset(t);
            if(asset == null) {
                asset = Data;
            }
            int[] bones = MotionEditor.GetInstance().GetBoneMapping();
            for(int i=0; i<GetRival().Bones.Length; i++) {
                Matrix4x4 m = asset.GetFrame(t).GetBoneTransformation(bones[i], editor.Mirror);
                GetRival().Bones[i].Transform.position = m.GetPosition();
                GetRival().Bones[i].Transform.rotation = m.GetRotation();
            }
        }
	}

    protected override void DerivedGUI(MotionEditor editor) {
    
    }

	protected override void DerivedDraw(MotionEditor editor) {
        UltiDraw.Begin();
        if(ShowSequence) {
            float start = editor.GetCurrentFrame().Timestamp - editor.PastWindow;
            float end = editor.GetCurrentFrame().Timestamp + editor.FutureWindow;
            for(float t=start; t<=end; t+=Data.GetDeltaTime()) {
                Vector3 position = GetBallPosition(t, editor.Mirror);
                UltiDraw.DrawSphere(position, Quaternion.identity, Radius, UltiDraw.Magenta);
            }
        }
        UltiDraw.End();
    }

	protected override void DerivedInspector(MotionEditor editor) {
        Rival = (Actor)EditorGUILayout.ObjectField("Rival", Rival, typeof(Actor), true);
        EditorGUILayout.ObjectField("Asset", GetRivalAsset(editor.GetCurrentFrame().Timestamp), typeof(MotionData), true);
        Ball = EditorGUILayout.Popup("Ball", Ball, Data.Source.GetBoneNames());
        LeftHand = EditorGUILayout.Popup("Left Hand", LeftHand, Data.Source.GetBoneNames());
        RightHand = EditorGUILayout.Popup("Right Hand", RightHand, Data.Source.GetBoneNames());
        Hips = EditorGUILayout.Popup("Hips", Hips, Data.Source.GetBoneNames());
        Head = EditorGUILayout.Popup("Head", Head, Data.Source.GetBoneNames());
        Axis = (Axis)EditorGUILayout.EnumPopup("Axis", Axis);

        Area = EditorGUILayout.FloatField("Area", Area);
        Radius = EditorGUILayout.FloatField("Radius", Radius);

        ShowSource = EditorGUILayout.Toggle("Show Source", ShowSource);
        ShowSequence = EditorGUILayout.Toggle("Show Sequence", ShowSequence);
	}

    public Actor GetRival() {
        if(Rival == null) {
            Actor[] instances = GameObject.FindObjectsOfType<Actor>();
            if(instances.Length > 0) {
                Rival = System.Array.Find(instances, x => x.name == "Rival");
            }
        }
        return Rival;
    }

    public MotionData GetRivalAsset(float timestamp) {
        return MotionEditor.GetInstance().GetAsset(GetInteractor(timestamp));
    }

    public RootSeries GetRivalRoot(TimeSeries global, float timestamp, bool mirrored) {
        MotionData asset = GetRivalAsset(timestamp);
        return asset == null ? null : (RootSeries)asset.GetModule<RootModule>().ExtractSeries(global, timestamp, mirrored);
    }

    public float GetControlRadius() {
        return Area / 2f;
    }

    public float GetInteractionRadius() {
        return 2f * Area;
    }

    private RootModule GetRootModule() {
        if(RootModule == null) {
            RootModule = Data.GetModule<RootModule>();
        }
        return RootModule;
    }

    private ContactModule GetContactModule() {
        if(ContactModule == null) {
            ContactModule = Data.GetModule<ContactModule>();
        }
        return ContactModule;
    }

    public bool IsDribbling(float timestamp, bool mirrored) {
        if(!InsideControlRadius()) {
            return false;
        }
        ContactModule contact = GetContactModule();
        if(contact == null) {
            return false;
        }
        ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
        ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
        if(left == null || right == null) {
            return false;
        }
        //Check holding contacts
        if(left.GetContact(timestamp, mirrored) == 1f && right.GetContact(timestamp, mirrored) == 1f) {
            return false;
        }
        //Check dribble contact
        if(left.GetContact(timestamp, mirrored) == 1f ^ right.GetContact(timestamp, mirrored) == 1f) {
            return true;
        }
        //Check if contact happened before and happens again after within the future and past window
        float window = 1f;
        Frame previous = contact.GetPreviousContactFrame(Data.GetFrame(timestamp), mirrored, left, right);
        if(previous == null || previous.Timestamp < timestamp - window) {
            return false;
        }
        Frame next = contact.GetNextContactFrame(Data.GetFrame(timestamp), mirrored, left, right);
        if(next != null && next.Timestamp - timestamp <= window) {
            return left.GetContact(next.Timestamp, mirrored) == 1f ^ right.GetContact(next.Timestamp, mirrored) == 1f;
        }
        return false;

        bool InsideControlRadius() {
            RootModule root = GetRootModule();
            if(root == null) {
                return false;
            }
            return Vector3.Distance(root.GetRootPosition(timestamp, mirrored).ZeroY(), GetBallPosition(timestamp, mirrored).ZeroY()) <= GetControlRadius();
        }
    }

    public bool IsHolding(float timestamp, bool mirrored) {
        ContactModule contact = GetContactModule();
        if(contact == null) {
            return false;
        }
        ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
        ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
        if(left == null || right == null) {
            return false;
        }
        return left.GetContact(timestamp, mirrored) == 1f && right.GetContact(timestamp, mirrored) == 1f;
    }

    public bool IsShooting(float timestamp, bool mirrored) {
        float ballHeight = Height(timestamp, mirrored);
        float ballUpMomentum = Speed(timestamp, mirrored);
        return (ballHeight > 1.5f) && (ballUpMomentum > 0.5f) && LeaveHand(timestamp, mirrored);
    }

    public bool LeaveHand(float timestamp, bool mirrored) {
        ContactModule contact = GetContactModule();
        if (contact == null) {
            return false;
        }
        float shootPeriod = 1.0f;
        ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
        ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
        Frame next = contact.GetNextContactEnd(Data.GetFrame(timestamp), mirrored, left, right);
        Frame previous = contact.GetPreviousContactEnd(Data.GetFrame(timestamp), mirrored, left, right);
        if (next != null && next.Timestamp - timestamp <= shootPeriod) {
            return true;
        }
        if (left.GetContact(timestamp, mirrored) == 0f && right.GetContact(timestamp, mirrored) == 0f &&
            previous != null && timestamp - previous.Timestamp <= shootPeriod) {
            return true;
        }
        return false;
    }

    private bool HasControl(float timestamp, bool mirrored) {
        return InsideRegion() && HasContact();

        bool InsideRegion() {
            RootModule root = GetRootModule();
            if(root == null) {
                return false;
            }
            return Vector3.Distance(root.GetRootPosition(timestamp, mirrored).ZeroY(), GetBallPosition(timestamp, mirrored).ZeroY()) <= GetControlRadius();
        }

        bool HasContact() {
            ContactModule contact = GetContactModule();
            if(contact == null) {
                return false;
            }
            ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
            ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
            if(left == null || right == null) {
                return false;
            }
            float window = 1f;
            foreach(Frame f in Data.GetFrames(timestamp - window, timestamp + window)) {
                if(contact.HasContact(f, mirrored, left, right)) {
                    return true;
                }
            }
            return false;
        }
    }

    public void ComputeInteraction() {
        Interactors = new string[Data.Frames.Length];
        for(int i=0; i<Interactors.Length; i++) {
            Interactors[i] = null;
        }

        bool IsValid(MotionData asset) {
            return asset.GetName().Contains("_P0");
        }
        
        string GetID(MotionData asset) {
            return asset.GetName().Substring(0, asset.GetName().LastIndexOf("_P0"));
        }

        MotionEditor editor = MotionEditor.GetInstance();
        if(editor.GetAsset() != Data) {
            Debug.Log("Asset " + Data.GetName() + " is not loaded.");
            return;
        }
        if(IsValid(Data)) {
            int pivot = editor.GetAssetIndex();

            //Collect all assets of same capture
            List<MotionData> assets = new List<MotionData>();
            assets.Add(Data);
            for(int i=pivot-1; i>=0; i--) {
                MotionData asset = editor.GetAsset(i);
                if(IsValid(asset) && GetID(Data) == GetID(asset)) {
                    assets.Add(asset);
                } else {
                    break;
                }
            }
            for(int i=pivot+1; i<editor.Assets.Length; i++) {
                MotionData asset = editor.GetAsset(i);
                if(IsValid(asset) && GetID(Data) == GetID(asset)) {
                    assets.Add(asset);
                } else {
                    break;
                }
            }
            
            //Find closest interaction inside area
            for(int i=0; i<Data.Frames.Length; i++) {
                Frame frame = Data.Frames[i];
                Matrix4x4 root = GetRoot(frame.Timestamp, false);
                float distance = GetInteractionRadius();
                foreach(MotionData data in assets) {
                    if(Data != data) {
                        RootModule m = Data.GetModule<RootModule>();
                        if(m != null) {
                            Matrix4x4 candidate = m.GetRootTransformation(frame.Timestamp, false); 
                            float d = Vector3.Distance(root.GetPosition(), candidate.GetPosition());
                            if(d < distance) {
                                distance = d;
                                Interactors[i] = Utility.GetAssetGUID(data);
                            }
                        }
                    }
                }
            }
        }
    }

    // public bool ComputeSpecialMove(float timestamp, bool mirrored, float delta) {
    //     ContactModule contact = (ContactModule)Data.GetModule(Module.ID.Contact);
    //     if(contact == null) {
    //         return IsSpecialMove(timestamp);
    //     }
    //     ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
    //     ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
    //     if(left == null || right == null) {
    //         return false;
    //     }
    //     Frame pivot = Data.GetFrame(timestamp);
    //     Frame start = contact.HasContact(pivot, mirrored, left, right) ? pivot : contact.GetPreviousContactEnd(pivot, mirrored, left, right);
    //     Frame end = contact.GetNextContactStart(pivot, mirrored, left, right);
    //     if(start != null && end != null) {
    //         if(
    //             left.GetContact(start, mirrored) == 1f && right.GetContact(end, mirrored) == 1f
    //             ||
    //             right.GetContact(start, mirrored) == 1f && left.GetContact(end, mirrored) == 1f
    //         ) {
    //             foreach(Frame f in Data.GetFrames(start.Timestamp, end.Timestamp)) {
    //                 if(IsSpecialMove(f.Timestamp)) {
    //                     return true;
    //                 }
    //             }
    //         }
    //     }
    //     return false;

    //     bool IsSpecialMove(float t) {
    //         Vector3 leftFoot = Data.GetFrame(t).GetBoneTransformation("Player 01:LeftToeBase", mirrored).GetPosition().ZeroY();
    //         Vector3 rightFoot = Data.GetFrame(t).GetBoneTransformation("Player 01:RightToeBase", mirrored).GetPosition().ZeroY();
    //         Vector3 center = 0.5f * (leftFoot + rightFoot);
    //         float radius = Vector3.Distance(leftFoot, rightFoot) / 2f;
    //         Vector3 ballPosition = GetBallPosition(t, mirrored);
    //         Vector3 ballVelocity = GetBallVelocity(t, mirrored, delta) * delta;
    //         bool crossPos = Math3D.AreLineSegmentsCrossing(ballPosition.ZeroY(), ballPosition.ZeroY()+ballVelocity.ZeroY(), leftFoot, rightFoot);
    //         bool crossNeg = Math3D.AreLineSegmentsCrossing(ballPosition.ZeroY(), ballPosition.ZeroY()-ballVelocity.ZeroY(), leftFoot, rightFoot);
    //         return Vector3.Distance(center, ballPosition) < radius && (crossPos || crossNeg);
    //     }
    // }

    private string GetInteractor(float timestamp) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return Interactors[Data.GetFrame(clamped).Index-1];
		} else {
			return Interactors[Data.GetFrame(timestamp).Index-1];
		}
    }

    public Vector3 GetBallPivot(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInversePivots[index] == null) {
				PrecomputedInversePivots[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularPivots[index] == null) {
				PrecomputedRegularPivots[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInversePivots[index].Value : PrecomputedRegularPivots[index].Value;
		}

        return Compute();
        Vector3 Compute() {
            SmoothingWindow = SmoothingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SmoothingWindow;
            SmoothingVectors = SmoothingVectors.Validate(SmoothingWindow.Length);
            SmoothingAngles = SmoothingAngles.Validate(SmoothingVectors.Length-1);
            for(int i=0; i<SmoothingVectors.Length; i++) {
                SmoothingVectors[i] = (GetBallPosition(timestamp + SmoothingWindow[i], mirrored).GetRelativePositionTo(GetRoot(timestamp + SmoothingWindow[i], mirrored))).ZeroY().normalized;
            }

			for(int i=0; i<SmoothingAngles.Length; i++) {
                SmoothingAngles[i] = Vector3.SignedAngle(SmoothingVectors[i], SmoothingVectors[i+1], Vector3.up) / (SmoothingWindow[i+1] - SmoothingWindow[i]);
            }
            float power = Mathf.Deg2Rad*Mathf.Abs(SmoothingAngles.Gaussian());

            return SmoothingVectors.Gaussian(power).SetY(Height(timestamp, mirrored));
        }
    }

    public Vector3 GetBallMomentum(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseMomentums[index] == null) {
				PrecomputedInverseMomentums[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularMomentums[index] == null) {
				PrecomputedRegularMomentums[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInverseMomentums[index].Value : PrecomputedRegularMomentums[index].Value;
		}

        return Compute();
        Vector3 Compute() {
            return ((GetBallPivot(timestamp, mirrored) - GetBallPivot(timestamp - Data.GetDeltaTime(), mirrored)) / Data.GetDeltaTime()).SetY(Speed(timestamp, mirrored));
        }
    }

    private float ContactPower(float timestamp, bool mirrored) {
        ContactModule contact = GetContactModule();
        if(contact == null) {
            return 0f;
        }
        ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
        ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
        if(left == null || right == null) {
            return 0f;
        }
        SmoothingWindow = SmoothingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SmoothingWindow;
        SmoothingContacts = SmoothingContacts.Validate(SmoothingWindow.Length);
        for(int i=0; i<SmoothingContacts.Length; i++) {
            SmoothingContacts[i] = left.GetContact(timestamp + SmoothingWindow[i], mirrored) + right.GetContact(timestamp + SmoothingWindow[i], mirrored);
        }
        return SmoothingContacts.Gaussian();
    }

    private float Height(float timestamp, bool mirrored) {
        float power = ContactPower(timestamp, mirrored);
        SmoothingWindow = SmoothingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SmoothingWindow;
        SmoothingHeights = SmoothingContacts.Validate(SmoothingWindow.Length);
        for(int i=0; i<SmoothingHeights.Length; i++) {
            SmoothingHeights[i] = GetBallPosition(timestamp + SmoothingWindow[i], mirrored).y;
        }
        return SmoothingHeights.Gaussian(power);
    }

    private float Speed(float timestamp, bool mirrored) {
        float power = ContactPower(timestamp, mirrored);
        SmoothingWindow = SmoothingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SmoothingWindow;
        SmoothingSpeeds = SmoothingSpeeds.Validate(SmoothingWindow.Length);
        for(int i=0; i<SmoothingSpeeds.Length; i++) {
            SmoothingSpeeds[i] = Mathf.Abs(GetBallVelocity(timestamp + SmoothingWindow[i], mirrored).y);
        }
        return SmoothingSpeeds.Gaussian(power);
    }

    public Matrix4x4 GetBallTransformation(float timestamp, bool mirrored) {
        return Matrix4x4.TRS(GetBallPosition(timestamp, mirrored), GetBallRotation(timestamp, mirrored), Vector3.one);
    }

    private Vector3 GetBallPosition(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
            float boundary = Mathf.Clamp(timestamp, start, end);
            float pivot = 2f*boundary - timestamp;
            float clamped = Mathf.Clamp(pivot, start, end);
            Matrix4x4 reference =  Data.GetFrame(clamped).GetBoneTransformation(Ball, mirrored);
            Vector3 position = 2f*Data.GetFrame(boundary).GetBoneTransformation(Ball, mirrored).GetPosition() - reference.GetPosition();
            position.y = reference.GetPosition().y;
            return position;
        } else {
            return Data.GetFrame(timestamp).GetBoneTransformation(Ball, mirrored).GetPosition();
        }
    }

    private Quaternion GetBallRotation(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return Data.GetFrame(clamped).GetBoneTransformation(Ball, mirrored).GetRotation();
		} else {
			return Data.GetFrame(timestamp).GetBoneTransformation(Ball, mirrored).GetRotation();
		}
    }

    private Vector3 GetBallVelocity(float timestamp, bool mirrored) {
		return (GetBallPosition(timestamp, mirrored) - GetBallPosition(timestamp - Data.GetDeltaTime(), mirrored)) / Data.GetDeltaTime();
    }

    public Matrix4x4[] CleanupBallTransformations(bool detectInvalid) {
        bool invalid = IsInvalid();
        if(!detectInvalid && invalid) {
            invalid = false;
            Debug.LogWarning("Ignored invalid detection in asset " + Data.GetName() + ".");
        }

        Matrix4x4[] motion = new Matrix4x4[Data.Frames.Length];
        for(int i=0; i<Data.Frames.Length; i++) {
            Matrix4x4 ball = GetTransformation(Ball, Data.Frames[i].Timestamp);
            Matrix4x4 root = GetRoot(Data.Frames[i].Timestamp, false);
            float radius = GetControlRadius();
            if(invalid) {
                //Project Surface
                Matrix4x4 head = GetTransformation(Head, Data.Frames[i].Timestamp);
                float offset = (head.GetPosition() - root.GetPosition()).ZeroY().magnitude;
                Vector3 projection = head.GetPosition() + (radius - offset) * (head.GetRotation()*Axis.GetAxis());
                Vector3 position = (root.GetPosition().ZeroY() + radius*(projection - root.GetPosition()).ZeroY().normalized).SetY(projection.y);
                position.y = Mathf.Max(position.y, root.GetPosition().y + Radius);
                Matrix4x4Extensions.SetPosition(ref ball, position);
            } else {
                //Clamp Magnitude
                Matrix4x4Extensions.SetPosition(ref ball, ball.GetPosition().ClampMagnitudeXZ(radius, root.GetPosition()));
            }
            motion[i] = ball;
        }
        return motion;

        Matrix4x4 GetTransformation(int bone, float timestamp) {
            float start = Data.GetFirstValidFrame().Timestamp;
            float end = Data.GetLastValidFrame().Timestamp;
            if(timestamp < start || timestamp > end) {
                float boundary = Mathf.Clamp(timestamp, start, end);
                float pivot = 2f*boundary - timestamp;
                float clamped = Mathf.Clamp(pivot, start, end);
                Matrix4x4 reference =  Data.GetFrame(clamped).GetBoneTransformation(bone, false);
                Vector3 position = 2f*Data.GetFrame(boundary).GetBoneTransformation(bone, false).GetPosition() - reference.GetPosition();
                position.y = reference.GetPosition().y;
                Quaternion rotation = reference.GetRotation();
                return Matrix4x4.TRS(position, rotation, Vector3.one);
            } else {
                return Data.GetFrame(timestamp).GetBoneTransformation(bone, false);
            }
        }

        bool IsInvalid() {
            //Get Range
            int start = Data.GetFirstValidFrame().Index-1;
            int end = Data.GetLastValidFrame().Index-1;
            //Check contacts
            ContactModule contact = GetContactModule();
            if(contact == null) {
                return false;
            }
            ContactModule.Sensor left = contact.GetSensor(Data.Source.Bones[LeftHand].Name);
            ContactModule.Sensor right = contact.GetSensor(Data.Source.Bones[RightHand].Name);
            if(left != null && right != null) {
                float[] leftContacts = left.Contacts.GatherByPivots(start, end);
                float[] rightContacts = left.Contacts.GatherByPivots(start, end);
                if(leftContacts.All(1f, 0.9f) ^ rightContacts.All(1f, 0.9f) && !ArrayExtensions.Equal(leftContacts, rightContacts).All(true, 0.1f)) {
                    Debug.LogWarning("Stuck-to-hand invalidation in file " + Data.GetName() + ".");
                    return true;
                }
            }
            //Check transformations
            Matrix4x4[] transformations = new Matrix4x4[Data.Frames.Length];
            for(int i=0; i<Data.Frames.Length; i++) {
                transformations[i] = Data.Frames[i].GetBoneTransformation(Ball, false);
            }
            if(transformations.GatherByPivots(start, end).Repeating(0.9f)) {
                Debug.LogWarning("Repeating ball transformation invalidation in file " + Data.GetName() + ".");
                return true;
            }
            //Check velocities
            Vector3[] velocities = new Vector3[Data.Frames.Length];
            for(int i=0; i<Data.Frames.Length; i++) {
                velocities[i] = Data.Frames[i].GetBoneVelocity(Ball, false);
            }
            if(velocities.GatherByPivots(start, end).Repeating(0.9f)) {
                Debug.LogWarning("Ball velocity invalidation in file " + Data.GetName() + ".");
                return true;
            }
            // //Check out-of-bounds
            // bool[] bounds = new bool[Data.Frames.Length];
            // for(int i=0; i<Data.Frames.Length; i++) {
            //     bounds[i] = Vector3.Distance(GetRoot(Data.Frames[i].Timestamp, false).GetPosition(), Data.Frames[i].GetBoneTransformation(Ball, false).GetPosition()) > 0.5f*GetInteractionArea();
            // }
            // if(bounds.All(true, 0.9f)) {
            //     return true;
            // }
            //All seems good
            return false;
        }
    }

    private Matrix4x4 GetRoot(float timestamp, bool mirrored) {
        RootModule root = GetRootModule();
        if(root == null) {
            return Matrix4x4.identity;
        }
        return root.GetRootTransformation(timestamp, mirrored);
    }

}
#endif