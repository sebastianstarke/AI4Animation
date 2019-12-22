using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class UltimateIK : MonoBehaviour {

    public enum ACTIVATION {Constant, Linear, Root, Square}
    public enum JOINT {Free, HingeX, HingeY, HingeZ, Ball}

    public bool AutoUpdate = true;

    public Transform Root = null;
    public Transform[] Objectives = new Transform[0];
    public Transform[] Targets = new Transform[0];

    public bool Draw = true;

    public Model Skeleton;
    public int SelectedBone = -1;
    public Vector3 SelectedPreview = Vector3.zero;
    public bool ShowPreview = false;

    void Reset() {
        Root = transform;
        Skeleton = new Model();
    }

    void LateUpdate() {
        if(AutoUpdate) {
            List<Matrix4x4> targets = new List<Matrix4x4>();
            for(int i=0; i<Objectives.Length; i++) {
                if(Objectives[i] != null) {
                    targets.Add(Targets[i] != null ? Targets[i].GetWorldMatrix(true) : Objectives[i].GetWorldMatrix(true));
                }
            }
            Skeleton.Solve(targets.ToArray());
        }
    }

    public void Rebuild() {
        SelectedBone = -1;
        SelectedPreview = Vector3.zero;
        Skeleton = BuildModel(Skeleton, Root, Objectives);
    }

    public static Model BuildModel(Transform root, params Transform[] objectives) {
        return BuildModel(null, root, objectives);
    }

    public static Model BuildModel(Model reference, Transform root, params Transform[] objectives) {
        if(reference == null) {
            reference = new Model();
        }
        if(reference.GetRoot() == root && reference.Objectives.Length == objectives.Length) {
            for(int i=0; i<objectives.Length; i++) {
                if(reference.Bones[reference.Objectives[i].Bone].Transform != objectives[i]) {
                    break;
                }
                if(i==objectives.Length-1) {
                    return reference;
                }
            }
        }
        objectives = Verify(root, objectives);
        Model skeleton = new Model();
        for(int i=0; i<objectives.Length; i++) {
            Transform[] chain = GetChain(root, objectives[i]);
            Objective objective = reference.FindObjective(objectives[i]);
            if(objective == null) {
                objective = new Objective();
            }
            for(int j=0; j<chain.Length; j++) {
                Bone bone = skeleton.FindBone(chain[j]);
                if(bone == null) {
                    bone = reference.FindBone(chain[j]);
                    if(bone != null) {
                        bone.Childs = new int[0];
                        bone.Objectives = new int[0];
                    }
                    if(bone == null) {
                        bone = new Bone();
                        bone.Transform = chain[j];
                        bone.ZeroPosition = chain[j].localPosition;
                        bone.ZeroRotation = chain[j].localRotation;
                    }
                    Bone parent = skeleton.FindBone(chain[j].parent);
                    if(parent != null) {
                        ArrayExtensions.Add(ref parent.Childs, skeleton.Bones.Length);
                    }
                    bone.Index = skeleton.Bones.Length;
                    ArrayExtensions.Add(ref skeleton.Bones, bone);
                }
                ArrayExtensions.Add(ref bone.Objectives, i);
            }
            objective.Bone = skeleton.FindBone(chain.Last()).Index;
            objective.TargetPosition = objectives[i].transform.position;
            objective.TargetRotation = objectives[i].transform.rotation;
            objective.Index = skeleton.Objectives.Length;
            ArrayExtensions.Add(ref skeleton.Objectives, objective);
        }
        skeleton.Iterations = reference.Iterations;
        skeleton.Threshold = reference.Threshold;
        skeleton.Activation = reference.Activation;
        skeleton.SeedZeroPose = reference.SeedZeroPose;
        return skeleton;
    }
    
    private static Transform[] Verify(Transform root, Transform[] objectives) {
        List<Transform> verified = new List<Transform>();
		if(root == null) {
			//Debug.Log("Given root was null. Extracting skeleton failed.");
			return verified.ToArray();
		}
		if(objectives.Length == 0) {
			//Debug.Log("No objectives given. Extracting skeleton failed.");
			return verified.ToArray();
		}
        for(int i=0; i<objectives.Length; i++) {
            if(objectives[i] == null) {
				//Debug.Log("A given objective was null and will be ignored.");
            } else if(verified.Contains(objectives[i])) {
                //Debug.Log("Given objective " + objectives[i].name + " is already contained and will be ignored.");
            } else if(!IsInsideHierarchy(root, objectives[i])) {
                //Debug.Log("Chain for " + objectives[i].name + " is not connected to " + root.name + " and will be ignored.");
            } else {
                verified.Add(objectives[i]);
            }
        }
        return verified.ToArray();
    }

    private static bool IsInsideHierarchy(Transform root, Transform t) {
        if(root == null || t == null) {
            return false;
        }
        while(t != root) {
            t = t.parent;
            if(t == null) {
                return false;
            }
        }
        return true;
    }

    private static Transform[] GetChain(Transform root, Transform end) {
        if(root == null || end == null) {
            return new Transform[0];
        }
        List<Transform> chain = new List<Transform>();
        Transform joint = end;
        chain.Add(joint);
        while(joint != root) {
            joint = joint.parent;
            if(joint == null) {
                return new Transform[0];
            } else {
                chain.Add(joint);
            }
        }
        chain.Reverse();
        return chain.ToArray();
    }

    [System.Serializable]
    public class Model {
        public int Iterations = 25;
        public float Threshold = 0.001f;
        public ACTIVATION Activation = ACTIVATION.Constant;
        public bool SeedZeroPose = false;
        public bool AvoidLocalOptima = false;

        public bool RootTranslationX, RootTranslationY, RootTranslationZ = false;

        public Bone[] Bones = new Bone[0];
        public Objective[] Objectives = new Objective[0];

        private float SolveTime = 0f;

        public bool IsSetup() {
            if(Bones.Length == 0 && Objectives.Length > 0) {
                Debug.Log("Oups! Bones are zero but objectives are not, this should not have happened!");
            }
            return Bones.Length > 0;
        }

        public Transform GetRoot() {
            return Bones.Length == 0 ? null : Bones.First().Transform;
        }

        public float GetSolveTime() {
            return SolveTime;
        }

        public void SetIterations(int value) {
            Iterations = Mathf.Max(value, 0);
        }

        public void SetThreshold(float value) {
            Threshold = Mathf.Max(value, 0f);
        }

        public Bone FindBone(Transform t) {
            return Array.Find(Bones, x => x.Transform == t);
        }

        public Objective FindObjective(Transform t) {
            return Array.Find(Objectives, x => x.Bone < Bones.Length && Bones[x.Bone].Transform == t);
        }

        public void SaveAsZeroPose() {
            foreach(Bone bone in Bones) {
                bone.ZeroPosition = bone.Transform.localPosition;
                bone.ZeroRotation = bone.Transform.localRotation;
            }
        }

        public void ComputeLevels() {
            Action<Bone, Bone> recursion = null;
            recursion = new Action<Bone, Bone>((parent, bone) => {
                bone.Level = parent == null ? 1 : bone.Active && parent.Active ? parent.Level + 1 : parent.Level;
                foreach(int index in bone.Childs) {
                    recursion(bone, Bones[index]);
                }
            });
            recursion(null, Bones.First());
        }

        public void SetWeights(params float[] weights) {
            if(Objectives.Length != weights.Length) {
                Debug.Log("Number of given weights <" + weights.Length + "> does not match number of objectives <" + Objectives.Length + ">.");
                return;
            }
            for(int i=0; i<Objectives.Length; i++) {
                Objectives[i].Weight = weights[i];
            }
        }

        public void Solve(params Matrix4x4[] targets) {
            if(Objectives.Length != targets.Length) {
                Debug.Log("Number of given targets <" + targets.Length + "> does not match number of objectives <" + Objectives.Length + ">.");
                return;
            }
            for(int i=0; i<Objectives.Length; i++) {
                Objectives[i].TargetPosition = targets[i].GetPosition();
                Objectives[i].TargetRotation = targets[i].GetRotation();
            }
            Solve();
        }

        public void Solve() {
            DateTime timestamp = Utility.GetTimestamp();
            if(IsSetup()) {
                ComputeLevels();
                if(SeedZeroPose) {
                    foreach(Bone bone in Bones) {
                        if(bone.Active) {
                            bone.Transform.localPosition = bone.ZeroPosition;
                            bone.Transform.localRotation = bone.ZeroRotation;
                        }
                    }
                }
                for(int i=0; i<Iterations; i++) {
                    if(!IsConverged()) {
                        if(RootTranslationX || RootTranslationY || RootTranslationZ) {
                            Vector3 delta = Vector3.zero;
                            int count = 0;
                            foreach(Objective o in Objectives) {
                                if(o.Active) {
                                    delta += o.TargetPosition - Bones[o.Bone].Transform.position;
                                    count += 1;
                                }
                            }
                            if(count > 0) {
                                delta.x *= RootTranslationX ? 1 : 0;
                                delta.y *= RootTranslationY ? 1 : 0;
                                delta.z *= RootTranslationZ ? 1 : 0;
                                GetRoot().position += delta / count;
                            }
                        }
                        Optimise(Bones.First());
                    }
                }
            }
            SolveTime = (float)Utility.GetElapsedTime(timestamp);
        }

        private void Optimise(Bone bone) {
            if(bone.Active) {
                Vector3 pos = bone.Transform.position;
                Quaternion rot = bone.Transform.rotation;
                Vector3 forward = Vector3.zero;
                Vector3 up = Vector3.zero;
                int count = 0;

                //Solve Objective Rotations
                foreach(int index in bone.Objectives) {
                    Objective o = Objectives[index];
                    if(o.Active && o.SolveRotation) {
                        Quaternion q = Quaternion.Slerp(
                            rot,
                            o.TargetRotation * Quaternion.Inverse(Bones[o.Bone].Transform.rotation) * rot,
                            GetWeight(bone, o)
                        );
                        forward += q*Vector3.forward;
                        up += q*Vector3.up;
                        count += 1;
                    }
                }

                //Solve Objective Positions
                foreach(int index in bone.Objectives) {
                    Objective o = Objectives[index];
                    if(o.Active && o.SolvePosition) {
                        Quaternion q = Quaternion.Slerp(
                            rot,
                            Quaternion.FromToRotation(Bones[o.Bone].Transform.position - pos, o.TargetPosition - pos) * rot,
                            GetWeight(bone, o)
                        );
                        forward += q*Vector3.forward;
                        up += q*Vector3.up;
                        count += 1;
                    }
                }

                if(count > 0) {
                    bone.Transform.rotation = Quaternion.LookRotation((forward/count).normalized, (up/count).normalized);
                }
            }

            bone.ResolveLimits(AvoidLocalOptima);

            foreach(int index in bone.Childs) {
                Optimise(Bones[index]);
            }
        }

        private bool IsConverged() {
            foreach(Objective o in Objectives) {
                if(o.GetError(this) > Threshold) {
                    return false;
                }
            }
            return true;
        }

        private float GetWeight(Bone bone, Objective objective) {
            switch(Activation) {
                case ACTIVATION.Constant:
                return objective.Weight;
                case ACTIVATION.Linear:
                return objective.Weight * (float)bone.Level/(float)Bones[objective.Bone].Level;
                case ACTIVATION.Root:
                return Mathf.Sqrt(objective.Weight * (float)bone.Level/(float)Bones[objective.Bone].Level);
                case ACTIVATION.Square:
                return Mathf.Pow(objective.Weight * (float)bone.Level/(float)Bones[objective.Bone].Level, 2f);
                default:
                return 1f;
            }
        }

    }

    [System.Serializable]
    public class Bone {
        public int Index = 0;
        public bool Active = true;
        public Transform Transform = null;
        public JOINT Joint = JOINT.Free;
        public float LowerLimit = 0f;
        public float UpperLimit = 0f;
        public Vector3 ZeroPosition;
        public Quaternion ZeroRotation;
        public int Level = 0;
        public int[] Childs = new int[0];
        public int[] Objectives = new int[0];

        public void SetJoint(JOINT joint) {
            if(Joint != joint) {
                Joint = joint;
                LowerLimit = 0f;
                UpperLimit = 0f;
            }
        }

        public void ClampLimits() {
            LowerLimit = Mathf.Clamp(LowerLimit, -180f, 0f);
            UpperLimit = Mathf.Clamp(UpperLimit, 0f, 180f);
        }

		public void ResolveLimits(bool avoidLocalOptima) {
            switch(Joint) {
                case JOINT.Free:
                break;

                case JOINT.HingeX:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetForward(), Vector3.ProjectOnPlane(Transform.localRotation.GetForward(), ZeroRotation.GetRight()), ZeroRotation.GetRight());
                    Transform.localRotation = ZeroRotation * (avoidLocalOptima ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.right):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.right));
                }
                break;

                case JOINT.HingeY:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetRight(), Vector3.ProjectOnPlane(Transform.localRotation.GetRight(), ZeroRotation.GetUp()), ZeroRotation.GetUp());
                    Transform.localRotation = ZeroRotation * (avoidLocalOptima ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.up):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.up));
                }
                break;

                case JOINT.HingeZ:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetUp(), Vector3.ProjectOnPlane(Transform.localRotation.GetUp(), ZeroRotation.GetForward()), ZeroRotation.GetForward());
                    Transform.localRotation = ZeroRotation * (avoidLocalOptima ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.forward):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.forward));
                }
                break;

                case JOINT.Ball:
                {
                    if(UpperLimit == 0f) {
                        Transform.localRotation = ZeroRotation;
                    } else {
                        Quaternion current = Transform.localRotation;
                        float angle = Quaternion.Angle(ZeroRotation, current);
                        if(angle > UpperLimit) {
                            Transform.localRotation = Quaternion.Slerp(ZeroRotation, current, UpperLimit / angle);
                        }
                    }
                }
                break;
            }
		}

    }

    [System.Serializable]
    public class Objective {
        public int Index = 0;
        public bool Active = true;
        public int Bone = 0;
        public Vector3 TargetPosition = Vector3.zero;
        public Quaternion TargetRotation = Quaternion.identity;
        public float Weight = 1f;
        public bool SolvePosition = true;
        public bool SolveRotation = true;

        public void SetTarget(Matrix4x4 matrix, float weight = 1f) {
            SetTarget(matrix.GetPosition(), weight);
            SetTarget(matrix.GetRotation(), weight);
        }

        public void SetTarget(Vector3 position, float weight = 1f) {
            if(weight == 1f) {
                TargetPosition = position;
            } else {
                TargetPosition = Vector3.Lerp(TargetPosition, position, weight);
            }
        }

        public void SetTarget(Quaternion rotation, float weight = 1f) {
            if(weight == 1f) {
                TargetRotation = rotation;
            } else {
                TargetRotation = Quaternion.Slerp(TargetRotation, rotation, weight);
            }
        }

        public float GetError(Model model) {
            if(!Active) {
                return 0f;
            }
            float error = 0f;
            if(SolvePosition) {
                error += Vector3.Distance(model.Bones[Bone].Transform.position, TargetPosition);
            }
            if(SolveRotation) {
                error += Mathf.Deg2Rad * Quaternion.Angle(model.Bones[Bone].Transform.rotation, TargetRotation);
            }
            return error;
        }
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(UltimateIK), true)]
	public class UltimateIK_Editor : Editor {

		public UltimateIK Target;

        private Color Background = new Color(0.25f, 0.25f, 0.25f, 1f); //Dark Grey
        private Color Header = new Color(212f/255f, 175f/255f, 55f/255f, 1f); //Gold
        private Color Content = Color.white;
        private Color Section = Color.white;
        private Color Panel = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
        private Color RegularField = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
        private Color ValidField = new Color(92/255f, 205/255f, 92/255f, 1f); //Light Green
        private Color InvalidField = new Color(205/255f, 92/255f, 92/255f, 1f); //Light Red
        private Color RegularButton = new Color(0.25f, 0.25f, 0.25f, 1f); //Dark Grey
        private Color PassiveButton = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
        private Color ActiveButton = new Color(0.4f, 0.5f, 0.6f, 1f); //Metal Blue
        private Color RegularFont = Color.white;
        private Color PassiveFont = new Color(1f/3f, 1f/3f, 1f/3f, 1f); //Grey
        private Color ActiveFont = Color.white;

		void Awake() {
			Target = (UltimateIK)target;
            ResetPreview();
		}

        void OnDestroy() {
            ResetPreview();
        }

        public void ApplyPreview() {
            if(Target != null && Target.SelectedBone != -1) {
                Target.Skeleton.Bones[Target.SelectedBone].Transform.localRotation = Target.Skeleton.Bones[Target.SelectedBone].ZeroRotation * Quaternion.Euler(Target.SelectedPreview);
            }
        }

        public void ResetPreview() {
            if(Target != null && Target.ShowPreview && Target.SelectedBone != -1) {
                Target.Skeleton.Bones[Target.SelectedBone].Transform.localRotation = Target.Skeleton.Bones[Target.SelectedBone].ZeroRotation;
                Target.SelectedPreview = Vector3.zero;
                Target.ShowPreview = false;
            }
        }

        void SetRoot(Transform t) {
            if(t == null || Target.Root == t || !IsInsideHierarchy(Target.transform, t)) {
                return;
            }
            Target.Root = t;
            Target.Rebuild();
        }

        void SetObjective(int index, Transform t) {
            if(Target.Objectives[index] == t) {
                return;
            }
            if(t != null) {
                if(ArrayExtensions.Contains(ref Target.Objectives, t) || !IsInsideHierarchy(Target.transform, t)) {
                    return;
                }
            }
            Target.Objectives[index] = t;
            Target.Rebuild();
        }

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

            Utility.SetGUIColor(Background);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();

                Utility.SetGUIColor(Header);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.LabelField("Setup");
                }
                Utility.SetGUIColor(Content);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.HelpBox("Solve Time: " + (1000f*Target.Skeleton.GetSolveTime()).ToString("F3") + "ms", MessageType.None);
                    Target.AutoUpdate = EditorGUILayout.Toggle("Auto Update", Target.AutoUpdate);
                    Target.Skeleton.SetIterations(EditorGUILayout.IntField("Iterations", Target.Skeleton.Iterations));
                    Target.Skeleton.SetThreshold(EditorGUILayout.FloatField("Threshold", Target.Skeleton.Threshold));
                    Target.Skeleton.Activation = (ACTIVATION)EditorGUILayout.EnumPopup("Activation", Target.Skeleton.Activation);
                    Target.Skeleton.AvoidLocalOptima = EditorGUILayout.Toggle("Avoid Local Optima", Target.Skeleton.AvoidLocalOptima);
                    EditorGUILayout.BeginHorizontal();
                    Target.Skeleton.SeedZeroPose = EditorGUILayout.Toggle("Seed Zero Pose", Target.Skeleton.SeedZeroPose);
                    if(Utility.GUIButton("Override", RegularButton, RegularFont, 100f, 18f)) {
                        Target.Skeleton.SaveAsZeroPose();
                    }
                    EditorGUILayout.EndHorizontal();
                    SetRoot((Transform)EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true));
                    EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Root Translation X", Target.Skeleton.RootTranslationX ? ActiveButton : PassiveButton, Target.Skeleton.RootTranslationX ? ActiveFont : PassiveFont)) {
                        Target.Skeleton.RootTranslationX = !Target.Skeleton.RootTranslationX;
                    }
                    if(Utility.GUIButton("Root Translation Y", Target.Skeleton.RootTranslationY ? ActiveButton : PassiveButton, Target.Skeleton.RootTranslationY ? ActiveFont : PassiveFont)) {
                        Target.Skeleton.RootTranslationY = !Target.Skeleton.RootTranslationY;
                    }
                    if(Utility.GUIButton("Root Translation Z", Target.Skeleton.RootTranslationZ ? ActiveButton : PassiveButton, Target.Skeleton.RootTranslationZ ? ActiveFont : PassiveFont)) {
                        Target.Skeleton.RootTranslationZ = !Target.Skeleton.RootTranslationZ;
                    }
                    EditorGUILayout.EndHorizontal();
                }
            }

            Utility.SetGUIColor(Background);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();

                Utility.SetGUIColor(Header);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.LabelField("Objectives");
                }
                Utility.SetGUIColor(Content);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    for(int i=0; i<Target.Objectives.Length; i++) {
                        InspectObjective(i);
                    }
                    EditorGUILayout.BeginHorizontal();
                    if(Utility.GUIButton("Add Objective", RegularButton, RegularFont)) {
                        ArrayExtensions.Expand(ref Target.Objectives);
                        ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                    }
                    EditorGUILayout.EndHorizontal();
                }
            }

            Utility.SetGUIColor(Background);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();

                Utility.SetGUIColor(Header);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.LabelField("Skeleton");
                }

                foreach(Bone bone in Target.Skeleton.Bones) {
                    InspectBone(bone);
                }

            }

            Utility.SetGUIColor(Background);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                if(Utility.GUIButton(Target.Draw ? "Drawing On" : "Drawing Off", Target.Draw ? ActiveButton : PassiveButton, Target.Draw ? ActiveFont : PassiveFont)) {
                    Target.Draw = !Target.Draw;
                }
            }

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

        void InspectBone(Bone bone) {
            Utility.SetGUIColor(Content);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                EditorGUILayout.BeginHorizontal();
                bone.Active = EditorGUILayout.Toggle(bone.Active, GUILayout.Width(20f));
                EditorGUILayout.BeginVertical();
                EditorGUI.BeginDisabledGroup(!bone.Active);
                
                if(bone.Index == Target.SelectedBone) {
                    if(!bone.Active || Utility.GUIButton(bone.Transform.name, ActiveButton, ActiveFont)) {
                        ResetPreview();
                        Target.SelectedBone = -1;
                    }

                    Utility.SetGUIColor(Panel);
                    using(new EditorGUILayout.VerticalScope ("Box")) {
                        Utility.ResetGUIColor();

                        bone.SetJoint((JOINT)EditorGUILayout.EnumPopup("Joint", bone.Joint));
                        switch(bone.Joint) {
                            case JOINT.Free:
                            break;

                            case JOINT.HingeX:
                            HingeInspector(bone);
                            break;

                            case JOINT.HingeY:
                            HingeInspector(bone);
                            break;

                            case JOINT.HingeZ:
                            HingeInspector(bone);
                            break;

                            case JOINT.Ball:
                            BallInspector(bone);
                            break;
                        }
                    }
                    bone.ClampLimits();

                    Utility.SetGUIColor(Panel);
                    using(new EditorGUILayout.VerticalScope ("Box")) {
                        Utility.ResetGUIColor();

                        if(Utility.GUIButton("Show Preview", Target.ShowPreview ? ActiveButton : PassiveButton, Target.ShowPreview ? ActiveFont : PassiveFont)) {
                            Target.ShowPreview = !Target.ShowPreview;
                            if(!Target.ShowPreview) {
                                ResetPreview();
                            }
                        }
                        EditorGUI.BeginDisabledGroup(!Target.ShowPreview);
                        EditorGUILayout.BeginHorizontal();
                        EditorGUILayout.LabelField("X", GUILayout.Width(50f));
                        Target.SelectedPreview.x = EditorGUILayout.Slider(Target.SelectedPreview.x, -180f, 180f);
                        EditorGUILayout.EndHorizontal();
                        EditorGUILayout.BeginHorizontal();
                        EditorGUILayout.LabelField("Y", GUILayout.Width(50f));
                        Target.SelectedPreview.y = EditorGUILayout.Slider(Target.SelectedPreview.y, -180f, 180f);
                        EditorGUILayout.EndHorizontal();
                        EditorGUILayout.BeginHorizontal();
                        EditorGUILayout.LabelField("Z", GUILayout.Width(50f));
                        Target.SelectedPreview.z = EditorGUILayout.Slider(Target.SelectedPreview.z, -180f, 180f);
                        EditorGUILayout.EndHorizontal();
                        EditorGUI.EndDisabledGroup();
                        if(Target.ShowPreview) {
                            ApplyPreview();
                        }
                    }

                    Utility.SetGUIColor(Panel);
                    using(new EditorGUILayout.VerticalScope ("Box")) {
                        Utility.ResetGUIColor();
                        EditorGUI.BeginDisabledGroup(true);
                        EditorGUILayout.Vector3Field("Zero Position", bone.ZeroPosition);
                        EditorGUILayout.Vector3Field("Zero Rotation", bone.ZeroRotation.eulerAngles);
                        EditorGUI.EndDisabledGroup();
                    }
                } else {
                    if(Utility.GUIButton(bone.Transform.name, RegularButton, RegularFont)) {
                        ResetPreview();
                        Target.SelectedBone = bone.Index;
                    }
                }
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndVertical();
                EditorGUILayout.EndHorizontal();
            }
        }

        void HingeInspector(Bone bone) {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Lower", GUILayout.Width(40f));
            bone.LowerLimit = EditorGUILayout.FloatField(bone.LowerLimit, GUILayout.Width(60f));
            EditorGUILayout.MinMaxSlider(ref bone.LowerLimit, ref bone.UpperLimit, -180f, 180f);
            bone.UpperLimit = EditorGUILayout.FloatField(bone.UpperLimit, GUILayout.Width(60f));
            EditorGUILayout.LabelField("Upper", GUILayout.Width(40f));
            EditorGUILayout.EndHorizontal();
        }

        void BallInspector(Bone bone) {
            float value = EditorGUILayout.Slider("Twist", bone.UpperLimit, 0f, 180f);
            bone.LowerLimit = -value;
            bone.UpperLimit = value;
        }

        void InspectObjective(int index) {
            if(Target.Objectives[index] != null && Target.Skeleton.FindObjective(Target.Objectives[index]) != null) {
                Objective o = Target.Skeleton.FindObjective(Target.Objectives[index]);
                Utility.SetGUIColor(Panel);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.BeginHorizontal();
                    o.Active = EditorGUILayout.Toggle(o.Active, GUILayout.Width(20f));
                    EditorGUI.BeginDisabledGroup(true);
                    Utility.SetGUIColor(ValidField);
                    EditorGUILayout.ObjectField(Target.Objectives[index], typeof(Transform), true);
                    Utility.ResetGUIColor();
                    EditorGUI.EndDisabledGroup();
                    if(Utility.GUIButton("X", InvalidField, RegularFont, 36f, 18f)) {
                        ArrayExtensions.RemoveAt(ref Target.Objectives, index);
                        ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                        Target.Rebuild();
                    } else {
                        EditorGUILayout.EndHorizontal();
                        Target.Targets[index] = (Transform)EditorGUILayout.ObjectField("Target", Target.Targets[index], typeof(Transform), true);
                        EditorGUILayout.BeginHorizontal();
                        o.Weight = EditorGUILayout.Slider("Weight", o.Weight, 0f, 1f);
                        if(Utility.GUIButton("Solve Position", o.SolvePosition ? ActiveButton : PassiveButton, o.SolvePosition ? ActiveFont : PassiveFont)) {
                            o.SolvePosition = !o.SolvePosition;
                        }
                        if(Utility.GUIButton("Solve Rotation", o.SolveRotation ? ActiveButton : PassiveButton, o.SolveRotation ? ActiveFont : PassiveFont)) {
                            o.SolveRotation = !o.SolveRotation;
                        }
                        EditorGUILayout.EndHorizontal();
                    }
                }
            } else {
                EditorGUILayout.BeginHorizontal();
                Utility.SetGUIColor(InvalidField);
                SetObjective(index, (Transform)EditorGUILayout.ObjectField("Bone", Target.Objectives[index], typeof(Transform), true));
                Utility.ResetGUIColor();
                if(Utility.GUIButton("X", InvalidField, RegularFont, 36f, 18f)) {
                    ArrayExtensions.RemoveAt(ref Target.Objectives, index);
                    ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                }
                EditorGUILayout.EndHorizontal();
            }
        }

        [DrawGizmo(GizmoType.Active | GizmoType.NotInSelectionHierarchy)]
        static void OnScene(Transform t, GizmoType gizmoType) {
            UltimateIK target = t.GetComponent<UltimateIK>();
            if(target != null) {
                if(target.Draw && target.Skeleton.IsSetup()) {
                    if(!Application.isPlaying) {
                        List<Matrix4x4> targets = new List<Matrix4x4>();
                        for(int i=0; i<target.Objectives.Length; i++) {
                            if(target.Objectives[i] != null) {
                                targets.Add(target.Targets[i] != null ? target.Targets[i].GetWorldMatrix(true) : target.Objectives[i].GetWorldMatrix(true));
                            }
                        }
                        for(int i=0; i<target.Skeleton.Objectives.Length; i++) {
                            target.Skeleton.Objectives[i].TargetPosition = targets[i].GetPosition();
                            target.Skeleton.Objectives[i].TargetRotation = targets[i].GetRotation();
                        }
                    }

                    DrawSkeleton(target, null, target.Skeleton.Bones.First());
                    foreach(Objective objective in target.Skeleton.Objectives) {
                        DrawObjective(target, objective);
                    }
                    foreach(Bone bone in target.Skeleton.Bones) {
                        DrawBone(target, bone);
                    }
                    DrawSelection(target);
                }
            }
        }

        static void DrawCoordinateSystem(Vector3 position, Quaternion rotation, float size) {
            Handles.color = Color.red;
            Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.right), size, EventType.Repaint);
            Handles.color = Color.green;
            Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.up), size, EventType.Repaint);
            Handles.color = Color.blue;
            Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.forward), size, EventType.Repaint);
        }

        static void DrawSelection(UltimateIK ik) {
            if(ik.SelectedBone != -1) {
                Bone bone = ik.Skeleton.Bones[ik.SelectedBone];
                Handles.color = Color.black.Transparent(0.5f);
                Handles.SphereHandleCap(0, bone.Transform.position, Quaternion.identity, 0.3f, EventType.Repaint);

                Handles.color = Color.magenta.Transparent(0.75f);
                Handles.CubeHandleCap(0, bone.Transform.position, bone.Transform.rotation, 0.05f, EventType.Repaint);

                Quaternion seed = bone.Transform.parent != null ? bone.Transform.parent.rotation : Quaternion.identity;
                switch(bone.Joint) {
                    case JOINT.Free:
                    DrawLimit(bone, Axis.XPositive, false);
                    DrawLimit(bone, Axis.YPositive, false);
                    DrawLimit(bone, Axis.ZPositive, false);
                    break;
                    
                    case JOINT.HingeX:
                    DrawLimit(bone, Axis.XPositive, true);
                    DrawLimit(bone, Axis.YPositive, false);
                    DrawLimit(bone, Axis.ZPositive, false);
                    break;

                    case JOINT.HingeY:
                    DrawLimit(bone, Axis.XPositive, false);
                    DrawLimit(bone, Axis.YPositive, true);
                    DrawLimit(bone, Axis.ZPositive, false);
                    break;

                    case JOINT.HingeZ:
                    DrawLimit(bone, Axis.XPositive, false);
                    DrawLimit(bone, Axis.YPositive, false);
                    DrawLimit(bone, Axis.ZPositive, true);
                    break;

                    case JOINT.Ball:
                    DrawLimit(bone, Axis.XPositive, true);
                    DrawLimit(bone, Axis.YPositive, true);
                    DrawLimit(bone, Axis.ZPositive, true);
                    break;
                }

            }
        }

        static void DrawLimit(Bone bone, Axis axis, bool active) {
            if(active) {
                Quaternion seed = bone.Transform.parent != null ? bone.Transform.parent.rotation : Quaternion.identity;
                switch(axis) {
                    case Axis.XPositive:
                    Handles.color = Color.red.Transparent(0.25f);
                    Handles.DrawSolidArc(bone.Transform.position, seed * bone.ZeroRotation.GetRight(), Quaternion.AngleAxis(bone.LowerLimit, seed * bone.ZeroRotation.GetRight()) * seed * bone.ZeroRotation.GetUp(), bone.UpperLimit - bone.LowerLimit, 0.15f);
                    Handles.color = Color.red;
                    Handles.ArrowHandleCap(0, bone.Transform.position, bone.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.right), 0.15f, EventType.Repaint);
                    break;

                    case Axis.YPositive:
                    Handles.color = Color.green.Transparent(0.25f);
                    Handles.DrawSolidArc(bone.Transform.position, seed * bone.ZeroRotation.GetUp(), Quaternion.AngleAxis(bone.LowerLimit, seed * bone.ZeroRotation.GetUp()) * seed * bone.ZeroRotation.GetForward(), bone.UpperLimit - bone.LowerLimit, 0.15f);
                    Handles.color = Color.green;
                    Handles.ArrowHandleCap(0, bone.Transform.position, bone.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.up), 0.15f, EventType.Repaint);
                    break;

                    case Axis.ZPositive:
                    Handles.color = Color.blue.Transparent(0.25f);
                    Handles.DrawSolidArc(bone.Transform.position, seed * bone.ZeroRotation.GetForward(), Quaternion.AngleAxis(bone.LowerLimit, seed * bone.ZeroRotation.GetForward()) * seed * bone.ZeroRotation.GetRight(), bone.UpperLimit - bone.LowerLimit, 0.15f);
                    Handles.color = Color.blue;
                    Handles.ArrowHandleCap(0, bone.Transform.position, bone.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.forward), 0.15f, EventType.Repaint);
                    break;
                }
            } else {
                Handles.color = Color.grey;
                Handles.ArrowHandleCap(0, bone.Transform.position, bone.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, axis.GetAxis()), 0.15f, EventType.Repaint);
            }
        }

        static void DrawBone(UltimateIK ik, Bone bone) {
            if(bone.Active && ik.SelectedBone != bone.Index) {
                Handles.color = Color.magenta.Transparent(0.75f);
                Handles.SphereHandleCap(0, bone.Transform.position, Quaternion.identity, 0.025f, EventType.Repaint);
            }
        }

        static void DrawObjective(UltimateIK ik, Objective o) {
            if(o.Active) {
                Handles.color = Color.green.Transparent(0.5f);
                Handles.SphereHandleCap(0, ik.Skeleton.Bones[o.Bone].Transform.position, Quaternion.identity, 0.1f, EventType.Repaint);
                DrawCoordinateSystem(ik.Skeleton.Bones[o.Bone].Transform.position, ik.Skeleton.Bones[o.Bone].Transform.rotation, 0.05f);
                Handles.color = Color.red.Transparent(0.75f);
                Handles.DrawDottedLine(ik.Skeleton.Bones[o.Bone].Transform.position, o.TargetPosition, 10f);
                Handles.SphereHandleCap(0, o.TargetPosition, Quaternion.identity, 0.025f, EventType.Repaint);
                DrawCoordinateSystem(o.TargetPosition, o.TargetRotation, 0.025f);
            }
        }

        static void DrawSkeleton(UltimateIK ik, Bone parent, Bone bone) {
            if(bone.Active) {
                parent = bone;
            }
            foreach(int child in bone.Childs) {
                if(parent != null && ik.Skeleton.Bones[child].Active) {
                    Handles.color = Color.cyan.Transparent(0.75f);
                    Handles.DrawAAPolyLine(7.5f, new Vector3[2]{parent.Transform.position, ik.Skeleton.Bones[child].Transform.position});
                    
                }
            }
            foreach(int child in bone.Childs) {
                DrawSkeleton(ik, parent, ik.Skeleton.Bones[child]);
            }
        }

	}
	#endif
    
}