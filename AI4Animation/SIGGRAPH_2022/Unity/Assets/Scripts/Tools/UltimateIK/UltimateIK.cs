using System;
using System.Collections.Generic;
using UnityEngine;

namespace UltimateIK {

    public enum ACTIVATION {Constant, Linear, Root, Square}
    public enum TYPE {Free, HingeX, HingeY, HingeZ, Ball}

    [System.Serializable]
    public class IK {
        public int Iterations = 25;
        public float Threshold = 0.001f;
        public ACTIVATION Activation = ACTIVATION.Linear;
        public bool SeedZeroPose = false;
        public bool AvoidJointLimits = false;
        public bool AllowRootUpdateX = false;
        public bool AllowRootUpdateY = false;
        public bool AllowRootUpdateZ = false;
        public float RootWeight = 1f;

        public Joint[] Joints = new Joint[0];
        public Objective[] Objectives = new Objective[0];

        private float SolveTime = 0f;

        public static IK Create(Transform root, params Transform[] objectives) {
            return Create(null, root, objectives);
        }

        public static IK Create(IK reference, Transform root, params Transform[] objectives) {
            if(reference == null) {
                reference = new IK();
            }
            if(reference.GetRoot() == root && reference.Objectives.Length == objectives.Length) {
                for(int i=0; i<objectives.Length; i++) {
                    if(reference.Joints[reference.Objectives[i].Joint].Transform != objectives[i]) {
                        break;
                    }
                    if(i==objectives.Length-1) {
                        return reference;
                    }
                }
            }
            objectives = Verify(root, objectives);
            IK instance = new IK();
            for(int i=0; i<objectives.Length; i++) {
                Transform[] chain = GetChain(root, objectives[i]);
                Objective objective = reference.FindObjective(objectives[i]);
                if(objective == null) {
                    objective = new Objective();
                }
                for(int j=0; j<chain.Length; j++) {
                    Joint joint = instance.FindJoint(chain[j]);
                    if(joint == null) {
                        joint = reference.FindJoint(chain[j]);
                        if(joint != null) {
                            joint.Childs = new int[0];
                            joint.Objectives = new int[0];
                        }
                        if(joint == null) {
                            joint = new Joint();
                            joint.Transform = chain[j];
                            joint.ZeroPosition = chain[j].localPosition;
                            joint.ZeroRotation = chain[j].localRotation;
                        }
                        Joint parent = instance.FindJoint(chain[j].parent);
                        if(parent != null) {
                            ArrayExtensions.Append(ref parent.Childs, instance.Joints.Length);
                        }
                        joint.Index = instance.Joints.Length;
                        ArrayExtensions.Append(ref instance.Joints, joint);
                    }
                    ArrayExtensions.Append(ref joint.Objectives, i);
                }
                objective.Joint = instance.FindJoint(chain.Last()).Index;
                objective.TargetPosition = objectives[i].transform.position;
                objective.TargetRotation = objectives[i].transform.rotation;
                objective.Index = instance.Objectives.Length;
                ArrayExtensions.Append(ref instance.Objectives, objective);
            }
            instance.Iterations = reference.Iterations;
            instance.Threshold = reference.Threshold;
            instance.Activation = reference.Activation;
            instance.AvoidJointLimits = reference.AvoidJointLimits;
            instance.AllowRootUpdateX = reference.AllowRootUpdateX;
            instance.AllowRootUpdateY = reference.AllowRootUpdateY;
            instance.AllowRootUpdateZ = reference.AllowRootUpdateZ;
            instance.SeedZeroPose = reference.SeedZeroPose;
            return instance;
        }

        private static Transform[] Verify(Transform root, Transform[] objectives) {
            List<Transform> verified = new List<Transform>();
            if(root == null) {
                Debug.Log("Given root was null. Extracting skeleton failed.");
                return verified.ToArray();
            }
            if(objectives.Length == 0) {
                Debug.Log("No objectives given. Extracting skeleton failed.");
                return verified.ToArray();
            }
            for(int i=0; i<objectives.Length; i++) {
                if(objectives[i] == null) {
                    Debug.Log("A given objective was null and will be ignored.");
                } else if(verified.Contains(objectives[i])) {
                    Debug.Log("Given objective " + objectives[i].name + " is already contained and will be ignored.");
                } else if(!IsInsideHierarchy(root, objectives[i])) {
                    Debug.Log("Chain for " + objectives[i].name + " is not connected to " + root.name + " and will be ignored.");
                } else {
                    verified.Add(objectives[i]);
                }
            }
            return verified.ToArray();
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

        public static bool IsInsideHierarchy(Transform root, Transform t) {
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

        public bool IsSetup() {
            if(Joints.Length == 0 && Objectives.Length > 0) {
                Debug.Log("Joint count is zero but objective count is not. This should not have happened!");
                return false;
            }
            return true;
        }

        public Transform GetRoot() {
            return Joints.Length == 0 ? null : Joints.First().Transform;
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

        public Joint FindJoint(string name) {
            return Array.Find(Joints, x => x.Transform.name == name);
        }

        public Joint FindJoint(Transform t) {
            return Array.Find(Joints, x => x.Transform == t);
        }

        public Objective FindObjective(Transform t) {
            return Array.Find(Objectives, x => x.Joint < Joints.Length && Joints[x.Joint].Transform == t);
        }

        public void SaveZeroPose() {
            foreach(Joint joint in Joints) {
                joint.ZeroPosition = joint.Transform.localPosition;
                joint.ZeroRotation = joint.Transform.localRotation;
            }
        }

        // public void ApplyZeroPose(Matrix4x4 root) {
        //     foreach(Joint joint in Joints) {
        //         if(joint.Active) {
        //             if(joint.Transform == GetRoot()) {
        //                 joint.Transform.position = root.GetPosition();
        //                 joint.Transform.rotation = root.GetRotation();
        //             } else {
        //                 joint.Transform.localPosition = joint.ZeroPosition;
        //                 joint.Transform.localRotation = joint.ZeroRotation;
        //             }
        //         }
        //     }
        // }

        public void SetWeights(params float[] weights) {
            if(Objectives.Length != weights.Length) {
                Debug.Log("Number of given weights <" + weights.Length + "> does not match number of objectives <" + Objectives.Length + ">.");
                return;
            }
            for(int i=0; i<Objectives.Length; i++) {
                Objectives[i].Weight = weights[i];
            }
        }

        public void SetTargets(params Matrix4x4[] targets) {
            if(Objectives.Length != targets.Length) {
                Debug.Log("Number of given targets <" + targets.Length + "> does not match number of objectives <" + Objectives.Length + ">.");
                return;
            }
            for(int i=0; i<Objectives.Length; i++) {
                Objectives[i].TargetPosition = targets[i].GetPosition();
                Objectives[i].TargetRotation = targets[i].GetRotation();
            }
        }

        public void Solve() {
            DateTime timestamp = Utility.GetTimestamp();
            if(IsSetup()) {
                //Compute Levels
                Action<Joint, Joint> recursion = null;
                recursion = new Action<Joint, Joint>((parent, joint) => {
                    joint.Length = parent == null ? 1 : joint.Active && parent.Active ? parent.Length + 1f : parent.Length;
                    // joint.Length = parent == null ? 0f : parent.Length + joint.Transform.localPosition.magnitude;
                    foreach(int index in joint.Childs) {
                        recursion(joint, Joints[index]);
                    }
                });
                recursion(null, Joints.First());

                //Apply Zero Pose
                if(SeedZeroPose) {
                    foreach(Joint joint in Joints) {
                        if(joint.Active) {
                            joint.Transform.localPosition = joint.ZeroPosition;
                            joint.Transform.localRotation = joint.ZeroRotation;
                        }
                    }
                }

                //Solve IK
                for(int i=0; i<Iterations; i++) {
                    if(!IsConverged()) {
                        if(AllowRootUpdateX || AllowRootUpdateY || AllowRootUpdateZ) {
                            Vector3 delta = Vector3.zero;
                            int count = 0;
                            foreach(Objective o in Objectives) {
                                if(o.Active) {
                                    delta += GetWeight(Joints[o.Joint], o) * (o.TargetPosition - Joints[o.Joint].Transform.position);
                                    count += 1;
                                }
                            }
                            delta.x *= AllowRootUpdateX ? RootWeight : 0f;
                            delta.y *= AllowRootUpdateY ? RootWeight : 0f;
                            delta.z *= AllowRootUpdateZ ? RootWeight : 0f;
                            if(count > 0) {
                                GetRoot().position += delta / count;
                            }
                        }
                        Optimise(Joints.First());
                    }
                }
            }
            SolveTime = (float)Utility.GetElapsedTime(timestamp);

            bool IsConverged() {
                foreach(Objective o in Objectives) {
                    if(o.GetError(this) > Threshold) {
                        return false;
                    }
                }
                return true;
            }

            float GetWeight(Joint joint, Objective objective) {
                // float weightMax = 0f;
                // float weightSum = 0f;
                // for(int i=0; i<Objectives.Length; i++) {
                //     weightSum += Objectives[i].Weight;
                //     weightMax = Mathf.Max(weightMax, Objectives[i].Weight);
                // }
                // float weight = objective.Weight / weightMax;
                // // float weight = (weightSum > 0f && weightSum < Objectives.Length) ? (objective.Weight * Objectives.Length / weightSum) : 1f;
                // switch(Activation) {
                //     case ACTIVATION.Constant:
                //     return weight;
                //     case ACTIVATION.Linear:
                //     return weight * (float)joint.Level/(float)Joints[objective.Joint].Level;
                //     case ACTIVATION.Root:
                //     return Mathf.Sqrt(weight * (float)joint.Level/(float)Joints[objective.Joint].Level);
                //     case ACTIVATION.Square:
                //     return Mathf.Pow(weight * (float)joint.Level/(float)Joints[objective.Joint].Level, 2f);
                //     default:
                //     return 1f;
                // }
                // float weight = (weightSum > 0f && weightSum < Objectives.Length) ? (objective.Weight * Objectives.Length / weightSum) : 1f;

                switch(Activation) {
                    case ACTIVATION.Constant:
                    return 1f/Objectives.Length;
                    case ACTIVATION.Linear:
                    return joint.Length/Joints[objective.Joint].Length;
                    case ACTIVATION.Root:
                    return Mathf.Sqrt(joint.Length/Joints[objective.Joint].Length);
                    case ACTIVATION.Square:
                    return Mathf.Pow(joint.Length/Joints[objective.Joint].Length, 2f);
                    default:
                    return 1f;
                }
            }

            void Optimise(Joint joint) {
                if(joint.Active) {
                    Vector3 pos = joint.Transform.position;
                    Quaternion rot = joint.Transform.rotation;
                    Vector3 forward = Vector3.zero;
                    Vector3 up = Vector3.zero;
                    int count = 0;

                    //Solve Objective Rotations
                    foreach(int index in joint.Objectives) {
                        Objective o = Objectives[index];
                        if(o.Active && o.SolveRotation) {
                            Quaternion q = Quaternion.Slerp(
                                rot,
                                o.TargetRotation * Quaternion.Inverse(Joints[o.Joint].Transform.rotation) * rot,
                                GetWeight(joint, o)
                            );
                            forward += q*Vector3.forward;
                            up += q*Vector3.up;
                            count += 1;
                        }
                    }

                    //Solve Objective Positions
                    foreach(int index in joint.Objectives) {
                        Objective o = Objectives[index];
                        if(o.Active && o.SolvePosition) {
                            Quaternion q = Quaternion.Slerp(
                                rot,
                                Quaternion.FromToRotation(Joints[o.Joint].Transform.position - pos, o.TargetPosition - pos) * rot,
                                GetWeight(joint, o)
                            );
                            forward += q*Vector3.forward;
                            up += q*Vector3.up;
                            count += 1;
                        }
                    }

                    if(count > 0) {
                        joint.Transform.rotation = Quaternion.LookRotation((forward/count).normalized, (up/count).normalized);
                        joint.ResolveLimits(AvoidJointLimits);
                    }
                }

                foreach(int index in joint.Childs) {
                    Optimise(Joints[index]);
                }
            }
        }

        public void PrintHierarchy() {
            string output = "Hierarchy" + "\n";
            void Traverse(Joint joint, int level) {
                if(joint.Active) {
                    for(int i=0; i<level; i++) {
                        output += "-> ";
                    }
                    output += joint.Transform.name + "\n";
                }
                for(int i=0; i<joint.Childs.Length; i++) {
                    Traverse(Joints[joint.Childs[i]], level+1);
                }
            }
            Traverse(Joints[0], 0);
            Debug.Log(output);
        }
    }

    [System.Serializable]
    public class Joint {
        public int Index = 0;
        public bool Active = true;

        public Transform Transform = null;
        
        [SerializeField] private TYPE Type = TYPE.Free;
        [SerializeField] private float LowerLimit = 0f;
        [SerializeField] private float UpperLimit = 0f;

        // public int Level = 0;
        public float Length = 0f;
        public int[] Childs = new int[0];
        public int[] Objectives = new int[0];

        public Vector3 ZeroPosition;
        public Quaternion ZeroRotation;

        public void SetJointType(TYPE type) {
            if(Type != type) {
                Type = type;
                LowerLimit = 0f;
                UpperLimit = 0f;
            }
        }

        public TYPE GetJointType() {
            return Type;
        }

        public void SaveZeroPose() {
            ZeroPosition = Transform.localPosition;
            ZeroRotation = Transform.localRotation;
        }

        public void SetLowerLimit(float value) {
            LowerLimit = Mathf.Clamp(value, -180f, 0f);
        }

        public float GetLowerLimit() {
            return LowerLimit;
        }

        public void SetUpperLimit(float value) {
            UpperLimit = Mathf.Clamp(value, 0f, 180f);
        }

        public float GetUpperLimit() {
            return UpperLimit;
        }

        public float GetJointAngle() {
            if(Type == TYPE.HingeX) {
                return Vector3.SignedAngle(ZeroRotation.GetForward(), Vector3.ProjectOnPlane(Transform.localRotation.GetForward(), ZeroRotation.GetRight()), ZeroRotation.GetRight());
            }
            if(Type == TYPE.HingeY) {
                return Vector3.SignedAngle(ZeroRotation.GetRight(), Vector3.ProjectOnPlane(Transform.localRotation.GetRight(), ZeroRotation.GetUp()), ZeroRotation.GetUp());
            }
            if(Type == TYPE.HingeZ) {
                return Vector3.SignedAngle(ZeroRotation.GetUp(), Vector3.ProjectOnPlane(Transform.localRotation.GetUp(), ZeroRotation.GetForward()), ZeroRotation.GetForward());
            }
            if(Type == TYPE.Ball) {
                return Quaternion.Angle(ZeroRotation, Transform.localRotation);
            }
            Debug.Log("Unsupported joint type for computing angle.");
            return 0f;
        }

        public void ResolveLimits(bool avoidBoneLimits) {
            switch(Type) {
                case TYPE.Free:
                break;

                case TYPE.HingeX:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetForward(), Vector3.ProjectOnPlane(Transform.localRotation.GetForward(), ZeroRotation.GetRight()), ZeroRotation.GetRight());
                    Transform.localRotation = ZeroRotation * (avoidBoneLimits ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.right):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.right));
                }
                break;

                case TYPE.HingeY:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetRight(), Vector3.ProjectOnPlane(Transform.localRotation.GetRight(), ZeroRotation.GetUp()), ZeroRotation.GetUp());
                    Transform.localRotation = ZeroRotation * (avoidBoneLimits ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.up):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.up));
                }
                break;

                case TYPE.HingeZ:
                {
                    float angle = Vector3.SignedAngle(ZeroRotation.GetUp(), Vector3.ProjectOnPlane(Transform.localRotation.GetUp(), ZeroRotation.GetForward()), ZeroRotation.GetForward());
                    Transform.localRotation = ZeroRotation * (avoidBoneLimits ?
                    Quaternion.AngleAxis(Mathf.Repeat(angle-LowerLimit, UpperLimit-LowerLimit) + LowerLimit, Vector3.forward):
                    Quaternion.AngleAxis(Mathf.Clamp(angle, LowerLimit, UpperLimit), Vector3.forward));
                }
                break;

                case TYPE.Ball:
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
        public int Joint = 0;
        public Vector3 TargetPosition = Vector3.zero;
        public Quaternion TargetRotation = Quaternion.identity;
        public float Weight = 1f;
        public bool SolvePosition = true;
        public bool SolveRotation = true;

        public void InterpolateTarget(Transform to, float amount) {
            SetTarget(Vector3.Lerp(TargetPosition, to.position, amount));
            SetTarget(Quaternion.Slerp(TargetRotation, to.rotation, amount));
        }

        public void SetTarget(Transform transform) {
            SetTarget(transform.position);
            SetTarget(transform.rotation);
        }
        
        public void SetTarget(Matrix4x4 matrix) {
            SetTarget(matrix.GetPosition());
            SetTarget(matrix.GetRotation());
        }

        public void SetTarget(Vector3 position, Quaternion rotation) {
            SetTarget(position);
            SetTarget(rotation);
        }

        public void SetTarget(Vector3 position) {
            TargetPosition = position;
        }

        public void SetTarget(Quaternion rotation) {
            TargetRotation = rotation;
        }

        public Matrix4x4 GetTarget() {
            return Matrix4x4.TRS(TargetPosition, TargetRotation, Vector3.one);
        }

        public float GetError(IK ik) {
            if(!Active) {
                return 0f;
            }
            float error = 0f;
            if(SolvePosition) {
                error += Vector3.Distance(ik.Joints[Joint].Transform.position, TargetPosition);
            }
            if(SolveRotation) {
                error += Mathf.Deg2Rad * Quaternion.Angle(ik.Joints[Joint].Transform.rotation, TargetRotation);
            }
            return error;
        }
    }
    
}