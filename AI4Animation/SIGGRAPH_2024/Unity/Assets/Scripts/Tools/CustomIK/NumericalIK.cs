using System;
using System.Collections.Generic;
using UnityEngine;

public class NumericalIK : MonoBehaviour {
    //You can remove this//
    public Transform Root = null;
    public Transform[] Objectives = new Transform[0];
    public Transform[] Targets = new Transform[0];
    private Quaternion[] SeedPose = new Quaternion[0];
    ///////////////////////
    public bool ResetSeedPose = true;
    public double FiniteStep = 0.1;

    public Vector3 Offset;

    private Model Skeleton;

    void Start() {
        Skeleton = BuildModel(Root, Objectives);
        SeedPose = new Quaternion[Skeleton.Bones.Length];
        for(int i=0; i<SeedPose.Length; i++) {
            SeedPose[i] = Skeleton.Bones[i].rotation;
        }
    }

    void LateUpdate() {
        Skeleton.Solve(ResetSeedPose ? SeedPose : null, Targets, FiniteStep);
    }

    public static Model BuildModel(Transform root, params Transform[] objectives) {
        Model model = new Model();
        objectives = Verify(root, objectives);
        for(int i=0; i<objectives.Length; i++) {
            Transform[] chain = GetChain(root, objectives[i]);
            for(int j=0; j<chain.Length; j++) {
                if(!model.FindBone(chain[j])) {
                    ArrayExtensions.Append(ref model.Bones, chain[j]);
                }
            }
            ArrayExtensions.Append(ref model.Objectives, objectives[i]);
        }
        return model;
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

    public class Model {
        public Transform[] Bones = new Transform[0];
        public Transform[] Objectives = new Transform[0];
        public float SolveTime = 0f;

        public bool IsSetup() {
            if(Bones.Length == 0 && Objectives.Length > 0) {
                Debug.Log("Bone count is zero but objective count is not. This should not have happened!");
                return false;
            }
            return true;
        }

        public Transform GetRoot() {
            return Bones.Length == 0 ? null : Bones.First();
        }

        public Transform FindBone(Transform t) {
            return Array.Find(Bones, x => x == t);
        }

        public Transform FindObjective(Transform t) {
            return Array.Find(Objectives, x => x == t);
        }

        //Replace this input with your signed distance field to create the cost function over the generated Objectives[i].position
        //Seed is just for this demo to start from the non-animated T-pose
        //Targets are just for this demo to make it interactive
        //FiniteStep is to compute the gradient of the cost function
        public void Solve(Quaternion[] seed, Transform[] targets, double finiteStep) {
            DateTime timestamp = Utility.GetTimestamp();
            if(finiteStep == 0.0) {
                return;
            }
            if(IsSetup()) {
                if(seed == null) {
                    //This is the current pose to start from which is given by the neural network
                    seed = new Quaternion[Bones.Length];
                    for(int i=0; i<Bones.Length; i++) {
                        seed[i] = Bones[i].rotation;
                    }
                }
                int numberOfVariables = 3*Bones.Length;
                Accord.Math.Optimization.Cobyla optimizer = new Accord.Math.Optimization.Cobyla(numberOfVariables, x => Cost(x)); //You can replace this with any other optimizer
                optimizer.Minimize(new double[numberOfVariables]);
                FK(optimizer.Solution);
                                
                double Cost(double[] angles) {
                    if(Objectives.Length == 0) {
                        return 0.0;
                    }
                    //Compute FK
                    FK(angles);
                    //Calculate Cost
                    double cost = 0.0;
                    for(int i=0; i<Objectives.Length; i++) {
                        double error = Vector3.Distance(Objectives[i].position, targets[i].position);
                        error *= error;
                        cost += error;
                    }
                    for(int i=0; i<Objectives.Length; i++) {
                        double error = Quaternion.Angle(Objectives[i].rotation, targets[i].rotation);
                        error *= error;
                        cost += error;
                    }
                    // cost /= Objectives.Length;
                    return cost;
                }

                // double[] Gradient(double[] angles) {
                //     double[] gradient = new double[angles.Length];
                //     double zeroCost = Cost(angles);
                //     for(int i=0; i<angles.Length; i++) {
                //         angles[i] += finiteStep;
                //         double cost = Cost(angles);
                //         angles[i] -= finiteStep;
                //         gradient[i] = (cost - zeroCost) / finiteStep;
                //     }
                //     return gradient;
                // }

                void FK(double[] angles) {
                    for(int i=0; i<Bones.Length; i++) {
                        float x = (float)angles[3*i + 0];
                        float y = (float)angles[3*i + 1];
                        float z = (float)angles[3*i + 2];
                        Bones[i].rotation = seed[i] * Quaternion.Euler(x, y, z);
                    }
                }
            }
            SolveTime = (float)Utility.GetElapsedTime(timestamp);
        }
    }
}