using System;
using System.Collections.Generic;
using UnityEngine;

namespace MagicIK {

    [Serializable]
    public class Solver {
        [SerializeReference] public Node[] Nodes = new Node[0];
        [SerializeReference] public Objective[] Objectives = new Objective[0];

        public int MaxIterations = 25;
        public float MaxError = 0.001f;
        public float RootPull = 0f;
        public bool SeedZeros = false;

        public List<string> Info {get; private set;}

        [NonSerialized] private static Color ActiveNodeColor = Color.green;
        [NonSerialized] private static Color InactiveNodeColor = Color.red;
        [NonSerialized] private static Color ObjectiveColor = Color.magenta;
        [NonSerialized] private static Color TargetColor = Color.red;
        [NonSerialized] private static Color LineColor = Color.black;
        [NonSerialized] private static float LineWidth = 0.005f;
        [NonSerialized] private static float NodeSize = 0.025f;
        [NonSerialized] private static float ObjectiveSize = 0.05f;

        public Solver(Solver reference, Transform root, params Transform[] objectives) {
            Info = new List<string>();
            if(root == null) {
                Info.Add("Given root was null."); return;
            }
            if(objectives.Length == 0) {
                Info.Add("No objectives given."); return;
            }

            MaxIterations = reference == null ? MaxIterations : reference.MaxIterations;
            MaxError = reference == null ? MaxError : reference.MaxError;
            RootPull = reference == null ? RootPull : reference.RootPull;
            SeedZeros = reference == null ? SeedZeros : reference.SeedZeros;

            List<Transform> verified = new List<Transform>();
            foreach(Transform objective in objectives) {
                //Validity Handling
                if(objective == null) {
                    Info.Add("A given objective was null and will be skipped"); continue;
                }
                if(verified.Contains(objective)) {
                    Info.Add("Duplicate objective found for " + objective.name + " and will be skipped."); continue;
                }
                if(!IsInsideHierarchy(root, objective)) {
                    Info.Add("Given objective " + objective.name + " is not connected to " + root.name + " and will be skipped.'"); continue;
                }
                verified.Add(objective);

                //Generate Hierarchy
                foreach(Transform pivot in GetChain(root, objective)) {
                    if(FindNode(pivot) == null) {
                        new Node(this, pivot, reference);
                    }
                }
                new Objective(this, objective, reference);

                //Compute Activations
                foreach(Node node in Nodes) {
                    node.ComputeActivations();
                }
            }
        }

        private Transform[] GetChain(Transform root, Transform end) {
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

        private bool IsInsideHierarchy(Transform root, Transform t) {
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

        public Node FindNode(Transform t) {
            return Array.Find(Nodes, x => x.Transform == t);
        }

        public Objective FindObjective(Transform t) {
            return Array.Find(Objectives, x => x.Node.Transform == t);
        }

        public float Solve() {
            DateTime timestamp = Utility.GetTimestamp();

            foreach(Node node in Nodes) {
                node.ComputeActivations();
            }

            if(Nodes.Length > 0) {
                //Apply Zero Pose
                foreach(Node node in Nodes) {
                    if(SeedZeros || node.SeedZero) {
                        node.Transform.localPosition = node.ZeroPosition;
                        node.Transform.localRotation = node.ZeroRotation;
                    }
                }

                //Solve IK
                for(int i=0; i<MaxIterations; i++) {
                    if(!IsConverged()) {
                        Optimise(Nodes[0]);
                    }
                }

                //Root Pulling
                {
                    Vector3 delta = Vector3.zero;
                    float sum = 0f;
                    foreach(Objective o in Objectives) {
                        delta += o.Weight * (o.Position - o.Node.Transform.position);
                        sum += o.Weight;
                    }
                    if(sum > 0) {
                        Nodes[0].Transform.position += RootPull * delta / sum;
                    }
                }
            }

            return (float)Utility.GetElapsedTime(timestamp);
        }

        private void Optimise(Node node) {
            if(node.Active) {
                Vector3 pos = node.Transform.position;
                Quaternion rot = node.Transform.rotation;
                Vector3 forward = Vector3.zero;
                Vector3 up = Vector3.zero;
                float sum = 0f;

                for(int i=0; i<node.Objectives.Length; i++) {
                    Objective o = node.Objectives[i];
                    if(node != o.Node) {
                        float weight = o.Weight * node.Activations[i];
                        Quaternion q = Quaternion.Slerp(
                            rot,
                            Quaternion.FromToRotation(o.Node.Transform.position - pos, o.Position - pos) * rot,
                            weight
                        );
                        forward += q*Vector3.forward;
                        up += q*Vector3.up;
                        sum += weight;
                    } else if(o.ApplyRotation) {
                        forward += o.Rotation*Vector3.forward;
                        up += o.Rotation*Vector3.up;
                        sum += 1f;
                    }
                }

                if(sum > 0f) {
                    node.Transform.rotation = Quaternion.LookRotation((forward/sum).normalized, (up/sum).normalized);
                }

                if(node.Limit != null) {
                    node.Limit.Solve();
                }
            }

            foreach(Node child in node.Childs) {
                Optimise(child);
            }
        }

        private bool IsConverged() {
            foreach(Objective o in Objectives) {
                if(o.GetError() > MaxError) {
                    return false;
                }
            }
            return true;
        }

        public void Print() {
            if(Nodes.Length == 0) {return;}

            string output = "Hierarchy" + "\n";
            void Traverse(Node node, int level) {
                for(int i=0; i<level; i++) {
                    output += "-> ";
                }
                output += "Node: " + node.Transform.name + "\n";
                output += "Childs:";
                foreach(Node child in node.Childs) {
                    output += " " + child.Transform.name;
                }
                output += "\n";
                output += "Objectives: " + node.Objectives.Length + "\n";
                output += "Depth: " + node.Depth + "\n";
                for(int i=0; i<node.Childs.Length; i++) {
                    Traverse(node.Childs[i], level+1);
                }
            }
            Traverse(Nodes[0], 0);
            Debug.Log(output);
        }

        public void Draw() {
            if(Nodes.Length == 0) {return;}

            UltiDraw.Begin();
            void Traverse(Node parent, Node node) {
                if(parent != null) {
                    UltiDraw.DrawLine(parent.Transform.position, node.Transform.position, LineWidth, LineColor);
                }
                UltiDraw.DrawSphere(node.Transform.position, Quaternion.identity, NodeSize, node.Active ? ActiveNodeColor : InactiveNodeColor);
                foreach(Node child in node.Childs) {
                    Traverse(node, child);
                }
            }
            Traverse(null, Nodes[0]);
            foreach(Objective objective in Objectives) {
                UltiDraw.DrawSphere(objective.Position, Quaternion.identity, ObjectiveSize, ObjectiveColor);
                // UltiDraw.DrawLine(objective.Node.Transform.position, objective.Position, TargetColor);
            }
            UltiDraw.End();
        }

        [Serializable]
        public class Node {
            [SerializeReference] public Node Parent = null;
            [SerializeReference] public Node[] Childs = new Node[0];
            [SerializeReference] public Objective[] Objectives = new Objective[0];
            public float[] Activations = new float[0];

            public Transform Transform;
            public Limit Limit = null;
            public int Depth = 0;
            public float Length = 0f;

            public Vector3 ZeroPosition = Vector3.zero;
            public Quaternion ZeroRotation = Quaternion.identity;
            public bool Active = true;
            public bool SeedZero = false;

            public Node(Solver solver, Transform transform, Solver reference) {
                Parent = solver.FindNode(transform.parent);
                Childs = new Node[0];
                Objectives = new Objective[0];
                if(Parent != null) {
                    ArrayExtensions.Append(ref Parent.Childs, this);
                }
                ArrayExtensions.Append(ref solver.Nodes, this);
                Activations.SetAll(0f);

                Transform = transform;
                Limit = Transform.GetComponent<Limit>();
                Depth = Parent == null ? 1 : (Parent.Depth+1);
                Length = Parent == null ? 0f : (Parent.Length + Transform.localPosition.magnitude);

                Node n = reference == null ? null : reference.FindNode(Transform);
                ZeroPosition = n == null ? Transform.localPosition : n.ZeroPosition;
                ZeroRotation = n == null ? Transform.localRotation : n.ZeroRotation;
                Active = n == null ? true : n.Active;
                SeedZero = n == null ? false : n.SeedZero;
            }

            public void ComputeActivations() {
                if(Active) {
                    for(int i=0; i<Objectives.Length; i++) {
                        float value = 0f;
                        float sum = 0f;
                        
                        void ChildRecursion(Node child) {
                            if(child.Active && child.Objectives.Contains(Objectives[i])) {
                                value += child.Length - Length;
                                sum += 1f;
                            } else {
                                foreach(Node n in child.Childs) {
                                    ChildRecursion(n);
                                }
                            }
                        }
                        foreach(Node n in Childs) {
                            ChildRecursion(n);
                        }
                        
                        Node pivot = this;
                        while(pivot.Parent != null && pivot.Parent.Active && pivot.Parent.Objectives.Contains(Objectives[i])) {
                            pivot = pivot.Parent;
                        }

                        Activations[i] = sum == 0f ? 0f : ((value / sum) / (Objectives[i].Node.Length - pivot.Length));
                    }
                }
            }
        }

        [Serializable]
        public class Objective {
            [SerializeReference] public Node Node;
            public Vector3 Position = Vector3.zero;
            public Quaternion Rotation = Quaternion.identity;
            public float Weight = 1f;
            public bool ApplyRotation = true;

            public Objective(Solver solver, Transform transform, Solver reference) {
                Node = solver.FindNode(transform);
                ArrayExtensions.Append(ref solver.Objectives, this);
                Position = transform.position;
                Rotation = transform.rotation;

                Objective o = reference == null ? null : reference.FindObjective(transform);
                Weight = o == null ? 1f : o.Weight;
                ApplyRotation = o == null ? true : o.ApplyRotation;

                Node node = Node;
                while(node != null) {
                    if(!node.Objectives.Contains(this)) {
                        ArrayExtensions.Append(ref node.Objectives, this);
                        ArrayExtensions.Append(ref node.Activations, 0f);
                    }
                    node = node.Parent;
                }
            }

            public float GetError() {
                return Vector3.Distance(Node.Transform.position, Position);
            }
        }
    }

}