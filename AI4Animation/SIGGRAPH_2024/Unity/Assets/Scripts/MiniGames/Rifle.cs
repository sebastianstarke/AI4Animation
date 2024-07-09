using UnityEngine;
using AI4Animation;
using UltimateIK;

namespace SIGGRAPH_2024 {
    public class Rifle : MonoBehaviour {
        public TrackingSystem TS;

        public Transform Muzzle;
        public Transform Hold;
        public Axis Forward;
        public Axis Up;
        public GameObject Projectile;
        public float Cooldown = 1f;
        public Vector3 LeftAimShift = Vector3.zero;
        public Vector3 RightAimShift = Vector3.zero;

        public Actor Target;

        public float ElbowConstraintWeight = 0.1f;

        public string LeftShoulder;
        public string LeftElbow;
        public string LeftWrist;
        public Transform LeftContact;

        public string RightShoulder;
        public string RightElbow;
        public string RightWrist;
        public Transform RightContact;

        public bool SolveIK = true;
        public bool DrawIK = false;

        private float Remaining = 0f;

        private IK LeftIK;
        private IK RightIK;

        void Awake() {
            LeftIK = IK.Create(Target.FindTransform(LeftShoulder), Target.FindTransforms(LeftElbow, LeftWrist));
            RightIK = IK.Create(Target.FindTransform(RightShoulder), Target.FindTransforms(RightElbow, RightWrist));

            LeftIK.FindJoint(Blueman.LeftWristTwistName).Active = false;
            // LeftIK.FindJoint("b_l_forearm").SetJointType(TYPE.HingeZ);
            // LeftIK.FindJoint("b_l_forearm").SetLowerLimit(-180f);
            // LeftIK.FindJoint("b_l_forearm").SetUpperLimit(20f);

            RightIK.FindJoint(Blueman.RightWristTwistName).Active = false;
            // RightIK.FindJoint("b_r_forearm").SetJointType(TYPE.HingeZ);
            // RightIK.FindJoint("b_r_forearm").SetLowerLimit(-180f);
            // RightIK.FindJoint("b_r_forearm").SetUpperLimit(20f);

            LeftIK.Objectives.First().SolvePosition = true;
            LeftIK.Objectives.First().SolveRotation = false;

            LeftIK.Objectives.Last().SolvePosition = true;
            LeftIK.Objectives.Last().SolveRotation = false;

            RightIK.Objectives.First().SolvePosition = true;
            RightIK.Objectives.First().SolveRotation = false;

            RightIK.Objectives.Last().SolvePosition = true;
            RightIK.Objectives.Last().SolveRotation = false;
        }

        void FixedUpdate() {
            Remaining = Mathf.Max(Remaining - Time.fixedDeltaTime, 0f);
            if(TS.GetTracker(UnityEngine.XR.XRNode.RightHand).GetButton(TrackingSystem.BUTTON.Trigger)) {
                Fire();
            }
        }

        public void Solve() {
            transform.position = GetLeftPalm();
            // transform.rotation = Quaternion.LookRotation(Vector3.Slerp((GetLeftPalm() - GetAnchor()).normalized, GetViewDirection(), 0.5f), Vector3.up);
            transform.rotation = Quaternion.LookRotation((GetLeftPalm() - GetRightPalm()).normalized, Vector3.up);

            if(SolveIK) {
                LeftIK.Objectives.First().Weight = ElbowConstraintWeight;
                RightIK.Objectives.First().Weight = ElbowConstraintWeight;

                LeftIK.SetTargets(Target.GetBoneTransformation(LeftElbow), LeftContact.GetWorldMatrix());
                RightIK.SetTargets(Target.GetBoneTransformation(RightElbow), RightContact.GetWorldMatrix());

                LeftIK.Solve();
                RightIK.Solve();

                LeftIK.Joints.Last().Transform.rotation = LeftContact.rotation;
                RightIK.Joints.Last().Transform.rotation = RightContact.rotation;
            }
        }

        public void ComputeWeapon() {
            transform.position = GetLeftPalm();
            transform.rotation = Quaternion.LookRotation((GetLeftPalm() - GetRightPalm()).normalized, Vector3.up);
        }

        public Matrix4x4 GetLeftTarget() {
            return LeftContact.GetWorldMatrix();
        }

        public Matrix4x4 GetRightTarget() {
            return RightContact.GetWorldMatrix();
        }

        // public void Solve(MagicIK.GenericIK solver) {
        //     if(SolveIK) {
        //         LeftIK.Objectives.First().Weight = ElbowConstraintWeight;
        //         RightIK.Objectives.First().Weight = ElbowConstraintWeight;

        //         LeftIK.SetTargets(Target.GetBoneTransformation(LeftElbow), LeftContact.GetWorldMatrix());
        //         RightIK.SetTargets(Target.GetBoneTransformation(RightElbow), RightContact.GetWorldMatrix());

        //         LeftIK.Solve();
        //         RightIK.Solve();

        //         LeftIK.Joints.Last().Transform.rotation = LeftContact.rotation;
        //         RightIK.Joints.Last().Transform.rotation = RightContact.rotation;
        //     }
        // }

        public bool Fire() {
            if(Remaining > 0f) {
                return false;
            }
            Projectile projectile = Instantiate(Projectile, Muzzle.position, Quaternion.LookRotation(GetMuzzleDirection(), GetMuzzleVertical())).GetComponent<Projectile>();
            projectile.Passthrough.Add(x => x.GetComponent<Projectile>() != null);
            projectile.Passthrough.Add(x => this != null && x.transform.root.gameObject == gameObject);
            Remaining = Cooldown;
            return true;
        }

        public Vector3 GetMuzzleDirection() {
            return transform.GetAxis(Forward);
        }

        public Vector3 GetMuzzleVertical() {
            return transform.GetAxis(Up);
        }

        private Vector3 GetLeftPalm() {
            Transform t = TS.Actor.FindTransform("l_palm_center_marker");
            return t.position + t.rotation * LeftAimShift + Target.GetRoot().position - TS.Root.GetPosition();
        }

        private Vector3 GetRightPalm() {
            Transform t = TS.Actor.FindTransform("r_palm_center_marker");
            return t.position + t.rotation * RightAimShift + Target.GetRoot().position - TS.Root.GetPosition();
        }

        // private Vector3 GetAnchor() {
        //     return (GetNeck() + GetRightShoulder() + GetRightPalm())/3f;
        //     // return GetNeck();
        // }

        // private Vector3 GetNeck() {
        //     return TS.Actor.FindTransform("b_neck0").position + Target.GetRoot().position - TS.Root.GetPosition();
        // }

        // private Vector3 GetRightShoulder() {
        //     return TS.Actor.FindTransform("p_r_scap").position + Target.GetRoot().position - TS.Root.GetPosition();
        // }

        // private Vector3 GetViewPoint() {
        //     return TS.CalculateViewPoint(0f) + Target.GetRoot().position - TS.Root.GetPosition();
        // }

        // private Vector3 GetViewDirection() {
        //     return TS.GetCameraDirection();
        // }

        void OnDrawGizmos() {
            Gizmos.color = Color.magenta;
            Gizmos.DrawLine(Muzzle.position, Muzzle.position + GetMuzzleDirection());
            Gizmos.DrawLine(Muzzle.position, Muzzle.position + 0.1f*GetMuzzleVertical());
            Gizmos.DrawSphere(Muzzle.position, 0.025f);
            Gizmos.DrawSphere(Muzzle.position + GetMuzzleDirection(), 0.025f);
            Gizmos.DrawSphere(Muzzle.position + 0.1f*GetMuzzleVertical(), 0.025f);

            if(Hold != null) {
                Gizmos.DrawSphere(Hold.position, 0.025f);
            }
            if(LeftContact != null) {
                Gizmos.DrawSphere(LeftContact.position, 0.025f);
            }
            if(RightContact != null) {
                Gizmos.DrawSphere(RightContact.position, 0.025f);
            }
        }

        void OnRenderObject() {
            if(DrawIK) {
                // UltiDraw.Begin();
                // UltiDraw.DrawLine(GetLeftReference(), GetRightReference(), 0.025f, UltiDraw.Cyan);
                // UltiDraw.DrawLine(GetMuzzlePosition(), GetMuzzlePosition() + 10f*GetMuzzleDirection().normalized, 0.025f, 0f, UltiDraw.White);
                // UltiDraw.End();

                UltiDraw.Begin();
                UltiDraw.DrawSphere(GetLeftPalm(), Quaternion.identity, 0.05f, UltiDraw.Cyan);
                UltiDraw.DrawSphere(GetRightPalm(), Quaternion.identity, 0.05f, UltiDraw.Cyan);
                UltiDraw.DrawLine(GetLeftPalm(), GetRightPalm(), 0.0125f, UltiDraw.Magenta);

                // UltiDraw.DrawCube(GetViewpoint(), TS.GetCameraRotation(), 0.25f, UltiDraw.White);
                // UltiDraw.DrawLine(GetViewPoint(), GetViewPoint() + 10f*GetViewDirection(), 0.025f, 0f, UltiDraw.White);
                // UltiDraw.DrawLine(GetViewPoint(), GetViewPoint() + 10f*(GetLeftPalm() - GetViewPoint()).normalized, 0.025f, 0f, UltiDraw.Magenta);
                UltiDraw.End();

                LeftIK.Draw();
                RightIK.Draw();
            }
        }
    }
}