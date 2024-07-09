#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SIGGRAPH_2024 {
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(HealthSystem))]
    [RequireComponent(typeof(Gun))]
    public class Drone : MonoBehaviour {

        public float Mobility = 5f;
        public float Smoothing = 0.5f;
        public float SearchFrequency = 1f;
        public int MaxSearchDepth = 5;
        public float HostileDistance = 15f;
        public float DistanceToTarget = 0f;
        public string TargetTag = "Player";

        public GameObject DestructionParticle;

        private Rigidbody RB;
        private HealthSystem HS;
        private Gun Gun;

        public PID_QuaternionController RotationController;

        public bool DrawSearch = false;
        public bool DrawHistory = false;

        private PathPlanner3D PathPlanner;
        private PathPlanner3D.Path Path;
        
        void Awake() {
            RB = GetComponent<Rigidbody>();
            HS = GetComponent<HealthSystem>();
            Gun = GetComponent<Gun>();
            
            PathPlanner = FindObjectOfType<PathPlanner3D>();
            StartCoroutine(Search());
        }

        void FixedUpdate() {
            Vector3 waypoint = GetNextWaypoint();
            if(waypoint != RB.position) {
                float distanceToTarget = Vector3.Distance(RB.position, waypoint);
                Vector3 targetVelocity = Mobility * (waypoint - RB.position).normalized.ClampMagnitude(0f, distanceToTarget);

                Vector3 tmp = Vector3.zero;
                RB.velocity = Vector3.SmoothDamp(
                    RB.velocity, 
                    targetVelocity, 
                    ref tmp, 
                    Smoothing
                );

                RB.MoveRotation(RotationController.Update(RB.rotation, Quaternion.LookRotation((GetTarget() - RB.position).ZeroY().normalized, Vector3.up), Time.fixedDeltaTime));
            }

            // if(Input.GetKey(KeyCode.F)) {
            //     Gun.Fire();
            // }
        }

        private IEnumerator Search() {
            while(true) {
                if(RB.position == GetTarget()) {
                    Path = null;
                } else {
                    Path = PathPlanner.Search(RB.position, GetTarget(), MaxSearchDepth, DistanceToTarget);
                }
                yield return new WaitForSeconds(1f/SearchFrequency);
            }
        }

        private Vector3 GetTarget() {
            GameObject go = GameObject.FindGameObjectWithTag(TargetTag);
            if(go == null || Vector3.Distance(RB.position, go.transform.position) > HostileDistance) {
                return RB.position;
            } else {
                return go.transform.position;
            }
        }

        private Vector3 GetNextWaypoint() {
            if(Path == null) {
                return RB.position;
            }
            return Vector3.zero;
            // return Path.GetTargetPoint(RB.position);
        }

        void OnDestroy() {
            if(this.IsDestroyedByPlaymodeChange()){return;}
            if(DestructionParticle != null) {
                Instantiate(DestructionParticle, RB.position, RB.rotation);
            }
        }

        void OnRenderObject() {
            if(DrawSearch) {
                if(Path != null) {
                    Path.Draw(DrawHistory);
                    UltiDraw.Begin();
                    Vector3 waypoint = GetNextWaypoint();
                    UltiDraw.DrawLine(RB.position, waypoint, UltiDraw.Magenta);
                    UltiDraw.DrawSphere(waypoint, Quaternion.identity, 0.25f, UltiDraw.Magenta);
                    UltiDraw.End();
                }
            }
        }
    }
}
#endif