using System;
using UnityEngine;
using UnityEditor;

namespace MagicIK {
    public abstract class Limit : MonoBehaviour {

        [SerializeField] [HideInInspector] public Quaternion ZeroRotation;

        void Reset() {
            ZeroRotation = transform.localRotation;
            foreach(IK ik in FindObjectsOfType<IK>()) {
                Solver.Node node = ik.Solver.FindNode(transform);
                if(node != null) {
                    node.Limit = this;
                }
            }
        }

        void OnDestroy() {
            Zero();
        }
        
        public void Zero() {
            transform.localRotation = ZeroRotation;
        }

        public abstract void Solve();

    }
}
