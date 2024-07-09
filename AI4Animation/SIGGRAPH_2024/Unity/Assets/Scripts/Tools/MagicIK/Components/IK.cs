using UnityEngine;

namespace MagicIK {
    public class IK : MonoBehaviour {

        [HideInInspector] public Solver Solver;

        void Reset() {
            Solver = new Solver(null, null, null);
        }
        
    }
}