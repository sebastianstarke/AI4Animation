using UnityEngine;

namespace HybridMode {
    public class Particle : MonoBehaviour {

        private float TimeToLive;

        void Awake() {
            TimeToLive = GetComponent<ParticleSystem>().main.duration;
        }

        void FixedUpdate() {
            TimeToLive -= Time.fixedDeltaTime;
            if(TimeToLive <= 0f) {
                Utility.Destroy(gameObject);
            }
        }
        
    }
}