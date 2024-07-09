using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

namespace SIGGRAPH_2024 {
    public class Projectile : MonoBehaviour {
        
        public int Damage = 10;
        public float Velocity = 10f;
        public float Noise = 1f;
        public float TimeToLive = 3f;
        public LayerMask CollisionMask = ~0;
        public GameObject CollisionParticle = null;

        public HashSet<Func<Collider, bool>> Passthrough = new HashSet<Func<Collider, bool>>();

        void FixedUpdate() {
            transform.position += UnityEngine.Random.Range(Velocity-Noise, Velocity+Noise) * transform.forward * Time.fixedDeltaTime;
            TimeToLive -= Time.fixedDeltaTime;
            if(TimeToLive <= 0f) {
                Destroy(gameObject);
            }
        }

        void OnTriggerEnter(Collider other) {
            if(!Passthrough.Any(x => x(other))) {
                if(other.GetComponentInParent<HealthSystem>()) {
                    other.GetComponentInParent<HealthSystem>().ApplyDamage(Damage);
                }
                Destroy(gameObject);
            }
        }

        void OnDestroy() {
            if(this.IsDestroyedByPlaymodeChange()){return;}
            if(CollisionParticle != null) {
                Instantiate(CollisionParticle, transform.position, transform.rotation);
            }
        }

    }
}