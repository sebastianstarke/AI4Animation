using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SIGGRAPH_2024 {
    public class PlatformerObstacle : MonoBehaviour
    {
        public float Speed = 5f;
        public float LifeTime = 30f;
        public float SpawnTime = 1f;

        void Awake(){
            gameObject.AddComponent<Transparency>().FadeIn(SpawnTime, 0f);
        }

        void FixedUpdate() {
            transform.position += Speed * transform.forward * Time.fixedDeltaTime;
            
            LifeTime -= Time.fixedDeltaTime;
            if(LifeTime <= 0) {
                Utility.Destroy(gameObject);
            }
        }
    }
}