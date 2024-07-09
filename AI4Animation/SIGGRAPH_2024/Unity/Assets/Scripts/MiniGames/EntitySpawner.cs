using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace HybridMode {
    public class EntitySpawner : MonoBehaviour {
        [Serializable]
        public class Entity {
            public GameObject Prefab;
            public float Interval;
            public Vector3 Position;
        }
        public Entity[] Entities;

        void Start() {
            foreach(Entity entity in Entities) {
                StartCoroutine(Spawner(entity));
            }
        }

        private IEnumerator Spawner(Entity entity) {
            while(true) {
                Instantiate(entity.Prefab, entity.Position, Quaternion.identity);
                yield return new WaitForSeconds(entity.Interval);
            }
        }
    }
}