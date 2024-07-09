using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SIGGRAPH_2024 {
    public class HealthSystem : MonoBehaviour {
        public int Health = 100;
        
        public void ApplyDamage(int value) {
            // Debug.Log("GameObject " + name + " took " + value + " damage.");
            Health -= value;
            if(Health <= 0) {
                Destroy(gameObject);
            }            
        }
    }
}