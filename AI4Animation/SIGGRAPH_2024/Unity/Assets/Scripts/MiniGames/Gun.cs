using UnityEngine;

namespace SIGGRAPH_2024 {
    public class Gun : MonoBehaviour {
        public Vector3 Muzzle;
        public Axis Forward;
        public Axis Up;
        public GameObject Projectile;
        public float Cooldown = 1f;

        private float Remaining = 0f;

        void FixedUpdate() {
            Remaining = Mathf.Max(Remaining - Time.fixedDeltaTime, 0f);
        }

        public bool Fire() {
            if(Remaining > 0f) {
                return false;
            }
            Projectile projectile = Instantiate(Projectile, GetMuzzlePosition(), Quaternion.LookRotation(GetMuzzleDirection(), GetMuzzleVertical())).GetComponent<Projectile>();
            projectile.Passthrough.Add(x => x.GetComponent<Projectile>() != null);
            projectile.Passthrough.Add(x => this != null && x.transform.root.gameObject == gameObject);
            Remaining = Cooldown;
            return true;
        }

        public Vector3 GetMuzzlePosition() {
            return Muzzle.PositionFrom(transform.GetGlobalMatrix());
        }

        public Vector3 GetMuzzleDirection() {
            return transform.GetAxis(Forward);
        }

        public Vector3 GetMuzzleVertical() {
            return transform.GetAxis(Up);
        }

        void OnDrawGizmosSelected() {
            Gizmos.color = Color.magenta;
            Gizmos.DrawLine(GetMuzzlePosition(), GetMuzzlePosition() + GetMuzzleDirection());
            Gizmos.DrawLine(GetMuzzlePosition(), GetMuzzlePosition() + 0.1f*GetMuzzleVertical());
            Gizmos.DrawSphere(GetMuzzlePosition(), 0.025f);
            Gizmos.DrawSphere(GetMuzzlePosition() + GetMuzzleDirection(), 0.025f);
            Gizmos.DrawSphere(GetMuzzlePosition() + 0.1f*GetMuzzleVertical(), 0.025f);
        }
    }
}