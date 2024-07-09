using System;
using UnityEngine;
using UnityEditor;

namespace SIGGRAPH_2024 {
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    [RequireComponent(typeof(BoxCollider))]
    public class Animal : MonoBehaviour {
        public enum ANIMAL_ID {Bison, Boar, Buffalo, Camel, Crocodile, Donkey, Elephant, Giraffe, Gorilla, Hedgehog, Hippo, Leopard, Mouse, Penguin, Raccoon, Rhino, Tiger, Unicorn, Whitebear, Whitefox}
        public float Speed = 5f;
        public float LifeTime = 5f;
        public float SpawnTime = 1f;
        public float SlicesLifeTime = 1f;
        public float DestructionForce = 10f;
        public GameObject DestructionParticle = null;
        public ANIMAL_ID ID = ANIMAL_ID.Bison;
        public Slicable[] Slicables = new Slicable[0];

        void Awake() {
            SetID((ANIMAL_ID)UnityEngine.Random.Range(0, Slicables.Length));
            gameObject.AddComponent<Transparency>().FadeIn(SpawnTime, 0f);
        }

        void FixedUpdate() {
            transform.position += Speed * transform.forward * Time.fixedDeltaTime;
            
            LifeTime -= Time.fixedDeltaTime;
            if(LifeTime <= 0) {
                Utility.Destroy(gameObject);
            }
        }

        public void SetID(ANIMAL_ID id) {
            if(ID != id) {
                ID = id;
                GetComponent<MeshFilter>().mesh = Slicables[(int)id].Mesh;
            }
        }

        void OnTriggerEnter(Collider collider) {
            if(collider.gameObject.layer == LayerMask.NameToLayer("LaserSword")) {
                Slice(collider.GetComponentInParent<LaserSword>());
            }
        }

        void OnDestroy() {
            if(this.IsDestroyedByPlaymodeChange()){return;}
            if(DestructionParticle != null) {
                Instantiate(DestructionParticle, transform.position, transform.rotation);
            }
        }

        public void Slice(LaserSword sword) {
            BoxCollider collider = GetComponent<BoxCollider>();

            Vector3 a = collider.ClosestPointOnBounds(sword.LaserPosition);
            Vector3 b = collider.ClosestPointOnBounds(sword.LaserStart.position + sword.LaserStartVelocity + Speed * transform.forward);
            Vector3 c = collider.ClosestPointOnBounds(sword.LaserEnd.position + sword.LaserEndVelocity - Speed * transform.forward);

            Plane plane = new Plane(a, b, c);
            Vector3 center = (a + b + c) / 3;

            // Slice and get pieces
            GameObject[] gos = MeshSlicer.CutIntoPieces(gameObject, center, plane.normal, null);

            // Set up Layer and Tag
            gos[0].gameObject.layer = gameObject.layer;
            gos[1].gameObject.layer = gameObject.layer;
            gos[0].gameObject.tag = "Untagged";
            gos[1].gameObject.tag = "Untagged";

            // Attach RigidBodies
            {
                gos[0].AddComponent<Transparency>().FadeOut(SlicesLifeTime/2f, SlicesLifeTime/2f);
                Rigidbody rb = gos[0].AddComponent<Rigidbody>();
                MeshCollider coll = gos[0].AddComponent<MeshCollider>();
                coll.convex = true;
                rb.AddForceAtPosition(-DestructionForce*plane.normal, transform.position, ForceMode.Impulse);
            }
            {
                gos[1].AddComponent<Transparency>().FadeOut(SlicesLifeTime/2f, SlicesLifeTime/2f);
                Rigidbody rb = gos[1].AddComponent<Rigidbody>();
                MeshCollider coll = gos[1].AddComponent<MeshCollider>();
                coll.convex = true;
                rb.AddForceAtPosition(+DestructionForce*plane.normal, transform.position, ForceMode.Impulse);
            }
            
            Utility.Destroy(gameObject);
            GameObject.Destroy(gos[0], SlicesLifeTime);
            GameObject.Destroy(gos[1], SlicesLifeTime);
        }

        [Serializable]
        public class Slicable {
            public bool Enabled = true;
            public Mesh Mesh = null;
        }

        #if UNITY_EDITOR
        [CustomEditor(typeof(Animal), true)]
        public class Animal_Editor : Editor {

            public Animal Target;

            void Awake() {
                Target = (Animal)target;
            }

            public override void OnInspectorGUI() {
                Undo.RecordObject(Target, Target.name);

                Target.Speed = EditorGUILayout.FloatField("Speed", Target.Speed);
                Target.LifeTime = EditorGUILayout.FloatField("Life Time", Target.LifeTime);
                Target.SlicesLifeTime = EditorGUILayout.FloatField("Slices Life Time", Target.SlicesLifeTime);
                Target.DestructionForce = EditorGUILayout.FloatField("Destruction Force", Target.DestructionForce);
                Target.DestructionParticle = EditorGUILayout.ObjectField("Destruction Particle", Target.DestructionParticle, typeof(GameObject), false) as GameObject;
                Target.SetID((ANIMAL_ID)EditorGUILayout.EnumPopup("ID", Target.ID));
                foreach(Slicable slicable in Target.Slicables) {
                    EditorGUILayout.BeginHorizontal();
                    slicable.Mesh = EditorGUILayout.ObjectField("Mesh", slicable.Mesh, typeof(Mesh), false) as Mesh;
                    slicable.Enabled = EditorGUILayout.Toggle(slicable.Enabled, GUILayout.Width(20f));
                    EditorGUILayout.EndHorizontal();
                }
                if(Utility.GUIButton("Add Slicable", UltiDraw.DarkGrey, UltiDraw.White)) {
                    ArrayExtensions.Expand(ref Target.Slicables);
                }

                if(GUI.changed) {
                    EditorUtility.SetDirty(Target);
                }
            }

        }
        #endif
    }
}