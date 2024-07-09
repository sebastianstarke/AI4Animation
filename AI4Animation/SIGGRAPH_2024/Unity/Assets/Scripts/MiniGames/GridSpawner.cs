using System.Collections;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;

namespace SIGGRAPH_2024 {
    public class GridSpawner : MonoBehaviour {
        public Vector2Int Cells = new Vector2Int(4, 3);
        public Vector2 Scale = Vector2.one;
        public Vector3 Resize = Vector3.one;
        public Color Color = Color.white;
        public List<GameObject> Obstacles;
        public float Cooldown = 3f;
        public int NumObstacles = 3;
        public bool FixedOrder = false;

        private float Width {get {return Scale.x / Cells.x;}}
        private float Height {get {return Scale.y / Cells.y;}}
        
        void Start() {
            StartCoroutine(SpawnObjects());
        }

        private Vector2Int GetCoordinates(int index) {
            int x = index % Cells.x;
            int y = index / Cells.x;
            return new Vector2Int(x,y);
        }

        private Vector3 Center(int x, int y) {
            return transform.position + transform.rotation * new Vector3(
                x.Ratio(0, Cells.x-1).Normalize(0f, 1f, -Scale.x/2f+Width/2f, Scale.x/2f-Width/2f),
                y.Ratio(0, Cells.y-1).Normalize(0f, 1f, -Scale.y/2f+Height/2f, Scale.y/2f-Height/2f),
                0f
            );
        }

        private Quaternion Rotation(int x, int y) {
            return transform.rotation * Quaternion.LookRotation(Vector3.forward);
        }

        public IEnumerator SpawnObjects() {
            int step = 0;
            while(true) {
                yield return new WaitForSeconds(Cooldown);
                step = (step + 1) % Obstacles.Count;
                if(FixedOrder) {
                    List<int> fields = new List<int>(ArrayExtensions.CreateEnumerated(Cells.x * Cells.y));
                    foreach(int index in fields.Random(NumObstacles)) {
                        Vector2Int coordinates = GetCoordinates(index);
                        GameObject instance = Instantiate(Obstacles[step], Center(coordinates.x, coordinates.y), Rotation(coordinates.x, coordinates.y));
                        instance.transform.localScale = Resize;
                    }
                } else {
                    List<int> fields = new List<int>(ArrayExtensions.CreateEnumerated(Cells.x * Cells.y));
                    foreach(int index in fields.Random(NumObstacles)) {
                        Vector2Int coordinates = GetCoordinates(index);
                        GameObject instance = Instantiate(Obstacles.Random(1)[0], Center(coordinates.x, coordinates.y), Rotation(coordinates.x, coordinates.y));
                        instance.transform.localScale = Vector3.Scale(instance.transform.localScale, Resize);
                    }
                }
            }
        }

        private void OnDrawGizmos() {
            Draw(Color);
        }

        public void Draw(Color color){
            UltiDraw.Begin();
            for(int x=0; x<Cells.x; x++) {
                for(int y=0; y<Cells.y; y++) {
                    UltiDraw.DrawSphere(Center(x,y), Rotation(x,y), 0.25f, color);
                    UltiDraw.DrawQuad(Center(x,y), transform.rotation * Quaternion.Euler(0f, 180f, 0f), Width, Height, color.Darken(0.5f));
                }
            }
            UltiDraw.End();
        }

        private class Cell {
            public Vector3 Center;
            public Quaternion Rotation;
            public float Width;
            public float Height;

            public Cell(Vector3 center, Quaternion rot, float width, float height){
                Center = center;
                Rotation = rot;
                Width = width;
                Height = height;
            }
        }
    }
}