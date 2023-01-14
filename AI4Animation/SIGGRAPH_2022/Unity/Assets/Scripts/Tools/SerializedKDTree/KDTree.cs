using UnityEngine;
using UnityEditor;

namespace SerializedKDTree {

    public class KDTree : ScriptableObject {

        public KDNode Root;
        public int Dimensions;
        public int BucketCapacity;

        public void Initialize(int dimensions, int bucketCapacity=24) {
            Dimensions = dimensions;
            BucketCapacity = bucketCapacity;
            Root = ScriptableObjectExtensions.Create<KDNode>(this).Initialize(this);
        }

        public void Finish() {
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssets();
        }

        public void AddPoint(float[] point, KDNode.Value value) {
            Root.AddPoint(new KDNode.Point(point), value);
            // KDNode.Point p = new KDNode.Point(point);
            // value.Point = p;
            // Root.AddPoint(p, value);
        }

        public NearestNeighbour NearestNeighbors(float[] point, int k, float maxDistance = -1f) {
            DistanceFunctions function = new SquareEuclideanDistanceFunction();
            return new NearestNeighbour(this, Root, point, function, k, maxDistance);
        }
    }
    
}