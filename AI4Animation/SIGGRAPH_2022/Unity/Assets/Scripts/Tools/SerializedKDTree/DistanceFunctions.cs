
namespace SerializedKDTree {

    public interface DistanceFunctions {
        float Distance(float[] p1, float[] p2);
        float DistanceToRectangle(float[] point, float[] min, float[] max);
    }

    public class SquareEuclideanDistanceFunction : DistanceFunctions {
        public float Distance(float[] p1, float[] p2) {
            float fSum = 0f;
            for (int i = 0; i < p1.Length; i++)
            {
                float fDifference = (p1[i] - p2[i]);
                fSum += fDifference * fDifference;
            }
            return fSum;
        }

        public float DistanceToRectangle(float[] point, float[] min, float[] max) {
            float fSum = 0f;
            float fDifference = 0f;
            for (int i = 0; i < point.Length; ++i)
            {
                fDifference = 0;
                if (point[i] > max[i])
                    fDifference = (point[i] - max[i]);
                else if (point[i] < min[i])
                    fDifference = (point[i] - min[i]);
                fSum += fDifference * fDifference;
            }
            return fSum;
        }
    }

}
