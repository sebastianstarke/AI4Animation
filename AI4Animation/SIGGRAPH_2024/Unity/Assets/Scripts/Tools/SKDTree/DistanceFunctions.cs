
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SKDTree
{
    /// <summary>
    /// An interface which enables flexible distance functions.
    /// </summary>
    public interface DistanceFunctions
    {
        /// <summary>
        /// Compute a distance between two n-dimensional points.
        /// </summary>
        /// <param name="p1">The first point.</param>
        /// <param name="p2">The second point.</param>
        /// <returns>The n-dimensional distance.</returns>
        float Distance(float[] p1, float[] p2);

        /// <summary>
        /// Find the shortest distance from a point to an axis aligned rectangle in n-dimensional space.
        /// </summary>
        /// <param name="point">The point of interest.</param>
        /// <param name="min">The minimum coordinate of the rectangle.</param>
        /// <param name="max">The maximum coorindate of the rectangle.</param>
        /// <returns>The shortest n-dimensional distance between the point and rectangle.</returns>
        float DistanceToRectangle(float[] point, float[] min, float[] max);
    }

    /// <summary>
    /// A distance function for our KD-Tree which returns squared euclidean distances.
    /// </summary>
    public class SquareEuclideanDistanceFunction : DistanceFunctions
    {
        /// <summary>
        /// Find the squared distance between two n-dimensional points.
        /// </summary>
        /// <param name="p1">The first point.</param>
        /// <param name="p2">The second point.</param>
        /// <returns>The n-dimensional squared distance.</returns>
        public float Distance(float[] p1, float[] p2)
        {
            float fSum = 0;
            for (int i = 0; i < p1.Length; i++)
            {
                float fDifference = (p1[i] - p2[i]);
                fSum += fDifference * fDifference;
            }
            return fSum;
        }

        /// <summary>
        /// Find the shortest distance from a point to an axis aligned rectangle in n-dimensional space.
        /// </summary>
        /// <param name="point">The point of interest.</param>
        /// <param name="min">The minimum coordinate of the rectangle.</param>
        /// <param name="max">The maximum coorindate of the rectangle.</param>
        /// <returns>The shortest squared n-dimensional squared distance between the point and rectangle.</returns>
        public float DistanceToRectangle(float[] point, float[] min, float[] max)
        {
            float fSum = 0;
            float fDifference = 0;
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
