using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KDTree
{
    /// <summary>
    /// A NearestNeighbour iterator for the KD-tree which intelligently iterates and captures relevant data in the search space.
    /// </summary>
    /// <typeparam name="T">The type of data the iterator should handle.</typeparam>
    public class NearestNeighbour<T> : IEnumerator<T>, IEnumerable<T>
    {
        /// <summary>The point from which are searching in n-dimensional space.</summary>
        private double[] tSearchPoint;
        /// <summary>A distance function which is used to compare nodes and value positions.</summary>
        private DistanceFunctions kDistanceFunction;
        /// <summary>The tree nodes which have yet to be evaluated.</summary>
        private MinHeap<KDNode<T>> pPending;
        /// <summary>The values which have been evaluated and selected.</summary>
        private IntervalHeap<T> pEvaluated;
        /// <summary>The root of the kd tree to begin searching from.</summary>
        private KDNode<T> pRoot = null;

        /// <summary>The max number of points we can return through this iterator.</summary>
        private int iMaxPointsReturned = 0;
        /// <summary>The number of points we can still test before conclusion.</summary>
        private int iPointsRemaining;
        /// <summary>Threshold to apply to tree iteration.  Negative numbers mean no threshold applied.</summary>
        private double fThreshold;

        /// <summary>Current value distance.</summary>
        private double _CurrentDistance = -1;
        /// <summary>Current value reference.</summary>
        private T _Current = default(T);

        /// <summary>
        /// Construct a new nearest neighbour iterator.
        /// </summary>
        /// <param name="pRoot">The root of the tree to begin searching from.</param>
        /// <param name="tSearchPoint">The point in n-dimensional space to search.</param>
        /// <param name="kDistance">The distance function used to evaluate the points.</param>
        /// <param name="iMaxPoints">The max number of points which can be returned by this iterator.  Capped to max in tree.</param>
        /// <param name="fThreshold">Threshold to apply to the search space.  Negative numbers indicate that no threshold is applied.</param>
        public NearestNeighbour(KDNode<T> pRoot, double[] tSearchPoint, DistanceFunctions kDistance, int iMaxPoints, double fThreshold)
        {
            // Check the dimensionality of the search point.
            if (tSearchPoint.Length != pRoot.iDimensions)
                throw new Exception("Dimensionality of search point and kd-tree are not the same.");

            // Store the search point.
            this.tSearchPoint = new double[tSearchPoint.Length];
            Array.Copy(tSearchPoint, this.tSearchPoint, tSearchPoint.Length);

            // Store the point count, distance function and tree root.
            this.iPointsRemaining = Math.Min(iMaxPoints, pRoot.Size);
            this.fThreshold = fThreshold;
            this.kDistanceFunction = kDistance;
            this.pRoot = pRoot;
            this.iMaxPointsReturned = iMaxPoints;
            _CurrentDistance = -1;

            // Create an interval heap for the points we check.
            this.pEvaluated = new IntervalHeap<T>();

            // Create a min heap for the things we need to check.
            this.pPending = new MinHeap<KDNode<T>>();
            this.pPending.Insert(0, pRoot);
        }

        /// <summary>
        /// Check for the next iterator item.
        /// </summary>
        /// <returns>True if we have one, false if not.</returns>
        public bool MoveNext()
        {
            // Bail if we are finished.
            if (iPointsRemaining == 0)
            {
                _Current = default(T);
                return false;
            }

            // While we still have paths to evaluate.
            while (pPending.Size > 0 && (pEvaluated.Size == 0 || (pPending.MinKey < pEvaluated.MinKey)))
            {
                // If there are pending paths possibly closer than the nearest evaluated point, check it out
                KDNode<T> pCursor = pPending.Min;
                pPending.RemoveMin();

                // Descend the tree, recording paths not taken
                while (!pCursor.IsLeaf)
                {
                    KDNode<T> pNotTaken;

                    // If the seach point is larger, select the right path.
                    if (tSearchPoint[pCursor.iSplitDimension] > pCursor.fSplitValue)
                    {
                        pNotTaken = pCursor.pLeft;
                        pCursor = pCursor.pRight;
                    }
                    else
                    {
                        pNotTaken = pCursor.pRight;
                        pCursor = pCursor.pLeft;
                    }

                    // Calculate the shortest distance between the search point and the min and max bounds of the kd-node.
                    double fDistance = kDistanceFunction.DistanceToRectangle(tSearchPoint, pNotTaken.tMinBound, pNotTaken.tMaxBound);

                    // If it is greater than the threshold, skip.
                    if (fThreshold >= 0 && fDistance > fThreshold)
                    {
                        //pPending.Insert(fDistance, pNotTaken);
                        continue;
                    }

                    // Only add the path we need more points or the node is closer than furthest point on list so far.
                    if (pEvaluated.Size < iPointsRemaining || fDistance <= pEvaluated.MaxKey)
                    {
                        pPending.Insert(fDistance, pNotTaken);
                    }
                }

                // If all the points in this KD node are in one place.
                if (pCursor.bSinglePoint)
                {
                    // Work out the distance between this point and the search point.
                    double fDistance = kDistanceFunction.Distance(pCursor.tPoints[0], tSearchPoint);

                    // Skip if the point exceeds the threshold.
                    // Technically this should never happen, but be prescise.
                    if (fThreshold >= 0 && fDistance >= fThreshold)
                        continue;

                    // Add the point if either need more points or it's closer than furthest on list so far.
                    if (pEvaluated.Size < iPointsRemaining || fDistance <= pEvaluated.MaxKey)
                    {
                        for (int i = 0; i < pCursor.Size; ++i)
                        {
                            // If we don't need any more, replace max
                            if (pEvaluated.Size == iPointsRemaining)
                                pEvaluated.ReplaceMax(fDistance, pCursor.tData[i]);

                            // Otherwise insert.
                            else
                                pEvaluated.Insert(fDistance, pCursor.tData[i]);
                        }
                    }
                }

                // If the points in the KD node are spread out.
                else
                {
                    // Treat the distance of each point seperately.
                    for (int i = 0; i < pCursor.Size; ++i)
                    {
                        // Compute the distance between the points.
                        double fDistance = kDistanceFunction.Distance(pCursor.tPoints[i], tSearchPoint);

                        // Skip if it exceeds the threshold.
                        if (fThreshold >= 0 && fDistance >= fThreshold)
                            continue;

                        // Insert the point if we have more to take.
                        if (pEvaluated.Size < iPointsRemaining)
                            pEvaluated.Insert(fDistance, pCursor.tData[i]);

                        // Otherwise replace the max.
                        else if (fDistance < pEvaluated.MaxKey)
                            pEvaluated.ReplaceMax(fDistance, pCursor.tData[i]);
                    }
                }
            }

            // Select the point with the smallest distance.
            if (pEvaluated.Size == 0)
                return false;

            iPointsRemaining--;
            _CurrentDistance = pEvaluated.MinKey;
            _Current = pEvaluated.Min;
            pEvaluated.RemoveMin();
            return true;
        }

        /// <summary>
        /// Reset the iterator.
        /// </summary>
        public void Reset()
        {
            // Store the point count and the distance function.
            this.iPointsRemaining = Math.Min(iMaxPointsReturned, pRoot.Size);
            _CurrentDistance = -1;

            // Create an interval heap for the points we check.
            this.pEvaluated = new IntervalHeap<T>();

            // Create a min heap for the things we need to check.
            this.pPending = new MinHeap<KDNode<T>>();
            this.pPending.Insert(0, pRoot);
        }

		public T Current
		{
			get { return _Current; }
		}

        /// <summary>
        /// Return the distance of the current value to the search point.
        /// </summary>
        public double CurrentDistance
        {
            get { return _CurrentDistance; }
        }

        /// <summary>
        /// Return the current value referenced by the iterator as an object.
        /// </summary>
        object IEnumerator.Current
        {
            get { return _Current; }
        }
  
        /// <summary>
        /// Return the current value referenced by the iterator.
        /// </summary>
        T IEnumerator<T>.Current
        {
            get { return _Current; }
        }

        public void Dispose()
        {
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this;
        }
    }
}
