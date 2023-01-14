using System;
using System.Collections;

namespace SerializedKDTree {

    public class NearestNeighbour : IEnumerator, IEnumerable {
        private float[] tSearchPoint;
        private DistanceFunctions kDistanceFunction;
        private MinHeap<KDNode> pPending;
        private IntervalHeap<KDNode.Value> pEvaluated;
        private KDNode pRoot = null;

        private int iMaxPointsReturned = 0;
        private int iPointsRemaining;
        private float fThreshold;

        private float _CurrentDistance = -1f;
        private KDNode.Value _Current = null;

        public NearestNeighbour(KDTree kdtree, KDNode pRoot, float[] tSearchPoint, DistanceFunctions kDistance, int iMaxPoints, float fThreshold) {
            // Check the dimensionality of the search point.
            if (tSearchPoint.Length != kdtree.Dimensions)
                throw new Exception("Dimensionality of search point and kd-tree are not the same.");

            // Store the search point.
            this.tSearchPoint = new float[tSearchPoint.Length];
            Array.Copy(tSearchPoint, this.tSearchPoint, tSearchPoint.Length);

            // Store the point count, distance function and tree root.
            this.iPointsRemaining = Math.Min(iMaxPoints, pRoot.Size);
            this.fThreshold = fThreshold;
            this.kDistanceFunction = kDistance;
            this.pRoot = pRoot;
            this.iMaxPointsReturned = iMaxPoints;
            _CurrentDistance = -1;

            // Create an interval heap for the points we check.
            this.pEvaluated = new IntervalHeap<KDNode.Value>();

            // Create a min heap for the things we need to check.
            this.pPending = new MinHeap<KDNode>();
            this.pPending.Insert(0, pRoot);
        }

        public bool MoveNext() {
            // Bail if we are finished.
            if (iPointsRemaining == 0)
            {
                _Current = null;
                return false;
            }

            // While we still have paths to evaluate.
            while (pPending.Size > 0 && (pEvaluated.Size == 0 || (pPending.MinKey < pEvaluated.MinKey)))
            {
                // If there are pending paths possibly closer than the nearest evaluated point, check it out
                KDNode pCursor = pPending.Min;
                pPending.RemoveMin();

                // Descend the tree, recording paths not taken
                while (!pCursor.IsLeaf())
                {
                    KDNode pNotTaken;

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
                    float fDistance = kDistanceFunction.DistanceToRectangle(tSearchPoint, pNotTaken.tMinBound, pNotTaken.tMaxBound);

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
                if (pCursor.SinglePoint)
                {
                    // Work out the distance between this point and the search point.
                    float fDistance = kDistanceFunction.Distance(pCursor.Points[0].Values, tSearchPoint);

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
                                pEvaluated.ReplaceMax(fDistance, pCursor.Values[i]);

                            // Otherwise insert.
                            else
                                pEvaluated.Insert(fDistance, pCursor.Values[i]);
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
                        float fDistance = kDistanceFunction.Distance(pCursor.Points[i].Values, tSearchPoint);

                        // Skip if it exceeds the threshold.
                        if (fThreshold >= 0 && fDistance >= fThreshold)
                            continue;

                        // Insert the point if we have more to take.
                        if (pEvaluated.Size < iPointsRemaining)
                            pEvaluated.Insert(fDistance, pCursor.Values[i]);

                        // Otherwise replace the max.
                        else if (fDistance < pEvaluated.MaxKey)
                            pEvaluated.ReplaceMax(fDistance, pCursor.Values[i]);
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

        public void Reset() {
            // Store the point count and the distance function.
            this.iPointsRemaining = Math.Min(iMaxPointsReturned, pRoot.Size);
            _CurrentDistance = -1;

            // Create an interval heap for the points we check.
            this.pEvaluated = new IntervalHeap<KDNode.Value>();

            // Create a min heap for the things we need to check.
            this.pPending = new MinHeap<KDNode>();
            this.pPending.Insert(0, pRoot);
        }

		public KDNode.Value Current {
			get { return _Current; }
		}

        public float CurrentDistance {
            get { return _CurrentDistance; }
        }

        object IEnumerator.Current {
            get { return _Current; }
        }

        public void Dispose() {
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }

        public IEnumerator GetEnumerator() {
            return this;
        }
    }

    public class MinHeap<T> {
        private static int DEFAULT_SIZE = 64;
        private T[] tData;
        private float[] tKeys;
        public MinHeap() : this(DEFAULT_SIZE) {

        }

        public MinHeap(int iCapacity) {
            this.tData = new T[iCapacity];
            this.tKeys = new float[iCapacity];
            this.Capacity = iCapacity;
            this.Size = 0;
        }

        public int Size { get; private set; }
        public int Capacity { get; private set; }

        public void Insert(float key, T value) {
            // If we need more room, double the space.
            if (Size >= Capacity)
            {
                // Calcualte the new capacity.
                Capacity *= 2;

                // Copy the data array.
                var newData = new T[Capacity];
                Array.Copy(tData, newData, tData.Length);
                tData = newData;

                // Copy the key array.
                var newKeys = new float[Capacity];
                Array.Copy(tKeys, newKeys, tKeys.Length);
                tKeys = newKeys;
            }

            // Insert new value at the end
            tData[Size] = value;
            tKeys[Size] = key;
            SiftUp(Size);
            Size++;
        }

        public void RemoveMin() {
            if (Size == 0)
                throw new Exception();

            Size--;
            tData[0] = tData[Size];
            tKeys[0] = tKeys[Size];
            tData[Size] = default(T);
            SiftDown(0);
        }

        public T Min {
            get
            {
                if (Size == 0)
                    throw new Exception();

                return tData[0];
            }
        }

        public float MinKey {
            get
            {
                if (Size == 0)
                    throw new Exception();

                return tKeys[0];
            }
        }

        private void SiftUp(int iChild) {
            // For each parent above the child, if the parent is smaller then bubble it up.
            for (int iParent = (iChild - 1) / 2; 
                iChild != 0 && tKeys[iChild] < tKeys[iParent]; 
                iChild = iParent, iParent = (iChild - 1) / 2)
            {
                T kData = tData[iParent];
                float dDist = tKeys[iParent];

                tData[iParent] = tData[iChild];
                tKeys[iParent] = tKeys[iChild];

                tData[iChild] = kData;
                tKeys[iChild] = dDist;
            }
        }

        private void SiftDown(int iParent) {
            // For each child.
            for (int iChild = iParent * 2 + 1; iChild < Size; iParent = iChild, iChild = iParent * 2 + 1)
            {
                // If the child is larger, select the next child.
                if (iChild + 1 < Size && tKeys[iChild] > tKeys[iChild + 1])
                    iChild++;

                // If the parent is larger than the largest child, swap.
                if (tKeys[iParent] > tKeys[iChild])
                {
                    // Swap the points
                    T pData = tData[iParent];
                    float pDist = tKeys[iParent];

                    tData[iParent] = tData[iChild];
                    tKeys[iParent] = tKeys[iChild];

                    tData[iChild] = pData;
                    tKeys[iChild] = pDist;
                }

                // TODO: REMOVE THE BREAK
                else
                {
                    break;
                }
            }
        }
    }

    public class IntervalHeap<T> {
        private const int DEFAULT_SIZE = 64;
        private T[] tData;
        private float[] tKeys;

        public IntervalHeap() : this(DEFAULT_SIZE) {

        }

        public IntervalHeap(int capacity) {
            this.tData = new T[capacity];
            this.tKeys = new float[capacity];
            this.Capacity = capacity;
            this.Size = 0;
        }

        public int Size { get; private set; }
        public int Capacity { get; private set; }

        public T Min {
            get
            {
                if (Size == 0)
                    throw new Exception();
                return tData[0];
            }
        }

        public T Max {
            get
            {
                if (Size == 0)
                {
                    throw new Exception();
                }
                else if (Size == 1)
                {
                    return tData[0];
                }

                return tData[1];
            }
        }

        public float MinKey {
            get
            {
                if (Size == 0)
                    throw new Exception();
                return tKeys[0];
            }
        }

        public float MaxKey {
            get
            {
                if (Size == 0)
                {
                    throw new Exception();
                }
                else if (Size == 1)
                {
                    return tKeys[0];
                }

                return tKeys[1];
            }
        }

        public void Insert(float key, T value) {
            // If more room is needed, double the array size.
            if (Size >= Capacity)
            {
                // Double the capacity.
                Capacity *= 2;

                // Expand the data array.
                var newData = new T[Capacity];
                Array.Copy(tData, newData, tData.Length);
                tData = newData;

                // Expand the key array.
                var newKeys = new float[Capacity];
                Array.Copy(tKeys, newKeys, tKeys.Length);
                tKeys = newKeys;
            }

            // Insert the new value at the end.
            Size++;
            tData[Size-1] = value;
            tKeys[Size-1] = key;

            // Ensure it is in the right place.
            SiftInsertedValueUp();
        }

        public void RemoveMin() {
            // Check for errors.
            if (Size == 0)
                throw new Exception();

            // Remove the item by 
            Size--;
            tData[0] = tData[Size];
            tKeys[0] = tKeys[Size];
            tData[Size] = default(T);
            SiftDownMin(0);
        }

        public void ReplaceMin(float key, T value) {
            // Check for errors.
            if (Size == 0)
                throw new Exception();

            // Add the data.
            tData[0] = value;
            tKeys[0] = key;

            // If we have more than one item.
            if (Size > 1)
            {
                // Swap with pair if necessary.
                if (tKeys[1] < key)
                    Swap(0, 1);
                SiftDownMin(0);
            }
        }

        public void RemoveMax() {
            // If we have no items in the queue.
            if (Size == 0)
            {
                throw new Exception();
            }

            // If we have one item, remove the min.
            else if (Size == 1)
            {
                RemoveMin();
                return;
            }

            // Remove the max.
            Size--;
            tData[1] = tData[Size];
            tKeys[1] = tKeys[Size];
            tData[Size] = default(T);
            SiftDownMax(1);
        }

        public void ReplaceMax(float key, T value) {
            if (Size == 0)
            {
                throw new Exception();
            }
            else if (Size == 1)
            {
                ReplaceMin(key, value);
                return;
            }

            tData[1] = value;
            tKeys[1] = key;
            // Swap with pair if necessary
            if (key < tKeys[0]) {
                Swap(0, 1);
            }
            SiftDownMax(1);
        }

        private int Swap(int x, int y) {
            // Store temp.
            T yData = tData[y];
            float yDist = tKeys[y];

            // Swap
            tData[y] = tData[x];
            tKeys[y] = tKeys[x];
            tData[x] = yData;
            tKeys[x] = yDist;

            // Return.
            return y;
        }

        private void SiftInsertedValueUp() {
            // Work out where the element was inserted.
            int u = Size-1;

            // If it is the only element, nothing to do.
            if (u == 0)
            {
            }

            // If it is the second element, sort with it's pair.
            else if (u == 1)
            {
                // Swap if less than paired item.
                if  (tKeys[u] < tKeys[u-1])
                    Swap(u, u-1);
            }

            // If it is on the max side, 
            else if (u % 2 == 1)
            {
                // Already paired. Ensure pair is ordered right
                int p = (u/2-1)|1; // The larger value of the parent pair
                if  (tKeys[u] < tKeys[u-1])
                { // If less than it's pair
                    u = Swap(u, u-1); // Swap with it's pair
                    if (tKeys[u] < tKeys[p-1])
                    { // If smaller than smaller parent pair
                        // Swap into min-heap side
                        u = Swap(u, p-1);
                        SiftUpMin(u);
                    }
                }
                else
                {
                    if (tKeys[u] > tKeys[p])
                    { // If larger that larger parent pair
                        // Swap into max-heap side
                        u = Swap(u, p);
                        SiftUpMax(u);
                    }
                }
            }
            else
            {
                // Inserted in the lower-value slot without a partner
                int p = (u/2-1)|1; // The larger value of the parent pair
                if (tKeys[u] > tKeys[p])
                { // If larger that larger parent pair
                    // Swap into max-heap side
                    u = Swap(u, p);
                    SiftUpMax(u);
                }
                else if (tKeys[u] < tKeys[p-1])
                { // If smaller than smaller parent pair
                    // Swap into min-heap side
                    u = Swap(u, p-1);
                    SiftUpMin(u);
                }
            }
        }

        private void SiftUpMin(int iChild) {
            // Min-side parent: (x/2-1)&~1
            for (int iParent = (iChild/2-1)&~1; 
                iParent >= 0 && tKeys[iChild] < tKeys[iParent]; 
                iChild = iParent, iParent = (iChild/2-1)&~1)
            {
                Swap(iChild, iParent);
            }
        }

        private void SiftUpMax(int iChild) {
            // Max-side parent: (x/2-1)|1
            for (int iParent = (iChild/2-1)|1; 
                iParent >= 0 && tKeys[iChild] > tKeys[iParent]; 
                iChild = iParent, iParent = (iChild/2-1)|1)
            {
                Swap(iChild, iParent);
            }
        }

        private void SiftDownMin(int iParent) {
            // For each child of the parent.
            for (int iChild = iParent * 2 + 2; iChild < Size; iParent = iChild, iChild = iParent * 2 + 2)
            {
                // If the next child is less than the current child, select the next one.
                if (iChild + 2 < Size && tKeys[iChild + 2] < tKeys[iChild])
                {
                    iChild += 2;
                }

                // If it is less than our parent swap.
                if (tKeys[iChild] < tKeys[iParent])
                {
                    Swap(iParent, iChild);

                    // Swap the pair if necessary.
                    if (iChild+1 < Size && tKeys[iChild+1] < tKeys[iChild])
                    {
                        Swap(iChild, iChild+1);
                    }
                }
                else
                {
                    break;
                }
            }
        }

        private void SiftDownMax(int iParent) {
            // For each child on the max side of the tree.
            for (int iChild = iParent * 2 + 1; iChild <= Size; iParent = iChild, iChild = iParent * 2 + 1)
            {
                // If the child is the last one (and only has half a pair).
                if (iChild == Size)
                {
                    // CHeck if we need to swap with th parent.
                    if (tKeys[iChild - 1] > tKeys[iParent])
                        Swap(iParent, iChild - 1);
                    break;
                }

                // If there is only room for a right child lower pair.
                else if (iChild + 2 == Size)
                {
                    // Swap the children.
                    if (tKeys[iChild + 1] > tKeys[iChild])
                    {
                        // Swap with the parent.
                        if (tKeys[iChild + 1] > tKeys[iParent])
                           Swap(iParent, iChild + 1);
                        break;
                    }
                }

                // 
                else if (iChild + 2 < Size)
                {
                    // If there is room for a right child upper pair
                    if (tKeys[iChild + 2] > tKeys[iChild])
                    {
                        iChild += 2;
                    }
                }
                if (tKeys[iChild] > tKeys[iParent])
                {
                    Swap(iParent, iChild);
                    // Swap with pair if necessary
                    if (tKeys[iChild-1] > tKeys[iChild])
                    {
                        Swap(iChild, iChild-1);
                    }
                }
                else
                {
                    break;
                }
            }
        }
    }

}
