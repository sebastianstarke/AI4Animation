using System;
using UnityEngine;

namespace SerializedKDTree {

    public class KDNode : ScriptableObject {
        [SerializeReference] public KDTree KDTree;
        [SerializeReference] public Point[] Points;
        [SerializeReference] public Value[] Values;
        [SerializeReference] public KDNode pLeft, pRight;
        public int iSplitDimension;
        public float fSplitValue;
        public float[] tMinBound, tMaxBound;
        public bool SinglePoint;
        public int Size;

        [Serializable]
        public class Point {
            public float[] Values;
            public Point(float[] values) {
                Values = values;
            }
        }

        [Serializable]
        public abstract class Value {
            // [SerializeReference] public Point Point;
        }

        public KDNode Initialize(KDTree kdtree) {
            KDTree = kdtree;

            // Variables.
            Size = 0;
            SinglePoint = true;

            // Setup leaf elements.
            Points = new Point[KDTree.BucketCapacity+1];
            Values = new Value[KDTree.BucketCapacity+1];

            return this;
        }

        public bool IsLeaf() {
            return pLeft == null && pRight == null;
        }

        public void AddPoint(Point point, Value kValue) {
            // Find the correct leaf node.
            KDNode pCursor = this;
            while (!pCursor.IsLeaf())
            {
                // Extend the size of the leaf.
                pCursor.ExtendBounds(point.Values);
                pCursor.Size++;

                // If it is larger select the right, or lower,  select the left.
                if (point.Values[pCursor.iSplitDimension] > pCursor.fSplitValue)
                {
                    pCursor = pCursor.pRight;
                }
                else
                {
                    pCursor = pCursor.pLeft;
                }
            }

            // Insert it into the leaf.
            pCursor.AddLeafPoint(point, kValue);
        }

        private void AddLeafPoint(Point point, Value kValue) {
            if(!IsLeaf()) {
                AddPoint(point, kValue);
            } else {
                // Add the data point to this node.
                Points[Size] = point;
                Values[Size] = kValue;
                ExtendBounds(point.Values);
                Size++;

                // Split if the node is getting too large in terms of data.
                if (Size == Points.Length - 1)
                {
                    // If the node is getting too physically large.
                    if (CalculateSplit())
                    {
                        // If the node successfully had it's split value calculated, split node.
                        SplitLeafNode();
                    }
                    else
                    {
                        // If the node could not be split, enlarge node data capacity.
                        IncreaseLeafCapacity();
                    }
                }
            }
        }

        private bool CheckBounds(float[] tPoint) {
            for (int i = 0; i < KDTree.Dimensions; ++i)
            {
                if (tPoint[i] > tMaxBound[i]) return false;
                if (tPoint[i] < tMinBound[i]) return false;
            }
            return true;
        }

        private void ExtendBounds(float[] tPoint) {
            // If we don't have bounds, create them using the new point then bail.
            if (tMinBound == null) 
            {
                tMinBound = new float[KDTree.Dimensions];
                tMaxBound = new float[KDTree.Dimensions];
                Array.Copy(tPoint, tMinBound, KDTree.Dimensions);
                Array.Copy(tPoint, tMaxBound, KDTree.Dimensions);
                return;
            }

            // For each dimension.
            for (int i = 0; i < KDTree.Dimensions; ++i)
            {
                if (Single.IsNaN(tPoint[i]))
                {
                    if (!Single.IsNaN(tMinBound[i]) || !Single.IsNaN(tMaxBound[i]))
                        SinglePoint = false;
                    
                    tMinBound[i] = Single.NaN;
                    tMaxBound[i] = Single.NaN;
                }
                else if (tMinBound[i] > tPoint[i])
                {
                    tMinBound[i] = tPoint[i];
                    SinglePoint = false;
                }
                else if (tMaxBound[i] < tPoint[i])
                {
                    tMaxBound[i] = tPoint[i];
                    SinglePoint = false;
                }
            }
        }

        private void IncreaseLeafCapacity() {
            Array.Resize<Point>(ref Points, Points.Length * 2);
            Array.Resize<Value>(ref Values, Values.Length * 2);
        }

        private bool CalculateSplit() {
            // Don't split if we are just one point.
            if (SinglePoint)
                return false;

            // Find the dimension with the largest range.  This will be our split dimension.
            float fWidth = 0;
            for (int i = 0; i < KDTree.Dimensions; i++)
            {
                float fDelta = (tMaxBound[i] - tMinBound[i]);
                if (Single.IsNaN(fDelta))
                    fDelta = 0;

                if (fDelta > fWidth)
                {
                    iSplitDimension = i;
                    fWidth = fDelta;
                }
            }

            // If we are not wide (i.e. all the points are in one place), don't split.
            if (fWidth == 0)
                return false;

            // Split in the middle of the node along the widest dimension.
            fSplitValue = (tMinBound[iSplitDimension] + tMaxBound[iSplitDimension]) * 0.5f;

            // Never split on infinity or NaN.
            if (fSplitValue == Single.PositiveInfinity) {
                fSplitValue = Single.MaxValue;
            } else if (fSplitValue == Single.NegativeInfinity) {
                fSplitValue = Single.MinValue;
            }
            
            // Don't let the split value be the same as the upper value as
            // can happen due to rounding errors!
            if (fSplitValue == tMaxBound[iSplitDimension]) {
                fSplitValue = tMinBound[iSplitDimension];
            }

            // Success
            return true;
        }

        private void SplitLeafNode() {
            // Create the new children.
            pRight = ScriptableObjectExtensions.Create<KDNode>(this).Initialize(KDTree);
            pLeft = ScriptableObjectExtensions.Create<KDNode>(this).Initialize(KDTree);

            // Move each item in this leaf into the children.
            for (int i = 0; i < Size; ++i)
            {
                // Store.
                Point tOldPoint = Points[i];
                Value kOldData = Values[i];

                // If larger, put it in the right.
                if (tOldPoint.Values[iSplitDimension] > fSplitValue)
                    pRight.AddLeafPoint(tOldPoint, kOldData);

                // If smaller, put it in the left.
                else
                    pLeft.AddLeafPoint(tOldPoint, kOldData);
            }

            // Wipe the data from this KDNode.
            Points = null;
            Values = null;
        }
    }

}
