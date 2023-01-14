using System;
using System.Collections;
using System.Collections.Generic;

namespace KDTree
{
    /// <summary>
    /// A MinHeap is a smallest-first queue based around a binary heap so it is quick to insert / remove items.
    /// </summary>
    /// <typeparam name="T">The type of data this MinHeap stores.</typeparam>
    /// <remarks>This is based on this: https://bitbucket.org/rednaxela/knn-benchmark/src/tip/ags/utils/dataStructures/trees/thirdGenKD/ </remarks>
    public class MinHeap<T>
    {
        /// <summary>
        /// The default size for a min heap.
        /// </summary>
        private static int DEFAULT_SIZE = 64;

        /// <summary>
        /// The data array.  This stores the data items in the heap.
        /// </summary>
        private T[] tData;

        /// <summary>
        /// The key array.  This determines how items are ordered. Smallest first.
        /// </summary>
        private double[] tKeys;

        /// <summary>
        /// Create a new min heap with the default capacity.
        /// </summary>
        public MinHeap() : this(DEFAULT_SIZE)
        {
        }

        /// <summary>
        /// Create a new min heap with a given capacity.
        /// </summary>
        /// <param name="iCapacity"></param>
        public MinHeap(int iCapacity)
        {
            this.tData = new T[iCapacity];
            this.tKeys = new double[iCapacity];
            this.Capacity = iCapacity;
            this.Size = 0;
        }

        /// <summary>
        /// The number of items in this queue.
        /// </summary>
        public int Size { get; private set; }

        /// <summary>
        /// The amount of space in this queue.
        /// </summary>
        public int Capacity { get; private set; }

        /// <summary>
        /// Insert a new element.
        /// </summary>
        /// <param name="key">The key which represents its position in the priority queue (ie. distance).</param>
        /// <param name="value">The value to be stored at the key.</param>
        public void Insert(double key, T value)
        {
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
                var newKeys = new double[Capacity];
                Array.Copy(tKeys, newKeys, tKeys.Length);
                tKeys = newKeys;
            }

            // Insert new value at the end
            tData[Size] = value;
            tKeys[Size] = key;
            SiftUp(Size);
            Size++;
        }

        /// <summary>
        /// Remove the smallest element.
        /// </summary>
        public void RemoveMin()
        {
            if (Size == 0)
                throw new Exception();

            Size--;
            tData[0] = tData[Size];
            tKeys[0] = tKeys[Size];
            tData[Size] = default(T);
            SiftDown(0);
        }

        /// <summary>
        /// Get the data stored at the minimum element.
        /// </summary>
        public T Min
        {
            get
            {
                if (Size == 0)
                    throw new Exception();

                return tData[0];
            }
        }

        /// <summary>
        /// Get the key which represents the minimum element.
        /// </summary>
        public double MinKey
        {
            get
            {
                if (Size == 0)
                    throw new Exception();

                return tKeys[0];
            }
        }

        /// <summary>
        /// Bubble a child item up the tree.
        /// </summary>
        /// <param name="iChild"></param>
        private void SiftUp(int iChild)
        {
            // For each parent above the child, if the parent is smaller then bubble it up.
            for (int iParent = (iChild - 1) / 2; 
                iChild != 0 && tKeys[iChild] < tKeys[iParent]; 
                iChild = iParent, iParent = (iChild - 1) / 2)
            {
                T kData = tData[iParent];
                double dDist = tKeys[iParent];

                tData[iParent] = tData[iChild];
                tKeys[iParent] = tKeys[iChild];

                tData[iChild] = kData;
                tKeys[iChild] = dDist;
            }
        }

        /// <summary>
        /// Bubble a parent down through the children so it goes in the right place.
        /// </summary>
        /// <param name="iParent">The index of the parent.</param>
        private void SiftDown(int iParent)
        {
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
                    double pDist = tKeys[iParent];

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
}