using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KDTree
{
    /// <summary>
    /// A binary interval heap is double-ended priority queue is a priority queue that it allows
    /// for efficient removal of both the maximum and minimum element.
    /// </summary>
    /// <typeparam name="T">The data type contained at each key.</typeparam>
    /// <remarks>This is based on this: https://bitbucket.org/rednaxela/knn-benchmark/src/tip/ags/utils/dataStructures/trees/thirdGenKD/ </remarks>
    public class IntervalHeap<T>
    {
        /// <summary>
        /// The default size for a new interval heap.
        /// </summary>
        private const int DEFAULT_SIZE = 64;

        /// <summary>
        /// The internal data array which contains the stored objects.
        /// </summary>
        private T[] tData;

        /// <summary>
        /// The array of keys which 
        /// </summary>
        private double[] tKeys;

        /// <summary>
        /// Construct a new interval heap with the default capacity.
        /// </summary>
        public IntervalHeap() : this(DEFAULT_SIZE)
        {
        }

        /// <summary>
        /// Construct a new interval heap with a custom capacity.
        /// </summary>
        /// <param name="capacity"></param>
        public IntervalHeap(int capacity)
        {
            this.tData = new T[capacity];
            this.tKeys = new double[capacity];
            this.Capacity = capacity;
            this.Size = 0;
        }

        /// <summary>
        /// The number of items in this interval heap.
        /// </summary>
        public int Size { get; private set; }

        /// <summary>
        /// The current capacity of this interval heap.
        /// </summary>
        public int Capacity { get; private set; }

        /// <summary>
        /// Get the data with the smallest key.
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
        /// Get the data with the largest key.
        /// </summary>
        public T Max
        {
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

        /// <summary>
        /// Get the smallest key.
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
        /// Get the largest key.
        /// </summary>
        public double MaxKey
        {
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

        /// <summary>
        /// Insert a new data item at a given key.
        /// </summary>
        /// <param name="key">The value which represents our data (i.e. a distance).</param>
        /// <param name="value">The data we want to store.</param>
        public void Insert(double key, T value)
        {
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
                var newKeys = new double[Capacity];
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

        /// <summary>
        /// Remove the item with the smallest key from the queue.
        /// </summary>
        public void RemoveMin()
        {
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

        /// <summary>
        /// Replace the item with the smallest key in the queue.
        /// </summary>
        /// <param name="key">The new minimum key.</param>
        /// <param name="value">The new minumum data value.</param>
        public void ReplaceMin(double key, T value)
        {
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

        /// <summary>
        /// Remove the item with the largest key in the queue.
        /// </summary>
        public void RemoveMax()
        {
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

        /// <summary>
        /// Swap out the item with the largest key in the queue.
        /// </summary>
        /// <param name="key">The new key for the largest item.</param>
        /// <param name="value">The new data for the largest item.</param>
        public void ReplaceMax(double key, T value)
        {
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


        /// <summary>
        /// Internal helper method which swaps two values in the arrays.
        /// This swaps both data and key entries.
        /// </summary>
        /// <param name="x">The first index.</param>
        /// <param name="y">The second index.</param>
        /// <returns>The second index.</returns>
        private int Swap(int x, int y)
        {
            // Store temp.
            T yData = tData[y];
            double yDist = tKeys[y];

            // Swap
            tData[y] = tData[x];
            tKeys[y] = tKeys[x];
            tData[x] = yData;
            tKeys[x] = yDist;

            // Return.
            return y;
        }

        /**
         * Min-side (u % 2 == 0):
         * - leftchild:  2u + 2
         * - rightchild: 2u + 4
         * - parent:     (x/2-1)&~1
         *
         * Max-side (u % 2 == 1):
         * - leftchild:  2u + 1
         * - rightchild: 2u + 3
         * - parent:     (x/2-1)|1
         */

        /// <summary>
        /// Place a newly inserted element a into the correct tree position.
        /// </summary>
        private void SiftInsertedValueUp()
        {
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

        /// <summary>
        /// Bubble elements up the min side of the tree.
        /// </summary>
        /// <param name="iChild">The child index.</param>
        private void SiftUpMin(int iChild)
        {
            // Min-side parent: (x/2-1)&~1
            for (int iParent = (iChild/2-1)&~1; 
                iParent >= 0 && tKeys[iChild] < tKeys[iParent]; 
                iChild = iParent, iParent = (iChild/2-1)&~1)
            {
                Swap(iChild, iParent);
            }
        }

        /// <summary>
        /// Bubble elements up the max side of the tree.
        /// </summary>
        /// <param name="iChild">The child index.</param>
        private void SiftUpMax(int iChild)
        {
            // Max-side parent: (x/2-1)|1
            for (int iParent = (iChild/2-1)|1; 
                iParent >= 0 && tKeys[iChild] > tKeys[iParent]; 
                iChild = iParent, iParent = (iChild/2-1)|1)
            {
                Swap(iChild, iParent);
            }
        }

        /// <summary>
        /// Bubble elements down the min side of the tree.
        /// </summary>
        /// <param name="iParent">The parent index.</param>
        private void SiftDownMin(int iParent)
        {
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

        /// <summary>
        /// Bubble elements down the max side of the tree.
        /// </summary>
        /// <param name="iParent"></param>
        private void SiftDownMax(int iParent)
        {
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