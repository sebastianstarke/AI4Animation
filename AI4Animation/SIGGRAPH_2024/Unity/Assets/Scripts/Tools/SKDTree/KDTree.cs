using System;
using System.Collections.Generic;
using UnityEngine;

namespace SKDTree {
    
    public class KDTree<T>: ScriptableObject {

        [Serializable]
        public class Sample {
            public float[] Key;
            public T Value;
            public Sample(float[] key, T value) {
                Key = key;
                Value = value;
            }
        }

        public List<Sample> Samples = new List<Sample>();

        private Datastructure Tree = null;

        private double SearchTime = 0.0;
        
        public void AddSample(float[] vector, T value) {
            Samples.Add(new Sample(vector, value));
        }

        public void Build(bool normalize=true, string[] groups=null, int bucketCapacity=24, int[] discretization=null) {
            Tree = new Datastructure(Samples.ToArray(), normalize, groups, bucketCapacity, discretization);
        }

        public bool IsInitialized() {
            return Tree != null;
        }

        public int GetDimensionality() {
            return Tree.GetDimensionality();
        }

        public T[] Query(float[] vector, int neighbors, int discretization=0, float[] distances=null) {
            if(Tree == null) {
                Debug.Log("KD tree has not been created. Call Build() first during runtime.");
                return null;
            }
            DateTime timestamp = Utility.GetTimestamp();
            NearestNeighbour<T> results = Tree.NearestNeighbors(vector, neighbors, discretization);
            List<T> values = new List<T>();
            while(results.MoveNext()) {
                values.Add(results.Current);
                if(distances != null) {
                    distances[values.Count-1] = results.CurrentDistance;
                }
            }
            SearchTime = Utility.GetElapsedTime(timestamp);
            return values.ToArray();
        }

        public double GetSearchTime() {
            return SearchTime;
        }

        public class Datastructure {
            private KDNode<T>[] Root;
            private int Dimensions;
            private float[] Mean;
            private float[] Sigma;

            public Datastructure(Sample[] samples, bool normalize=true, string[] groups=null, int bucketCapacity=24, int[] discretization=null) {
                Dimensions = samples.First().Key.Length;
                Debug.Log("Building KD Tree with " + Dimensions + " dimensions and " + samples.Length + " samples.");
                Mean = new float[Dimensions];
                Sigma = new float[Dimensions];
                if(normalize) {
                    RunningStatistics[] means = new RunningStatistics[Dimensions];
                    RunningStatistics[] sigmas = new RunningStatistics[Dimensions];
                    for(int i=0; i<Dimensions; i++) {
                        means[i] = new RunningStatistics();
                        sigmas[i] = new RunningStatistics();
                    }

                    if(groups == null) {
                        for(int i=0; i<samples.Length; i++) {
                            for(int j=0; j<Dimensions; j++) {
                                means[j].Add(samples[i].Key[j]);
                                sigmas[j].Add(samples[i].Key[j]);
                            }
                        }
                    } else {
                        for(int i=0; i<samples.Length; i++) {
                            for(int j=0; j<Dimensions; j++) {
                                means[j].Add(samples[i].Key[j]);
                            }
                        }
                        for(int j=0; j<Dimensions; j++) {
                            for(int k=0; k<groups.Length; k++) {
                                if(groups[k] == groups[j]) {
                                    for(int i=0; i<samples.Length; i++) {
                                        sigmas[k].Add(samples[i].Key[j]);
                                    }
                                }
                            }

                        }
                    }

                    // for(int i=0; i<samples.Length; i++) {
                    //     for(int j=0; j<Dimensions; j++) {
                    //         means[j].Add(samples[i].Key[j]);
                    //         if(groups == null) {
                    //             // means[j].Add(samples[i].Key[j]);
                    //             sigmas[j].Add(samples[i].Key[j]);
                    //         } else {
                    //             for(int k=0; k<groups.Length; k++) {
                    //                 if(groups[k] == groups[j]) {
                    //                     // means[k].Add(samples[i].Key[j]);
                    //                     sigmas[k].Add(samples[i].Key[j]);
                    //                 }
                    //             }
                    //         }
                    //     }
                    // }
                    for(int i=0; i<means.Length; i++) {
                        Mean[i] = means[i].Mean();
                    }
                    for(int i=0; i<sigmas.Length; i++) {
                        Sigma[i] = sigmas[i].Sigma();
                    }
                } else {
                    Mean.SetAll(0f);
                    Sigma.SetAll(1f);
                }

                if(discretization == null) {
                    Root = new KDNode<T>[1];
                    Root[0] = new KDNode<T>(Dimensions, bucketCapacity);
                    for(int i=0; i<samples.Length; i++) {
                        Root[0].AddPoint(Normalize(samples[i].Key), samples[i].Value);
                    }
                } else {
                    Debug.Log("Discretizing KD Tree with " + discretization + " clusters.");
                    Root = new KDNode<T>[discretization.Max()+1];
                    for(int i=0; i<Root.Length; i++) {
                        Root[i] = new KDNode<T>(Dimensions, bucketCapacity);
                    }
                    for(int i=0; i<samples.Length; i++) {
                        Root[discretization[i]].AddPoint(Normalize(samples[i].Key), samples[i].Value);
                    }
                }
            }

            public int GetDimensionality() {
                return Dimensions;
            }

            private float[] Normalize(float[] x) {
                if(x.Length != Mean.Length) {
                    Debug.Log("Feature: " + x.Length + " / " + Mean.Length);
                }
                float[] y = new float[x.Length];
                for(int i=0; i<x.Length; i++) {
                    y[i] = (x[i] - Mean[i]) / Sigma[i];
                }
                return y;
            }

            /// <summary>
            /// Get the nearest neighbours to a point in the kd tree using a user defined distance function.
            /// </summary>
            /// <param name="tSearchPoint">The point of interest.</param>
            /// <param name="iMaxReturned">The maximum number of points which can be returned by the iterator.</param>
            /// <param name="kDistanceFunction">The distance function to use.</param>
            /// <param name="fDistance">A threshold distance to apply. Optional. Negative values mean that it is not applied.</param>
            /// <returns>A new nearest neighbour iterator with the given parameters.</returns>
            public NearestNeighbour<T> NearestNeighbors(float[] tSearchPoint, int iMaxReturned, int discretization) {
                return new NearestNeighbour<T>(Root[discretization], Normalize(tSearchPoint), new SquareEuclideanDistanceFunction(), iMaxReturned, -1f);
            }
        }
    }

}