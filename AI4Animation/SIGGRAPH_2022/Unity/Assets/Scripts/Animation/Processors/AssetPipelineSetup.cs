#if UNITY_EDITOR
using System;
using System.Collections;
using UnityEngine;

namespace AI4Animation {
    public abstract class AssetPipelineSetup : ScriptableObject {
        [NonSerialized] public AssetPipeline Pipeline;
        public abstract void Inspector();
        public abstract void Inspector(AssetPipeline.Item item);
        public abstract bool CanProcess();
        public abstract void Begin();
        public abstract void Callback();
        public abstract void Finish();
        public abstract IEnumerator Iterate(MotionAsset asset);
    }
}
#endif