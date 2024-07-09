#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections;
using AI4Animation;

namespace SIGGRAPH_2024 {
    public class ProcessingPipeline : AssetPipelineSetup {

        public enum MODE {Blueman};
        public MODE Mode = MODE.Blueman;

        public override void Inspector() {
            Mode = (MODE)EditorGUILayout.EnumPopup("Mode", Mode);
        }
        
        public override void Inspector(AssetPipeline.Item item) {

        }

        public override bool CanProcess() {
            return true;
        }

        public override void Begin() {

        }

        public override IEnumerator Iterate(MotionAsset asset) {
            Debug.Log("Asset: " + asset.name);
            Pipeline.GetEditor().AutoSave = false;
            Pipeline.GetEditor().LoadSession(Utility.GetAssetGUID(asset));
            switch(Mode) {
                case MODE.Blueman:
                ProcessBlueman(asset);
                break;
            }
            yield return new WaitForSeconds(0f);
            Pipeline.GetEditor().AutoSave = true;
        }

        public override void Callback() {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Resources.UnloadUnusedAssets();
        }

        public override void Finish() {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Resources.UnloadUnusedAssets();
        }

        private void ProcessBlueman(MotionAsset asset) {
            asset.RemoveAllModules<RootModule>();

            asset.MirrorAxis = Axis.ZPositive;
            asset.Model = "Character";
            asset.Scale = 1.0f;

            asset.DetectSymmetry();
            asset.Source.Bones[0].Correction = new Vector3(0f, 180f, 0f);
            for(int i=1; i<asset.Source.Bones.Length; i++) {
                if(asset.Symmetry[i] != i) {
                    asset.Source.Bones[i].Correction = new Vector3(0f, 0f, 180f);
                } else {
                    asset.Source.Bones[i].Correction = Vector3.zero;
                }
            }

            {
                RootModule module = asset.AddOrGetModule<RootModule>("BodyWorld");
                module.Primary = true;
                module.Topology = RootModule.TOPOLOGY.Bone;
                module.Ground = 0;
                module.SmoothPositions = false;
                module.SmoothRotations = false;
                module.SmoothingWindow = 0f;
                module.LockThreshold = 0.1f;
                module.RootBone = asset.Source.GetBoneIndex("body_world");
                module.Hips = asset.Source.GetBoneIndex(Blueman.HipsName);
                module.Neck = asset.Source.GetBoneIndex(Blueman.NeckName);
                module.LeftHip = asset.Source.GetBoneIndex(Blueman.LeftHipName);
                module.RightHip = asset.Source.GetBoneIndex(Blueman.RightHipName);
                module.LeftShoulder = asset.Source.GetBoneIndex(Blueman.LeftShoulderName);
                module.RightShoulder = asset.Source.GetBoneIndex(Blueman.RightShoulderName);
            }

            {
                ContactModule module = asset.AddOrGetModule<ContactModule>();
                module.Clear();

                module.AddSensor("b_l_talocrural", Vector3.zero, Vector3.zero, 0.1f*Vector3.one, 0.25f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("b_l_ball", Vector3.zero, Vector3.zero, 0.05f*Vector3.one, 0.25f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                
                module.AddSensor("b_r_talocrural", Vector3.zero, Vector3.zero, 0.1f*Vector3.one, 0.25f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("b_r_ball", Vector3.zero, Vector3.zero, 0.05f*Vector3.one, 0.25f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                
                module.CaptureContacts(Pipeline.GetEditor());
            }

            {
                asset.RemoveModule<StyleModule>();
                StyleModule module = asset.AddOrGetModule<StyleModule>();
                module.Clear();
                StyleModule.Function function = module.AddFunction("Jump");
                int head = asset.Source.GetBoneIndex(Blueman.HeadName);
                int hip = asset.Source.GetBoneIndex(Blueman.HipsName);
                int leftFoot = asset.Source.GetBoneIndex(Blueman.LeftAnkleName);
                int rightFoot = asset.Source.GetBoneIndex(Blueman.RightAnkleName);
                for(int i=0; i<asset.Frames.Length; i++) {
                    function.StandardValues[i] = module.EstimateJumping(asset.Frames[i].Timestamp, false, head, hip, leftFoot, rightFoot, 1.6f, 0.95f, 0.1f) ? 1f : 0f;
                    function.MirroredValues[i] = module.EstimateJumping(asset.Frames[i].Timestamp, true, head, hip, leftFoot, rightFoot, 1.6f, 0.95f, 0.1f) ? 1f : 0f;
                }
            }

            {
                MotionModule module = asset.AddOrGetModule<MotionModule>();
            }

            {
                AvatarModule module = asset.AddOrGetModule<AvatarModule>();
                module.Head = asset.Source.FindBone(Blueman.HeadName).Index;
                module.Offset = new Vector3(-0.1f, 0f, 0f);
                module.View = Axis.YPositive;
                module.FOV = 100f;
            }

            asset.MarkDirty(true, false);
        }
    }
}
#endif