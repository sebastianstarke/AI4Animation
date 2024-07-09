using AI4Animation;

namespace SIGGRAPH_2024 {
    public class Blueman {
        public const string RootName = "body_world";
        public const string HipsName = "b_root";
        public const string NeckName = "b_neck0";
        public const string HeadName = "b_head";
        public const string Spine0Name = "b_spine0";
        public const string Spine1Name = "b_spine1";
        public const string Spine2Name = "b_spine2";
        public const string SpineName = "b_spine3";
        public const string LeftShoulderName = "p_l_scap";
        public const string LeftArmName = "b_l_arm";
        public const string LeftElbowName = "b_l_forearm";
        public const string LeftWristTwistName = "b_l_wrist_twist";
        public const string LeftWristName = "b_l_wrist";
        public const string RightShoulderName = "p_r_scap";
        public const string RightArmName = "b_r_arm";
        public const string RightElbowName = "b_r_forearm";
        public const string RightWristTwistName = "b_r_wrist_twist";
        public const string RightWristName = "b_r_wrist";

        public const string LeftHipName = "b_l_upleg";
        public const string LeftKneeName = "b_l_leg";
        public const string LeftFootTwistName = "b_l_foot_twist";
        public const string LeftAnkleName = "b_l_talocrural";
        public const string LeftHeelName = "b_l_subtalar";
        public const string LeftToeName = "b_l_ball";

        public const string RightHipName = "b_r_upleg";
        public const string RightKneeName = "b_r_leg";
        public const string RightFootTwistName = "b_r_foot_twist";
        public const string RightAnkleName = "b_r_talocrural";
        public const string RightHeelName = "b_r_subtalar";
        public const string RightToeName = "b_r_ball";

        public static string[] FullBodyNames = new string[] {
            "b_root",
            "b_l_upleg",
            "b_l_leg",
            "b_l_talocrural",
            "b_l_ball",
            "b_r_upleg",
            "b_r_leg",
            "b_r_talocrural",
            "b_r_ball",
            "b_spine0",
            "b_spine1",
            "b_spine2",
            "b_spine3",
            "b_neck0",
            "b_head",
            "b_l_shoulder",
            "p_l_scap",
            "b_l_arm",
            "b_l_forearm",
            "b_l_wrist_twist",
            "b_l_wrist",
            "b_r_shoulder",
            "p_r_scap",
            "b_r_arm",
            "b_r_forearm",
            "b_r_wrist_twist",
            "b_r_wrist"
        };
        public static string[] LowerBodyNames = new string[] {
            "b_root",
            "b_l_upleg",
            "b_l_leg",
            "b_l_talocrural",
            "b_l_ball",
            "b_r_upleg",
            "b_r_leg",
            "b_r_talocrural",
            "b_r_ball"
        };
        public static string[] UpperBodyNames = new string[] {
            "b_root",
            "b_spine0",
            "b_spine1",
            "b_spine2",
            "b_spine3",
            "b_neck0",
            "b_head",
            "b_l_shoulder",
            "p_l_scap",
            "b_l_arm",
            "b_l_forearm",
            "b_l_wrist_twist",
            "b_l_wrist",
            "b_r_shoulder",
            "p_r_scap",
            "b_r_arm",
            "b_r_forearm",
            "b_r_wrist_twist",
            "b_r_wrist"
        };
        public static string[] TrackerNames = new string[] {
            "b_head",
            "b_l_wrist",
            "b_r_wrist"
        };

        public static int[] FullBodyIndices = null;
        public static int[] LowerBodyIndices = null;
        public static int[] UpperBodyIndices = null;
        public static int[] TrackerIndices = null;
        public static int HipsIndex = -1;
        public static int LeftHipIndex = -1;
        public static int RightHipIndex = -1;
        public static int HeadIndex = -1;
        public static int LeftWristIndex = -1;
        public static int RightWristIndex = -1;
        public static int LeftKneeIndex = -1;
        public static int RightKneeIndex = -1;
        public static int LeftAnkleIndex = -1;
        public static int RightAnkleIndex = -1;
        public static int LeftToeIndex = -1;
        public static int RightToeIndex = -1;

        #if UNITY_EDITOR
        public static void RegisterIndices(MotionAsset asset) {
            FullBodyIndices = asset.Source.GetBoneIndices(FullBodyNames);
            LowerBodyIndices = asset.Source.GetBoneIndices(LowerBodyNames);
            UpperBodyIndices = asset.Source.GetBoneIndices(UpperBodyNames);
            TrackerIndices = asset.Source.GetBoneIndices(TrackerNames);
            HipsIndex = asset.Source.GetBoneIndex(HipsName);
            LeftHipIndex = asset.Source.GetBoneIndex(LeftHipName);
            RightHipIndex = asset.Source.GetBoneIndex(RightHipName);
            HeadIndex = asset.Source.GetBoneIndex(HeadName);
            LeftWristIndex = asset.Source.GetBoneIndex(LeftWristName);
            RightWristIndex = asset.Source.GetBoneIndex(RightWristName);
            LeftKneeIndex = asset.Source.GetBoneIndex(LeftKneeName);
            RightKneeIndex = asset.Source.GetBoneIndex(RightKneeName);
            LeftAnkleIndex = asset.Source.GetBoneIndex(LeftAnkleName);
            RightAnkleIndex = asset.Source.GetBoneIndex(RightAnkleName);
            LeftToeIndex = asset.Source.GetBoneIndex(LeftToeName);
            RightToeIndex = asset.Source.GetBoneIndex(RightToeName);
        }
        #endif
    }
}
