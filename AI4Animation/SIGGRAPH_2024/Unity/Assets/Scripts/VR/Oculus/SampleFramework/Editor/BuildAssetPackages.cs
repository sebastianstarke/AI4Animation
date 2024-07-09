using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class BuildAssetPackages
{
    enum BuildConfiguration
    {
        Windows,
        Android
    }

    public static void Build()
    {
        Debug.Log("Building Deliverables");
        ExportPackages();
    }

    public static void ExportPackages()
    {
        string[] assets = AssetDatabase.FindAssets("t:Object", null).Select(s=>AssetDatabase.GUIDToAssetPath(s)).ToArray();
        assets = assets.Where(s=>
            s.StartsWith("Assets/Oculus/AudioManager/") ||
            s.StartsWith("Assets/Oculus/LipSync/") ||
            s.StartsWith("Assets/Oculus/Platform/") ||
            s.StartsWith("Assets/Oculus/Spatializer/") ||
            s.StartsWith("Assets/Oculus/Voice/") ||
            s.StartsWith("Assets/Oculus/Interaction/") ||
            s.StartsWith("Assets/Oculus/VoiceMod/") ||
            s.StartsWith("Assets/Oculus/VR/") ||
            s.StartsWith("Assets/Oculus/SampleFramework/")
        ).ToArray();
        AssetDatabase.ExportPackage(assets, "OculusIntegration.unitypackage");
    }
}
