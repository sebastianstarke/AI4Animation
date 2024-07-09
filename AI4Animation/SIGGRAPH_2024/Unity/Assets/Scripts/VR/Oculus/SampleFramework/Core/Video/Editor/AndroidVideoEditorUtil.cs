// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections.Generic;
using UnityEditor;
using System.IO;

public class AndroidVideoEditorUtil
{
    private static readonly string videoPlayerFileName = "Assets/Oculus/SampleFramework/Core/Video/Plugins/Android/java/com/oculus/videoplayer/NativeVideoPlayer.java";
    private static readonly string audio360PluginPath = "Assets/Oculus/SampleFramework/Core/Video/Plugins/Android/Audio360/audio360.aar";
    private static readonly string audio360Exo218PluginPath = "Assets/Oculus/SampleFramework/Core/Video/Plugins/Android/Audio360/audio360-exo218.aar";

    [MenuItem("Oculus/Samples/Video/Enable Native Android Video Player")]
    public static void EnableNativeVideoPlayer()
    {
        // Enable native video player
        PluginImporter nativeVideoPlayer = (PluginImporter)AssetImporter.GetAtPath(videoPlayerFileName);
        if (nativeVideoPlayer != null)
        {
            nativeVideoPlayer.SetCompatibleWithPlatform(BuildTarget.Android, true);
            nativeVideoPlayer.SaveAndReimport();
        }

        // Enable audio plugins
        PluginImporter audio360 = (PluginImporter)AssetImporter.GetAtPath(audio360PluginPath);
        PluginImporter audio360exo218 = (PluginImporter)AssetImporter.GetAtPath(audio360Exo218PluginPath);

        if (audio360 != null && audio360exo218 != null)
        {
            audio360.SetCompatibleWithPlatform(BuildTarget.Android, true);
            audio360exo218.SetCompatibleWithPlatform(BuildTarget.Android, true);
            audio360.SaveAndReimport();
            audio360exo218.SaveAndReimport();
        }

        // Enable gradle build with exoplayer
        Gradle.Configuration.UseGradle();
        var template = Gradle.Configuration.OpenTemplate();
        template.AddDependency("com.google.android.exoplayer:exoplayer", "2.18.2");
        template.Save();

        var properties = Gradle.Configuration.OpenProperties();
        properties.SetProperty("android.useAndroidX", "true");
        properties.Save();

        // Set the API level to 31
        PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions) 31;
    }

    [MenuItem("Oculus/Samples/Video/Disable Native Android Video Player")]
    public static void DisableNativeVideoPlayer()
    {
        // Disable native video player
        PluginImporter nativeVideoPlayer = (PluginImporter)AssetImporter.GetAtPath(videoPlayerFileName);
        if (nativeVideoPlayer != null)
        {
            nativeVideoPlayer.SetCompatibleWithPlatform(BuildTarget.Android, false);
            nativeVideoPlayer.SaveAndReimport();
        }

        // Disable audio plugins
        PluginImporter audio360 = (PluginImporter)AssetImporter.GetAtPath(audio360PluginPath);
        PluginImporter audio360exo218 = (PluginImporter)AssetImporter.GetAtPath(audio360Exo218PluginPath);

        if (audio360 != null && audio360exo218 != null)
        {
            audio360.SetCompatibleWithPlatform(BuildTarget.Android, false);
            audio360exo218.SetCompatibleWithPlatform(BuildTarget.Android, false);
            audio360.SaveAndReimport();
            audio360exo218.SaveAndReimport();
        }

        // remove exoplayer and sourcesets from gradle file (leave other parts since they are harmless).
        if (Gradle.Configuration.IsUsingGradle())
        {
            var template = Gradle.Configuration.OpenTemplate();
            template.RemoveDependency("com.google.android.exoplayer:exoplayer");
            template.RemoveAndroidSetting("sourceSets.main.java.srcDir");
            template.Save();
        }

    }
}
