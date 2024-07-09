using UnityEditor;

public class FirebaseEnabler
{
    private static readonly string FirebaseBuildDefinition = "OVR_SAMPLES_ENABLE_FIREBASE";

    [MenuItem("Oculus/Samples/Firebase/Enable Firebase Sample")]
    public static void EnableFirebaseSample()
    {
        var defineString = PlayerSettings.GetScriptingDefineSymbolsForGroup(BuildTargetGroup.Android);
        PlayerSettings.SetScriptingDefineSymbolsForGroup(BuildTargetGroup.Android, $"{defineString};{FirebaseBuildDefinition}");
    }

    [MenuItem("Oculus/Samples/Firebase/Disable Firebase Sample")]
    public static void DisableFirebaseSample()
    {
        var defineString = PlayerSettings.GetScriptingDefineSymbolsForGroup(BuildTargetGroup.Android);
        var defines = defineString.Split(new char[] { ';' }, System.StringSplitOptions.RemoveEmptyEntries);
        var filtered = System.Array.FindAll(defines, d => d != FirebaseBuildDefinition);
        PlayerSettings.SetScriptingDefineSymbolsForGroup(BuildTargetGroup.Android, string.Join(";", filtered));
    }
}
