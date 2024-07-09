using UnityEditor;
 
public class FBXImporterSettings : AssetPostprocessor {
    void OnPreprocessModel() {
        ModelImporter importer = assetImporter as ModelImporter;
        if(assetPath.Contains(".fbx")) {
            importer.animationCompression = ModelImporterAnimationCompression.Off;
        }
    }
}