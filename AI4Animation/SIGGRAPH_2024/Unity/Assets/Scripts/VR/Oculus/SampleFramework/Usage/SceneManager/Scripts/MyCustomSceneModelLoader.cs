using System.Collections;
using UnityEngine;

public class MyCustomSceneModelLoader : OVRSceneModelLoader
{
  IEnumerator DelayedLoad()
  {
    yield return new WaitForSeconds(1.0f);
    Debug.Log("[MyCustomSceneLoader] calling OVRSceneManager.LoadSceneModel() delayed by 1 second");
    SceneManager.LoadSceneModel();
  }

  protected override void OnStart()
  {
    // Don't load immediately, wait some time
    StartCoroutine(DelayedLoad());
  }

  protected override void OnNoSceneModelToLoad()
  {
    // Don't trigger capture flow in case there is no scene, just log a message
    Debug.Log("[MyCustomSceneLoader] There is no scene to load, but we don't want to trigger scene capture.");
  }
}
