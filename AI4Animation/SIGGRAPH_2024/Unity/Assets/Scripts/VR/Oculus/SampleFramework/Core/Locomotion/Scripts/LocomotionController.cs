/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using System.Collections;
using JetBrains.Annotations;
using UnityEngine.Assertions;
#if UNITY_EDITOR
using UnityEngine.SceneManagement;
#endif

/// <summary>
/// Simply aggregates accessors.
/// </summary>
public class LocomotionController : MonoBehaviour
{
    public OVRCameraRig CameraRig;
    //public CharacterController CharacterController;
    public CapsuleCollider CharacterController;
	//public OVRPlayerController PlayerController;
	public SimpleCapsuleWithStickMovement PlayerController;

    void Start()
    {
		/*
        if (CharacterController == null)
        {
            CharacterController = GetComponentInParent<CharacterController>();
        }
        Assert.IsNotNull(CharacterController);
		*/
        //if (PlayerController == null)
        //{
            //PlayerController = GetComponentInParent<OVRPlayerController>();
        //}
        //Assert.IsNotNull(PlayerController);
        if(CameraRig == null)
        {
            CameraRig = FindObjectOfType<OVRCameraRig>();
        }
        Assert.IsNotNull(CameraRig);
#if UNITY_EDITOR
        OVRPlugin.SendEvent("locomotion_controller", (SceneManager.GetActiveScene().name == "Locomotion").ToString(), "sample_framework");
#endif
	}
}
