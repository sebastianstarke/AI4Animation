// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using AI4Animation;

// // XR Rig Controller use the TrackingSystem to control the XR Rig for Gameplay logic
// public abstract class XRRigController : MonoBehaviour
// {
//     protected LiveTracking Device = null;

//     protected XRRig _xrRig = null;

//     public Camera GetCamera()
//     {
//         return _xrRig.HMDAnchor.GetComponent<Camera>();
//     }

//     public LiveTracking GetDevice()
//     {
//         return Device;
//     }

//     public XRRig XRRig
//     {
//         get { return _xrRig; }
//     }

//     public virtual void Setup(LiveTracking device, XRRig xrRig)
//     {
//         Device = device;
//         _xrRig = xrRig;
//     }

//     private void Update()
//     {
//         if (Initialized())
//         {
//             UpdateXRRig();
//         }
//     }

//     // Update the transformations of XR Rig anchors
//     protected abstract void UpdateXRRig();

//     protected bool Initialized()
//     {
//         return Device != null && _xrRig != null;
//     }

//     public virtual void SetEnabled(bool enabled)
//     {
//         if (GetCamera())
//             GetCamera().enabled = enabled;
//         if (_xrRig)
//             _xrRig.enabled = enabled;
//     }

//     // protected Vector3 GetDevicePosition(Matrix4x4 globalTransform, Vector3 deltaLocalPosition, Vector3 deltaLocalRotation)
//     // {
//     //     var deltaInv = Quaternion.Euler(deltaLocalRotation).GetInverse();
//     //     var rot = globalTransform.GetRotation() * deltaInv;
//     //     var worldSpace = Matrix4x4.TRS(globalTransform.GetPosition(), rot, Vector3.one);
//     //     return worldSpace.MultiplyPoint(-deltaLocalPosition);
//     // }
// }
