// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using AI4Animation;

// // An XR Rig that controls where the HMD is, and update the controller transformations accordingly.
// public abstract class ManualXRRigController : XRRigController
// {
//     protected override void UpdateXRRig()
//     {
//         UpdateHMD();
//         UpdateControllers(); 
//     }

//     protected abstract void UpdateHMD();

//     protected void UpdateControllers()
//     {
//         Transform camera = _xrRig.HMDAnchor.transform;

//         Matrix4x4 GetRawTransformation(LiveTracking.TRACKER_TYPE type)
//         {
//             var tracker = Device.GetTracker(type);
//             if(tracker == null)
//                 return Matrix4x4.identity;

//             var liveTracker = tracker as LiveTracking.LiveTracker;
//             if(liveTracker != null)
//             {
//                 return Matrix4x4.TRS(liveTracker.GetRawPosition(), liveTracker.GetRawRotation(), Vector3.one);
//             }
//             return tracker.GetTransformation();
//         }        
        
//         var hmd = GetRawTransformation(LiveTracking.TRACKER_TYPE.HEAD);
//         var rightController = GetRawTransformation(LiveTracking.TRACKER_TYPE.RIGHT_WRIST);
//         var leftController = GetRawTransformation(LiveTracking.TRACKER_TYPE.LEFT_WRIST);
//         var realRightController = camera.GetGlobalMatrix() * hmd.inverse * rightController;
//         var realLeftController = camera.GetGlobalMatrix() * hmd.inverse * leftController;
//         _xrRig.RightControllerAnchor.position = realRightController.GetPosition();
//         _xrRig.RightControllerAnchor.rotation = realRightController.GetRotation();
//         _xrRig.LeftControllerAnchor.position = realLeftController.GetPosition();
//         _xrRig.LeftControllerAnchor.rotation = realLeftController.GetRotation();
//     }

// }
