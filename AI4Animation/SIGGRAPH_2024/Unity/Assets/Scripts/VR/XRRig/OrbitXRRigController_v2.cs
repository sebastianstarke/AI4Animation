// using System.Collections;
// using System.Collections.Generic;
// using AI4Animation;
// using UnityEngine;

// public class OrbitXRRigController_v2 : OrbitXRRigController_v1
// {
//     private Quaternion OffsetRotation = Quaternion.identity;

//     private Quaternion GetHMDRotation()
//     {
//         var headTracker = Device.GetTracker(LiveTracking.TRACKER_TYPE.HEAD);
//         var liveHmd = headTracker as LiveTracking.LiveTracker;
//         return liveHmd == null ? headTracker.GetTransformation().GetRotation() : liveHmd.GetRawRotation();
//     }

//     protected override void UpdateHMD()
//     {
//         if(!LockFocus()) {
//             Transform camera = _xrRig.HMDAnchor;
//             camera.rotation = OffsetRotation * GetHMDRotation();
//         }
//         else
//         {
//             LookAtCharacter();
//         }
//     }

//     void LookAtCharacter()
//     {
//         Transform camera = _xrRig.HMDAnchor;
//         Matrix4x4 head = Device.GetTransformation(LiveTracking.TRACKER_TYPE.HEAD);

//         //Save previous coordinates
//         Vector3 previousPosition = camera.position;
//         Quaternion previousRotation = camera.rotation;

//         //Positioning the camera around the target and focusing at it
//         Vector3 selfOffset = SelfOffset;
//         Vector3 targetOffset = TargetOffset + new Vector3(0f, head.GetPosition().y, 0f);
//         Vector3 pivot = Character.transform.position + Character.transform.rotation * targetOffset;
//         camera.position = Character.transform.position + Character.transform.rotation * (Mathf.Abs(Orbiting.y) * selfOffset);
//         camera.rotation = head.GetRotation();
//         camera.RotateAround(pivot, Vector3.up, -Orbiting.x);
//         camera.LookAt(pivot);

//         //Lerp camera from previous to new coordinates
//         camera.position = Vector3.Lerp(previousPosition, camera.position, 1f-Damping);
//         camera.rotation = Quaternion.Slerp(previousRotation, camera.rotation, 1f-Damping);


//         OffsetRotation = camera.rotation * GetHMDRotation().GetInverse();
//     }

//     private bool LockFocus()
//     {
//         var tracker = Device.GetTracker(LiveTracking.TRACKER_TYPE.RIGHT_WRIST);
//         var liveTracker = tracker as LiveTracking.LiveTracker;
//         if(liveTracker == null)
//             return false;
        
//         Vector2 axis = liveTracker.GetJoystickAxis();
//         return axis.x != 0 || axis.y != 0;
//     }
// }
