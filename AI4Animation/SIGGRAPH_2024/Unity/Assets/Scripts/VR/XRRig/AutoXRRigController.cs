// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using AI4Animation;

// public class AutoXRRigController : XRRigController
// {
//     protected override void UpdateXRRig()
//     {
//         // DO NOTHING: XR Rig anchors will be automatically driven by other scripts
//     }

//     private Transform GetRigRoot()
//     {
//         return _xrRig.gameObject.transform;
//     }

//     public Vector3 GetRigOffset()
//     {
//         return  GetRigRoot().position;
//     }

//     public Quaternion GetRigOrientation()
//     {
//         return  GetRigRoot().rotation;
//     }

//     public void AddOffset(Vector3 delta)
//     {
//          GetRigRoot().position += delta;
//     }

//     public void SetOffset(Vector3 offset)
//     {
//          GetRigRoot().position = offset;
//     }

//     public void AddOrientationOffset(Quaternion delta)
//     {
//          GetRigRoot().rotation *= delta;
//     }

//     public void SetOrientationOffset(Quaternion offset)
//     {
//          GetRigRoot().rotation = offset;
//     }
// }

