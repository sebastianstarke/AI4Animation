using System;
using System.Collections;
using UnityEngine;
using AI4Animation;

public class VRCamera : MonoBehaviour {

    public enum MODE {FirstPerson, ThirdPerson, FixedView, Embodied, Static, None}
    public enum WAIT {EndOfFrame, FixedUpdate}

    public MODE Mode = MODE.FirstPerson;
    public WAIT Wait = WAIT.EndOfFrame;
    [Range(0f,1f)] public float SmoothTime = 0.1f;

    [Serializable]
    public class FirstPersonSettings {
        public Transform HMD = null;
        public Vector3 DeltaRotation = Vector3.zero;
    }
    public FirstPersonSettings FirstPerson;

    [Serializable]
    public class ThirdPersonSettings {
        public Transform Anchor = null;
        public Vector3 SelfOffset = Vector3.zero;
        public Vector3 TargetOffset = Vector3.zero;
    }
    public ThirdPersonSettings ThirdPerson;

    [Serializable]
    public class FixedViewSettings {
        public Transform Root = null;
        public Vector3 SelfOffset = Vector3.zero;
        public Vector3 TargetOffset = Vector3.zero;
    }
    public FixedViewSettings FixedView;

    [Serializable]
    public class EmbodiedSettings {
        public Transform HMD = null;
        public Transform Head = null;
        public float Distance = 1f;
        public Axis Axis = Axis.ZPositive;
        public Vector3 DeltaRotation = Vector3.zero;
    }
    public EmbodiedSettings Embodied;

    [Serializable]
    public class StaticSettings {
        public Vector3 OriginPosition = Vector3.one;
        public Vector3 OriginRotation = Vector3.zero;
        public Transform HMD = null;
        public Vector3 DeltaRotation = Vector3.zero;
    }
    public StaticSettings Static;

    private Camera Camera = null;
    private MODE ActiveMode = MODE.None;
    private Vector3 LinearVelocity = Vector3.zero;
    private Vector3 ForwardVelocity = Vector3.zero;
    private Vector3 UpVelocity = Vector3.zero;

    void Awake() {
        Camera = GetComponent<Camera>();
    }

    void Update() {
        //TODO: Implement camera switching logic via controller button press.

        if(ActiveMode != Mode) {
            ActiveMode = Mode;
            StopAllCoroutines();
            Debug.Log("Starting Mode: "+ ActiveMode.ToString());
            if(Mode == MODE.FirstPerson) {
                StartCoroutine(FirstPersonCamera());
            }
            if(Mode == MODE.ThirdPerson) {
                StartCoroutine(ThirdPersonCamera());
            }
            if(Mode == MODE.FixedView) {
                StartCoroutine(FixedViewCamera());
            }
            if(Mode == MODE.Embodied) {
                StartCoroutine(EmbodiedCamera());
            }
            if(Mode == MODE.Static) {
                StartCoroutine(StaticCamera());
            }
        }
    }

    private Transform GetTarget(Transform target) {
        if(target != null) {
            return target;
        }
        return FindObjectOfType<Actor>().transform;
    }

    private IEnumerator FirstPersonCamera() {
        while(true) {
            switch(Wait) {
                case WAIT.EndOfFrame: yield return new WaitForEndOfFrame(); break;
                case WAIT.FixedUpdate: yield return new WaitForFixedUpdate(); break;
            }
            Transform target = GetTarget(FirstPerson.HMD);
            Transform camera = Camera.transform;
            Matrix4x4 reference = target.GetWorldMatrix();

            //Save previous coordinates
            Vector3 previousPosition = camera.position;
            Quaternion previousRotation = camera.rotation;

            //Calculate new coordinates
            Vector3 position = reference.GetPosition();
            Quaternion rotation = reference.GetRotation() * Quaternion.Euler(FirstPerson.DeltaRotation);

            //Lerp camera from previous to new coordinates
            camera.position = Vector3.SmoothDamp(previousPosition, position, ref LinearVelocity, SmoothTime);
            camera.rotation = Quaternion.LookRotation(
                Vector3.SmoothDamp(previousRotation.GetForward(), rotation.GetForward(), ref ForwardVelocity, SmoothTime).normalized,
                Vector3.SmoothDamp(previousRotation.GetUp(), rotation.GetUp(), ref UpVelocity, SmoothTime).normalized
            );
        }
    }

    private IEnumerator ThirdPersonCamera() {
        while(true) {
            switch(Wait) {
                case WAIT.EndOfFrame: yield return new WaitForEndOfFrame(); break;
                case WAIT.FixedUpdate: yield return new WaitForFixedUpdate(); break;
            }
            Transform target = GetTarget(ThirdPerson.Anchor);
            Transform camera = Camera.transform;
            Matrix4x4 reference = target.GetWorldMatrix();

            //Save previous coordinates
            Vector3 previousPosition = camera.position;
            Quaternion previousRotation = camera.rotation;

            //Calculate new coordinates
            Actor character = target.GetComponent<Actor>();
            Vector3 selfOffset = ThirdPerson.SelfOffset.SetY(ThirdPerson.SelfOffset.y + reference.GetPosition().y);
            Vector3 targetOffset = ThirdPerson.TargetOffset.SetY(ThirdPerson.TargetOffset.y + reference.GetPosition().y);
            Vector3 pivot = character.transform.position + character.transform.rotation * targetOffset;
            camera.position = character.transform.position + character.transform.rotation * selfOffset;
            camera.rotation = reference.GetRotation();
            camera.LookAt(pivot);

            //Lerp camera from previous to new coordinates
            camera.position = Vector3.SmoothDamp(previousPosition, camera.position, ref LinearVelocity, SmoothTime);
            camera.rotation = Quaternion.LookRotation(
                Vector3.SmoothDamp(previousRotation.GetForward(), camera.rotation.GetForward(), ref ForwardVelocity, SmoothTime).normalized,
                Vector3.SmoothDamp(previousRotation.GetUp(), camera.rotation.GetUp(), ref UpVelocity, SmoothTime).normalized
            );
        }
    }

    private IEnumerator FixedViewCamera() {
		while(Mode == MODE.FixedView) {
            switch(Wait) {
                case WAIT.EndOfFrame: yield return new WaitForEndOfFrame(); break;
                case WAIT.FixedUpdate: yield return new WaitForFixedUpdate(); break;
            }
            Transform target = GetTarget(FixedView.Root);
            Transform camera = Camera.transform;
            Matrix4x4 reference = target.GetWorldMatrix();

            //Save previous coordinates
            Vector3 previousPosition = camera.position;
            Quaternion previousRotation = camera.rotation;

            //Calculate new coordinates
			transform.position = reference.GetPosition() + FixedView.SelfOffset;
			transform.LookAt(reference.GetPosition() + FixedView.TargetOffset);

            //Lerp camera from previous to new coordinates
            camera.position = Vector3.SmoothDamp(previousPosition, camera.position, ref LinearVelocity, SmoothTime);
            camera.rotation = Quaternion.LookRotation(
                Vector3.SmoothDamp(previousRotation.GetForward(), camera.rotation.GetForward(), ref ForwardVelocity, SmoothTime).normalized,
                Vector3.SmoothDamp(previousRotation.GetUp(), camera.rotation.GetUp(), ref UpVelocity, SmoothTime).normalized
            );
		}
    }

    private IEnumerator EmbodiedCamera() {
		while(Mode == MODE.Embodied) {
            switch(Wait) {
                case WAIT.EndOfFrame: yield return new WaitForEndOfFrame(); break;
                case WAIT.FixedUpdate: yield return new WaitForFixedUpdate(); break;
            }
            Transform truth = GetTarget(Embodied.HMD);
            Transform target = GetTarget(Embodied.Head);
            Transform camera = Camera.transform;
            Matrix4x4 reference = target.GetWorldMatrix();

            //Save previous coordinates
            Vector3 previousPosition = camera.position;
            Quaternion previousRotation = camera.rotation;

            //Calculate new coordinates
            Vector3 position = reference.GetPosition() - Embodied.Distance * reference.GetAxis(Embodied.Axis);
            position.y = truth.position.y;
            Quaternion rotation = reference.GetRotation() * Quaternion.Euler(Embodied.DeltaRotation);
            //Lerp camera from previous to new coordinates
            camera.position = Vector3.SmoothDamp(previousPosition, position, ref LinearVelocity, SmoothTime);
            camera.rotation = Quaternion.LookRotation(
                Vector3.SmoothDamp(previousRotation.GetForward(), rotation.GetForward(), ref ForwardVelocity, SmoothTime).normalized,
                Vector3.SmoothDamp(previousRotation.GetUp(), rotation.GetUp(), ref UpVelocity, SmoothTime).normalized
            );
		}
    }
    private IEnumerator StaticCamera() {
		while(Mode == MODE.Static) {
            switch(Wait) {
                case WAIT.EndOfFrame: yield return new WaitForEndOfFrame(); break;
                case WAIT.FixedUpdate: yield return new WaitForFixedUpdate(); break;
            }
            Transform target = GetTarget(Static.HMD);
            Transform camera = Camera.transform;
            Matrix4x4 reference = target.GetWorldMatrix();

            //Save previous coordinates
            Vector3 previousPosition = camera.position;
            Quaternion previousRotation = camera.rotation;

            //Calculate new coordinates
            Vector3 position = new Vector3(Static.OriginPosition.x, reference.GetPosition().y, Static.OriginPosition.z);
            Quaternion rotation = Quaternion.Euler(Static.OriginRotation) * reference.GetRotation() * Quaternion.Euler(Static.DeltaRotation);
            //Lerp camera from previous to new coordinates
            camera.position = Vector3.SmoothDamp(previousPosition, position, ref LinearVelocity, SmoothTime);
            camera.rotation = Quaternion.LookRotation(
                Vector3.SmoothDamp(previousRotation.GetForward(), rotation.GetForward(), ref ForwardVelocity, SmoothTime).normalized,
                Vector3.SmoothDamp(previousRotation.GetUp(), rotation.GetUp(), ref UpVelocity, SmoothTime).normalized
            );
		}
    }
}