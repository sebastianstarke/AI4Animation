using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MotionRecorder : MonoBehaviour {

    public enum Action {Record, Playback};

    public string SavePath = string.Empty;

    public Action Mode = Action.Record;
    public Actor[] Actors;
    public MotionRecording Recording;

    public int Framerate = 30;
    public int Index = 0;

    #if UNITY_EDITOR
    [ContextMenu("Create Assets")]
    public void CreateAssets() {
        Recording  = (MotionRecording)ScriptableObject.CreateInstance<MotionRecording>();
        AssetDatabase.CreateAsset(Recording, SavePath + "Recording.asset");
    }
    #endif

    void Start() {
        foreach(Actor actor in Actors) {
            AnimationController animation = actor.GetComponent<AnimationController>();
            if(animation != null) {
                animation.enabled = Mode == Action.Record;
            }
        }
    }

    void OnEnable() {
        #if UNITY_EDITOR
        if(Mode == Action.Record) {
            Recording.Reset();
        }
        #endif
    }

    void OnDisable() {
        #if UNITY_EDITOR
        if(Mode == Action.Record) {
            EditorUtility.SetDirty(Recording);
            AssetDatabase.SaveAssets();
        }
        #endif
    }

    void Update() {
        Application.targetFrameRate = Framerate;

        #if UNITY_EDITOR
        if(Mode == Action.Record) {
            Recording.AddFrame(Actors);
        }
        #endif

        if(Mode == Action.Playback) {
            Index = Mathf.Clamp(Index, 0, Recording.Frames.Length-1);
            for(int i=0; i<Actors.Length; i++) {
                Matrix4x4[] transformations = Recording.GetTransformations(Index, i);
                Actors[i].transform.position = Utility.ProjectGround(transformations.First().GetPosition(), LayerMask.GetMask("Ground"));
                Actors[i].transform.rotation = Quaternion.LookRotation(Vector3.ProjectOnPlane(transformations.First().GetForward(), Vector3.up), Vector3.up);
                Actors[i].SetBoneTransformations(Recording.GetTransformations(Index, i));
            }
            Index = (Index + 1) % Recording.Frames.Length;
        }
    }
}
