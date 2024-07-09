#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using UnityEditor.Animations;
using System.IO;
using UnityEditor.Formats.Fbx.Exporter;

public class MotionFBXRecorder
{   
    public static int BatchSize = 15;
    private enum MODE {
        Manual,
        FrameInterval,
        TimeInterval
    };
    private MODE RecordingMode = MODE.Manual;
    private IntervalData FrameInterval = new IntervalData(0f, 1f);
    private IntervalData TimeInterval = new IntervalData(0f, 1f);
    private string FileName;
    private string Destination = string.Empty;
    private bool ShowExportOptions = false;
    private ExportModelOptions ExportOptions = new ExportModelOptions();
    private GameObjectRecorder Recorder;
    private EditorCoroutines.EditorCoroutine Coroutine = null;

    private IRecorder Controller;
    public MotionFBXRecorder(IRecorder controller) {
        Controller = controller;
        FileName = controller.Target.name;
        CreateRecorder();

        ExportOptions = new ExportModelOptions
        {
            ModelAnimIncludeOption = Include.ModelAndAnim,
            ExportFormat = ExportFormat.Binary,
            LODExportType = LODExportType.Lowest,
            ObjectPosition = ObjectPosition.LocalCentered,
            AnimateSkinnedMesh = false,
            UseMayaCompatibleNames = false,
            ExportUnrendered = false,
            PreserveImportSettings = false,
            KeepInstances = false,
            EmbedTextures = false
        };
    }

    private void CreateRecorder(bool iterateToFirstNode = true) {
        GameObject rootNode = Controller.Target.gameObject;
        if(iterateToFirstNode){
            rootNode = rootNode.transform.root.gameObject;
        }
        Recorder = new GameObjectRecorder(rootNode);
        Recorder.BindComponentsOfType(Recorder.root, typeof(Transform), true);
    }

    public void StartRecordingManual() {
        if(!CanRecord()) { return; }

        CreateRecorder();
        Coroutine = null;
        try {
            Coroutine = EditorCoroutines.StartCoroutine(RecordManual(), this); 
        } catch(Exception e) {
            Debug.LogError(e);
            StopCoroutine();
        }
    }

    public void StartRecordingTimeInterval(float startTimestamp, float endTimestamp) {
        if(!CanRecord()) { return; }

        CreateRecorder();
        Coroutine = null;
        try{
        Coroutine = EditorCoroutines.StartCoroutine(RecordTimeInterval(new IntervalData(startTimestamp, endTimestamp)), this);
        } catch(Exception e) {
            Debug.LogError(e);
            StopCoroutine();
        }
    }

    private IEnumerator RecordManual() {
        int counter = 0;
        float start = Controller.GetTimestamp();
        while(true) {
            float timestamp = start + counter/Controller.ExportFramerate;
            Controller.Callback(timestamp);
            Controller.Animate();
            Snapshot();
            counter += 1;
            
            if(counter % BatchSize == 0){
                yield return new WaitForSeconds(0f);
            }
        }
    }

    private IEnumerator RecordTimeInterval(IntervalData interval) {
        //Controller.Callback(interval.Start);
        float totalTime = Mathf.Max(0f, interval.End - interval.Start);
        int counter = 0;

        while((float)counter/Controller.ExportFramerate <= totalTime) {
            float timestamp = interval.Start + counter/Controller.ExportFramerate;
            Controller.Callback(timestamp);
            Controller.Animate();
            Snapshot();
            counter += 1;
            
            if(counter % BatchSize == 0){
                yield return new WaitForSeconds(0f);
            }
        }
        Export();
    }

    private void Snapshot(){
        Recorder.TakeSnapshot(1f/Controller.ExportFramerate);          
    }

    private void StopCoroutine() {
        EditorCoroutines.StopAllCoroutines(this);
        Coroutine = null;
    }

    public bool IsRunning() {
        if(Coroutine == null)  {
            return false;
        }
        return Recorder.isRecording || !Coroutine.finished;
    }

    private bool CanRecord() {
        if(Destination == "") { Debug.Log("Recording not allowed. Export <Destination Path> is empty."); }
        if(FileName == "") { Debug.Log("Recording not allowed. Export <FileName> is empty."); }
        if(IsRunning()) { Debug.Log("Already recording."); }
        bool value = !IsRunning() && Destination != "" && FileName != "";
        return value;
    }
    public void SetDestination(string path) {
        if(Destination != path) {
            Destination = path;
        }
    }
    public void SetFileName(string name) {
        if(FileName != name) {
            FileName = name;
        }
    }

    private CurveFilterOptions GetCurveFilterOptions() {
        CurveFilterOptions curveOptions = new CurveFilterOptions();
        curveOptions.positionError = 0f;
        curveOptions.rotationError = 0f;
        curveOptions.scaleError = 0f;
        curveOptions.floatError = 0f;
        curveOptions.keyframeReduction = false;
        curveOptions.unrollRotation = false;
        return curveOptions;
    }
    private void Export() {
        AnimationClip gameObjectClip = new AnimationClip();
        gameObjectClip.name = "Take 001";

        Recorder.SaveToClip(gameObjectClip, Controller.ExportFramerate, GetCurveFilterOptions());
        Recorder.ResetRecording();
        
        //AssetDatabase.CreateAsset(gameObjectClip, "Assets/" + FileName + ".anim");

        Animator animator = Recorder.root.GetComponent<Animator>();
        if(animator == null) {
            animator = Recorder.root.AddComponent<Animator>();
        }
        animator.runtimeAnimatorController = AnimatorController.CreateAnimatorControllerAtPathWithClip("Assets/" + FileName + ".controller", gameObjectClip); 

        string filePath = Path.Combine(Destination, FileName + ".fbx");
        ModelExporter.ExportObject(filePath, Recorder.root, ExportOptions);
        AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(animator.runtimeAnimatorController)); 
        Utility.Destroy(Recorder.root.GetComponent<Animator>());

        StopCoroutine();
    }


    public void Inspector() {
        Utility.SetGUIColor(IsRunning() ? UltiDraw.Red : UltiDraw.Gold);
        using(new EditorGUILayout.VerticalScope ("Box")) {
            Utility.ResetGUIColor();

            EditorGUI.BeginDisabledGroup(IsRunning());
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.ObjectField("Target", Controller.Target.gameObject, typeof(GameObject), true);
            EditorGUI.EndDisabledGroup();
            Controller.ExportFramerate = EditorGUILayout.FloatField("Export Framerate", Controller.ExportFramerate);
            DestinationUI();
            FileName = EditorGUILayout.TextField("Filename", FileName);
            RecordingMode = (MODE)EditorGUILayout.EnumPopup("Recording Mode", RecordingMode);
            
            if(RecordingMode == MODE.FrameInterval) {
                FrameInterval.Start = Mathf.Clamp(EditorGUILayout.IntField("Start", (int)FrameInterval.Start), 1, Controller.MaxFrame);
                FrameInterval.End = Mathf.Clamp(EditorGUILayout.IntField("End", (int)FrameInterval.End), 1, Controller.MaxFrame);
            } 

            if(RecordingMode == MODE.TimeInterval) {
                TimeInterval.Start = Mathf.Clamp(EditorGUILayout.FloatField("Start", TimeInterval.Start), 0f, Controller.MaxTimestamp);
                TimeInterval.End = Mathf.Clamp(EditorGUILayout.FloatField("End", TimeInterval.End), 0f, Controller.MaxTimestamp);
            }

            ExportOptionsUI(); 
            EditorGUI.EndDisabledGroup();

            if(IsRunning()) {
                if(Utility.GUIButton("Stop Recording", UltiDraw.DarkRed, UltiDraw.White)) {
                    Export();
                }
            } else {
                if(Utility.GUIButton("Start Recording", UltiDraw.Green, UltiDraw.White)) {
                    if(!CanRecord()) { return; }

                    CreateRecorder(Controller.Target.gameObject);
                    switch (RecordingMode)
                    {
                        case MODE.FrameInterval:
                            IntervalData interval = new IntervalData((float)(FrameInterval.Start/Controller.Framerate), (float)(FrameInterval.End/Controller.Framerate));
                            interval.Print();
                            Coroutine = EditorCoroutines.StartCoroutine(RecordTimeInterval(interval), this);
                            break;
                        case MODE.TimeInterval:
                            TimeInterval.Print();
                            Coroutine = EditorCoroutines.StartCoroutine(RecordTimeInterval(TimeInterval), this);
                            break;
                        default:
                            Coroutine = EditorCoroutines.StartCoroutine(RecordManual(), this);
                            break;
                    }
                }
            }
        }
    }

    public void DestinationUI() {
        if(Utility.GUIButton("Destination Folder: " + Destination, Destination != "" ? UltiDraw.DarkGrey : UltiDraw.DarkRed, UltiDraw.White)) {
            Destination = EditorUtility.OpenFolderPanel("FBX Exporter", Destination, "");
        }
    }

    public void ExportOptionsUI() {
        GUIStyle foldoutStyle = new GUIStyle(EditorStyles.foldout);
        foldoutStyle.fontStyle = FontStyle.Bold;
        foldoutStyle.normal.textColor = UltiDraw.Cyan;

        ShowExportOptions = EditorGUILayout.Foldout(ShowExportOptions, new GUIContent("Export Settings"), true, foldoutStyle);
        if(ShowExportOptions){
            ExportOptions.ModelAnimIncludeOption = (Include)EditorGUILayout.EnumPopup("Export Mesh", ExportOptions.ModelAnimIncludeOption);
            ExportOptions.ExportFormat = (ExportFormat)EditorGUILayout.EnumPopup("Export Format", ExportOptions.ExportFormat);
            ExportOptions.LODExportType = (LODExportType)EditorGUILayout.EnumPopup("LOD Type", ExportOptions.LODExportType);
            ExportOptions.ObjectPosition = (ObjectPosition)EditorGUILayout.EnumPopup("Object Position", ExportOptions.ObjectPosition);
            EditorGUILayout.BeginHorizontal();
            if(Utility.GUIButton("Animate Skinned Mesh", ExportOptions.AnimateSkinnedMesh ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.AnimateSkinnedMesh = !ExportOptions.AnimateSkinnedMesh;
            }
            if(Utility.GUIButton("Maya Compatible Names", ExportOptions.UseMayaCompatibleNames ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.UseMayaCompatibleNames = !ExportOptions.UseMayaCompatibleNames;
            }
            if(Utility.GUIButton("Export Unrendered", ExportOptions.ExportUnrendered ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.ExportUnrendered = !ExportOptions.ExportUnrendered;
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            if(Utility.GUIButton("Preserve Import Settings", ExportOptions.PreserveImportSettings ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.PreserveImportSettings = !ExportOptions.PreserveImportSettings;
            }
            if(Utility.GUIButton("Keep Instances", ExportOptions.KeepInstances ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.KeepInstances = !ExportOptions.KeepInstances;
            }
            if(Utility.GUIButton("Embed Textures", ExportOptions.EmbedTextures ? UltiDraw.Cyan : UltiDraw.DarkGrey, UltiDraw.White)) {
                ExportOptions.EmbedTextures = !ExportOptions.EmbedTextures;
            }
            EditorGUILayout.EndHorizontal();
        }
    }

    private struct IntervalData
    {
        public float Start;
        public float End;

        public IntervalData(float start, float end){
            Start = start;
            End = end;
        }

        public void Print() {
            Debug.Log("Start: " + Start + " | End: " + End);
        }
    }
}
#endif
