using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;

public class SampleUI : MonoBehaviour
{
    RectTransform collectionButton;
    RectTransform inputText;
    RectTransform valueText;
    bool inMenu;

    void Start()
    {
#if OVR_SAMPLES_ENABLE_FIREBASE
        DebugUIBuilder.instance.AddButton("Log", Log);
        DebugUIBuilder.instance.AddButton("Record Exception", RecordException);
        collectionButton = DebugUIBuilder.instance.AddButton("Toggle Crashlytics Collection (true)", ToggleCrashlyticsCollection);
        DebugUIBuilder.instance.AddButton("Set Custom Key", ToggleCrashlyticsCollection);
        DebugUIBuilder.instance.AddButton("Set User ID", SetUserID);
        DebugUIBuilder.instance.AddButton("Crash", Crash);
        
        DebugUIBuilder.instance.AddLabel("(Text input used by most methods)", DebugUIBuilder.DEBUG_PANE_RIGHT);
        inputText = DebugUIBuilder.instance.AddTextField("Input Text", DebugUIBuilder.DEBUG_PANE_RIGHT);
        DebugUIBuilder.instance.AddLabel("(The value of Set Custom Key)", DebugUIBuilder.DEBUG_PANE_RIGHT);
        valueText = DebugUIBuilder.instance.AddTextField("Value", DebugUIBuilder.DEBUG_PANE_RIGHT);
#else
        DebugUIBuilder.instance.AddLabel("Enable Firebase in your project before running this sample", DebugUIBuilder.DEBUG_PANE_RIGHT);
#endif
        DebugUIBuilder.instance.Show();
        inMenu = true;
    }

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.Two) || OVRInput.GetDown(OVRInput.Button.Start))
        {
            if (inMenu) DebugUIBuilder.instance.Hide();
            else DebugUIBuilder.instance.Show();
            inMenu = !inMenu;
        }
    }

    string GetText()
    {
        return inputText.GetComponentInChildren<InputField>().text;
    }
#if OVR_SAMPLES_ENABLE_FIREBASE
    void Log()
    {
        Firebase.Crashlytics.Crashlytics.Log(GetText());
    }

    void RecordException()
    {
        Firebase.Crashlytics.Crashlytics.LogException(new System.Exception(GetText()));
    }

    void ToggleCrashlyticsCollection()
    {
        Firebase.Crashlytics.Crashlytics.IsCrashlyticsCollectionEnabled = !Firebase.Crashlytics.Crashlytics.IsCrashlyticsCollectionEnabled;
        Text buttonText = collectionButton.GetComponentInChildren<Text>();
        if(buttonText)
        {
            buttonText.text = string.Format("Toggle Crashlytics Collection ({0})", Firebase.Crashlytics.Crashlytics.IsCrashlyticsCollectionEnabled);
        }        
    }

    void SetCustomKey()
    {
        Firebase.Crashlytics.Crashlytics.SetCustomKey(GetText(), valueText.GetComponentInChildren<InputField>().text);
    }

    void SetUserID()
    {
        Firebase.Crashlytics.Crashlytics.SetUserId(GetText());
    }

    void Crash()
    {
        unsafe
        {
            int* i = null;
            *i = 0;
        }
    }
#endif
}
