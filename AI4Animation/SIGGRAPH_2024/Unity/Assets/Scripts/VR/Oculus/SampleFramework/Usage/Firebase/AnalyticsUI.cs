using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

#if OVR_SAMPLES_ENABLE_FIREBASE
using Firebase.Analytics;
#endif

public class AnalyticsUI : MonoBehaviour
{
#if OVR_SAMPLES_ENABLE_FIREBASE
    const int target = DebugUIBuilder.DEBUG_PANE_LEFT;

    bool analyticsCollectionEnabled = true;
    RectTransform textFieldSessionTimeoutDuration;
    RectTransform textFieldUserId;

    RectTransform textFieldName;
    RectTransform textFieldProperty;

    RectTransform textFieldEventName;

    List<(RectTransform, RectTransform)> intParams = new List<(RectTransform, RectTransform)>();//, FltParms, StrParams;
    List<(RectTransform, RectTransform)> fltParams = new List<(RectTransform, RectTransform)>();//, FltParms, StrParams;
    List<(RectTransform, RectTransform)> strParams = new List<(RectTransform, RectTransform)>();//, FltParms, StrParams;

    // Start is called before the first frame update
    void Start()
    {
        DebugUIBuilder ui = DebugUIBuilder.instance;
        ui.AddButton("ResetAnalyticsData", ResetAnalyticsData, target);
        ui.AddButton("SetAnalyticsCollectionEnabled", SetAnalyticsCollectionEnabled, target);
        ui.AddToggle("CollectionEnabled", delegate (Toggle t) { analyticsCollectionEnabled = t.enabled; },
            analyticsCollectionEnabled, target);

        //ui.AddButton("SetDefaultEventParameters", SetDefaultEventParameters, target);

        ui.AddButton("SetSessionTimeoutDuration", SetSessionTimeoutDuration, target);
        textFieldSessionTimeoutDuration = ui.AddTextField("500000", target);

        ui.AddButton("SetUserId", SetUserId, target);
        textFieldUserId = ui.AddTextField("UserId", target);

        ui.AddButton("SetUserProperty", SetUserProperty, target);
        textFieldName = ui.AddTextField("Name", target);
        textFieldProperty = ui.AddTextField("Nroperty", target);

        // log event UI
        ui.AddButton("LogEvent", LogEvent, 3);
        textFieldEventName = ui.AddTextField("EventName", 3);
        ui.AddButton("AddInt", AddInt, 3);
        ui.AddButton("AddFlt", AddFlt, 3);
        ui.AddButton("AddStr", AddStr, 3);
        ui.AddButton("Clear", Clear, 3);

        ui.Show();
    }

    void Clear()
    {
        DebugUIBuilder ui = DebugUIBuilder.instance;
        var field = typeof(DebugUIBuilder).GetField("insertedElements", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var relayout = typeof(DebugUIBuilder).GetMethod("Relayout", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        List<RectTransform> elements = ((List<RectTransform>[])field.GetValue(ui))[3];
        foreach ((RectTransform a, RectTransform b) in intParams)
        {
            elements.Remove(a);
            elements.Remove(b);
            a.SetParent(null);
            b.SetParent(null);
        }
        foreach ((RectTransform a, RectTransform b) in fltParams)
        {
            elements.Remove(a);
            elements.Remove(b);
            a.SetParent(null);
            b.SetParent(null);
        }
        foreach ((RectTransform a, RectTransform b) in strParams)
        {
            elements.Remove(a);
            elements.Remove(b);
            a.SetParent(null);
            b.SetParent(null);
        }
        relayout.Invoke(ui, new object[] { });
    }

    void AddInt()
    {
        DebugUIBuilder ui = DebugUIBuilder.instance;
        var name = ui.AddTextField("INT_PARAM", 3);
        var value = ui.AddTextField("0", 3);
        intParams.Add((name, value));
    }

    void AddFlt()
    {
        DebugUIBuilder ui = DebugUIBuilder.instance;
        var name = ui.AddTextField("FLT_PARAM", 3);
        var value = ui.AddTextField("0.0", 3);
        intParams.Add((name, value));
    }

    void AddStr()
    {
        DebugUIBuilder ui = DebugUIBuilder.instance;
        var name = ui.AddTextField("STR_PARAM", 3);
        var value = ui.AddTextField("None", 3);
        intParams.Add((name, value));
    }

    private List<Parameter> GetParameterList()
    {
        List<Parameter> parameters = new List<Parameter>();
        foreach ((RectTransform a, RectTransform b) in intParams)
        {
            string name = a.GetComponentInChildren<Text>().text;
            int value;
            if(int.TryParse(b.GetComponentInChildren<Text>().text, out value))
            {
                parameters.Add(new Parameter(name, value));
            }
        }
        foreach ((RectTransform a, RectTransform b) in fltParams)
        {
            string name = a.GetComponentInChildren<Text>().text;
            float value;
            if (float.TryParse(b.GetComponentInChildren<Text>().text, out value))
            {
                parameters.Add(new Parameter(name, value));
            }
        }
        foreach ((RectTransform a, RectTransform b) in fltParams)
        {
            string name = a.GetComponentInChildren<Text>().text;
            string value = b.GetComponentInChildren<Text>().text;
            parameters.Add(new Parameter(name, value));
        }
        return parameters;
    }

    void LogEvent()
    {
        Debug.Log("LogEvent");
        List<Parameter> parameters = GetParameterList();
        string eventName = textFieldEventName.GetComponentInChildren<Text>().text;
        FirebaseAnalytics.LogEvent(eventName, parameters.ToArray());

    }

    void ResetAnalyticsData()
    {
        Debug.Log("ResetAnalyticsData");
        FirebaseAnalytics.ResetAnalyticsData();
    }

    void SetAnalyticsCollectionEnabled()
    {
        Debug.Log(string.Format("SetAnalyticsCollectionEnabled({0})", analyticsCollectionEnabled));
        FirebaseAnalytics.SetAnalyticsCollectionEnabled(analyticsCollectionEnabled);
    }

    void SetSessionTimeoutDuration()
    {
        long ms;
        if( long.TryParse(textFieldSessionTimeoutDuration.GetComponentInChildren<Text>().text, out ms) )
        {
            Debug.Log(string.Format("SetSessionTimeoutDuration({0})", ms));
            FirebaseAnalytics.SetSessionTimeoutDuration(System.TimeSpan.FromMilliseconds(ms));
        }
    }

    void SetUserId()
    {
        string userId = textFieldUserId.GetComponentInChildren<Text>().text;
        Debug.Log(string.Format("SetUserId({0})", userId));
        FirebaseAnalytics.SetUserId(userId);
    }

    void SetUserProperty()
    {
        string name = textFieldName.GetComponentInChildren<Text>().text;
        string property = textFieldProperty.GetComponentInChildren<Text>().text;
        FirebaseAnalytics.SetUserProperty(name, property);
    }
#endif
}
