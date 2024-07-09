/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
#if UNITY_EDITOR
using UnityEngine.SceneManagement;
#endif
using TMPro;

public class DebugUIBuilder : MonoBehaviour
{
  // room for extension:
  // support update funcs
  // fix bug where it seems to appear at a random offset
  // support remove

  // Convenience consts for clarity when using multiple debug panes.
  // But note that you can an arbitrary number of panes if you add them in the inspector.
  public const int DEBUG_PANE_CENTER = 0;
  public const int DEBUG_PANE_RIGHT = 1;
  public const int DEBUG_PANE_LEFT = 2;

  [SerializeField]
  private RectTransform buttonPrefab = null;

  [SerializeField]
  private RectTransform[] additionalButtonPrefab = null;

  [SerializeField]
  private RectTransform labelPrefab = null;
  [SerializeField]
  private RectTransform sliderPrefab = null;
  [SerializeField]
  private RectTransform dividerPrefab = null;
  [SerializeField]
  private RectTransform togglePrefab = null;
  [SerializeField]
  private RectTransform radioPrefab = null;
  [SerializeField]
  private RectTransform textPrefab = null;

  [SerializeField]
  private GameObject uiHelpersToInstantiate = null;

  [SerializeField]
  private Transform[] targetContentPanels = null;

  private bool[] reEnable;

  [SerializeField]
  private List<GameObject> toEnable = null;
  [SerializeField]
  private List<GameObject> toDisable = null;

  public static DebugUIBuilder instance;

  public delegate void OnClick();
  public delegate void OnToggleValueChange(Toggle t);
  public delegate void OnSlider(float f);
  public delegate bool ActiveUpdate();

  public float elementSpacing = 16.0f;
  public float marginH = 16.0f;
  public float marginV = 16.0f;
  private Vector2[] insertPositions;
  private List<RectTransform>[] insertedElements;
  private Vector3 menuOffset;
  OVRCameraRig rig;
  private Dictionary<string, ToggleGroup> radioGroups = new Dictionary<string, ToggleGroup>();
  LaserPointer lp;
  LineRenderer lr;

  public LaserPointer.LaserBeamBehavior laserBeamBehavior;
  public bool isHorizontal = false;
  public bool usePanelCentricRelayout = false;

  public void Awake()
  {
    Debug.Assert(instance == null);
    instance = this;
    menuOffset = transform.position; // TODO: this is unpredictable/busted
    gameObject.SetActive(false);
    rig = FindObjectOfType<OVRCameraRig>();
    for (int i = 0; i < toEnable.Count; ++i)
    {
      toEnable[i].SetActive(false);
    }

    insertPositions = new Vector2[targetContentPanels.Length];
    for (int i = 0; i < insertPositions.Length; ++i)
    {
      insertPositions[i].x = marginH;
      insertPositions[i].y = -marginV;
    }
    insertedElements = new List<RectTransform>[targetContentPanels.Length];
    for (int i = 0; i < insertedElements.Length; ++i)
    {
      insertedElements[i] = new List<RectTransform>();
    }

    if (uiHelpersToInstantiate)
    {
      GameObject.Instantiate(uiHelpersToInstantiate);
    }

    lp = FindObjectOfType<LaserPointer>();
    if (!lp)
    {
      Debug.LogError("Debug UI requires use of a LaserPointer and will not function without it. Add one to your scene, or assign the UIHelpers prefab to the DebugUIBuilder in the inspector.");
      return;
    }
    lp.laserBeamBehavior = laserBeamBehavior;

    if (!toEnable.Contains(lp.gameObject))
    {
      toEnable.Add(lp.gameObject);
    }
    GetComponent<OVRRaycaster>().pointer = lp.gameObject;
    lp.gameObject.SetActive(false);
#if UNITY_EDITOR
    string scene = SceneManager.GetActiveScene().name;
    OVRPlugin.SendEvent("debug_ui_builder",
      ((scene == "DebugUI") ||
       (scene == "DistanceGrab") ||
       (scene == "OVROverlay") ||
       (scene == "Locomotion")).ToString(),
      "sample_framework");
#endif
  }

  public void Show()
  {
    Relayout();
    gameObject.SetActive(true);
    transform.position = rig.transform.TransformPoint(menuOffset);
    Vector3 newEulerRot = rig.transform.rotation.eulerAngles;
    newEulerRot.x = 0.0f;
    newEulerRot.z = 0.0f;
    transform.eulerAngles = newEulerRot;

    if (reEnable == null || reEnable.Length < toDisable.Count) reEnable = new bool[toDisable.Count];
    reEnable.Initialize();
    int len = toDisable.Count;
    for (int i = 0; i < len; ++i)
    {
      if (toDisable[i])
      {
        reEnable[i] = toDisable[i].activeSelf;
        toDisable[i].SetActive(false);
      }
    }
    len = toEnable.Count;
    for (int i = 0; i < len; ++i)
    {
      toEnable[i].SetActive(true);
    }

    int numPanels = targetContentPanels.Length;
    for (int i = 0; i < numPanels; ++i)
    {
      targetContentPanels[i].gameObject.SetActive(insertedElements[i].Count > 0);
    }
  }

  public void Hide()
  {
    gameObject.SetActive(false);

    for (int i = 0; i < reEnable.Length; ++i)
    {
      if (toDisable[i] && reEnable[i])
      {
        toDisable[i].SetActive(true);
      }
    }

    int len = toEnable.Count;
    for (int i = 0; i < len; ++i)
    {
      toEnable[i].SetActive(false);
    }
  }

  // Currently a slow brute-force method that lays out every element.
  // As this is intended as a debug UI, it might be fine, but there are many simple optimizations we can make.
  private void StackedRelayout()
  {

    for (int panelIdx = 0; panelIdx < targetContentPanels.Length; ++panelIdx)
    {
      RectTransform canvasRect = targetContentPanels[panelIdx].GetComponent<RectTransform>();
      List<RectTransform> elems = insertedElements[panelIdx];
      int elemCount = elems.Count;
      float x = marginH;
      float y = -marginV;
      float maxWidth = 0.0f;
      for (int elemIdx = 0; elemIdx < elemCount; ++elemIdx)
      {
        RectTransform r = elems[elemIdx];
        r.anchoredPosition = new Vector2(x, y);

        if (isHorizontal){
          x += (r.rect.width + elementSpacing);
        }
        else
        {
          y -= (r.rect.height + elementSpacing);
        }
        maxWidth = Mathf.Max(r.rect.width + 2 * marginH, maxWidth);
      }
      canvasRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, maxWidth);
      canvasRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, -y + marginV);
    }

  }

  private void PanelCentricRelayout(){
    if(!isHorizontal){
      Debug.Log("Error:Panel Centeric relayout is implemented only for horizontal panels");
      return;
    }

    for (int panelIdx = 0; panelIdx < targetContentPanels.Length; ++panelIdx)
    {
      RectTransform canvasRect = targetContentPanels[panelIdx].GetComponent<RectTransform>();
      List<RectTransform> elems = insertedElements[panelIdx];
      int elemCount = elems.Count;
      float x = marginH;
      float y = -marginV;
      float maxWidth = x;
      for (int elemIdx = 0; elemIdx < elemCount; ++elemIdx)
      {
        RectTransform r = elems[elemIdx];
        maxWidth += (r.rect.width + elementSpacing);
      }
      maxWidth -=elementSpacing;
      maxWidth += marginH;
      float totalmaxWidth = maxWidth;
      x = -0.5f * totalmaxWidth;
      y = -marginV;
      //Offset the UI  elements half of total lenght of the panel.
      for (int elemIdx = 0; elemIdx < elemCount; ++elemIdx)
      {
        RectTransform r = elems[elemIdx];
        if(elemIdx ==0){
          x += marginH;
        }
        x += 0.5f*r.rect.width;
        r.anchoredPosition = new Vector2(x , y);
        x +=r.rect.width*0.5f+elementSpacing;
        maxWidth = Mathf.Max(r.rect.width + 2 * marginH, maxWidth);
      }
      canvasRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, maxWidth);
      canvasRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, -y + marginV);
    }
  }

  private void Relayout()
  {
    if(usePanelCentricRelayout){
      PanelCentricRelayout();
    }else{
      StackedRelayout();
    }
  }
  private void AddRect(RectTransform r, int targetCanvas)
  {
    if (targetCanvas > targetContentPanels.Length)
    {
      Debug.LogError("Attempted to add debug panel to canvas " + targetCanvas + ", but only " + targetContentPanels.Length + " panels were provided. Fix in the inspector or pass a lower value for target canvas.");
      return;
    }

    r.transform.SetParent(targetContentPanels[targetCanvas], false);
    insertedElements[targetCanvas].Add(r);
    if (gameObject.activeInHierarchy)
    {
      Relayout();
    }
  }

  public RectTransform AddButton(string label, OnClick handler = null, int buttonIndex = -1, int targetCanvas = 0, bool highResolutionText = false)
  {
    RectTransform buttonRT = null;
    if(buttonIndex == -1)
        buttonRT = GameObject.Instantiate(buttonPrefab).GetComponent<RectTransform>();
    else
        buttonRT = GameObject.Instantiate(additionalButtonPrefab[buttonIndex]).GetComponent<RectTransform>();

    Button button = buttonRT.GetComponentInChildren<Button>();
    if(handler != null)
      button.onClick.AddListener(delegate { handler(); });


      if(highResolutionText){
        ((TextMeshProUGUI)(buttonRT.GetComponentsInChildren(typeof(TextMeshProUGUI), true)[0])).text = label;
      }
      else{
        ((Text)(buttonRT.GetComponentsInChildren(typeof(Text), true)[0])).text = label;
      }

    AddRect(buttonRT, targetCanvas);
    return buttonRT;
  }

  public RectTransform AddLabel(string label, int targetCanvas = 0)
  {
    RectTransform rt = GameObject.Instantiate(labelPrefab).GetComponent<RectTransform>();
    rt.GetComponent<Text>().text = label;
    AddRect(rt, targetCanvas);
    return rt;
  }

  public RectTransform AddSlider(string label, float min, float max, OnSlider onValueChanged, bool wholeNumbersOnly = false, int targetCanvas = 0)
  {
    RectTransform rt = (RectTransform)GameObject.Instantiate(sliderPrefab);
    Slider s = rt.GetComponentInChildren<Slider>();
    s.minValue = min;
    s.maxValue = max;
    s.onValueChanged.AddListener(delegate (float f) { onValueChanged(f); });
    s.wholeNumbers = wholeNumbersOnly;
    AddRect(rt, targetCanvas);
    return rt;
  }

  public RectTransform AddDivider(int targetCanvas = 0)
  {
    RectTransform rt = (RectTransform)GameObject.Instantiate(dividerPrefab);
    AddRect(rt, targetCanvas);
    return rt;
  }

  public RectTransform AddToggle(string label, OnToggleValueChange onValueChanged, int targetCanvas = 0)
  {
    RectTransform rt = (RectTransform)GameObject.Instantiate(togglePrefab);
    AddRect(rt, targetCanvas);
    Text buttonText = rt.GetComponentInChildren<Text>();
    buttonText.text = label;
    Toggle t = rt.GetComponentInChildren<Toggle>();
    t.onValueChanged.AddListener(delegate { onValueChanged(t); });
    return rt;
  }

  public RectTransform AddToggle(string label, OnToggleValueChange onValueChanged, bool defaultValue, int targetCanvas = 0)
  {
    RectTransform rt = (RectTransform)GameObject.Instantiate(togglePrefab);
    AddRect(rt, targetCanvas);
    Text buttonText = rt.GetComponentInChildren<Text>();
    buttonText.text = label;
    Toggle t = rt.GetComponentInChildren<Toggle>();
    t.isOn = defaultValue;
    t.onValueChanged.AddListener(delegate { onValueChanged(t); });
    return rt;
  }

  public RectTransform AddRadio(string label, string group, OnToggleValueChange handler, int targetCanvas = 0)
  {
    RectTransform rt = (RectTransform)GameObject.Instantiate(radioPrefab);
    AddRect(rt, targetCanvas);
    Text buttonText = rt.GetComponentInChildren<Text>();
    buttonText.text = label;
    Toggle tb = rt.GetComponentInChildren<Toggle>();
    if (group == null) group = "default";
    ToggleGroup tg = null;
    bool isFirst = false;
    if (!radioGroups.ContainsKey(group))
    {
      tg = tb.gameObject.AddComponent<ToggleGroup>();
      radioGroups[group] = tg;
      isFirst = true;
    }
    else
    {
      tg = radioGroups[group];
    }
    tb.group = tg;
    tb.isOn = isFirst;
    tb.onValueChanged.AddListener(delegate { handler(tb); });
    return rt;
  }

  public RectTransform AddTextField(string label, int targetCanvas = 0)
  {
      RectTransform textRT = GameObject.Instantiate(textPrefab).GetComponent<RectTransform>();
      InputField inputField = textRT.GetComponentInChildren<InputField>();
      inputField.text = label;
      AddRect(textRT, targetCanvas);
      return textRT;
  }

  public void ToggleLaserPointer(bool isOn)
  {
    if (lp)
    {
      if (isOn)
      {
        lp.enabled = true;
      }
      else
      {
        lp.enabled = false;
      }
    }
  }
}
