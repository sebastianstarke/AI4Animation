// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;
using UnityEngine.Serialization;

/// <summary>
/// Specific functionality for spawned anchors
/// </summary>
[RequireComponent(typeof(OVRSpatialAnchor))]
public class Anchor : MonoBehaviour
{
    public const string NumUuidsPlayerPref = "numUuids";

    [SerializeField, FormerlySerializedAs("canvas_")]
    private Canvas _canvas;

    [SerializeField, FormerlySerializedAs("pivot_")]
    private Transform _pivot;

    [SerializeField, FormerlySerializedAs("anchorMenu_")]
    private GameObject _anchorMenu;

    private bool _isSelected;

    private bool _isHovered;

    [SerializeField, FormerlySerializedAs("anchorName_")]
    private TextMeshProUGUI _anchorName;

    [SerializeField, FormerlySerializedAs("saveIcon_")]
    private GameObject _saveIcon;

    [SerializeField, FormerlySerializedAs("labelImage_")]
    private Image _labelImage;

    [SerializeField, FormerlySerializedAs("labelBaseColor_")]
    private Color _labelBaseColor;

    [SerializeField, FormerlySerializedAs("labelHighlightColor_")]
    private Color _labelHighlightColor;

    [SerializeField, FormerlySerializedAs("labelSelectedColor_")]
    private Color _labelSelectedColor;

    [SerializeField, FormerlySerializedAs("uiManager_")]
    private AnchorUIManager _uiManager;

    [SerializeField, FormerlySerializedAs("renderers_")]
    private MeshRenderer[] _renderers;

    private int _menuIndex = 0;

    [SerializeField, FormerlySerializedAs("buttonList_")]
    private List<Button> _buttonList;

    private Button _selectedButton;

    private OVRSpatialAnchor _spatialAnchor;

    private GameObject _icon;

    #region Monobehaviour Methods

    private void Awake()
    {
        _anchorMenu.SetActive(false);
        _renderers = GetComponentsInChildren<MeshRenderer>();
        _canvas.worldCamera = Camera.main;
        _selectedButton = _buttonList[0];
        _selectedButton.OnSelect(null);
        _spatialAnchor = GetComponent<OVRSpatialAnchor>();
        _icon = GetComponent<Transform>().FindChildRecursive("Sphere").gameObject;
    }

    private IEnumerator Start()
    {
        while (_spatialAnchor && !_spatialAnchor.Created)
        {
            yield return null;
        }

        if (_spatialAnchor)
        {
            _anchorName.text = _spatialAnchor.Uuid.ToString("D");
        }
        else
        {
            // Creation must have failed
            Destroy(gameObject);
        }
    }

    private void Update()
    {
        // Billboard the boundary
        BillboardPanel(_canvas.transform);

        // Billboard the menu
        BillboardPanel(_pivot);

        HandleMenuNavigation();

        //Billboard the icon
        BillboardPanel(_icon.transform);
    }

    #endregion // MonoBehaviour Methods

    #region UI Event Listeners

    /// <summary>
    /// UI callback for the anchor menu's Save button
    /// </summary>
    public void OnSaveLocalButtonPressed()
    {
        if (!_spatialAnchor) return;

        _spatialAnchor.Save((anchor, success) =>
        {
            if (!success) return;

            // Enables save icon on the menu
            ShowSaveIcon = true;

            // Write uuid of saved anchor to file
            if (!PlayerPrefs.HasKey(NumUuidsPlayerPref))
            {
                PlayerPrefs.SetInt(NumUuidsPlayerPref, 0);
            }

            int playerNumUuids = PlayerPrefs.GetInt(NumUuidsPlayerPref);
            PlayerPrefs.SetString("uuid" + playerNumUuids, anchor.Uuid.ToString());
            PlayerPrefs.SetInt(NumUuidsPlayerPref, ++playerNumUuids);
        });
    }

    /// <summary>
    /// UI callback for the anchor menu's Hide button
    /// </summary>
    public void OnHideButtonPressed()
    {
        Destroy(gameObject);
    }

    /// <summary>
    /// UI callback for the anchor menu's Erase button
    /// </summary>
    public void OnEraseButtonPressed()
    {
        if (!_spatialAnchor) return;

        _spatialAnchor.Erase((anchor, success) =>
        {
            if (success)
            {
                _saveIcon.SetActive(false);
            }
        });
    }

    #endregion // UI Event Listeners

    #region Public Methods

    public bool ShowSaveIcon
    {
        set => _saveIcon.SetActive(value);
    }

    /// <summary>
    /// Handles interaction when anchor is hovered
    /// </summary>
    public void OnHoverStart()
    {
        if (_isHovered)
        {
            return;
        }
        _isHovered = true;

        // Yellow highlight
        foreach (MeshRenderer renderer in _renderers)
        {
            renderer.material.SetColor("_EmissionColor", Color.yellow);
        }
        _labelImage.color = _labelHighlightColor;
    }

    /// <summary>
    /// Handles interaction when anchor is no longer hovered
    /// </summary>
    public void OnHoverEnd()
    {
        if (!_isHovered)
        {
            return;
        }
        _isHovered = false;

        // Go back to normal
        foreach (MeshRenderer renderer in _renderers)
        {
            renderer.material.SetColor("_EmissionColor", Color.clear);
        }

        if (_isSelected)
        {
            _labelImage.color = _labelSelectedColor;
        }
        else
        {
            _labelImage.color = _labelBaseColor;
        }
    }

    /// <summary>
    /// Handles interaction when anchor is selected
    /// </summary>
    public void OnSelect()
    {
        if (_isSelected)
        {
            // Hide Anchor menu on deselect
            _anchorMenu.SetActive(false);
            _isSelected = false;
            _selectedButton = null;
            if (_isHovered)
            {
                _labelImage.color = _labelHighlightColor;
            }
            else
            {
                _labelImage.color = _labelBaseColor;
            }
        }
        else
        {
            // Show Anchor Menu on select
            _anchorMenu.SetActive(true);
            _isSelected = true;
            _menuIndex = -1;
            NavigateToIndexInMenu(true);
            if (_isHovered)
            {
                _labelImage.color = _labelHighlightColor;
            }
            else
            {
                _labelImage.color = _labelSelectedColor;
            }
        }
    }

    #endregion // Public Methods

    #region Private Methods

    private void BillboardPanel(Transform panel)
    {
        // The z axis of the panel faces away from the side that is rendered, therefore this code is actually looking away from the camera
        panel.LookAt(new Vector3(panel.position.x * 2 - Camera.main.transform.position.x, panel.position.y * 2 - Camera.main.transform.position.y, panel.position.z * 2 - Camera.main.transform.position.z), Vector3.up);
    }

    private void HandleMenuNavigation()
    {
        if (!_isSelected)
        {
            return;
        }
        if (OVRInput.GetDown(OVRInput.RawButton.RThumbstickUp))
        {
            NavigateToIndexInMenu(false);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.RThumbstickDown))
        {
            NavigateToIndexInMenu(true);
        }
        if (OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
        {
            _selectedButton.OnSubmit(null);
        }
    }

    private void NavigateToIndexInMenu(bool moveNext)
    {
        if (moveNext)
        {
            _menuIndex++;
            if (_menuIndex > _buttonList.Count - 1)
            {
                _menuIndex = 0;
            }
        }
        else
        {
            _menuIndex--;
            if (_menuIndex < 0)
            {
                _menuIndex = _buttonList.Count - 1;
            }
        }
        if (_selectedButton)
        {
            _selectedButton.OnDeselect(null);
        }
        _selectedButton = _buttonList[_menuIndex];
        _selectedButton.OnSelect(null);
    }

    #endregion // Private Methods
}
