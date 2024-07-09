// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

/// <summary>
/// Manages UI of anchor sample.
/// </summary>
[RequireComponent(typeof(SpatialAnchorLoader))]
public class AnchorUIManager : MonoBehaviour
{
    /// <summary>
    /// Anchor UI manager singleton instance
    /// </summary>
    public static AnchorUIManager Instance;

    /// <summary>
    /// Anchor Mode switches between create and select
    /// </summary>
    public enum AnchorMode { Create, Select };

    [SerializeField, FormerlySerializedAs("createModeButton_")]
    private GameObject _createModeButton;

    [SerializeField, FormerlySerializedAs("selectModeButton_")]
    private GameObject _selectModeButton;

    [SerializeField, FormerlySerializedAs("trackedDevice_")]
    private Transform _trackedDevice;

    private Transform _raycastOrigin;

    private bool _drawRaycast = false;

    [SerializeField, FormerlySerializedAs("lineRenderer_")]
    private LineRenderer _lineRenderer;

    private Anchor _hoveredAnchor;

    private Anchor _selectedAnchor;

    private AnchorMode _mode = AnchorMode.Create;

    [SerializeField, FormerlySerializedAs("buttonList_")]
    private List<Button> _buttonList;

    private int _menuIndex = 0;

    private Button _selectedButton;

    [SerializeField]
    private Anchor _anchorPrefab;

    public Anchor AnchorPrefab => _anchorPrefab;

    [SerializeField, FormerlySerializedAs("placementPreview_")]
    private GameObject _placementPreview;

    [SerializeField, FormerlySerializedAs("anchorPlacementTransform_")]
    private Transform _anchorPlacementTransform;

    private delegate void PrimaryPressDelegate();

    private PrimaryPressDelegate _primaryPressDelegate;

    private bool _isFocused = true;

    #region Monobehaviour Methods

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(this);
        }
    }

    private void Start()
    {
        _raycastOrigin = _trackedDevice;

        _selectedButton = _buttonList[0];
        _buttonList[0].OnSelect(null);

        _lineRenderer.startWidth = 0.005f;
        _lineRenderer.endWidth = 0.005f;

        ToggleCreateMode();
    }

    private void Update()
    {
        if (_drawRaycast)
        {
            ControllerRaycast();
        }

        if (_selectedAnchor == null)
        {
            // Refocus menu
            _selectedButton.OnSelect(null);
            _isFocused = true;
        }

        HandleMenuNavigation();

        if (OVRInput.GetDown(OVRInput.RawButton.A))
        {
            _primaryPressDelegate?.Invoke();
        }
    }

    #endregion // Monobehaviour Methods


    #region Menu UI Callbacks

    /// <summary>
    /// Create mode button pressed UI callback. Referenced by the Create button in the menu.
    /// </summary>
    public void OnCreateModeButtonPressed()
    {
        ToggleCreateMode();
        _createModeButton.SetActive(!_createModeButton.activeSelf);
        _selectModeButton.SetActive(!_selectModeButton.activeSelf);
    }

    /// <summary>
    /// Load anchors button pressed UI callback. Referenced by the Load Anchors button in the menu.
    /// </summary>
    public void OnLoadAnchorsButtonPressed()
    {
        GetComponent<SpatialAnchorLoader>().LoadAnchorsByUuid();
    }

    #endregion // Menu UI Callbacks

    #region Mode Handling

    private void ToggleCreateMode()
    {
        if (_mode == AnchorMode.Select)
        {
            _mode = AnchorMode.Create;
            EndSelectMode();
            StartPlacementMode();
        }
        else
        {
            _mode = AnchorMode.Select;
            EndPlacementMode();
            StartSelectMode();
        }
    }

    private void StartPlacementMode()
    {
        ShowAnchorPreview();
        _primaryPressDelegate = PlaceAnchor;
    }

    private void EndPlacementMode()
    {
        HideAnchorPreview();
        _primaryPressDelegate = null;
    }

    private void StartSelectMode()
    {
        ShowRaycastLine();
        _primaryPressDelegate = SelectAnchor;
    }

    private void EndSelectMode()
    {
        HideRaycastLine();
        _primaryPressDelegate = null;
    }

    #endregion // Mode Handling


    #region Private Methods

    private void HandleMenuNavigation()
    {
        if (!_isFocused)
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
        _selectedButton.OnDeselect(null);
        _selectedButton = _buttonList[_menuIndex];
        _selectedButton.OnSelect(null);
    }

    private void ShowAnchorPreview()
    {
        _placementPreview.SetActive(true);
    }

    private void HideAnchorPreview()
    {
        _placementPreview.SetActive(false);
    }

    private void PlaceAnchor()
    {
        Instantiate(_anchorPrefab, _anchorPlacementTransform.position, _anchorPlacementTransform.rotation);
    }

    private void ShowRaycastLine()
    {
        _drawRaycast = true;
        _lineRenderer.gameObject.SetActive(true);
    }

    private void HideRaycastLine()
    {
        _drawRaycast = false;
        _lineRenderer.gameObject.SetActive(false);
    }

    private void ControllerRaycast()
    {
        Ray ray = new Ray(_raycastOrigin.position, _raycastOrigin.TransformDirection(Vector3.forward));
        _lineRenderer.SetPosition(0, _raycastOrigin.position);
        _lineRenderer.SetPosition(1, _raycastOrigin.position + _raycastOrigin.TransformDirection(Vector3.forward) * 10f);

        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, Mathf.Infinity))
        {
            Anchor anchorObject = hit.collider.GetComponent<Anchor>();
            if (anchorObject != null)
            {
                _lineRenderer.SetPosition(1, hit.point);

                HoverAnchor(anchorObject);
                return;
            }
        }
        UnhoverAnchor();
    }

    private void HoverAnchor(Anchor anchor)
    {
        _hoveredAnchor = anchor;
        _hoveredAnchor.OnHoverStart();
    }

    private void UnhoverAnchor()
    {
        if (_hoveredAnchor == null)
        {
            return;
        }
        _hoveredAnchor.OnHoverEnd();
        _hoveredAnchor = null;
    }

    private void SelectAnchor()
    {
        if (_hoveredAnchor != null)
        {
            if (_selectedAnchor != null)
            {
                // Deselect previous Anchor
                _selectedAnchor.OnSelect();
                _selectedAnchor = null;
            }

            // Select new Anchor
            _selectedAnchor = _hoveredAnchor;
            _selectedAnchor.OnSelect();

            // Defocus menu
            _selectedButton.OnDeselect(null);
            _isFocused = false;
        }
        else
        {
            if (_selectedAnchor != null)
            {
                // Deselect previous Anchor
                _selectedAnchor.OnSelect();
                _selectedAnchor = null;

                // Refocus menu
                _selectedButton.OnSelect(null);
                _isFocused = true;
            }
        }
    }

    #endregion // Private Methods
}
