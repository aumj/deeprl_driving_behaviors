/****************************************************************************/
/// @file    GNEDialog_FixAdditionalPositions.h
/// @author  Pablo Alvarez Lopez
/// @date    Jul 2017
/// @version $Id: GNEDialog_FixAdditionalPositions.h 25918 2017-09-07 19:38:16Z behrisch $
///
// Dialog used to fix invalid stopping places
/****************************************************************************/
// SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
// Copyright (C) 2001-2017 DLR (http://www.dlr.de/) and contributors
/****************************************************************************/
//
//   This file is part of SUMO.
//   SUMO is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
/****************************************************************************/
#ifndef GNEDialog_FixAdditionalPositions_h
#define GNEDialog_FixAdditionalPositions_h

// ===========================================================================
// included modules
// ===========================================================================

#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include <fx.h>

// ===========================================================================
// class declarations
// ===========================================================================
class GNEStoppingPlace;
class GNEDetector;
class GNEViewNet;

// ===========================================================================
// class definitions
// ===========================================================================

/**
 * @class GNEDialog_FixAdditionalPositions
 * @brief Dialog for edit rerouters
 */
class GNEDialog_FixAdditionalPositions : public FXDialogBox {
    /// @brief FOX-declaration
    FXDECLARE(GNEDialog_FixAdditionalPositions)

public:
    /// @brief Constructor
    GNEDialog_FixAdditionalPositions(GNEViewNet* viewNet, const std::vector<GNEStoppingPlace*>& invalidStoppingPlaces, const std::vector<GNEDetector*>& invalidDetectors);

    /// @brief destructor
    ~GNEDialog_FixAdditionalPositions();

    /// @name FOX-callbacks
    /// @{
    /// @brief event when user select a option
    long onCmdSelectOption(FXObject* obj, FXSelector, void*);

    /// @brief event after press accept button
    long onCmdAccept(FXObject*, FXSelector, void*);

    /// @brief event after press cancel button
    long onCmdCancel(FXObject*, FXSelector, void*);
    /// @}

protected:
    /// @brief FOX needs this
    GNEDialog_FixAdditionalPositions() {}

    /// @brief view net
    GNEViewNet* myViewNet;

    /// @brief vector with the invalid stoppingplaces
    std::vector<GNEStoppingPlace*> myInvalidStoppingPlaces;

    /// @brief vector with the invalid stoppingplaces
    std::vector<GNEDetector*> myInvalidDetectors;

    /// @brief list with the stoppingPlaces and detectors
    FXTable* myTable;

    /// @brief Option "Activate friendlyPos and save"
    FXRadioButton* myOptionA;

    /// @brief Option "Fix Positions and save"
    FXRadioButton* myOptionB;

    /// @brief Option "Save invalid"
    FXRadioButton* myOptionC;

    /// @brief Option "Select invalid stops and cancel"
    FXRadioButton* myOptionD;

    /// @brief accept button
    FXButton* myAcceptButton;

    /// @brief cancel button
    FXButton* myCancelButton;

private:
    /// @brief Invalidated copy constructor.
    GNEDialog_FixAdditionalPositions(const GNEDialog_FixAdditionalPositions&) = delete;

    /// @brief Invalidated assignment operator.
    GNEDialog_FixAdditionalPositions& operator=(const GNEDialog_FixAdditionalPositions&) = delete;
};

#endif
