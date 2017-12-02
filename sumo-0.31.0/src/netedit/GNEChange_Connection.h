/****************************************************************************/
/// @file    GNEChange_Connection.h
/// @author  Jakob Erdmann
/// @date    May 2011
/// @version $Id: GNEChange_Connection.h 24223 2017-05-07 11:15:04Z palcraft $
///
// A network change in which a single connection is created or deleted
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
#ifndef GNEChange_Connection_h
#define GNEChange_Connection_h


// ===========================================================================
// included modules
// ===========================================================================
#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include <fx.h>
#include <string>
#include <utils/foxtools/fxexdefs.h>
#include <netbuild/NBEdge.h>
#include "GNEChange.h"

// ===========================================================================
// class declarations
// ===========================================================================
class GNEEdge;
class GNEConnection;

// ===========================================================================
// class definitions
// ===========================================================================
/**
 * @class GNEChange_Connection
 * A network change in which a single lane is created or deleted
 */
class GNEChange_Connection : public GNEChange {
    FXDECLARE_ABSTRACT(GNEChange_Connection)

public:

    /**@brief Constructor for creating/deleting a connection
     * @param[in] edge The source edge of the connection
     * @param[in] nbCon The data of the connection
     * @param[in] check if in the moment of change connection was selected
     * @param[in] forward Whether to create/delete (true/false)
     */
    GNEChange_Connection(GNEEdge* edge, NBEdge::Connection nbCon, bool selected, bool forward);

    /// @brief Destructor
    ~GNEChange_Connection();

    /// @name inherited from GNEChange
    /// @{
    /// @brief get undo Name
    FXString undoName() const;

    /// @brief get Redo name
    FXString redoName() const;

    /// @brief undo action
    void undo();

    /// @brief redo action
    void redo();
    /// @}


private:
    // @brief the connection object to be removed/re-added
    GNEEdge* myEdge;

    /// @brief the data which must be copied because the original reference does not persist
    NBEdge::Connection myNBEdgeConnection;

    /// @brief flag to indicates if crossing was previously selected
    bool mySelected;
};

#endif
/****************************************************************************/
