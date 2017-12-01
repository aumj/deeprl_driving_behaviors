/****************************************************************************/
/// @file    MSJunctionControl.cpp
/// @author  Christian Roessel
/// @author  Daniel Krajzewicz
/// @author  Michael Behrisch
/// @date    Tue, 06 Mar 2001
/// @version $Id: MSJunctionControl.cpp 24352 2017-05-18 09:28:53Z behrisch $
///
// Container for junctions; performs operations on all stored junctions
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


// ===========================================================================
// included modules
// ===========================================================================
#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include <algorithm>
#include "MSInternalJunction.h"
#include "MSJunctionControl.h"


// ===========================================================================
// member method definitions
// ===========================================================================
MSJunctionControl::MSJunctionControl() {
}


MSJunctionControl::~MSJunctionControl() {
}


void
MSJunctionControl::postloadInitContainer() {
    const std::map<std::string, MSJunction*>& junctionMap = getMyMap();
    // initialize normal junctions before internal junctions
    // (to allow calling getIndex() during initialization of internal junction links)
    for (std::map<std::string, MSJunction*>::const_iterator i = junctionMap.begin(); i != junctionMap.end(); ++i) {
        if (i->second->getType() != NODETYPE_INTERNAL) {
            i->second->postloadInit();
        }
    }
    for (std::map<std::string, MSJunction*>::const_iterator i = junctionMap.begin(); i != junctionMap.end(); ++i) {
        if (i->second->getType() == NODETYPE_INTERNAL) {
            i->second->postloadInit();
        }
    }
}


/****************************************************************************/

