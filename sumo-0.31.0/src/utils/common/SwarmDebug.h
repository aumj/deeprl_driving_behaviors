/****************************************************************************/
/// @file    SwarmDebug.h
/// @author  Riccardo Belletti
/// @date    2014-03-21
/// @version $Id: SwarmDebug.h 25296 2017-07-22 18:29:42Z behrisch $
///
// Used for additional optional debug messages
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
#pragma once


//#ifndef ASSERT_H
//#define ASSERT_H
#include <assert.h>
//#endif


#ifndef SWARM_DEBUG
#define DBG(X) {}
#else
#define DBG(X) {X}
#endif/* DEBUG_H_ */
