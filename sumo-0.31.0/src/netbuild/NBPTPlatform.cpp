/****************************************************************************/
/// @file    NBPTPlatform.cpp
/// @author  Gregor Laemmel
/// @date    Tue, 24 Aug 2017
/// @version $Id: NBPTPlatform.cpp 25741 2017-08-30 11:58:01Z behrisch $
///
// The representation of a pt platform
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
#include "NBPTPlatform.h"
NBPTPlatform::NBPTPlatform(Position position, double d): myPos(position), myLength(d) {

}
Position* NBPTPlatform::getMyPos()  {
    return &myPos;
}
double NBPTPlatform::getMyLength() const {
    return myLength;
}
