/****************************************************************************/
/// @file    MSPModel.h
/// @author  Jakob Erdmann
/// @date    Mon, 13 Jan 2014
/// @version $Id: MSPModel.h 25918 2017-09-07 19:38:16Z behrisch $
///
// The pedestrian following model (prototype)
/****************************************************************************/
// SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
// Copyright (C) 2014-2017 DLR (http://www.dlr.de/) and contributors
/****************************************************************************/
//
//   This file is part of SUMO.
//   SUMO is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
/****************************************************************************/
#ifndef MSPModel_h
#define MSPModel_h

// ===========================================================================
// included modules
// ===========================================================================
#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include <string>
#include <limits>
#include <utils/common/SUMOTime.h>
#include <utils/common/Command.h>
#include <utils/geom/GeomHelper.h>
#include <microsim/pedestrians/MSPerson.h>

// ===========================================================================
// class declarations
// ===========================================================================
class MSNet;
class MSLane;
class MSJunction;

typedef std::pair<const MSPerson*, double> PersonDist;

// ===========================================================================
// class definitions
// ===========================================================================
/**
 * @class MSPModel
 * @brief The pedestrian following model
 *
 */
class MSPModel {
public:

    static MSPModel* getModel();

    /// @brief remove state at simulation end
    static void cleanup();

    virtual ~MSPModel() {};

    /// @brief register the given person as a pedestrian
    virtual PedestrianState* add(MSPerson* person, MSPerson::MSPersonStage_Walking* stage, SUMOTime now) = 0;

    /// @brief remove the specified person from the pedestrian simulation
    virtual void remove(PedestrianState* state) = 0;

    /** @brief whether a pedestrian is blocking the crossing of lane for the given vehicle bondaries
     * @param[in] lane The crossing to check
     * @param[in] vehSide The offset to the vehicle side near the start of the crossing
     * @param[in] vehWidth The width of the vehicle
     * @param[in] oncomingGap The distance which the vehicle wants to keep from oncoming pedestrians
     * @param[in] collectBlockers The list of persons blocking the crossing
     * @return Whether the vehicle must wait
     */
    virtual bool blockedAtDist(const MSLane* lane, double vehSide, double vehWidth,
                               double oncomingGap, std::vector<const MSPerson*>* collectBlockers) {
        UNUSED_PARAMETER(lane);
        UNUSED_PARAMETER(vehSide);
        UNUSED_PARAMETER(vehWidth);
        UNUSED_PARAMETER(oncomingGap);
        UNUSED_PARAMETER(collectBlockers);
        return false;
    }

    /// @brief whether the given lane has pedestrians on it
    virtual bool hasPedestrians(const MSLane* lane) {
        UNUSED_PARAMETER(lane);
        return false;
    }

    /// @brief returns the next pedestrian beyond minPos that is laterally between minRight and maxLeft or 0
    virtual PersonDist nextBlocking(const MSLane* lane, double minPos, double minRight, double maxLeft, double stopTime = 0) {
        UNUSED_PARAMETER(lane);
        UNUSED_PARAMETER(minPos);
        UNUSED_PARAMETER(minRight);
        UNUSED_PARAMETER(maxLeft);
        UNUSED_PARAMETER(stopTime);
        return PersonDist((const MSPerson*)0, -1);
    }

    virtual void cleanupHelper() {};

    // @brief walking directions
    static const int FORWARD;
    static const int BACKWARD;
    static const int UNDEFINED_DIRECTION;

    // @brief the safety gap to keep between the car and the pedestrian in all directions
    static const double SAFETY_GAP;

    /// @brief the offset for computing person positions when walking on edges without a sidewalk
    static const double SIDEWALK_OFFSET;

    /// @brief return whether the route may traversed with the given starting direction
    static bool canTraverse(int dir, const ConstMSEdgeVector& route);

private:
    static MSPModel* myModel;

};


/// @brief abstract base class for managing callbacks to retrieve various state information from the model
class PedestrianState {
public:
    virtual ~PedestrianState() {};

    /// @brief return the offset from the start of the current edge measured in its natural direction
    virtual double getEdgePos(const MSPerson::MSPersonStage_Walking& stage, SUMOTime now) const = 0;

    /// @brief return the network coordinate of the person
    virtual Position getPosition(const MSPerson::MSPersonStage_Walking& stage, SUMOTime now) const = 0;

    /// @brief return the direction in which the person faces in degrees
    virtual double getAngle(const MSPerson::MSPersonStage_Walking& stage, SUMOTime now) const = 0;

    /// @brief return the time the person spent standing
    virtual SUMOTime getWaitingTime(const MSPerson::MSPersonStage_Walking& stage, SUMOTime now) const = 0;

    /// @brief return the current speed of the person
    virtual double getSpeed(const MSPerson::MSPersonStage_Walking& stage) const = 0;

    /// @brief return the list of internal edges if the pedestrian is on an intersection
    virtual const MSEdge* getNextEdge(const MSPerson::MSPersonStage_Walking& stage) const = 0;
};



#endif /* MSPModel_h */

