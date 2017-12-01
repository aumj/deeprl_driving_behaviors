/****************************************************************************/
/// @file    GNERouteProbe.h
/// @author  Pablo Alvarez Lopez
/// @date    May 2016
/// @version $Id: GNERouteProbe.h 25918 2017-09-07 19:38:16Z behrisch $
///
//
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
#ifndef GNERouteProbe_h
#define GNERouteProbe_h


// ===========================================================================
// included modules
// ===========================================================================
#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include "GNEAdditional.h"

// ===========================================================================
// class declarations
// ===========================================================================

class GNEEdge;

// ===========================================================================
// class definitions
// ===========================================================================
/**
 * @class GNERouteProbe
 * ------------
 */
class GNERouteProbe : public GNEAdditional {
public:
    /**@brief Constructor
     * @param[in] id The storage of gl-ids to get the one for this lane representation from
     * @param[in] viewNet pointer to GNEViewNet of this additional element belongs
     * @param[in] edge edge in which this routeProbe is placed
     * @param[in] frequency The frequency in which to report the distribution
     * @param[in] filename The file for generated output
     * @param[in] begin The time at which to start generating output
     */
    GNERouteProbe(const std::string& id, GNEViewNet* viewNet, GNEEdge* edge, double frequency, const std::string& filename, double begin);

    /// @brief Destructor
    ~GNERouteProbe();

    /**@brief writte additional element into a xml file
     * @param[in] device device in which write parameters of additional element
     * @param[in] volatileOptionsEnabled flag to indicate that additionals are being saved with volatile options enabled
     */
    void writeAdditional(OutputDevice& device, bool volatileOptionsEnabled) const;

    /// @brief get filename of RouteProbe
    const std::string& getFilename() const;

    /// @brief get frequency of RouteProbe
    double getFrequency() const;

    /// @brief get begin of RouteProbe
    double getBegin() const;

    /// @brief set filename of RouteProbe
    void setFilename(const std::string& filename);

    /// @brief set frequency of RouteProbe
    void setFrequency(double frequency);

    /// @brief set begin of RouteProbe
    void setBegin(double begin);

    /// @name Functions related with geometry of element
    /// @{
    /**@brief change the position of the element geometry without saving in undoList
     * @param[in] newPosition new position of geometry
     * @note should't be called in drawGL(...) functions to avoid smoothness issues
     */
    void moveGeometry(const Position& newPosition);

    /**@brief commit geometry changes in the attributes of an element after use of moveGeometry(...)
     * @param[in] oldPos the old position of additional
     * @param[in] undoList The undoList on which to register changes
     */
    void commitGeometryMoving(const Position& oldPos, GNEUndoList* undoList);

    /// @brief update pre-computed geometry information
    void updateGeometry();

    /// @brief Returns position of additional in view
    Position getPositionInView() const;
    /// @}

    /// @name inherited from GUIGlObject
    /// @{
    /// @brief Returns the name of the parent object
    /// @return This object's parent id
    const std::string& getParentName() const;

    /**@brief Draws the object
     * @param[in] s The settings for the current view (may influence drawing)
     * @see GUIGlObject::drawGL
     */
    void drawGL(const GUIVisualizationSettings& s) const;
    /// @}

    /// @name inherited from GNEAttributeCarrier
    /// @{
    /* @brief method for getting the Attribute of an XML key
     * @param[in] key The attribute key
     * @return string with the value associated to key
     */
    std::string getAttribute(SumoXMLAttr key) const;

    /* @brief method for setting the attribute and letting the object perform additional changes
     * @param[in] key The attribute key
     * @param[in] value The new value
     * @param[in] undoList The undoList on which to register changes
     */
    void setAttribute(SumoXMLAttr key, const std::string& value, GNEUndoList* undoList);

    /* @brief method for checking if the key and their correspond attribute are valids
     * @param[in] key The attribute key
     * @param[in] value The value asociated to key key
     * @return true if the value is valid, false in other case
     */
    bool isValid(SumoXMLAttr key, const std::string& value);
    /// @}

protected:
    /// @brief Frequency of RouteProbe
    double myFrequency;

    /// @brief filename of RouteProbe
    std::string myFilename;

    /// @brief begin of rerouter
    double myBegin;

    /// @brief route probe logo offset
    Position myRouteProbeLogoOffset;

    /// @brief number of lanes of edge (To improve efficiency)
    int myNumberOfLanes;

    /// @brief relative position regarding to other route probes
    int myRelativePositionY;

private:
    /// @brief set attribute after validation
    void setAttribute(SumoXMLAttr key, const std::string& value);

    /// @brief Invalidated copy constructor.
    GNERouteProbe(const GNERouteProbe&);

    /// @brief Invalidated assignment operator.
    GNERouteProbe& operator=(const GNERouteProbe&);
};

#endif

/****************************************************************************/
