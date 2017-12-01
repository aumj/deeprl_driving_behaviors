/****************************************************************************/
/// @file    GNEViewNet.cpp
/// @author  Jakob Erdmann
/// @author  Pablo Alvarez Lopez
/// @date    Feb 2011
/// @version $Id: GNEViewNet.cpp 25995 2017-09-12 15:17:27Z palcraft $
///
// A view on the network being edited (adapted from GUIViewTraffic)
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

#include <iostream>
#include <utility>
#include <cmath>
#include <limits>
#include <foreign/polyfonts/polyfonts.h>
#include <utils/gui/globjects/GLIncludes.h>
#include <utils/foxtools/MFXImageHelper.h>
#include <utils/foxtools/MFXCheckableButton.h>
#include <utils/common/RGBColor.h>
#include <utils/options/OptionsCont.h>
#include <utils/geom/PositionVector.h>
#include <utils/gui/windows/GUISUMOAbstractView.h>
#include <utils/gui/windows/GUIDanielPerspectiveChanger.h>
#include <utils/gui/windows/GUIAppEnum.h>
#include <utils/gui/images/GUIIconSubSys.h>
#include <utils/gui/images/GUITextureSubSys.h>
#include <utils/gui/settings/GUICompleteSchemeStorage.h>
#include <utils/gui/images/GUITexturesHelper.h>
#include <utils/gui/windows/GUIDialog_ViewSettings.h>
#include <utils/gui/globjects/GUIGlObjectStorage.h>
#include <utils/gui/div/GLHelper.h>
#include <utils/gui/div/GUIGlobalSelection.h>
#include <utils/gui/div/GUIDesigns.h>
#include <utils/xml/XMLSubSys.h>
#include <netbuild/NBEdge.h>

#include "GNEViewNet.h"
#include "GNEEdge.h"
#include "GNELane.h"
#include "GNEJunction.h"
#include "GNEPOI.h"
#include "GNEApplicationWindow.h"
#include "GNEViewParent.h"
#include "GNENet.h"
#include "GNEUndoList.h"
#include "GNEInspectorFrame.h"
#include "GNESelectorFrame.h"
#include "GNEConnectorFrame.h"
#include "GNETLSEditorFrame.h"
#include "GNEAdditionalFrame.h"
#include "GNECrossingFrame.h"
#include "GNEPolygonFrame.h"
#include "GNEDeleteFrame.h"
#include "GNEAdditionalHandler.h"
#include "GNEPoly.h"
#include "GNECrossing.h"
#include "GNEAdditional.h"
#include "GNEAdditionalDialog.h"
#include "GNERerouter.h"
#include "GNEConnection.h"
#include "GNEChange_POI.h"
#include "GNEChange_Poly.h"


// ===========================================================================
// FOX callback mapping
// ===========================================================================
FXDEFMAP(GNEViewNet) GNEViewNetMap[] = {
    // Modes
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_CREATE_EDGE,                GNEViewNet::onCmdSetModeCreateEdge),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_MOVE,                       GNEViewNet::onCmdSetModeMove),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_DELETE,                     GNEViewNet::onCmdSetModeDelete),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_INSPECT,                    GNEViewNet::onCmdSetModeInspect),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_SELECT,                     GNEViewNet::onCmdSetModeSelect),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_CONNECT,                    GNEViewNet::onCmdSetModeConnect),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_TLS,                        GNEViewNet::onCmdSetModeTLS),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_ADDITIONAL,                 GNEViewNet::onCmdSetModeAdditional),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_CROSSING,                   GNEViewNet::onCmdSetModeCrossing),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_MODE_POLYGON,                    GNEViewNet::onCmdSetModePolygon),
    // Viewnet
    FXMAPFUNC(SEL_COMMAND, MID_GNE_VIEWNET_SHOW_CONNECTIONS,        GNEViewNet::onCmdToogleShowConnection),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_VIEWNET_SELECT_EDGES,            GNEViewNet::onCmdToogleSelectEdges),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_VIEWNET_SHOW_BUBBLES,            GNEViewNet::onCmdToogleShowBubbles),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_VIEWNET_SHOW_GRID,               GNEViewNet::onCmdShowGrid),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_VIEWNET_DELETE_GEOMETRY_POINT,   GNEViewNet::onCmdDeleteGeometryPoint),
    // Junctions
    FXMAPFUNC(SEL_COMMAND, MID_GNE_JUNCTION_EDIT_SHAPE,             GNEViewNet::onCmdEditJunctionShape),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_JUNCTION_REPLACE,                GNEViewNet::onCmdReplaceJunction),
    // Edges
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_SPLIT,                      GNEViewNet::onCmdSplitEdge),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_SPLIT_BIDI,                 GNEViewNet::onCmdSplitEdgeBidi),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_REVERSE,                    GNEViewNet::onCmdReverseEdge),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_ADD_REVERSE,                GNEViewNet::onCmdAddReversedEdge),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_SET_ENDPOINT,               GNEViewNet::onCmdSetEdgeEndpoint),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_RESET_ENDPOINT,             GNEViewNet::onCmdResetEdgeEndpoint),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_EDGE_STRAIGHTEN,                 GNEViewNet::onCmdStraightenEdges),
    // Lanes
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_DUPLICATE,                  GNEViewNet::onCmdDuplicateLane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_TRANSFORM_SIDEWALK,         GNEViewNet::onCmdRestrictLaneSidewalk),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_TRANSFORM_BIKE,             GNEViewNet::onCmdRestrictLaneBikelane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_TRANSFORM_BUS,              GNEViewNet::onCmdRestrictLaneBuslane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_REVERT_TRANSFORMATION,      GNEViewNet::onCmdRevertRestriction),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_ADD_SIDEWALK,               GNEViewNet::onCmdAddRestrictedLaneSidewalk),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_ADD_BIKE,                   GNEViewNet::onCmdAddRestrictedLaneBikelane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_ADD_BUS,                    GNEViewNet::onCmdAddRestrictedLaneBuslane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_REMOVE_SIDEWALK,            GNEViewNet::onCmdRemoveRestrictedLaneSidewalk),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_REMOVE_BIKE,                GNEViewNet::onCmdRemoveRestrictedLaneBikelane),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_LANE_REMOVE_BUS,                 GNEViewNet::onCmdRemoveRestrictedLaneBuslane),
    // Polygons
    FXMAPFUNC(SEL_COMMAND, MID_GNE_POLYGON_SIMPLIFY_SHAPE,          GNEViewNet::onCmdSimplifyShape),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_POLYGON_CLOSE,                   GNEViewNet::onCmdClosePolygon),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_POLYGON_OPEN,                    GNEViewNet::onCmdOpenPolygon),
    FXMAPFUNC(SEL_COMMAND, MID_GNE_POLYGON_SET_FIRST_POINT,         GNEViewNet::onCmdSetFirstGeometryPoint),
};


// Object implementation
FXIMPLEMENT(GNEViewNet, GUISUMOAbstractView, GNEViewNetMap, ARRAYNUMBER(GNEViewNetMap))


// ===========================================================================
// member method definitions
// ===========================================================================

GNEViewNet::GNEViewNet(FXComposite* tmpParent, FXComposite* actualParent, GUIMainWindow& app,
                       GNEViewParent* viewParent, GNENet* net, GNEUndoList* undoList,
                       FXGLVisual* glVis, FXGLCanvas* share, FXToolBar* toolBar) :
    GUISUMOAbstractView(tmpParent, app, viewParent, net->getVisualisationSpeedUp(), glVis, share),
    myViewParent(viewParent),
    myNet(net),
    myEditMode(GNE_MODE_MOVE),
    myCurrentFrame(0),
    myShowConnections(false),
    mySelectEdges(true),
    myCreateEdgeSource(0),
    myJunctionToMove(0),
    myEdgeToMove(0),
    myPolyToMove(0),
    myPoiToMove(0),
    myAdditionalToMove(0),
    myMovingIndexShape(-1),
    myMovingSelection(false),
    myAmInRectSelect(false),
    myToolbar(toolBar),
    myEditModeCreateEdge(0),
    myEditModeMove(0),
    myEditModeDelete(0),
    myEditModeInspect(0),
    myEditModeSelect(0),
    myEditModeConnection(0),
    myEditModeTrafficLight(0),
    myEditModeAdditional(0),
    myEditModeCrossing(0),
    myEditModePolygon(0),
    myEditModeNames(),
    myUndoList(undoList),
    myEditJunctionShapePoly(0),
    myTestingMode(OptionsCont::getOptions().getBool("gui-testing")) {
    // view must be the final member of actualParent
    reparent(actualParent);

    buildEditModeControls();
    myUndoList->mark();
    myNet->setViewNet(this);

    ((GUIDanielPerspectiveChanger*)myChanger)->setDragDelay(100000000); // 100 milliseconds

    if (myTestingMode && OptionsCont::getOptions().isSet("window-size")) {
        std::vector<std::string> windowSize = OptionsCont::getOptions().getStringVector("window-size");
        assert(windowSize.size() == 2);
        myTestingWidth = GNEAttributeCarrier::parse<int>(windowSize[0]);
        myTestingHeight = GNEAttributeCarrier::parse<int>(windowSize[1]);
    }
}


GNEViewNet::~GNEViewNet() { }


void
GNEViewNet::doInit() {}


void
GNEViewNet::buildViewToolBars(GUIGlChildWindow& cw) {
    // build coloring tools
    {
        for (auto it_names : gSchemeStorage.getNames()) {
            cw.getColoringSchemesCombo().appendItem(it_names.c_str());
            if (it_names == myVisualizationSettings->name) {
                cw.getColoringSchemesCombo().setCurrentItem(cw.getColoringSchemesCombo().getNumItems() - 1);
            }
        }
        cw.getColoringSchemesCombo().setNumVisible(MAX2(5, (int)gSchemeStorage.getNames().size() + 1));
    }
    // for junctions
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate Junction\tLocate a junction within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATEJUNCTION), &cw, MID_LOCATEJUNCTION,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
    // for edges
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate Street\tLocate a street within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATEEDGE), &cw, MID_LOCATEEDGE,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
    // for tls
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate TLS\tLocate a tls within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATETLS), &cw, MID_LOCATETLS,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
    // for additional stuff
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate Additional\tLocate an additional structure within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATEADD), &cw, MID_LOCATEADD,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
    // for pois
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate PoI\tLocate a PoI within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATEPOI), &cw, MID_LOCATEPOI,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
    // for polygons
    new FXButton(cw.getLocatorPopup(),
                 "\tLocate Polygon\tLocate a Polygon within the network.",
                 GUIIconSubSys::getIcon(ICON_LOCATEPOLY), &cw, MID_LOCATEPOLY,
                 ICON_ABOVE_TEXT | FRAME_THICK | FRAME_RAISED);
}


bool
GNEViewNet::setColorScheme(const std::string& name) {
    if (!gSchemeStorage.contains(name)) {
        return false;
    }
    if (myVisualizationChanger != 0) {
        if (myVisualizationChanger->getCurrentScheme() != name) {
            myVisualizationChanger->setCurrentScheme(name);
        }
    }
    myVisualizationSettings = &gSchemeStorage.get(name.c_str());
    update();
    return true;
}


void
GNEViewNet::buildColorRainbow(GUIColorScheme& scheme, int active, GUIGlObjectType objectType) {
    if (objectType == GLO_LANE) {
        assert(!scheme.isFixed());
        // retrieve range
        double minValue = std::numeric_limits<double>::infinity();
        double maxValue = -std::numeric_limits<double>::infinity();
        // XXX (see #3409) multi-colors are not currently handled. this is a quick hack
        if (active == 9) {
            active = 8; // segment height, fall back to start height
        }
        const std::vector<GNELane*> lanes = myNet->retrieveLanes();
        for (auto it : lanes) {
            const double val = it->getColorValue(active);
            minValue = MIN2(minValue, val);
            maxValue = MAX2(maxValue, val);
        }
        scheme.clear();
        // add new thresholds
        double range = maxValue - minValue;
        scheme.addColor(RGBColor::RED, (minValue));
        scheme.addColor(RGBColor::ORANGE, (minValue + range * 1 / 6.0));
        scheme.addColor(RGBColor::YELLOW, (minValue + range * 2 / 6.0));
        scheme.addColor(RGBColor::GREEN, (minValue + range * 3 / 6.0));
        scheme.addColor(RGBColor::CYAN, (minValue + range * 4 / 6.0));
        scheme.addColor(RGBColor::BLUE, (minValue + range * 5 / 6.0));
        scheme.addColor(RGBColor::MAGENTA, (maxValue));
    }
}


void
GNEViewNet::setStatusBarText(const std::string& text) {
    myApp->setStatusBarText(text);
}


bool
GNEViewNet::selectEdges() {
    return mySelectEdges;
}


bool
GNEViewNet::showConnections() {
    if (myEditMode == GNE_MODE_CONNECT) {
        return true;
    } else if (myMenuCheckShowConnections->shown() == false) {
        return false;
    } else {
        return (myVisualizationSettings->showLane2Lane);
    }
}


bool
GNEViewNet::autoSelectNodes() {
    return (myMenuCheckExtendToEdgeNodes->getCheck() != 0);
}


void
GNEViewNet::setSelectionScaling(double selectionScale) {
    myVisualizationSettings->selectionScale = selectionScale;
}


bool
GNEViewNet::changeAllPhases() const {
    return (myMenuCheckChangeAllPhases->getCheck() != 0);
}


bool
GNEViewNet::showJunctionAsBubbles() const {
    return (myEditMode == GNE_MODE_MOVE) && (myMenuCheckShowBubbleOverJunction->getCheck());
}


void
GNEViewNet::startEditShapeJunction(GNEJunction* junction) {
    if ((myEditJunctionShapePoly == NULL) && (junction != NULL) && (junction->getNBNode()->getShape().size() > 1)) {
        // save current edit mode before starting
        myPreviousEditMode = myEditMode;
        setEditModeFromHotkey(MID_GNE_MODE_MOVE);
        junction->getNBNode()->computeNodeShape(-1);
        // add special GNEPoly fo edit shapes
        myEditJunctionShapePoly = myNet->addPolygonForEditShapes(junction);
        // update view net to show the new myEditJunctionShapePoly
        update();
    }
}


void
GNEViewNet::stopEditShapeJunction() {
    // stop edit shape junction deleting myEditJunctionShapePoly
    if (myEditJunctionShapePoly != 0) {
        myNet->removePolygonOfView(myEditJunctionShapePoly);
        myNet->removePolygon(myEditJunctionShapePoly->getMicrosimID());
        myEditJunctionShapePoly = 0;
        // restore previous edit mode
        if (myEditMode != myPreviousEditMode) {
            setEditMode(myPreviousEditMode);
        }
    }

}


int
GNEViewNet::doPaintGL(int mode, const Boundary& bound) {
    // init view settings
    glRenderMode(mode);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // visualize rectangular selection
    if (myAmInRectSelect) {
        glPushMatrix();
        glTranslated(0, 0, GLO_MAX - 1);
        GLHelper::setColor(GNENet::selectionColor);
        glLineWidth(2);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_QUADS);
        glVertex2d(mySelCorner1.x(), mySelCorner1.y());
        glVertex2d(mySelCorner1.x(), mySelCorner2.y());
        glVertex2d(mySelCorner2.x(), mySelCorner2.y());
        glVertex2d(mySelCorner2.x(), mySelCorner1.y());
        glEnd();
        glPopMatrix();
    }

    // compute lane width
    double lw = m2p(SUMO_const_laneWidth);
    // draw decals (if not in grabbing mode)
    if (!myUseToolTips) {
        drawDecals();
        // depending of the visualizationSettings, enable or disable check box show grid
        if (myVisualizationSettings->showGrid) {
            myMenuCheckShowGrid->setCheck(true);
            paintGLGrid();
        } else {
            myMenuCheckShowGrid->setCheck(false);
        }
        myMenuCheckShowConnections->setCheck(myVisualizationSettings->showLane2Lane);
        if (myTestingMode) {
            if (myTestingWidth > 0 && (getWidth() != myTestingWidth || getHeight() != myTestingHeight)) {
                // only resize once to avoid flickering
                //std::cout << " before resize: view=" << getWidth() << ", " << getHeight() << " app=" << myApp->getWidth() << ", " << myApp->getHeight() << "\n";
                myApp->resize(myTestingWidth + myTestingWidth - getWidth(), myTestingHeight + myTestingHeight - getHeight());
                //std::cout << " directly after resize: view=" << getWidth() << ", " << getHeight() << " app=" << myApp->getWidth() << ", " << myApp->getHeight() << "\n";
                myTestingWidth = 0;
            }
            //std::cout << " fixed: view=" << getWidth() << ", " << getHeight() << " app=" << myApp->getWidth() << ", " << myApp->getHeight() << "\n";
            // draw pink square in the upper left corner on top of everything
            glPushMatrix();
            const double size = p2m(32);
            Position center = screenPos2NetPos(8, 8);
            GLHelper::setColor(RGBColor::MAGENTA);
            glTranslated(center.x(), center.y(), GLO_MAX - 1);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBegin(GL_QUADS);
            glVertex2d(0, 0);
            glVertex2d(0, -size);
            glVertex2d(size, -size);
            glVertex2d(size, 0);
            glEnd();
            glPopMatrix();

            // Reset textures due bug #2780. This solution is only provisional, and must be updated in the future
            GUITextureSubSys::resetTextures();
        }
    }

    // draw temporal shape of polygon during drawing
    if (myViewParent->getPolygonFrame()->getDrawingMode()->isDrawing()) {
        const PositionVector& temporalDrawingShape = myViewParent->getPolygonFrame()->getDrawingMode()->getTemporalShape();
        // draw blue line with the current drawed shape
        glPushMatrix();
        glLineWidth(2);
        GLHelper::setColor(RGBColor::BLUE);
        GLHelper::drawLine(temporalDrawingShape);
        glPopMatrix();
        // draw red line from the last point of shape to the current mouse position
        if (temporalDrawingShape.size() > 0) {
            glPushMatrix();
            glLineWidth(2);
            GLHelper::setColor(RGBColor::RED);
            GLHelper::drawLine(temporalDrawingShape.back(), snapToActiveGrid(getPositionInformation()));
            glPopMatrix();
        }
    }

    glLineWidth(1);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    const float minB[2] = { (float)bound.xmin(), (float)bound.ymin() };
    const float maxB[2] = { (float)bound.xmax(), (float)bound.ymax() };
    myVisualizationSettings->scale = lw;
    glEnable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_POLYGON_OFFSET_LINE);
    myVisualizationSettings->editMode = myEditMode;
    int hits2 = myGrid->Search(minB, maxB, *myVisualizationSettings);

    glTranslated(0, 0, GLO_ADDITIONAL);
    for (std::map<const GUIGlObject*, int>::iterator i = myAdditionallyDrawn.begin(); i != myAdditionallyDrawn.end(); ++i) {
        (i->first)->drawGLAdditional(this, *myVisualizationSettings);
    }

    glPopMatrix();
    return hits2;
}


long
GNEViewNet::onLeftBtnPress(FXObject*, FXSelector, void* eventData) {
    FXEvent* e = (FXEvent*) eventData;
    setFocus();
    // interpret object under curser
    if (makeCurrent()) {
        int id = getObjectUnderCursor();
        GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
        GUIGlObjectStorage::gIDStorage.unblockObject(id);
        GNEJunction* pointed_junction = 0;
        GNELane* pointed_lane = 0;
        GNEEdge* pointed_edge = 0;
        GNEPOI* pointed_poi = 0;
        GNEPoly* pointed_poly = 0;
        GNECrossing* pointed_crossing = 0;
        GNEAdditional* pointed_additional = 0;
        GNEConnection* pointed_connection = 0;
        if (pointed) {
            switch (pointed->getType()) {
                case GLO_JUNCTION:
                    pointed_junction = (GNEJunction*)pointed;
                    break;
                case GLO_EDGE:
                    pointed_edge = (GNEEdge*)pointed;
                    break;
                case GLO_LANE:
                    pointed_lane = (GNELane*)pointed;
                    pointed_edge = &(pointed_lane->getParentEdge());
                    break;
                case GLO_POI:
                    pointed_poi = (GNEPOI*)pointed;
                    break;
                case GLO_POLYGON:
                    pointed_poly = (GNEPoly*)pointed;
                    break;
                case GLO_CROSSING:
                    pointed_crossing = (GNECrossing*)pointed;
                    break;
                case GLO_ADDITIONAL:
                    pointed_additional = (GNEAdditional*)pointed;
                    break;
                case GLO_CONNECTION:
                    pointed_connection = (GNEConnection*)pointed;
                    break;
                default:
                    pointed = 0;
                    break;
            }
        }

        // decide what to do based on mode
        switch (myEditMode) {
            case GNE_MODE_CREATE_EDGE: {
                if ((e->state & CONTROLMASK) == 0) {
                    // allow moving when control is held down
                    if (!myUndoList->hasCommandGroup()) {
                        myUndoList->p_begin("create new " + toString(SUMO_TAG_EDGE));
                    }
                    if (!pointed_junction) {
                        pointed_junction = myNet->createJunction(snapToActiveGrid(getPositionInformation()), myUndoList);
                    }
                    if (myCreateEdgeSource == 0) {
                        myCreateEdgeSource = pointed_junction;
                        myCreateEdgeSource->markAsCreateEdgeSource();
                        update();
                    } else {
                        if (myCreateEdgeSource != pointed_junction) {
                            // may fail to prevent double edges
                            GNEEdge* newEdge = myNet->createEdge(
                                                   myCreateEdgeSource, pointed_junction, myViewParent->getInspectorFrame()->getEdgeTemplate(), myUndoList);
                            if (newEdge) {
                                // create another edge, if create opposite edge is enabled
                                if (myAutoCreateOppositeEdge->getCheck()) {
                                    myNet->createEdge(pointed_junction, myCreateEdgeSource, myViewParent->getInspectorFrame()->getEdgeTemplate(), myUndoList, "-" + newEdge->getNBEdge()->getID());
                                }
                                myCreateEdgeSource->unMarkAsCreateEdgeSource();

                                if (myUndoList->hasCommandGroup()) {
                                    myUndoList->p_end();
                                } else {
                                    std::cout << "edge created without an open CommandGroup )-:\n";
                                }
                                if (myChainCreateEdge->getCheck()) {
                                    myCreateEdgeSource = pointed_junction;
                                    myCreateEdgeSource->markAsCreateEdgeSource();
                                    myUndoList->p_begin("create new " + toString(SUMO_TAG_EDGE));
                                } else {
                                    myCreateEdgeSource = 0;
                                }
                            } else {
                                setStatusBarText("An " + toString(SUMO_TAG_EDGE) + " with the same geometry already exists!");
                            }
                        } else {
                            setStatusBarText("Start- and endpoint for an " + toString(SUMO_TAG_EDGE) + " must be distinct!");
                        }
                        update();
                    }
                }
                // process click
                processClick(e, eventData);
                break;
            }
            case GNE_MODE_MOVE: {
                if (pointed_poly) {
                    // set Poly to move
                    myPolyToMove = pointed_poly;
                    // save original shape (needed for commit change)
                    myMovingOriginalShape = myPolyToMove->getShape();
                    // obtain index of vertex to move if shape isn't blocked
                    if ((myPolyToMove->isShapeBlocked() == false) && (myPolyToMove->isMovementBlocked() == false)) {
                        // obtain index of vertex to move and moving reference
                        myMovingIndexShape = myPolyToMove->getVertexIndex(getPositionInformation());
                    } else {
                        myMovingIndexShape = -1;
                    }
                    // obtain moving reference
                    myMovingReference = getPositionInformation();
                } else if (pointed_poi) {
                    myPoiToMove = pointed_poi;
                    // Save original Position of Element and obtain moving reference
                    myMovingOriginalPosition = myPoiToMove->getPositionInView();
                    myMovingReference = getPositionInformation();
                } else if (pointed_junction) {
                    if (gSelected.isSelected(GLO_JUNCTION, pointed_junction->getGlID())) {
                        myMovingSelection = true;
                        // save position of current selected junctions (Needed when mouse is released)
                        std::vector<GNEJunction*> selectedJunctions = myNet->retrieveJunctions(true);
                        for (std::vector<GNEJunction*>::const_iterator i = selectedJunctions.begin(); i != selectedJunctions.end(); i++) {
                            myOriginPostionOfMovedJunctions[*i] = (*i)->getPositionInView();
                        }
                    } else {
                        myJunctionToMove = pointed_junction;
                    }
                    // Save original Position of Element and obtain moving reference
                    myMovingOriginalPosition = pointed_junction->getPositionInView();
                    myMovingReference = getPositionInformation();
                } else if (pointed_edge) {
                    if (gSelected.isSelected(GLO_EDGE, pointed_edge->getGlID())) {
                        myMovingSelection = true;
                    } else {
                        myEdgeToMove = pointed_edge;
                        myMovingOriginalShape = myEdgeToMove->getNBEdge()->getInnerGeometry();
                    }
                    myMovingOriginalPosition = getPositionInformation();
                } else if (pointed_additional) {
                    if (gSelected.isSelected(GLO_ADDITIONAL, pointed_additional->getGlID())) {
                        myMovingSelection = true;
                    } else {
                        myAdditionalToMove = pointed_additional;
                        if (myAdditionalToMove->getLane()) {
                            if (GNEAttributeCarrier::hasAttribute(myAdditionalToMove->getTag(), SUMO_ATTR_STARTPOS)) {
                                // Obtain start position
                                double startPos = GNEAttributeCarrier::parse<double>(myAdditionalToMove->getAttribute(SUMO_ATTR_STARTPOS));
                                if (GNEAttributeCarrier::hasAttribute(myAdditionalToMove->getTag(), SUMO_ATTR_ENDPOS)) {
                                    // Obtain end position
                                    double endPos = GNEAttributeCarrier::parse<double>(myAdditionalToMove->getAttribute(SUMO_ATTR_ENDPOS));
                                    // Save both values in myMovingOriginalPosition
                                    myMovingOriginalPosition.set(startPos, endPos);
                                } else if (GNEAttributeCarrier::hasAttribute(myAdditionalToMove->getTag(), SUMO_ATTR_LENGTH)) {
                                    // Obtain length attribute
                                    double length = GNEAttributeCarrier::parse<double>(myAdditionalToMove->getAttribute(SUMO_ATTR_LENGTH));
                                    // Save both values in myMovingOriginalPosition
                                    myMovingOriginalPosition.set(startPos, length);
                                } else {
                                    // Save only startpos in myMovingOriginalPosition
                                    myMovingOriginalPosition.set(startPos, 0);
                                }
                            } else if (GNEAttributeCarrier::hasAttribute(myAdditionalToMove->getTag(), SUMO_ATTR_POSITION)) {
                                myMovingOriginalPosition.set(GNEAttributeCarrier::parse<double>(myAdditionalToMove->getAttribute(SUMO_ATTR_POSITION)), 0);
                            }
                            // Set myMovingReference
                            myMovingReference.set(pointed_additional->getLane()->getShape().nearest_offset_to_point2D(getPositionInformation(), false), 0, 0);
                        } else {
                            // Save original Position of Element and obtain moving reference
                            myMovingOriginalPosition = pointed_additional->getPositionInView();
                            myMovingReference = getPositionInformation();
                        }
                    }
                } else {
                    // process click
                    processClick(e, eventData);
                }
                update();
                break;
            }
            case GNE_MODE_DELETE: {
                // Check if Control key is pressed
                bool markElementMode = (((FXEvent*)eventData)->state & CONTROLMASK) != 0;
                GNEAttributeCarrier* ac = dynamic_cast<GNEAttributeCarrier*>(pointed);
                if ((pointed_lane != NULL) && mySelectEdges) {
                    ac = pointed_edge;
                }
                if (ac) {
                    // if pointed element is an attribute carrier, remove it or mark it
                    if (markElementMode) {
                        if (myViewParent->getDeleteFrame()->getMarkedAttributeCarrier() != ac) {
                            myViewParent->getDeleteFrame()->markAttributeCarrier(ac);
                        }
                    } else if (myViewParent->getDeleteFrame()->getMarkedAttributeCarrier() != NULL) {
                        myViewParent->getDeleteFrame()->markAttributeCarrier(NULL);
                    } else {
                        myViewParent->getDeleteFrame()->removeAttributeCarrier(ac);
                    }
                } else {
                    // process click
                    processClick(e, eventData);
                }
                break;
            }
            case GNE_MODE_INSPECT: {
                GNEAttributeCarrier* pointedAC = 0;
                GUIGlObject* pointedO = 0;
                if (pointed_junction) {
                    pointedAC = pointed_junction;
                    pointedO = pointed_junction;
                } else if (pointed_lane) { // implies pointed_edge
                    if (mySelectEdges) {
                        pointedAC = pointed_edge;
                        pointedO = pointed_edge;
                    } else {
                        pointedAC = pointed_lane;
                        pointedO = pointed_lane;
                    }
                } else if (pointed_edge) {
                    pointedAC = pointed_edge;
                    pointedO = pointed_edge;
                } else if (pointed_crossing) {
                    pointedAC = pointed_crossing;
                    pointedO = pointed_crossing;
                } else if (pointed_additional) {
                    pointedAC = pointed_additional;
                    pointedO = pointed_additional;
                } else if (pointed_connection) {
                    pointedAC = pointed_connection;
                    pointedO = pointed_connection;
                } else if (pointed_poly) {
                    pointedAC = pointed_poly;
                    pointedO = pointed_poly;
                } else if (pointed_poi) {
                    pointedAC = pointed_poi;
                    pointedO = pointed_poi;
                }
                // obtain selected ACs
                std::vector<GNEAttributeCarrier*> selectedElements;
                std::vector<GNEAttributeCarrier*> selectedFilteredElements;
                if (pointedO && gSelected.isSelected(pointedO->getType(), pointedO->getGlID())) {
                    std::set<GUIGlID> selectedIDs = gSelected.getSelected(pointedO->getType());
                    selectedElements = myNet->retrieveAttributeCarriers(selectedIDs, pointedO->getType());
                    // filter selected elements (example: if we have two E2 and one busStop selected, and user click over one E2,
                    // attribues of busstop musn't be shown
                    for (auto i : selectedElements) {
                        if (i->getTag() == pointedAC->getTag()) {
                            selectedFilteredElements.push_back(i);
                        }
                    }
                }
                // Inspect seleted ACs, or single clicked AC
                if (selectedFilteredElements.size() > 0) {
                    myViewParent->getInspectorFrame()->inspectMultisection(selectedFilteredElements);
                } else if (pointedAC != NULL) {
                    myViewParent->getInspectorFrame()->inspectElement(pointedAC);
                }
                // process click
                processClick(e, eventData);
                // focus upper element of inspector frame
                if ((selectedFilteredElements.size() > 0) || (pointedAC != NULL)) {
                    myViewParent->getInspectorFrame()->focusUpperElement();
                }
                update();
                break;
            }
            case GNE_MODE_SELECT:
                if ((pointed_lane != NULL) && mySelectEdges) {
                    gSelected.toggleSelection(pointed_edge->getGlID());
                } else if (pointed) {
                    gSelected.toggleSelection(pointed->getGlID());
                }

                myAmInRectSelect = (((FXEvent*)eventData)->state & SHIFTMASK) != 0;
                if (myAmInRectSelect) {
                    mySelCorner1 = getPositionInformation();
                    mySelCorner2 = getPositionInformation();
                } else {
                    // process click
                    processClick(e, eventData);
                }
                update();
                break;

            case GNE_MODE_CONNECT: {
                if (pointed_lane) {
                    const bool mayPass = (((FXEvent*)eventData)->state & SHIFTMASK) != 0;
                    const bool allowConflict = (((FXEvent*)eventData)->state & CONTROLMASK) != 0;
                    myViewParent->getConnectorFrame()->handleLaneClick(pointed_lane, mayPass, allowConflict, true);
                    update();
                }
                // process click
                processClick(e, eventData);
                break;
            }
            case GNE_MODE_TLS: {
                if (pointed_junction) {
                    myViewParent->getTLSEditorFrame()->editJunction(pointed_junction);
                    update();
                }
                // process click
                processClick(e, eventData);
                break;
            }
            case GNE_MODE_ADDITIONAL: {
                if (pointed_additional == NULL) {
                    GNENetElement* netElement = dynamic_cast<GNENetElement*>(pointed);
                    GNEAdditionalFrame::AddAdditionalResult result = myViewParent->getAdditionalFrame()->addAdditional(netElement, this);
                    // process click or update view depending of the result of "add additional"
                    if ((result == GNEAdditionalFrame::ADDADDITIONAL_SUCCESS) || (result == GNEAdditionalFrame::ADDADDITIONAL_INVALID_PARENT)) {
                        update();
                        // process click
                        processClick(e, eventData);
                    }
                }

                break;
            }
            case GNE_MODE_CROSSING: {
                if (pointed_crossing == NULL) {
                    GNENetElement* netElement = dynamic_cast<GNENetElement*>(pointed);
                    if (myViewParent->getCrossingFrame()->addCrossing(netElement)) {
                        update();
                    }
                }
                // process click
                processClick(e, eventData);
                break;
            }
            case GNE_MODE_POLYGON: {
                if (pointed_poi == NULL) {
                    GNEPolygonFrame::AddShapeResult result = myViewParent->getPolygonFrame()->processClick(snapToActiveGrid(getPositionInformation()));
                    // view net must be always update
                    update();
                    // process clickw depending of the result of "add additional"
                    if ((result != GNEPolygonFrame::ADDSHAPE_NEWPOINT)) {
                        // process click
                        processClick(e, eventData);
                    }
                }
                break;
            }
            default: {
                // process click
                processClick(e, eventData);
            }
        }
        makeNonCurrent();
    }
    return 1;
}


long
GNEViewNet::onLeftBtnRelease(FXObject* obj, FXSelector sel, void* eventData) {
    GUISUMOAbstractView::onLeftBtnRelease(obj, sel, eventData);
    if (myPolyToMove) {
        myPolyToMove->commitShapeChange(myMovingOriginalShape, myUndoList);
        myPolyToMove = 0;
    } else if (myPoiToMove) {
        myPoiToMove->commitGeometryMoving(myMovingOriginalPosition, myUndoList);
        myPoiToMove = 0;
    } else if (myJunctionToMove) {
        // position is already up to date but we must register with myUndoList
        if (!mergeJunctions(myJunctionToMove)) {
            myJunctionToMove->commitGeometryMoving(myMovingOriginalPosition, myUndoList);
        }
        myJunctionToMove = 0;
    } else if (myEdgeToMove) {
        // set cleaned shape
        myEdgeToMove->commitGeometryMoving(myMovingOriginalShape,
                                           MAX2(POSITION_EPS, GNEJunction::BUBBLE_RADIUS * myVisualizationSettings->junctionSize.exaggeration),
                                           myUndoList);
        myEdgeToMove = 0;
    } else if (myAdditionalToMove) {
        myAdditionalToMove->commitGeometryMoving(myMovingOriginalPosition, myUndoList);
        myAdditionalToMove = 0;
    } else if (myMovingSelection) {
        // positions and shapes are already up to date but we must register with myUndoList
        myNet->finishMoveSelection(myUndoList);
        // commit new positions of selected junctions
        if (myOriginPostionOfMovedJunctions.size() > 0) {
            myUndoList->p_begin("position of selected elements");
            for (std::map<GNEJunction*, Position>::const_iterator i = myOriginPostionOfMovedJunctions.begin(); i != myOriginPostionOfMovedJunctions.end(); i++) {
                i->first->commitGeometryMoving(i->second, myUndoList);
            }
            myUndoList->p_end();
            myOriginPostionOfMovedJunctions.clear();
        }
        myMovingSelection = false;
    } else if (myAmInRectSelect) {
        myAmInRectSelect = false;
        // shift held down on mouse-down and mouse-up
        if (((FXEvent*)eventData)->state & SHIFTMASK) {
            if (makeCurrent()) {
                Boundary b;
                b.add(mySelCorner1);
                b.add(mySelCorner2);
                myViewParent->getSelectorFrame()->handleIDs(getObjectsInBoundary(b), mySelectEdges);
                makeNonCurrent();
            }
        }
        update();
    }
    return 1;
}


long GNEViewNet::onRightBtnPress(FXObject* obj, FXSelector sel, void* eventData) {
    if ((myEditMode == GNE_MODE_POLYGON) && myViewParent->getPolygonFrame()->getDrawingMode()->isDrawing()) {
        // during drawing of a polygon, right click removes the last created point
        myViewParent->getPolygonFrame()->getDrawingMode()->removeLastPoint();
        // update view
        update();
        return 1;
    } else {
        return GUISUMOAbstractView::onRightBtnPress(obj, sel, eventData);
    }
}


long GNEViewNet::onRightBtnRelease(FXObject* obj, FXSelector sel, void* eventData) {
    if ((myEditMode == GNE_MODE_POLYGON) && myViewParent->getPolygonFrame()->getDrawingMode()->isDrawing()) {
        // during drawing of a polygon, right click removes the last created point
        return 1;
    } else {
        return GUISUMOAbstractView::onRightBtnRelease(obj, sel, eventData);
    }
}


long
GNEViewNet::onDoubleClicked(FXObject*, FXSelector, void*) {
    // If current edit mode is INSPECT or ADDITIONAL
    if (myEditMode == GNE_MODE_INSPECT || myEditMode == GNE_MODE_ADDITIONAL) {
        setFocus();
        // interpret object under curser
        if (makeCurrent()) {
            int id = getObjectUnderCursor();
            GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
            GUIGlObjectStorage::gIDStorage.unblockObject(id);
            GNEAdditional* pointed_additional = dynamic_cast<GNEAdditional*>(pointed);
            // If pointed element is an additional
            if (pointed_additional != NULL) {
                // If additional has a additional dialog, open it.
                pointed_additional->openAdditionalDialog();
            }
            makeNonCurrent();
        }
    }
    return 1;
}


long
GNEViewNet::onMouseMove(FXObject* obj, FXSelector sel, void* eventData) {
    GUISUMOAbstractView::onMouseMove(obj, sel, eventData);
    // in delete mode object under cursor must be checked in every mouse movement
    if (myEditMode == GNE_MODE_DELETE) {
        setFocus();
        // show object information in delete frame
        if (makeCurrent()) {
            // obtain ac of globjectID
            int glid = getObjectUnderCursor();
            GNEAttributeCarrier* ac = dynamic_cast<GNEAttributeCarrier*>(GUIGlObjectStorage::gIDStorage.getObjectBlocking(glid));
            GUIGlObjectStorage::gIDStorage.unblockObject(glid);
            // Update current label of delete frame
            myViewParent->getDeleteFrame()->updateCurrentLabel(ac);
        }
    } else {
        if (myPolyToMove) {
            // Calculate movement offset and move geometry of shape
            Position offsetPosition = myMovingReference - getPositionInformation();
            if (myPolyToMove->isShapeBlocked()) {
                myPolyToMove->moveEntireShape(myMovingOriginalShape, snapToActiveGrid(offsetPosition));
            } else {
                myMovingIndexShape = myPolyToMove->moveVertexShape(myMovingIndexShape, snapToActiveGrid(getPositionInformation()));
            }
        } else if (myPoiToMove) {
            // Calculate movement offset and move geometry of junction
            Position offsetPosition = myMovingReference - getPositionInformation();
            myPoiToMove->moveGeometry(snapToActiveGrid(myMovingOriginalPosition - offsetPosition));
        } else if (myJunctionToMove) {
            // check if  one of their junctions neighboors is in the position objective
            std::vector<GNEJunction*> junctionNeighbours = myJunctionToMove->getJunctionNeighbours();
            for (auto i : junctionNeighbours) {
                if (i->getPositionInView() == getPositionInformation()) {
                    return 0;
                }
            }
            // Calculate movement offset and move geometry of junction
            Position offsetPosition = myMovingReference - getPositionInformation();
            myJunctionToMove->moveJunctionGeometry2D(snapToActiveGrid(myMovingOriginalPosition - offsetPosition));
        } else if (myEdgeToMove) {
            Position newPosition = snapToActiveGrid(getPositionInformation());
            double minimumMovingRadius = GNEJunction::BUBBLE_RADIUS / 2;
            if ((newPosition.distanceTo2D(myEdgeToMove->getGNEJunctionSource()->getPositionInView()) >= minimumMovingRadius) &&
                    (newPosition.distanceTo2D(myEdgeToMove->getGNEJunctionDestiny()->getPositionInView()) >= minimumMovingRadius)) {
                myMovingOriginalPosition = myEdgeToMove->moveGeometry(myMovingOriginalPosition, newPosition);
            }
        } else if (myAdditionalToMove  && (myAdditionalToMove->isAdditionalBlocked() == false)) {
            // If additional is placed over lane, move it across it
            if (myAdditionalToMove->getLane()) {
                double posOfMouseOverLane = myAdditionalToMove->getLane()->getShape().nearest_offset_to_point2D(getPositionInformation(), false);
                myAdditionalToMove->moveGeometry(Position(posOfMouseOverLane - myMovingReference.x(), 0));
                myMovingReference.set(posOfMouseOverLane, 0, 0);
            } else {
                // Calculate movement offset and move geometry of junction
                Position offsetPosition = myMovingReference - getPositionInformation();
                myAdditionalToMove->moveGeometry(snapToActiveGrid(myMovingOriginalPosition - offsetPosition));
            }
        } else if (myMovingSelection) {
            Position moveTarget = getPositionInformation();
            myNet->moveSelection(myMovingOriginalPosition, moveTarget);
            myMovingOriginalPosition = moveTarget;
        } else if (myAmInRectSelect) {
            mySelCorner2 = getPositionInformation();
        }
    }

    // update view
    update();
    return 1;
}


long
GNEViewNet::onKeyPress(FXObject* o, FXSelector sel, void* eventData) {
    return GUISUMOAbstractView::onKeyPress(o, sel, eventData);
}


long
GNEViewNet::onKeyRelease(FXObject* o, FXSelector sel, void* eventData) {
    if (myAmInRectSelect && ((((FXEvent*)eventData)->state & SHIFTMASK) == false)) {
        myAmInRectSelect = false;
        update();
    }
    return GUISUMOAbstractView::onKeyRelease(o, sel, eventData);
}


void
GNEViewNet::abortOperation(bool clearSelection) {
    // steal focus from any text fields
    setFocus();
    if (myCreateEdgeSource != NULL) {
        // remove current created edge source
        myCreateEdgeSource->unMarkAsCreateEdgeSource();
        myCreateEdgeSource = 0;
    } else if (myEditMode == GNE_MODE_SELECT) {
        myAmInRectSelect = false;
        if (clearSelection) {
            gSelected.clear();
        }
    } else if (myEditMode == GNE_MODE_CONNECT) {
        myViewParent->getConnectorFrame()->onCmdCancel(0, 0, 0);
    } else if (myEditMode == GNE_MODE_TLS) {
        myViewParent->getTLSEditorFrame()->onCmdCancel(0, 0, 0);
    } else if (myEditMode == GNE_MODE_MOVE) {
        stopEditShapeJunction();
    } else if (myEditMode == GNE_MODE_POLYGON) {
        // abort current drawing
        myViewParent->getPolygonFrame()->getDrawingMode()->abortDrawing();
    }
    myUndoList->p_abort();
}


void
GNEViewNet::hotkeyDel() {
    if (myEditMode == GNE_MODE_CONNECT || myEditMode == GNE_MODE_TLS) {
        setStatusBarText("Cannot delete in this mode");
    } else {
        myUndoList->p_begin("delete selection");
        deleteSelectedConnections();
        deleteSelectedCrossings();
        deleteSelectedAdditionals();
        deleteSelectedLanes();
        deleteSelectedEdges();
        deleteSelectedJunctions();
        deleteSelectedPOIs();
        deleteSelectedPolygons();
        myUndoList->p_end();
    }
}


void
GNEViewNet::hotkeyEnter() {
    if (myEditMode == GNE_MODE_CONNECT) {
        myViewParent->getConnectorFrame()->onCmdOK(0, 0, 0);
    } else if (myEditMode == GNE_MODE_TLS) {
        myViewParent->getTLSEditorFrame()->onCmdOK(0, 0, 0);
    } else if ((myEditMode == GNE_MODE_MOVE) && (myEditJunctionShapePoly != 0)) {
        // save edited junction's shape
        if (myEditJunctionShapePoly != 0) {
            myUndoList->p_begin("custom junction shape");
            myEditJunctionShapePoly->getShapeEditedJunction()->setAttribute(SUMO_ATTR_SHAPE, toString(myEditJunctionShapePoly->getShape()), myUndoList);
            myUndoList->p_end();
            stopEditShapeJunction();
            update();
        }
    } else if (myEditMode == GNE_MODE_POLYGON) {
        if (myViewParent->getPolygonFrame()->getDrawingMode()->isDrawing()) {
            // stop current drawing
            myViewParent->getPolygonFrame()->getDrawingMode()->stopDrawing();
        } else {
            // start drawing
            myViewParent->getPolygonFrame()->getDrawingMode()->startDrawing();
        }
    }
}


void
GNEViewNet::hotkeyFocusFrame() {
    // if there is a visible frame, set focus over it. In other case, set focus over ViewNet
    if (myCurrentFrame != NULL) {
        myCurrentFrame->focusUpperElement();
    } else {
        setFocus();
    }
}


void
GNEViewNet::setEditModeFromHotkey(FXushort selid) {
    switch (selid) {
        case MID_GNE_MODE_CREATE_EDGE:
            setEditMode(GNE_MODE_CREATE_EDGE);
            break;
        case MID_GNE_MODE_MOVE:
            setEditMode(GNE_MODE_MOVE);
            break;
        case MID_GNE_MODE_DELETE:
            setEditMode(GNE_MODE_DELETE);
            break;
        case MID_GNE_MODE_INSPECT:
            setEditMode(GNE_MODE_INSPECT);
            break;
        case MID_GNE_MODE_SELECT:
            setEditMode(GNE_MODE_SELECT);
            break;
        case MID_GNE_MODE_CONNECT:
            setEditMode(GNE_MODE_CONNECT);
            break;
        case MID_GNE_MODE_TLS:
            setEditMode(GNE_MODE_TLS);
            break;
        case MID_GNE_MODE_ADDITIONAL:
            setEditMode(GNE_MODE_ADDITIONAL);
            break;
        case MID_GNE_MODE_CROSSING:
            setEditMode(GNE_MODE_CROSSING);
            break;
        case MID_GNE_MODE_POLYGON:
            setEditMode(GNE_MODE_POLYGON);
            break;
        default:
            throw ProcessError("invalid edit mode called by hotkey");
            break;
    }
}


void
GNEViewNet::markPopupPosition() {
    myPopupSpot = getPositionInformation();
}


GNEViewParent*
GNEViewNet::getViewParent() const {
    return myViewParent;
}


GNENet*
GNEViewNet::getNet() const {
    return myNet;
}


GNEUndoList*
GNEViewNet::getUndoList() const {
    return myUndoList;
}


EditMode
GNEViewNet::getCurrentEditMode() const {
    return myEditMode;
}


bool
GNEViewNet::showLockIcon() const {
    return (myEditMode == GNE_MODE_MOVE || myEditMode == GNE_MODE_INSPECT || myEditMode == GNE_MODE_ADDITIONAL);
}


GNEJunction*
GNEViewNet::getJunctionAtCursorPosition(Position& /* pos */) {
    GNEJunction* junction = 0;
    if (makeCurrent()) {
        int id = getObjectAtPosition(myPopupSpot);
        GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
        GUIGlObjectStorage::gIDStorage.unblockObject(id);
        if (pointed) {
            switch (pointed->getType()) {
                case GLO_JUNCTION:
                    junction = (GNEJunction*)pointed;
                    break;
                default:
                    break;
            }
        }
    }
    return junction;
}


GNEEdge*
GNEViewNet::getEdgeAtCursorPosition(Position& /* pos */) {
    GNEEdge* edge = 0;
    if (makeCurrent()) {
        int id = getObjectAtPosition(myPopupSpot);
        GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
        GUIGlObjectStorage::gIDStorage.unblockObject(id);
        if (pointed) {
            switch (pointed->getType()) {
                case GLO_EDGE:
                    edge = (GNEEdge*)pointed;
                    break;
                case GLO_LANE:
                    edge = &(((GNELane*)pointed)->getParentEdge());
                    break;
                default:
                    break;
            }
        }
    }
    return edge;
}


GNELane*
GNEViewNet::getLaneAtCurserPosition(Position& /* pos */) {
    GNELane* lane = 0;
    if (makeCurrent()) {
        int id = getObjectAtPosition(myPopupSpot);
        GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
        GUIGlObjectStorage::gIDStorage.unblockObject(id);
        if (pointed) {
            if (pointed->getType() == GLO_LANE) {
                lane = (GNELane*)pointed;
            }
        }
    }
    return lane;
}


std::set<GNEEdge*>
GNEViewNet::getEdgesAtCursorPosition(Position& /* pos */) {
    std::set<GNEEdge*> result;
    if (makeCurrent()) {
        const std::vector<GUIGlID> ids = getObjectsAtPosition(myPopupSpot, 1.0);
        for (auto it : ids) {
            GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(it);
            GUIGlObjectStorage::gIDStorage.unblockObject(it);
            if (pointed) {
                switch (pointed->getType()) {
                    case GLO_EDGE:
                        result.insert((GNEEdge*)pointed);
                        break;
                    case GLO_LANE:
                        result.insert(&(((GNELane*)pointed)->getParentEdge()));
                        break;
                    default:
                        break;
                }
            }
        }
    }
    return result;
}


GNEPoly*
GNEViewNet::getPolygonAtCursorPosition(Position& pos) {
    if (makeCurrent()) {
        int id = getObjectAtPosition(pos);
        GUIGlObject* pointed = GUIGlObjectStorage::gIDStorage.getObjectBlocking(id);
        GUIGlObjectStorage::gIDStorage.unblockObject(id);
        if (pointed && (pointed->getType() == GLO_POLYGON)) {
            return dynamic_cast<GNEPoly*>(pointed);
        }
    }
    return 0;
}


long
GNEViewNet::onCmdSetModeCreateEdge(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_CREATE_EDGE);
    return 1;
}


long
GNEViewNet::onCmdSetModeMove(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_MOVE);
    return 1;
}


long
GNEViewNet::onCmdSetModeDelete(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_DELETE);
    return 1;
}


long
GNEViewNet::onCmdSetModeInspect(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_INSPECT);
    return 1;
}


long
GNEViewNet::onCmdSetModeSelect(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_SELECT);
    return 1;
}


long
GNEViewNet::onCmdSetModeConnect(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_CONNECT);
    return 1;
}


long
GNEViewNet::onCmdSetModeTLS(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_TLS);
    return 1;
}


long
GNEViewNet::onCmdSetModeAdditional(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_ADDITIONAL);
    return 1;
}


long
GNEViewNet::onCmdSetModeCrossing(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_CROSSING);
    return 1;
}


long
GNEViewNet::onCmdSetModePolygon(FXObject*, FXSelector, void*) {
    setEditMode(GNE_MODE_POLYGON);
    return 1;
}


long
GNEViewNet::onCmdSplitEdge(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        myNet->splitEdge(edge, edge->getSplitPos(myPopupSpot), myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdSplitEdgeBidi(FXObject*, FXSelector, void*) {
    std::set<GNEEdge*> edges = getEdgesAtCursorPosition(myPopupSpot);
    if (edges.size() != 0) {
        myNet->splitEdgesBidi(edges, (*edges.begin())->getSplitPos(myPopupSpot), myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdReverseEdge(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        myNet->reverseEdge(edge, myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdAddReversedEdge(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        myNet->addReversedEdge(edge, myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdSetEdgeEndpoint(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        edge->setEndpoint(myPopupSpot, myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdResetEdgeEndpoint(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        edge->resetEndpoint(myPopupSpot, myUndoList);
    }
    return 1;
}


long
GNEViewNet::onCmdStraightenEdges(FXObject*, FXSelector, void*) {
    GNEEdge* edge = getEdgeAtCursorPosition(myPopupSpot);
    if (edge != 0) {
        if (gSelected.isSelected(GLO_EDGE, edge->getGlID())) {
            myUndoList->p_begin("straighten selected " + toString(SUMO_TAG_EDGE) + "s");
            std::vector<GNEEdge*> edges = myNet->retrieveEdges(true);
            for (auto it : edges) {
                it->setAttribute(SUMO_ATTR_SHAPE, "", myUndoList);
            }
            myUndoList->p_end();
        } else {
            myUndoList->p_begin("straighten " + toString(SUMO_TAG_EDGE));
            edge->setAttribute(SUMO_ATTR_SHAPE, "", myUndoList);
            myUndoList->p_end();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdSimplifyShape(FXObject*, FXSelector, void*) {
    if (myEditJunctionShapePoly != 0) {
        myEditJunctionShapePoly->simplifyShape(false);
        update();
    } else {
        GNEPoly* polygonUnderMouse = getPolygonAtCursorPosition(myPopupSpot);
        if (polygonUnderMouse) {
            polygonUnderMouse->simplifyShape();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdDeleteGeometryPoint(FXObject*, FXSelector, void*) {
    if (myEditJunctionShapePoly != 0) {
        myEditJunctionShapePoly->deleteGeometryNear(myPopupSpot, false);
        update();
    } else {
        GNEPoly* polygonUnderMouse = getPolygonAtCursorPosition(myPopupSpot);
        if (polygonUnderMouse) {
            polygonUnderMouse->deleteGeometryNear(myPopupSpot);
        }
    }
    return 1;
}


long
GNEViewNet::onCmdClosePolygon(FXObject*, FXSelector, void*) {
    if (myEditJunctionShapePoly != 0) {
        myEditJunctionShapePoly->closePolygon(false);
        update();
    } else {
        GNEPoly* polygonUnderMouse = getPolygonAtCursorPosition(myPopupSpot);
        if (polygonUnderMouse) {
            polygonUnderMouse->closePolygon();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdOpenPolygon(FXObject*, FXSelector, void*) {
    if (myEditJunctionShapePoly != 0) {
        myEditJunctionShapePoly->openPolygon(false);
        update();
    } else {
        GNEPoly* polygonUnderMouse = getPolygonAtCursorPosition(myPopupSpot);
        if (polygonUnderMouse) {
            polygonUnderMouse->openPolygon();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdSetFirstGeometryPoint(FXObject*, FXSelector, void*) {
    if (myEditJunctionShapePoly != 0) {
        myEditJunctionShapePoly->changeFirstGeometryPoint(myEditJunctionShapePoly->getVertexIndex(myPopupSpot, false), false);
        update();
    } else {
        GNEPoly* polygonUnderMouse = getPolygonAtCursorPosition(myPopupSpot);
        if (polygonUnderMouse) {
            polygonUnderMouse->changeFirstGeometryPoint(polygonUnderMouse->getVertexIndex(myPopupSpot, false));
        }
    }
    return 1;
}


long
GNEViewNet::onCmdDuplicateLane(FXObject*, FXSelector, void*) {
    GNELane* lane = getLaneAtCurserPosition(myPopupSpot);
    if (lane != 0) {
        if (gSelected.isSelected(GLO_LANE, lane->getGlID())) {
            myUndoList->p_begin("duplicate selected " + toString(SUMO_TAG_LANE) + "s");
            std::vector<GNELane*> lanes = myNet->retrieveLanes(true);
            for (auto it : lanes) {
                myNet->duplicateLane(it, myUndoList);
            }
            myUndoList->p_end();
        } else {
            myUndoList->p_begin("duplicate " + toString(SUMO_TAG_LANE));
            myNet->duplicateLane(lane, myUndoList);
            myUndoList->p_end();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdRestrictLaneSidewalk(FXObject*, FXSelector, void*) {
    return restrictLane(SVC_PEDESTRIAN);
}


long
GNEViewNet::onCmdRestrictLaneBikelane(FXObject*, FXSelector, void*) {
    return restrictLane(SVC_BICYCLE);
}


long
GNEViewNet::onCmdRestrictLaneBuslane(FXObject*, FXSelector, void*) {
    return restrictLane(SVC_BUS);
}


long
GNEViewNet::onCmdAddRestrictedLaneSidewalk(FXObject*, FXSelector, void*) {
    return addRestrictedLane(SVC_PEDESTRIAN);
}


long
GNEViewNet::onCmdAddRestrictedLaneBikelane(FXObject*, FXSelector, void*) {
    return addRestrictedLane(SVC_BICYCLE);
}


long
GNEViewNet::onCmdAddRestrictedLaneBuslane(FXObject*, FXSelector, void*) {
    return addRestrictedLane(SVC_BUS);
}


long
GNEViewNet::onCmdRemoveRestrictedLaneSidewalk(FXObject*, FXSelector, void*) {
    return removeRestrictedLane(SVC_PEDESTRIAN);
}


long
GNEViewNet::onCmdRemoveRestrictedLaneBikelane(FXObject*, FXSelector, void*) {
    return removeRestrictedLane(SVC_BICYCLE);
}


long
GNEViewNet::onCmdRemoveRestrictedLaneBuslane(FXObject*, FXSelector, void*) {
    return removeRestrictedLane(SVC_BUS);
}


bool
GNEViewNet::restrictLane(SUMOVehicleClass vclass) {
    GNELane* lane = getLaneAtCurserPosition(myPopupSpot);
    if (lane != 0) {
        // Get selected lanes
        std::vector<GNELane*> lanes = myNet->retrieveLanes(true); ;
        // Declare map of edges and lanes
        std::map<GNEEdge*, GNELane*> mapOfEdgesAndLanes;
        // Iterate over selected lanes
        for (auto i : lanes) {
            mapOfEdgesAndLanes[myNet->retrieveEdge(i->getParentEdge().getID())] = i;
        }
        // Throw warning dialog if there hare multiple lanes selected in the same edge
        if (mapOfEdgesAndLanes.size() != lanes.size()) {
            FXMessageBox::information(getApp(), MBOX_OK,
                                      "Multiple lane in the same edge selected", "%s",
                                      ("There are selected lanes that belong to the same edge.\n Only one lane per edge will be restricted for " + toString(vclass) + ".").c_str());
        }
        // If we handeln a set of lanes
        if (mapOfEdgesAndLanes.size() > 0) {
            // declare counter for number of Sidewalks
            int counter = 0;
            // iterate over selected lanes
            for (auto i : mapOfEdgesAndLanes) {
                if (i.first->hasRestrictedLane(vclass)) {
                    counter++;
                }
            }
            // if all edges parent own a Sidewalk, stop function
            if (counter == (int)mapOfEdgesAndLanes.size()) {
                FXMessageBox::information(getApp(), MBOX_OK,
                                          ("Set vclass for " + toString(vclass) + " to selected lanes").c_str(), "%s",
                                          ("All lanes own already another lane in the same edge with a restriction for " + toString(vclass)).c_str());
                return 0;
            } else {
                if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                    WRITE_WARNING("Opening FXMessageBox 'restrict lanes'");
                }
                // Ask confirmation to user
                FXuint answer = FXMessageBox::question(getApp(), MBOX_YES_NO,
                                                       ("Set vclass for " + toString(vclass) + " to selected lanes").c_str(), "%s",
                                                       (toString(mapOfEdgesAndLanes.size() - counter) + " lanes will be restricted for " + toString(vclass) + ". continue?").c_str());
                if (answer != 1) { //1:yes, 2:no, 4:esc
                    // write warning if netedit is running in testing mode
                    if ((answer == 2) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'No'");
                    } else if ((answer == 4) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'ESC'");
                    }
                    return 0;
                } else {
                    // write warning if netedit is running in testing mode
                    if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'Yes'");
                    }
                }
            }
            // begin undo operation
            myUndoList->p_begin("restrict lanes to " + toString(vclass));
            // iterate over selected lanes
            for (std::map<GNEEdge*, GNELane*>::iterator i = mapOfEdgesAndLanes.begin(); i != mapOfEdgesAndLanes.end(); i++) {
                // Transform lane to Sidewalk
                myNet->restrictLane(vclass, i->second, myUndoList);
            }
            // end undo operation
            myUndoList->p_end();
        } else {
            // If only have a single lane, start undo/redo operation
            myUndoList->p_begin("restrict lane to " + toString(vclass));
            // Transform lane to Sidewalk
            myNet->restrictLane(vclass, lane, myUndoList);
            // end undo operation
            myUndoList->p_end();
        }
    }
    return 1;
}


bool
GNEViewNet::addRestrictedLane(SUMOVehicleClass vclass) {
    GNELane* lane = getLaneAtCurserPosition(myPopupSpot);
    if (lane != 0) {
        // Get selected edges
        std::vector<GNEEdge*> edges = myNet->retrieveEdges(true);
        // get selected lanes
        std::vector<GNELane*> lanes = myNet->retrieveLanes(true);
        // Declare set of edges
        std::set<GNEEdge*> setOfEdges;
        // Fill set of edges with vector of edges
        for (auto i : edges) {
            setOfEdges.insert(i);
        }
        // iterate over selected lanes
        for (auto it : lanes) {
            // Insert pointer to edge into set of edges (To avoid duplicates)
            setOfEdges.insert(myNet->retrieveEdge(it->getParentEdge().getID()));
        }
        // If we handeln a set of edges
        if (setOfEdges.size() > 0) {
            // declare counter for number of restrictions
            int counter = 0;
            // iterate over set of edges
            for (std::set<GNEEdge*>::iterator it = setOfEdges.begin(); it != setOfEdges.end(); it++) {
                // update counter if edge has already a restricted lane of type "vclass"
                if ((*it)->hasRestrictedLane(vclass)) {
                    counter++;
                }
            }
            // if all lanes own a Sidewalk, stop function
            if (counter == (int)setOfEdges.size()) {
                FXMessageBox::information(getApp(), MBOX_OK,
                                          ("Add vclass for" + toString(vclass) + " to selected lanes").c_str(), "%s",
                                          ("All lanes own already another lane in the same edge with a restriction for " + toString(vclass)).c_str());
                return 0;
            } else {
                if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                    WRITE_WARNING("Opening FXMessageBox 'restrict lanes'");
                }
                // Ask confirmation to user
                FXuint answer = FXMessageBox::question(getApp(), MBOX_YES_NO,
                                                       ("Add vclass for " + toString(vclass) + " to selected lanes").c_str(), "%s",
                                                       (toString(setOfEdges.size() - counter) + " restrictions for " + toString(vclass) + " will be added. continue?").c_str());
                if (answer != 1) { //1:yes, 2:no, 4:esc
                    // write warning if netedit is running in testing mode
                    if ((answer == 2) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'No'");
                    } else if ((answer == 4) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'ESC'");
                    }
                    return 0;
                } else {
                    // write warning if netedit is running in testing mode
                    if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'Yes'");
                    }
                }
            }
            // begin undo operation
            myUndoList->p_begin("Add restrictions for " + toString(vclass));
            // iterate over set of edges
            for (std::set<GNEEdge*>::iterator it = setOfEdges.begin(); it != setOfEdges.end(); it++) {
                // add Sidewalk
                myNet->addSRestrictedLane(vclass, *(*it), myUndoList);
            }
            // end undo operation
            myUndoList->p_end();
        } else {
            // If only have a single lane, start undo/redo operation
            myUndoList->p_begin("Add vclass for " + toString(vclass));
            // Add Sidewalk
            myNet->addSRestrictedLane(vclass, lane->getParentEdge(), myUndoList);
            // end undo/redo operation
            myUndoList->p_end();
        }
    }
    return 1;
}


bool
GNEViewNet::removeRestrictedLane(SUMOVehicleClass vclass) {
    GNELane* lane = getLaneAtCurserPosition(myPopupSpot);
    if (lane != 0) {
        // Get selected edges
        std::vector<GNEEdge*> edges = myNet->retrieveEdges(true);
        // get selected lanes
        std::vector<GNELane*> lanes = myNet->retrieveLanes(true);
        // Declare set of edges
        std::set<GNEEdge*> setOfEdges;
        // Fill set of edges with vector of edges
        for (auto i : edges) {
            setOfEdges.insert(i);
        }
        // iterate over selected lanes
        for (auto it : lanes) {
            // Insert pointer to edge into set of edges (To avoid duplicates)
            setOfEdges.insert(myNet->retrieveEdge(it->getParentEdge().getID()));
        }
        // If we handeln a set of edges
        if (setOfEdges.size() > 0) {
            // declare counter for number of restrictions
            int counter = 0;
            // iterate over set of edges
            for (std::set<GNEEdge*>::iterator it = setOfEdges.begin(); it != setOfEdges.end(); it++) {
                // update counter if edge has already a restricted lane of type "vclass"
                if ((*it)->hasRestrictedLane(vclass)) {
                    counter++;
                }
            }
            // if all lanes don't own a Sidewalk, stop function
            if (counter == 0) {
                FXMessageBox::information(getApp(), MBOX_OK,
                                          ("Remove vclass for " + toString(vclass) + " to selected lanes").c_str(), "%s",
                                          ("Selected lanes and edges haven't a restriction for " + toString(vclass)).c_str());
                return 0;
            } else {
                if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                    WRITE_WARNING("Opening FXMessageBox 'restrict lanes'");
                }
                // Ask confirmation to user
                FXuint answer = FXMessageBox::question(getApp(), MBOX_YES_NO,
                                                       ("Remove vclass for " + toString(vclass) + " to selected lanes").c_str(), "%s",
                                                       (toString(counter) + " restrictions for " + toString(vclass) + " will be removed. continue?").c_str());
                if (answer != 1) { //1:yes, 2:no, 4:esc
                    // write warning if netedit is running in testing mode
                    if ((answer == 2) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'No'");
                    } else if ((answer == 4) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'ESC'");
                    }
                    return 0;
                } else {
                    // write warning if netedit is running in testing mode
                    if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'Yes'");
                    }
                }
            }
            // begin undo operation
            myUndoList->p_begin("Remove restrictions for " + toString(vclass));
            // iterate over set of edges
            for (std::set<GNEEdge*>::iterator it = setOfEdges.begin(); it != setOfEdges.end(); it++) {
                // add Sidewalk
                myNet->removeRestrictedLane(vclass, *(*it), myUndoList);
            }
            // end undo operation
            myUndoList->p_end();
        } else {
            // If only have a single lane, start undo/redo operation
            myUndoList->p_begin("Remove vclass for " + toString(vclass));
            // Remove Sidewalk
            myNet->removeRestrictedLane(vclass, lane->getParentEdge(), myUndoList);
            // end undo/redo operation
            myUndoList->p_end();
        }
    }
    return 1;
}


void
GNEViewNet::processClick(FXEvent* e, void* eventData) {
    // process click
    destroyPopup();
    setFocus();
    myChanger->onLeftBtnPress(eventData);
    grab();
    // Check there are double click
    if (e->click_count == 2) {
        handle(this, FXSEL(SEL_DOUBLECLICKED, 0), eventData);
    }
}

long
GNEViewNet::onCmdRevertRestriction(FXObject*, FXSelector, void*) {
    GNELane* lane = getLaneAtCurserPosition(myPopupSpot);
    if (lane != 0) {
        // Declare vector of lanes
        std::vector<GNELane*> lanes;
        // Check if we have a set of selected edges or lanes
        if (gSelected.isSelected(GLO_EDGE, lane->getParentEdge().getGlID())) {
            // Get selected edgeds
            std::vector<GNEEdge*> edges = myNet->retrieveEdges(true);
            // fill vector of lanes with the lanes of selected edges
            for (auto i : edges) {
                for (auto j : i->getLanes()) {
                    lanes.push_back(j);
                }
            }
        } else if (gSelected.isSelected(GLO_LANE, lane->getGlID())) {
            // get selected lanes
            lanes = myNet->retrieveLanes(true);
        }
        // If we handeln a set of lanes
        if (lanes.size() > 0) {
            // declare counter for number of Sidewalks
            int counter = 0;
            // iterate over selected lanes
            for (auto it : lanes) {
                if ((it->isRestricted(SVC_PEDESTRIAN)) || (it->isRestricted(SVC_BICYCLE)) || (it->isRestricted(SVC_BUS))) {
                    counter++;
                }
            }
            // if none of selected lanes has a transformation, stop
            if (counter == 0) {
                FXMessageBox::information(getApp(), MBOX_OK,
                                          "Revert restriction", "%s",
                                          "None of selected lanes has a previous restriction");
                return 0;
            } else {
                if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                    WRITE_WARNING("Opening FXMessageBox 'restrict lanes'");
                }
                // Ask confirmation to user
                FXuint answer = FXMessageBox::question(getApp(), MBOX_YES_NO,
                                                       "Revert restriction", "%s",
                                                       (toString(counter) + " restrictions of lanes lanes will be reverted. continue?").c_str());
                if (answer != 1) { //1:yes, 2:no, 4:esc
                    // write warning if netedit is running in testing mode
                    if ((answer == 2) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'No'");
                    } else if ((answer == 4) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'ESC'");
                    }
                    return 0;
                } else {
                    // write warning if netedit is running in testing mode
                    if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                        WRITE_WARNING("Closed FXMessageBox 'restrict lanes' with 'Yes'");
                    }
                }
            }
            // begin undo operation
            myUndoList->p_begin("revert restrictions");
            // iterate over selected lanes
            for (auto it : lanes) {
                // revert transformation
                myNet->revertLaneRestriction(it, myUndoList);
            }
            // end undo operation
            myUndoList->p_end();
        } else {
            // If only have a single lane, start undo/redo operation
            myUndoList->p_begin("revert restriction");
            // revert transformation
            myNet->revertLaneRestriction(lane, myUndoList);
            // end undo operation
            myUndoList->p_end();
        }
    }
    return 1;
}


long
GNEViewNet::onCmdEditJunctionShape(FXObject*, FXSelector, void*) {
    // Obtain junction under mouse
    GNEJunction* junction = getJunctionAtCursorPosition(myPopupSpot);
    if (junction) {
        startEditShapeJunction(junction);
    }
    return 1;
}


long
GNEViewNet::onCmdReplaceJunction(FXObject*, FXSelector, void*) {
    GNEJunction* junction = getJunctionAtCursorPosition(myPopupSpot);
    if (junction != 0) {
        myNet->replaceJunctionByGeometry(junction, myUndoList);
        update();
    }
    return 1;
}


long
GNEViewNet::onCmdToogleShowConnection(FXObject*, FXSelector, void*) {
    if (!myShowConnections) {
        getNet()->initGNEConnections();
        myShowConnections = true;
    }
    myVisualizationSettings->showLane2Lane = myMenuCheckShowConnections->getCheck() == 1;
    // Update viewnNet to show/hide conections
    update();
    // Hide/show connections requiere recompute
    getNet()->requireRecompute();
    return 1;
}


long
GNEViewNet::onCmdToogleSelectEdges(FXObject*, FXSelector, void*) {
    if (myMenuCheckSelectEdges->getCheck()) {
        mySelectEdges = true;
    } else {
        mySelectEdges = false;
    }
    return 1;
}


long
GNEViewNet::onCmdToogleShowBubbles(FXObject*, FXSelector, void*) {
    // Update view net Shapes
    update();
    return 1;
}


long
GNEViewNet::onCmdShowGrid(FXObject*, FXSelector, void*) {
    // show or hidde grid depending of myMenuCheckShowGrid
    if (myMenuCheckShowGrid->getCheck()) {
        myVisualizationSettings->showGrid = true;
    } else {
        myVisualizationSettings->showGrid = false;
    }
    update();
    return 1;
}


// ===========================================================================
// private
// ===========================================================================

void
GNEViewNet::setEditMode(EditMode mode) {
    setStatusBarText("");
    abortOperation(false);
    // stop editing of shape's junction
    stopEditShapeJunction();

    if (mode == myEditMode) {
        setStatusBarText("Mode already selected");
        if (myCurrentFrame != NULL) {
            myCurrentFrame->focusUpperElement();
        }
    } else {
        myEditMode = mode;
        switch (mode) {
            case GNE_MODE_CONNECT:
            case GNE_MODE_TLS:
                // modes which depend on computed data
                myNet->computeEverything((GNEApplicationWindow*)myApp);
                break;
            default:
                break;
        }
        updateModeSpecificControls();
    }
}


void
GNEViewNet::buildEditModeControls() {
    // initialize mappings
    myEditModeNames.insert("(e) Create Edge", GNE_MODE_CREATE_EDGE);
    myEditModeNames.insert("(m) Move", GNE_MODE_MOVE);
    myEditModeNames.insert("(d) Delete", GNE_MODE_DELETE);
    myEditModeNames.insert("(i) Inspect", GNE_MODE_INSPECT);
    myEditModeNames.insert("(s) Select", GNE_MODE_SELECT);
    myEditModeNames.insert("(c) Connect", GNE_MODE_CONNECT);
    myEditModeNames.insert("(t) Traffic Lights", GNE_MODE_TLS);
    myEditModeNames.insert("(a) Additionals", GNE_MODE_ADDITIONAL);
    myEditModeNames.insert("(r) Crossings", GNE_MODE_CROSSING);
    myEditModeNames.insert("(p) Polygons", GNE_MODE_POLYGON);

    // initialize buttons for modes
    myEditModeCreateEdge = new MFXCheckableButton(false, myToolbar, "\tset create edge mode\tMode for creating junction and edges.",
            GUIIconSubSys::getIcon(ICON_MODECREATEEDGE), this, MID_GNE_MODE_CREATE_EDGE, GUIDesignButtonToolbarCheckable);
    myEditModeMove = new MFXCheckableButton(false, myToolbar, "\tset move mode\tMode for move elements.",
                                            GUIIconSubSys::getIcon(ICON_MODEMOVE), this, MID_GNE_MODE_MOVE, GUIDesignButtonToolbarCheckable);
    myEditModeDelete = new MFXCheckableButton(false, myToolbar, "\tset delete mode\tMode for delete elements.",
            GUIIconSubSys::getIcon(ICON_MODEDELETE), this, MID_GNE_MODE_DELETE, GUIDesignButtonToolbarCheckable);
    myEditModeInspect = new MFXCheckableButton(false, myToolbar, "\tset inspect mode\tMode for inspect elements and change their attributes.",
            GUIIconSubSys::getIcon(ICON_MODEINSPECT), this, MID_GNE_MODE_INSPECT, GUIDesignButtonToolbarCheckable);
    myEditModeSelect = new MFXCheckableButton(false, myToolbar, "\tset select mode\tMode for select elements.",
            GUIIconSubSys::getIcon(ICON_MODESELECT), this, MID_GNE_MODE_SELECT, GUIDesignButtonToolbarCheckable);
    myEditModeConnection = new MFXCheckableButton(false, myToolbar, "\tset connection mode\tMode for edit connections between lanes.",
            GUIIconSubSys::getIcon(ICON_MODECONNECTION), this, MID_GNE_MODE_CONNECT, GUIDesignButtonToolbarCheckable);
    myEditModeTrafficLight = new MFXCheckableButton(false, myToolbar, "\tset traffic light mode\tMode for edit traffic lights over junctions.",
            GUIIconSubSys::getIcon(ICON_MODETLS), this, MID_GNE_MODE_TLS, GUIDesignButtonToolbarCheckable);
    myEditModeAdditional = new MFXCheckableButton(false, myToolbar, "\tset additional mode\tMode for adding additional elements.",
            GUIIconSubSys::getIcon(ICON_MODEADDITIONAL), this, MID_GNE_MODE_ADDITIONAL, GUIDesignButtonToolbarCheckable);
    myEditModeCrossing = new MFXCheckableButton(false, myToolbar, "\tset crossing mode\tMode for creating crossings between edges.",
            GUIIconSubSys::getIcon(ICON_MODECROSSING), this, MID_GNE_MODE_CROSSING, GUIDesignButtonToolbarCheckable);
    myEditModePolygon = new MFXCheckableButton(false, myToolbar, "\tset polygon mode\tMode for creating polygons and POIs.",
            GUIIconSubSys::getIcon(ICON_MODEPOLYGON), this, MID_GNE_MODE_POLYGON, GUIDesignButtonToolbarCheckable);

    // @ToDo add here new FXToolBarGrip(myNavigationToolBar, NULL, 0, GUIDesignToolbarGrip);

    // initialize mode specific controls
    myChainCreateEdge = new FXMenuCheck(myToolbar, ("chain\t\tCreate consecutive " + toString(SUMO_TAG_EDGE) + "s with a single click (hit ESC to cancel chain).").c_str(), this, 0);
    myChainCreateEdge->setCheck(false);

    myAutoCreateOppositeEdge = new FXMenuCheck(myToolbar, ("two-way\t\tAutomatically create an " + toString(SUMO_TAG_EDGE) + " in the opposite direction").c_str(), this, 0);
    myAutoCreateOppositeEdge->setCheck(false);

    myMenuCheckSelectEdges = new FXMenuCheck(myToolbar, ("select edges\t\tToggle whether clicking should select " + toString(SUMO_TAG_EDGE) + "s or " + toString(SUMO_TAG_LANE) + "s").c_str(), this, MID_GNE_VIEWNET_SELECT_EDGES);
    myMenuCheckSelectEdges->setCheck(true);

    myMenuCheckShowConnections = new FXMenuCheck(myToolbar, ("show " + toString(SUMO_TAG_CONNECTION) + "s\t\tToggle show " + toString(SUMO_TAG_CONNECTION) + "s over " + toString(SUMO_TAG_JUNCTION) + "s").c_str(), this, MID_GNE_VIEWNET_SHOW_CONNECTIONS);
    myMenuCheckShowConnections->setCheck(myVisualizationSettings->showLane2Lane);

    myMenuCheckExtendToEdgeNodes = new FXMenuCheck(myToolbar, ("auto-select " + toString(SUMO_TAG_JUNCTION) + "s\t\tToggle whether selecting multiple " + toString(SUMO_TAG_EDGE) + "s should automatically select their " + toString(SUMO_TAG_JUNCTION) + "s").c_str(), this, 0);

    myMenuCheckWarnAboutMerge = new FXMenuCheck(myToolbar, ("ask for merge\t\tAsk for confirmation before merging " + toString(SUMO_TAG_JUNCTION) + ".").c_str(), this, 0);
    myMenuCheckWarnAboutMerge->setCheck(true);

    myMenuCheckShowBubbleOverJunction = new FXMenuCheck(myToolbar, ("Show bubbles over " + toString(SUMO_TAG_JUNCTION) + "s \t\tShow bubbles over " + toString(SUMO_TAG_JUNCTION) + "'s shapes.").c_str(), this, MID_GNE_VIEWNET_SHOW_BUBBLES);
    myMenuCheckShowBubbleOverJunction->setCheck(false);

    myMenuCheckChangeAllPhases = new FXMenuCheck(myToolbar, ("apply change to all phases\t\tToggle whether clicking should apply state changes to all phases of the current " + toString(SUMO_TAG_TRAFFIC_LIGHT) + " plan").c_str(), this, 0);
    myMenuCheckChangeAllPhases->setCheck(false);

    myMenuCheckShowGrid = new FXMenuCheck(myToolbar, "show grid\t\tshow grid with size defined in visualization options", this, MID_GNE_VIEWNET_SHOW_GRID);
    myMenuCheckShowGrid->setCheck(false);
}


void
GNEViewNet::updateModeSpecificControls() {
    // hide grid
    myMenuCheckShowGrid->setCheck(myVisualizationSettings->showGrid);
    // hide all controls (checkboxs)
    myChainCreateEdge->hide();
    myAutoCreateOppositeEdge->hide();
    myMenuCheckSelectEdges->hide();
    myMenuCheckShowConnections->hide();
    myMenuCheckExtendToEdgeNodes->hide();
    myMenuCheckChangeAllPhases->hide();
    myMenuCheckWarnAboutMerge->hide();
    myMenuCheckShowBubbleOverJunction->hide();
    myMenuCheckShowGrid->hide();
    // unckeck all edit modes
    myEditModeCreateEdge->setChecked(false);
    myEditModeMove->setChecked(false);
    myEditModeDelete->setChecked(false);
    myEditModeInspect->setChecked(false);
    myEditModeSelect->setChecked(false);
    myEditModeConnection->setChecked(false);
    myEditModeTrafficLight->setChecked(false);
    myEditModeAdditional->setChecked(false);
    myEditModeCrossing->setChecked(false);
    myEditModePolygon->setChecked(false);
    myViewParent->hideAllFrames();
    // enable selected controls
    switch (myEditMode) {
        case GNE_MODE_CREATE_EDGE:
            myChainCreateEdge->show();
            myAutoCreateOppositeEdge->show();
            myEditModeCreateEdge->setChecked(true);
            myMenuCheckShowGrid->show();
            break;
        case GNE_MODE_MOVE:
            myMenuCheckWarnAboutMerge->show();
            myMenuCheckShowBubbleOverJunction->show();
            myEditModeMove->setChecked(true);
            myMenuCheckShowGrid->show();
            break;
        case GNE_MODE_DELETE:
            myViewParent->getDeleteFrame()->show();
            myViewParent->getDeleteFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getDeleteFrame();
            myMenuCheckShowConnections->show();
            myMenuCheckSelectEdges->show();
            myEditModeDelete->setChecked(true);
            break;
        case GNE_MODE_INSPECT:
            myViewParent->getInspectorFrame()->show();
            myViewParent->getInspectorFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getInspectorFrame();
            myMenuCheckSelectEdges->show();
            myMenuCheckShowConnections->show();
            myEditModeInspect->setChecked(true);
            break;
        case GNE_MODE_SELECT:
            myViewParent->getSelectorFrame()->show();
            myViewParent->getSelectorFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getSelectorFrame();
            myMenuCheckSelectEdges->show();
            myMenuCheckShowConnections->show();
            myMenuCheckExtendToEdgeNodes->show();
            myEditModeSelect->setChecked(true);
            break;
        case GNE_MODE_CONNECT:
            myViewParent->getConnectorFrame()->show();
            myViewParent->getConnectorFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getConnectorFrame();
            myEditModeConnection->setChecked(true);
            break;
        case GNE_MODE_TLS:
            myViewParent->getTLSEditorFrame()->show();
            myViewParent->getTLSEditorFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getTLSEditorFrame();
            myMenuCheckChangeAllPhases->show();
            myEditModeTrafficLight->setChecked(true);
            break;
        case GNE_MODE_ADDITIONAL:
            myViewParent->getAdditionalFrame()->show();
            myViewParent->getAdditionalFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getAdditionalFrame();
            myEditModeAdditional->setChecked(true);
            myMenuCheckShowGrid->show();
            break;
        case GNE_MODE_CROSSING:
            myViewParent->getCrossingFrame()->show();
            myViewParent->getCrossingFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getCrossingFrame();
            myEditModeCrossing->setChecked(true);
            myMenuCheckShowGrid->setCheck(false);
            break;
        case GNE_MODE_POLYGON:
            myViewParent->getPolygonFrame()->show();
            myViewParent->getPolygonFrame()->focusUpperElement();
            myCurrentFrame = myViewParent->getPolygonFrame();
            myEditModePolygon->setChecked(true);
            myMenuCheckShowGrid->show();
            break;
        default:
            break;
    }
    // Update buttons
    myEditModeCreateEdge->update();
    myEditModeMove->update();
    myEditModeDelete->update();
    myEditModeInspect->update();
    myEditModeSelect->update();
    myEditModeConnection->update();
    myEditModeTrafficLight->update();
    myEditModeAdditional->update();
    myEditModeCrossing->update();
    myEditModePolygon->update();
    // force repaint because different modes draw different things
    myToolbar->recalc();
    onPaint(0, 0, 0);
    update();
}


void
GNEViewNet::deleteSelectedJunctions() {
    std::vector<GNEJunction*> junctions = myNet->retrieveJunctions(true);
    if (junctions.size() > 0) {
        std::string plural = junctions.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_JUNCTION) + plural);
        for (auto i : junctions) {
            myNet->deleteJunction(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedLanes() {
    std::vector<GNELane*> lanes = myNet->retrieveLanes(true);
    if (lanes.size() > 0) {
        std::string plural = lanes.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_LANE) + plural);
        for (auto i : lanes) {
            myNet->deleteLane(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedEdges() {
    std::vector<GNEEdge*> edges = myNet->retrieveEdges(true);
    if (edges.size() > 0) {
        std::string plural = edges.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_EDGE) + plural);
        for (auto i : edges) {
            myNet->deleteEdge(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedAdditionals() {
    std::vector<GNEAdditional*> additionals = myNet->retrieveAdditionals(true);
    if (additionals.size() > 0) {
        std::string plural = additionals.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected additional" + plural);
        for (auto i : additionals) {
            getViewParent()->getAdditionalFrame()->removeAdditional(i);
        }
        myUndoList->p_end();
    }
}



void
GNEViewNet::deleteSelectedCrossings() {
    // obtain selected crossings
    std::vector<GNEJunction*> junctions = myNet->retrieveJunctions();
    std::vector<GNECrossing*> crossings;
    for (auto i : junctions) {
        for (auto j : i->getGNECrossings()) {
            if (gSelected.isSelected(GLO_CROSSING, j->getGlID())) {
                crossings.push_back(j);
            }
        }
    }
    // remove selected crossings
    if (crossings.size() > 0) {
        std::string plural = crossings.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_CROSSING) + "s");
        for (auto i : crossings) {
            myNet->deleteCrossing(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedConnections() {
    // obtain selected connections
    std::vector<GNEEdge*> edges = myNet->retrieveEdges();
    std::vector<GNEConnection*> connections;
    for (auto i : edges) {
        for (auto j : i->getGNEConnections()) {
            if (gSelected.isSelected(GLO_CONNECTION, j->getGlID())) {
                connections.push_back(j);
            }
        }
    }
    // remove selected connections
    if (connections.size() > 0) {
        std::string plural = connections.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_CONNECTION) + plural);
        for (auto i : connections) {
            myNet->deleteConnection(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedPOIs() {
    // obtain selected POIs
    std::vector<GNEPOI*> selectedPOIs;
    for (auto i : myNet->getPOIs().getMyMap()) {
        GNEPOI* poi = reinterpret_cast<GNEPOI*>(i.second);
        if (gSelected.isSelected(GLO_POI, poi->getGlID())) {
            selectedPOIs.push_back(poi);
        }
    }
    if (selectedPOIs.size() > 0) {
        std::string plural = selectedPOIs.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_POI) + plural);
        for (auto i : selectedPOIs) {
            myNet->deletePOI(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


void
GNEViewNet::deleteSelectedPolygons() {
    // obtain selected Polygons
    std::vector<GNEPoly*> selectedPolygons;
    for (auto i : myNet->getPolygons().getMyMap()) {
        GNEPoly* Polygon = reinterpret_cast<GNEPoly*>(i.second);
        if (gSelected.isSelected(GLO_POLYGON, Polygon->getGlID())) {
            selectedPolygons.push_back(Polygon);
        }
    }
    if (selectedPolygons.size() > 0) {
        std::string plural = selectedPolygons.size() == 1 ? ("") : ("s");
        myUndoList->p_begin("delete selected " + toString(SUMO_TAG_POLY) + plural);
        for (auto i : selectedPolygons) {
            myNet->deletePolygon(i, myUndoList);
        }
        myUndoList->p_end();
    }
}


bool
GNEViewNet::mergeJunctions(GNEJunction* moved) {
    const Position& newPos = moved->getNBNode()->getPosition();
    GNEJunction* mergeTarget = 0;
    // try to find another junction to merge with
    if (makeCurrent()) {
        Boundary selection;
        selection.add(newPos);
        selection.grow(0.1);
        const std::vector<GUIGlID> ids = getObjectsInBoundary(selection);
        GUIGlObject* object = 0;
        for (auto it_ids : ids) {
            if (it_ids == 0) {
                continue;
            }
            object = GUIGlObjectStorage::gIDStorage.getObjectBlocking(it_ids);
            if (!object) {
                throw ProcessError("Unkown object in selection (id=" + toString(it_ids) + ").");
            }
            if ((object->getType() == GLO_JUNCTION) && (it_ids != moved->getGlID())) {
                mergeTarget = dynamic_cast<GNEJunction*>(object);
            }
            GUIGlObjectStorage::gIDStorage.unblockObject(it_ids);
        }
    }
    if (mergeTarget) {
        // optionally ask for confirmation
        if (myMenuCheckWarnAboutMerge->getCheck()) {
            if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                WRITE_WARNING("Opening FXMessageBox 'merge junctions'");
            }
            // open question box
            FXuint answer = FXMessageBox::question(this, MBOX_YES_NO,
                                                   "Confirm Junction Merger", "%s",
                                                   ("Do you wish to merge junctions '" + moved->getMicrosimID() +
                                                    "' and '" + mergeTarget->getMicrosimID() + "'?\n" +
                                                    "('" + moved->getMicrosimID() +
                                                    "' will be eliminated and its roads added to '" +
                                                    mergeTarget->getMicrosimID() + "')").c_str());
            if (answer != 1) { //1:yes, 2:no, 4:esc
                // write warning if netedit is running in testing mode
                if ((answer == 2) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                    WRITE_WARNING("Closed FXMessageBox 'merge junctions' with 'No'");
                } else if ((answer == 4) && (OptionsCont::getOptions().getBool("gui-testing-debug"))) {
                    WRITE_WARNING("Closed FXMessageBox 'merge junctions' with 'ESC'");
                }
                return false;
            } else {
                // write warning if netedit is running in testing mode
                if (OptionsCont::getOptions().getBool("gui-testing-debug")) {
                    WRITE_WARNING("Closed FXMessageBox 'merge junctions' with 'Yes'");
                }
            }
        }
        myNet->mergeJunctions(moved, mergeTarget, myUndoList);
        return true;
    } else {
        return false;
    }
}


void
GNEViewNet::updateControls() {
    switch (myEditMode) {
        case GNE_MODE_INSPECT:
            myViewParent->getInspectorFrame()->update();
            break;
        default:
            break;
    }
}
/****************************************************************************/
