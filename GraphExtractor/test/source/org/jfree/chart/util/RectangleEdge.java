/* ===========================================================
 * JFreeChart : a free chart library for the Java(tm) platform
 * ===========================================================
 *
 * (C) Copyright 2000-2008, by Object Refinery Limited and Contributors.
 *
 * Project Info:  http://www.jfree.org/jfreechart/index.html
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 *
 * [Java is a trademark or registered trademark of Sun Microsystems, Inc.
 * in the United States and other countries.]
 *
 * -------------
 * RectangleEdge
 * -------------
 * (C) Copyright 2003-2008, by Object Refinery Limited.
 *
 * Original Author:  David Gilbert (for Object Refinery Limited);
 * Contributor(s):   -;
 *
 * Changes:
 * --------
 * 14-Jul-2003 : Version 1(DG);
 * 20-Jun-2007 : New class copied from JCommon (DG);
 *
 */

package org.jfree.chart.util;

import java.awt.geom.Rectangle2D;
import java.io.ObjectStreamException;
import java.io.Serializable;

/**
 * Used to indicate the edge of a rectangle.
 */
public final class RectangleEdge implements Serializable {

    /** For serialization. */
    private static final long serialVersionUID = -7400988293691093548L;

    /** Top. */
    public static final RectangleEdge TOP = new RectangleEdge(
            "RectangleEdge.TOP");

    /** Bottom. */
    public static final RectangleEdge BOTTOM = new RectangleEdge(
            "RectangleEdge.BOTTOM");

    /** Left. */
    public static final RectangleEdge LEFT = new RectangleEdge(
            "RectangleEdge.LEFT");

    /** Right. */
    public static final RectangleEdge RIGHT = new RectangleEdge(
            "RectangleEdge.RIGHT");

    /** The name. */
    private String name;

    /**
     * Private constructor.
     *
     * @param name  the name.
     */
    private RectangleEdge(String name) {
        this.name = name;
    }

    /**
     * Returns a string representing the object.
     *
     * @return The string.
     */
    public String toString() {
        return this.name;
    }

    /**
     * Returns <code>true</code> if this object is equal to the specified
     * object, and <code>false</code> otherwise.
     *
     * @param obj  the other object.
     *
     * @return A boolean.
     */
    public boolean equals(Object obj) {

        if (this == obj) {
            return true;
        }
        if (!(obj instanceof RectangleEdge)) {
            return false;
        }

        RectangleEdge order = (RectangleEdge) obj;
        if (!this.name.equals(order.name)) {
            return false;
        }

        return true;

    }

    /**
     * Returns a hash code value for the object.
     *
     * @return the hashcode
     */
    public int hashCode() {
        return this.name.hashCode();
    }

    /**
     * Returns <code>true</code> if the edge is <code>TOP</code> or
     * <code>BOTTOM</code>, and <code>false</code> otherwise.
     *
     * @param edge  the edge.
     *
     * @return A boolean.
     */
    public static boolean isTopOrBottom(RectangleEdge edge) {
        return (edge == RectangleEdge.TOP || edge == RectangleEdge.BOTTOM);
    }

    /**
     * Returns <code>true</code> if the edge is <code>LEFT</code> or
     * <code>RIGHT</code>, and <code>false</code> otherwise.
     *
     * @param edge  the edge.
     *
     * @return A boolean.
     */
    public static boolean isLeftOrRight(RectangleEdge edge) {
        return (edge == RectangleEdge.LEFT || edge == RectangleEdge.RIGHT);
    }

    /**
     * Returns the opposite edge.
     *
     * @param edge  an edge.
     *
     * @return The opposite edge.
     */
    public static RectangleEdge opposite(RectangleEdge edge) {
        RectangleEdge result = null;
        if (edge == RectangleEdge.TOP) {
            result = RectangleEdge.BOTTOM;
        }
        else if (edge == RectangleEdge.BOTTOM) {
            result = RectangleEdge.TOP;
        }
        else if (edge == RectangleEdge.LEFT) {
            result = RectangleEdge.RIGHT;
        }
        else if (edge == RectangleEdge.RIGHT) {
            result = RectangleEdge.LEFT;
        }
        return result;
    }

    /**
     * Returns the x or y coordinate of the specified edge.
     *
     * @param rectangle  the rectangle.
     * @param edge  the edge.
     *
     * @return The coordinate.
     */
    public static double coordinate(Rectangle2D rectangle,
                                    RectangleEdge edge) {
        double result = 0.0;
        if (edge == RectangleEdge.TOP) {
            result = rectangle.getMinY();
        }
        else if (edge == RectangleEdge.BOTTOM) {
            result = rectangle.getMaxY();
        }
        else if (edge == RectangleEdge.LEFT) {
            result = rectangle.getMinX();
        }
        else if (edge == RectangleEdge.RIGHT) {
            result = rectangle.getMaxX();
        }
        return result;
    }

    /**
     * Ensures that serialization returns the unique instances.
     *
     * @return The object.
     *
     * @throws ObjectStreamException if there is a problem.
     */
    private Object readResolve() throws ObjectStreamException {
        RectangleEdge result = null;
        if (this.equals(RectangleEdge.TOP)) {
            result = RectangleEdge.TOP;
        }
        else if (this.equals(RectangleEdge.BOTTOM)) {
            result = RectangleEdge.BOTTOM;
        }
        else if (this.equals(RectangleEdge.LEFT)) {
            result = RectangleEdge.LEFT;
        }
        else if (this.equals(RectangleEdge.RIGHT)) {
            result = RectangleEdge.RIGHT;
        }
        return result;
    }

}