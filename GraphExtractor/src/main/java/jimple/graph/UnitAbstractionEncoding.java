package jimple.graph;

import org.json.JSONObject;

import soot.Unit;
import soot.jimple.internal.JAssignStmt;
import soot.jimple.internal.JBreakpointStmt;
import soot.jimple.internal.JEnterMonitorStmt;
import soot.jimple.internal.JExitMonitorStmt;
import soot.jimple.internal.JGotoStmt;
import soot.jimple.internal.JIdentityStmt;
import soot.jimple.internal.JIfStmt;
import soot.jimple.internal.JInvokeStmt;
import soot.jimple.internal.JLookupSwitchStmt;
import soot.jimple.internal.JNopStmt;
import soot.jimple.internal.JRetStmt;
import soot.jimple.internal.JReturnStmt;
import soot.jimple.internal.JReturnVoidStmt;
import soot.jimple.internal.JTableSwitchStmt;
import soot.jimple.internal.JThrowStmt;

public class UnitAbstractionEncoding {

	public JSONObject explainUnit(Unit u) {
		if (u instanceof JBreakpointStmt) {

		} else if (u instanceof JIdentityStmt) {

		} else if (u instanceof JAssignStmt) {

		} else if (u instanceof JGotoStmt) {

		} else if (u instanceof JIfStmt) {

		} else if (u instanceof JInvokeStmt) {

		} else if (u instanceof JEnterMonitorStmt) {

		} else if (u instanceof JExitMonitorStmt) {

		} else if (u instanceof JNopStmt) {

		} else if (u instanceof JRetStmt) {

		} else if (u instanceof JReturnStmt) {

		} else if (u instanceof JReturnVoidStmt) {

		} else if (u instanceof JLookupSwitchStmt) {

		} else if (u instanceof JTableSwitchStmt) {

		} else if (u instanceof JThrowStmt) {

		}
		return null;
	}
	
	//public 
	
}
