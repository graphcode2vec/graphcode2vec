package jimple.graph;

import soot.Unit;
import soot.dava.internal.javaRep.DVariableDeclarationStmt;
import soot.jimple.BreakpointStmt;
import soot.jimple.DefinitionStmt;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.MonitorStmt;
import soot.jimple.NopStmt;
import soot.jimple.RetStmt;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.SwitchStmt;
import soot.jimple.ThrowStmt;
import soot.jimple.internal.AbstractStmt;

public class Node implements AbstractNode{
	private Unit node=null;
	private int id=0;
	public NodeType type = NodeType.Default;
	public Node() {}
	public Node(NodeType type, Unit unit) {
		this.type = type;
		this.node = unit;
	}
	public Node(Unit unit, int id) {
		this.node = unit;
		this.getUnitType(unit);
		this.id = id;
	}
	
	public NodeType getType() {
		return this.type;
	}
	public int getID() {
		return this.id;
	}
	@Override
	public String toString() {
		if(node!=null)
		  return node.toString();
		else
			return "";
	}
	
	public NodeType getUnitType(Unit unit) 
	{
		if(unit instanceof IdentityStmt ) {
			this.type = NodeType.IdentityStmt;
		}else if ( unit instanceof AbstractStmt) {
			this.type = NodeType.AbstractStmt;
		}else if ( unit instanceof DVariableDeclarationStmt) {
			this.type = NodeType.DVariableDeclarationStmt;
		}else if ( unit instanceof BreakpointStmt) {
			this.type = NodeType.BreakpointStmt;
		}else if ( unit instanceof DefinitionStmt) {
			this.type = NodeType.DefinitionStmt;
		}else if( unit instanceof GotoStmt) {
			this.type = NodeType.GotoStmt;
		}else if( unit instanceof IfStmt) {
			this.type = NodeType.IfStmt;
		}else if( unit instanceof InvokeStmt) {
			this.type = NodeType.InvokeStmt;
		}else if( unit instanceof MonitorStmt) {
			this.type = NodeType.MonitorStmt;
		}else if( unit instanceof NopStmt) {
			this.type = NodeType.NopStmt;
		}else if( unit instanceof RetStmt) {
			this.type = NodeType.RetStmt;
		}else if( unit instanceof ReturnStmt) {
			this.type = NodeType.ReturnStmt;
		}else if( unit instanceof ReturnVoidStmt) {
			this.type = NodeType.ReturnVoidStmt;
		}else if( unit instanceof SwitchStmt ) {
			this.type = NodeType.SwitchStmt;
		}else if( unit instanceof ThrowStmt) {
			this.type = NodeType.ThrowStmt;
		}else {
			assert false: "Undefined Unit type " + unit.getClass().getName();
		}
		return this.type;
	}
	
	public Unit getUnit() {
		return this.node;
	}
	public void setUnit(Unit unit) {
		 this.node = unit;
	}
	public String getTypeName() {
		return this.type.name();
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + id;
		result = prime * result + ((node == null) ? 0 : node.hashCode());
		result = prime * result + ((type == null) ? 0 : type.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Node other = (Node) obj;
		if (id != other.id)
			return false;
		if (node == null) {
			if (other.node != null)
				return false;
		} else if (!node.equals(other.node))
			return false;
		if (type != other.type)
			return false;
		return true;
	}

}
