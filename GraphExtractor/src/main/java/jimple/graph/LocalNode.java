package jimple.graph;

import soot.Local;

public class LocalNode  implements AbstractNode {
   private Local node = null;
   private NodeType type = NodeType.Declariation;
   private int id=0;
   public LocalNode(Local node, int id) {
	   this.node = node;
	   this.id = id;
   }
   
   @Override
	public String toString() {
			return node.getType().toString() +" "+node.toString();
	}
	
	public NodeType getLocalType() 
	{
		return this.type;
	}



	@Override
	public NodeType getType() {
		// TODO Auto-generated method stub
		return this.type;
	}

	@Override
	public String getTypeName() {
		// TODO Auto-generated method stub
		return this.type.name();
	}

	@Override
	public int getID() {
		// TODO Auto-generated method stub
		return this.id;
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
		LocalNode other = (LocalNode) obj;
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
