package jimple.graph;

public class Exist implements AbstractNode {
	private String name = "Exist";
	private NodeType type = NodeType.Entry;
	private int id = 0;
	public Exist(int id) {
		this.id = id;
	}
	public NodeType getType() {
		return this.type;
	}
	public String getTypeName() {
		return this.type.name();
	}
	@Override
	public String toString() {
			return name;
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
		result = prime * result + ((name == null) ? 0 : name.hashCode());
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
		Exist other = (Exist) obj;
		if (id != other.id)
			return false;
		if (name == null) {
			if (other.name != null)
				return false;
		} else if (!name.equals(other.name))
			return false;
		if (type != other.type)
			return false;
		return true;
	}
}
