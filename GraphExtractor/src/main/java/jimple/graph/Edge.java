package jimple.graph;
public class Edge{
	 public String name;
	 public EdgeType type;
	 public String identifier;
	 public AbstractNode starting;
	 public AbstractNode dest;
	 public Edge() {}
	 
	 public Edge(AbstractNode starting, AbstractNode dest, EdgeType type) {
		 this.starting = starting;
		 this.dest = dest;
		 this.type = type;
	 }
	 
	 
	public String getTypeName() {
		return type.name();
	} 
	 
	 @Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((dest == null) ? 0 : dest.hashCode());
		result = prime * result + ((identifier == null) ? 0 : identifier.hashCode());
		result = prime * result + ((name == null) ? 0 : name.hashCode());
		result = prime * result + ((starting == null) ? 0 : starting.hashCode());
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
		Edge other = (Edge) obj;
		if (dest == null) {
			if (other.dest != null)
				return false;
		} else if (!dest.equals(other.dest))
			return false;
		if (identifier == null) {
			if (other.identifier != null)
				return false;
		} else if (!identifier.equals(other.identifier))
			return false;
		if (name == null) {
			if (other.name != null)
				return false;
		} else if (!name.equals(other.name))
			return false;
		if (starting == null) {
			if (other.starting != null)
				return false;
		} else if (!starting.equals(other.starting))
			return false;
		if (type != other.type)
			return false;
		return true;
	}
	 
}
